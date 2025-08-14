from __future__ import annotations

from typing import Any, Dict, List
import re

import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class ChordTransitionHeatmapAnalyzer(BaseAnalyzer):
    """
    Analyzes chord transitions and visualizes them as a heatmap
    expressed as global percentages out of all detected transitions.
    Each cell (a -> b) holds (count(a->b) / sum_all_counts) * 100.
    """

    def __init__(self) -> None:
        """
        Initialize the analyzer and internal transition matrix cache.
        """
        self.transition_matrix: pd.DataFrame = pd.DataFrame()

    def parse_chords_sequence(self, chord_str: str) -> List[str]:
        """
        Parse a chord sequence string into a list of chord tokens.

        Args:
            chord_str: Raw chord sequence.

        Returns:
            List[str]: Parsed chord tokens.
        """
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [t for t in tokens if not re.match(r"^<.*>$", t)]

    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the analysis over the dataset and cache a globally-normalized transition matrix.

        Args:
            data: Input dataframe with at least 'year', 'genre', 'chords' columns.
            genres: Optional list of genres to include. If empty, no genre filter.
            start_year: Inclusive start year for filtering.
            end_year: Inclusive end year for filtering.

        Returns:
            Dict[str, Any]: Summary payload including:
                - total_chords: number of distinct chords (matrix dimension).
                - nonzero_transitions: count of non-zero A->B cells.
                - from_share_all: mapping {FromChord: % of total transitions starting from it},
                  sorted descending.
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        # Count raw transitions A->B
        transitions: Dict[str, Dict[str, int]] = {}
        for chord_str in df["chords"]:
            seq = self.parse_chords_sequence(chord_str)
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if a not in transitions:
                    transitions[a] = {}
                transitions[a][b] = transitions[a].get(b, 0) + 1

        # Build full matrix over union of sources & targets
        all_chords = sorted(set(transitions) | set(t for row in transitions.values() for t in row))
        matrix = pd.DataFrame(0.0, index=all_chords, columns=all_chords, dtype=float)

        for a, row in transitions.items():
            for b, cnt in row.items():
                matrix.at[a, b] = float(cnt)

        # Global normalization to percentages: (cell / total) * 100
        total_transitions = float(matrix.values.sum())
        if total_transitions > 0.0:
            matrix = (matrix / total_transitions) * 100.0

        self.transition_matrix = matrix

        return {
            "total_chords": int(len(self.transition_matrix)),
            "nonzero_transitions": int((self.transition_matrix > 0).sum().sum()),
            "from_share_all": (
                self.transition_matrix.sum(axis=1)
                .sort_values(ascending=False)
                .to_dict()
            ),
        }

    def create_visualization(self):
        """
        Build and return a heatmap of the top-20 chords (by outgoing share)
        in % of total transitions.

        Returns:
            plotly.graph_objects.Figure: A heatmap. If no data is available,
            returns a placeholder figure.
        """
        if self.transition_matrix.empty:
            return px.imshow([[0]], text_auto=True)

        top_chords = (
            self.transition_matrix.sum(axis=1)
            .sort_values(ascending=False)
            .head(20)
            .index
            .tolist()
        )
        trimmed = self.transition_matrix.loc[top_chords, top_chords]

        fig = px.imshow(
            trimmed,
            text_auto=".1f",  # show percentages with one decimal
            color_continuous_scale="Viridis",
            labels=dict(x="To", y="From", color="Percentage"),
            title="Chord Transition Heatmap (global %, out of all transitions)",
            origin="lower"
        )
        fig.update_layout(
            height=700,
            xaxis=dict(categoryorder="array", categoryarray=top_chords),
            yaxis=dict(categoryorder="array", categoryarray=top_chords),
        )
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Full chord-to-chord transition matrix (global % of total transitions).",
            "result_table": (
                self.transition_matrix.reset_index()
                .melt(id_vars=self.transition_matrix.index.name or 'From', var_name='To', value_name='Percentage')
                .query("Percentage > 0")
                .sort_values("Percentage", ascending=False)
                .to_dict(orient='records')
                if not self.transition_matrix.empty else []
            )
    }
