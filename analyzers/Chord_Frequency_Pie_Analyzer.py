from __future__ import annotations

from typing import Any, Dict, List

import re
import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class ChordFrequencyPieAnalyzer(BaseAnalyzer):
    """
    Analyzer that computes chord frequency statistics and renders a pie chart
    of the most frequent chords in the filtered dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the analyzer and internal frequency cache.
        """
        self.freq: pd.Series = pd.Series(dtype=int)

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
        Run the analysis over the dataset and cache frequency results.

        Args:
            data: Input dataframe with at least 'year', 'genre', 'chords' columns.
            genres: Optional list of genres to include. If empty, no genre filter.
            start_year: Inclusive start year for filtering.
            end_year: Inclusive end year for filtering.

        Returns:
            Dict[str, Any]: Summary payload including the total unique chord count
            and a full mapping of all chords to their frequencies.
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        chord_counts: Dict[str, int] = {}

        for chord_str in df["chords"]:
            chords = self.parse_chords_sequence(chord_str)
            for chord in chords:
                chord_counts[chord] = chord_counts.get(chord, 0) + 1

        self.freq = pd.Series(chord_counts).sort_values(ascending=False)

        return {
            "unique_chords": int(len(self.freq)),
            "all_chords": self.freq.to_dict()
        }

    def create_visualization(self):
        """
        Build and return a pie chart showing the 15 most frequent chords.
        Any additional chords beyond the top 15 are aggregated
        into a single 'Other' slice.

        Returns:
            plotly.graph_objects.Figure: A pie chart. If no data is
            available, returns a placeholder pie with a single 'None' slice.
        """
        if self.freq.empty:
            return px.pie(names=["None"], values=[1])

        # Top 15
        top = self.freq.head(15).copy()

        # Aggregate the rest under "Other"
        other_count = int(self.freq.iloc[15:].sum())
        if other_count > 0:
            top.loc["Other"] = other_count

        top_df = top.reset_index()
        top_df.columns = ["Chord", "Count"]

        fig = px.pie(
            top_df,
            names="Chord",
            values="Count",
            title="Top 15 Chords Distribution (with Others)",
            hole=0.3
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def get_report(self) -> Dict[str, Any]:
        """
        Return a full, serializable summary of the analysis for export/logging.

        Returns:
            Dict[str, Any]: Textual summary and a full mapping of all chords
            to their frequencies.
        """
        return {
            "summary": "Full list of chords with their frequencies in the dataset.",
            "all_chords": self.freq.to_dict()
        }
