from __future__ import annotations

from typing import Any, Dict, List, Tuple
from collections import Counter
import re

import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class ChordLoopDetector(BaseAnalyzer):
    """
    Analyzer that detects common repeating chord loops
    that appear as two consecutive, identical segments in a chord sequence.
    """

    def __init__(self, min_loop_len: int = 3, max_loop_len: int = 6) -> None:
        """
        Initialize detector configuration and internal counters.

        Args:
            min_loop_len: Minimum loop length to consider (inclusive).
            max_loop_len: Maximum loop length to consider (inclusive).
        """
        self.min_loop_len = min_loop_len
        self.max_loop_len = max_loop_len
        self.loop_counter: Counter[Tuple[str, ...]] = Counter()

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

    def detect_loops_in_sequence(self, chords: List[str]) -> List[Tuple[str, ...]]:
        """
        Detect immediate repeating loops within a single chord sequence.

        A loop is defined as a segment of length L that repeats twice in a row:
        [x0 ... x{L-1}] [x0 ... x{L-1}].

        Args:
            chords: Parsed list of chord tokens.

        Returns:
            List[Tuple[str, ...]]: All loops found in this sequence (may contain duplicates).
        """
        loops: List[Tuple[str, ...]] = []
        n = len(chords)
        for size in range(self.min_loop_len, self.max_loop_len + 1):
            # require space for two adjacent segments of length 'size'
            for i in range(0, n - size * 2 + 1):
                first = chords[i:i + size]
                second = chords[i + size:i + 2 * size]
                if first == second:
                    loops.append(tuple(first))
        return loops

    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the analysis over the dataset and cache loop frequency results.

        Args:
            data: Input dataframe with at least 'year', 'genre', 'chords' columns.
            genres: Optional list of genres to include. If empty, no genre filter.
            start_year: Inclusive start year for filtering.
            end_year: Inclusive end year for filtering.

        Returns:
            Dict[str, Any]: Summary payload including:
                - total_loops: total number of loop occurrences detected.
                - top_loops: list of the 10 most common loops with their counts.
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        self.loop_counter.clear()

        for chord_str in df["chords"]:
            sequence = self.parse_chords_sequence(chord_str)
            loops = self.detect_loops_in_sequence(sequence)
            self.loop_counter.update(loops)

        top_loops = self.loop_counter.most_common(10)

        return {
            "total_loops": int(sum(self.loop_counter.values())),
            "top_loops": [
                {"loop": " → ".join(loop), "count": count}
                for loop, count in top_loops
            ],
        }

    def create_visualization(self):
        """
        Build and return a horizontal bar chart of the top 10 repeating loops.

        Returns:
            plotly.graph_objects.Figure: A bar chart. If no loops are available,
            returns a placeholder chart with an informative title.
        """
        top_items = self.loop_counter.most_common(10)
        if not top_items:
            return px.scatter(title="No repeating chord loops found")

        viz_df = pd.DataFrame({
            "Chord Loop": [" → ".join(loop) for loop, _ in top_items],
            "Count": [count for _, count in top_items],
        })

        fig = px.bar(
            viz_df,
            x="Count",
            y="Chord Loop",
            orientation="h",
            title="Most Common Repeating Chord Loops"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

       def get_report(self) -> Dict[str, Any]:
        """
        Return a serializable summary of the top 100 detected loops for export/logging.
    
        Returns:
            Dict[str, Any]:
                - summary: textual description.
                - top_100_loops: mapping of the top 100 loops (as strings) to their counts.
        """
        top_items = self.loop_counter.most_common(100)
        top_100_loops: Dict[str, int] = {" → ".join(loop): count for loop, count in top_items}
    
        return {
            "summary": "Top 100 repeating chord loops across selected songs.",
            "top_100_loops": top_100_loops,
        }
