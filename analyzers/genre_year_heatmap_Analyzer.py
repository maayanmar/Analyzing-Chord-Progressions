from __future__ import annotations

from typing import Any, Dict, List
import re

import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class GenreYearHeatmap(BaseAnalyzer):
    """
    Heatmap of average unique chords per song by genre and year.
    """

    def __init__(self) -> None:
        """
        Initialize internal results cache.
        """
        self.results: pd.DataFrame | None = None

    def parse_chords_sequence(self, chord_str: str) -> List[str]:
        """
        Parse a chord sequence string into chord tokens.

        Args:
            chord_str: Raw chord sequence.

        Returns:
            List[str]: Parsed chord tokens.
        """
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [t for t in tokens if not re.match(r"^<.*>$", t)]

    def _unique_count(self, chord_str: str) -> int:
        """
        Count unique chords in a single song.
        """
        return len(set(self.parse_chords_sequence(chord_str)))

    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the analysis and cache a table of average unique chords per song,
        grouped by (genre, year).

        Args:
            data: DataFrame with columns: 'year', 'genre', 'chords'.
            genres: Optional list of genres to include. If empty, no genre filter.
            start_year: Inclusive start year for filtering.
            end_year: Inclusive end year for filtering.

        Returns:
            Dict[str, Any]: Serializable summary and table suitable for UI/export:
                - summary: textual description.
                - result_table: [{ 'genre': str, 'year': int, 'avg_unique_chords': float }, ...]
                - cells_count: number of (genre, year) rows in the result.
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        df["unique_chords"] = df["chords"].apply(self._unique_count)

        grouped = (
            df.groupby(["genre", "year"], as_index=False)["unique_chords"]
              .mean()
              .rename(columns={"unique_chords": "avg_unique_chords"})
        )

        self.results = grouped

        result_table = grouped.to_dict(orient="records")
        summary = "A heatmap of the average number of unique chords per song, per genre and year."

        return {
            "summary": summary,
            "result_table": result_table,
            "cells_count": int(len(grouped)),
        }

    def create_visualization(self):
        """
        Build and return a heatmap of avg unique chords by (genre, year).
        """
        if self.results is None or self.results.empty:
            return px.imshow([[0]], title="No Data Available")

        pivot_df = self.results.pivot(index="genre", columns="year", values="avg_unique_chords")

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Year", y="Genre", color="Avg Unique Chords"),
            aspect="auto",
            color_continuous_scale="Blues",
            title="Heatmap: Average Unique Chords by Genre and Year"
        )
        return fig

    def get_report(self) -> Dict[str, Any]:
        """
        Return a serializable summary of the average number of unique chords
        per song by genre and year for export/logging.
        """
        return {
            "summary": "A heatmap of the average number of unique chords per song, per genre and year.",
            "result_table": (
                self.results.to_dict(orient="records") if self.results is not None else []
            ),
        }
