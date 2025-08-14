from __future__ import annotations

from typing import Any, Dict, List
import re

import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class ChordCountByGenre(BaseAnalyzer):
    """
    Calculates the average number of unique chords per song for each genre.
    """

    def __init__(self) -> None:
        """
        Initialize internal results cache.
        """
        self.results: pd.DataFrame | None = None

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

    def count_unique_chords(self, chord_str: str) -> int:
        """
        Count the number of unique chord tokens (after filtering markup).
        """
        chords = self.parse_chords_sequence(chord_str)
        return len(set(chords))

    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the analysis and cache a table of average *unique* chords per song by genre.

        Args:
            data: DataFrame with columns: 'year', 'genre', 'chords'.
            genres: Optional list of genres to include. If empty, no genre filter.
            start_year: Inclusive start year for filtering.
            end_year: Inclusive end year for filtering.

        Returns:
            Dict[str, Any]: Serializable summary and table suitable for UI/export:
                - summary: textual description.
                - result_table: [{ 'genre': str, 'avg_unique_chords_per_song': float }, ...]
                - genres_count: number of genres in the result table.
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        df["unique_chord_count"] = df["chords"].apply(self.count_unique_chords)

        grouped = (
            df.groupby("genre", as_index=False)["unique_chord_count"]
              .mean()
              .rename(columns={"unique_chord_count": "avg_unique_chords_per_song"})
        )

        self.results = grouped

        result_table = grouped.to_dict(orient="records")
        summary = "Average number of unique chords per song for each genre."

        return {
            "summary": summary,
            "result_table": result_table,
            "genres_count": int(len(grouped)),
        }

    def create_visualization(self):
        """
        Build and return a bar chart of average unique chords per song by genre.
        """
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x="genre",
            y="avg_unique_chords_per_song",
            title="Average Number of Unique Chords per Song by Genre",
            labels={"avg_unique_chords_per_song": "Avg Unique Chords per Song", "genre": "Genre"},
            color="avg_unique_chords_per_song",
            color_continuous_scale="Blues",
        )
        return fig

    def get_report(self) -> Dict[str, Any]:
        """
        Return a serializable summary of the average number of unique
        chords per song by genre for export/logging.
        """
        summary = "Average number of unique chords per song for each genre."
        result_table = (
            self.results.to_dict(orient="records") if self.results is not None else []
        )
        return {
            "summary": summary,
            "result_table": result_table,
            "genres_count": int(len(self.results)) if self.results is not None else 0,
        }
