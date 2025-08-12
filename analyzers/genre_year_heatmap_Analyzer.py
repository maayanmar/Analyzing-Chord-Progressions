import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer
import re

class GenreYearHeatmap(BaseAnalyzer):
    """
    Heatmap of average unique chords per song by genre and year.
    """

    def __init__(self):
        self.results = None

    def parse_chords(self, chord_str):
        if not isinstance(chord_str, str):
            return set()
        tokens = chord_str.strip().split()
        chords_only = [token for token in tokens if not re.match(r'^<.*>$', token)]
        return set(chords_only)

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]

        if genres:
            df = df[df['genre'].isin(genres)]

        df['unique_chords'] = df['chords'].apply(lambda x: len(self.parse_chords(x)))

        grouped = df.groupby(['genre', 'year'])['unique_chords'].mean().reset_index()
        grouped.rename(columns={'unique_chords': 'avg_unique_chords'}, inplace=True)

        self.results = grouped
        return grouped.to_dict(orient='records')

    def create_visualization(self):
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

    def get_report(self) -> dict:
        return {
            "summary": "Displays a heatmap of the average number of unique chords per song, per genre and year.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
