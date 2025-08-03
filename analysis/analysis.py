import pandas as pd
from analyzers.base_analyzer import BaseAnalyzer
import re
import plotly.express as px

class ChordStatistics(BaseAnalyzer):
    """
    Analyzes average unique chords per song across genres.
    """

    def __init__(self):
        self.results = None

    def parse_chords(self, chord_str):
        """Helper function to extract actual chords from a string with markers like <chorus_1>."""
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

        grouped = df.groupby('genre')['unique_chords'].mean().reset_index()
        grouped.rename(columns={'unique_chords': 'avg_unique_chords'}, inplace=True)

        self.results = grouped
        return grouped.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='genre',
            y='avg_unique_chords',
            title='Average Unique Chords per Song by Genre',
            labels={'avg_unique_chords': 'Average Unique Chords', 'genre': 'Genre'},
            color='avg_unique_chords',
            color_continuous_scale='Plasma'
        )
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Shows average number of unique chords per song across genres.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }

