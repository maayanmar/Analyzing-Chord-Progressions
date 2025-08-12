import pandas as pd
import re
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class ChordCountByGenre(BaseAnalyzer):
    """
    Calculates the average number of chords per song for each genre.
    """

    def __init__(self):
        self.results = None

    def count_chords(self, chord_str):
        if not isinstance(chord_str, str):
            return 0
        tokens = chord_str.strip().split()
        chords_only = [token for token in tokens if not re.match(r'^<.*>$', token)]
        return len(chords_only)

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]

        if genres:
            df = df[df['genre'].isin(genres)]

        df['chord_count'] = df['chords'].apply(self.count_chords)

        grouped = df.groupby('genre')['chord_count'].mean().reset_index()
        grouped.rename(columns={'chord_count': 'avg_chords_per_song'}, inplace=True)

        self.results = grouped
        return grouped.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='genre',
            y='avg_chords_per_song',
            title='Average Number of Chords per Song by Genre',
            labels={'avg_chords_per_song': 'Avg Chords per Song', 'genre': 'Genre'},
            color='avg_chords_per_song',
            color_continuous_scale='Blues'
        )
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Shows average number of total (not unique) chords per song for each genre.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
