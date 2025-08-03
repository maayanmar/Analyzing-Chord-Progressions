import pandas as pd
import re
from analyzers.base_analyzer import BaseAnalyzer
import plotly.express as px

class ChordComplexityAnalyzer(BaseAnalyzer):
    """
    Computes a chord complexity score per genre or year range.
    Complexity is defined as:
        - Number of unique chords
        - Number of transitions
        - Bonus for complex chord notations (e.g., maj7, dim, aug, sus4)
    """

    def __init__(self):
        self.results = None

    def parse_chords(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        chords = [t for t in tokens if not re.match(r'<.*?>', t)]
        return chords

    def chord_complexity_score(self, chords):
        unique_chords = set(chords)
        num_unique = len(unique_chords)
        transitions = sum(1 for i in range(1, len(chords)) if chords[i] != chords[i-1])
        complex_bonus = sum(1 for c in chords if re.search(r'maj7|dim|aug|sus|add|#|b', c))
        return num_unique + transitions + 0.5 * complex_bonus

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]

        if genres:
            df = df[df['genre'].isin(genres)]

        df['chords_list'] = df['chords'].apply(self.parse_chords)
        df['complexity'] = df['chords_list'].apply(self.chord_complexity_score)

        grouped = df.groupby('genre')['complexity'].mean().reset_index()
        grouped.rename(columns={'complexity': 'avg_complexity_score'}, inplace=True)
        self.results = grouped
        return grouped.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='genre',
            y='avg_complexity_score',
            title='Average Chord Complexity Score by Genre',
            labels={'avg_complexity_score': 'Avg Complexity Score', 'genre': 'Genre'},
            color='avg_complexity_score',
            color_continuous_scale='Magma'
        )
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Calculates average chord complexity score per genre.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
