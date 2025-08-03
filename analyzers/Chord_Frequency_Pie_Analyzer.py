import pandas as pd
import re
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class ChordFrequencyPieAnalyzer(BaseAnalyzer):
    """
    Analyzes chord frequency and visualizes as a pie chart.
    """

    def __init__(self):
        self.freq = pd.Series(dtype=int)

    def parse_chords_sequence(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [token for token in tokens if not re.match(r'^<.*>$', token)]

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        chord_counts = {}

        for chord_str in df['chords']:
            chords = self.parse_chords_sequence(chord_str)
            for chord in chords:
                chord_counts[chord] = chord_counts.get(chord, 0) + 1

        self.freq = pd.Series(chord_counts).sort_values(ascending=False)
        return {
            "unique_chords": len(self.freq),
            "most_common_chords": self.freq.head(10).to_dict()
        }

    def create_visualization(self):
        if self.freq.empty:
            return px.pie(names=["None"], values=[1])

        top = self.freq.head(15).reset_index()
        top.columns = ["Chord", "Count"]

        fig = px.pie(
            top,
            names="Chord",
            values="Count",
            title="Top 15 Chords Distribution",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Pie chart showing the most frequent chords in the dataset.",
            "top_chords": self.freq.head(10).to_dict()
        }
