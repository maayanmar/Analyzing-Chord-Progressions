import pandas as pd
import re
from collections import defaultdict, Counter
from itertools import tee, islice
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class AssociationRulesAnalyzer(BaseAnalyzer):
    """
    Finds the most common chord transitions across selected genres and years.
    """
    def __init__(self):
        self.results = None

    def parse_chords(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [token for token in tokens if not re.match(r'^<.*>$', token)]

    def pairwise(self, iterable):
        """Generates consecutive chord pairs"""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        transitions = Counter()

        for chords in df['chords']:
            chord_seq = self.parse_chords(chords)
            for a, b in self.pairwise(chord_seq):
                transitions[(a, b)] += 1

        if not transitions:
            self.results = pd.DataFrame()
            return []

        most_common = transitions.most_common(20)
        result_df = pd.DataFrame(most_common, columns=['pair', 'count'])
        result_df['from'] = result_df['pair'].apply(lambda x: x[0])
        result_df['to'] = result_df['pair'].apply(lambda x: x[1])
        result_df['percentage'] = result_df['count'] / result_df['count'].sum() * 100

        self.results = result_df
        return result_df.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='pair',
            y='count',
            title='Most Common Chord Transitions',
            labels={'count': 'Frequency', 'pair': 'Chord Transition'},
            text='percentage',
            color='count',
            color_continuous_scale='Sunsetdark'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Most common chord transitions and their frequencies.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
