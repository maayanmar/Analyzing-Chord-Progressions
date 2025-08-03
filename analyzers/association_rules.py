import pandas as pd
import re
from collections import Counter
from itertools import islice
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class AssociationRulesAnalyzer(BaseAnalyzer):
    """
    Finds the most common chord n-grams (default 2, i.e. bigrams) across selected genres and years.
    """
    def __init__(self, n=2):
        self.n = n
        self.results = None

    def parse_chords(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [token for token in tokens if not re.match(r'^<.*>$', token)]

    def n_grams(self, sequence, n):
        return zip(*(islice(sequence, i, None) for i in range(n)))

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        transitions = Counter()

        for chords in df['chords']:
            chord_seq = self.parse_chords(chords)
            for gram in self.n_grams(chord_seq, self.n):
                transitions[gram] += 1

        if not transitions:
            self.results = pd.DataFrame()
            return []

        most_common = transitions.most_common(20)
        result_df = pd.DataFrame(most_common, columns=['sequence', 'count'])
        result_df['sequence_str'] = result_df['sequence'].apply(lambda x: ' → '.join(x))
        result_df['percentage'] = result_df['count'] / result_df['count'].sum() * 100

        self.results = result_df
        return result_df.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='sequence_str',
            y='count',
            title=f'Most Common Chord {self.n}-grams',
            labels={'count': 'Frequency', 'sequence_str': 'Chord Sequence'},
            text='percentage',
            color='count',
            color_continuous_scale='Sunsetdark'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def get_report(self) -> dict:
        return {
            "summary": f"Most common chord {self.n}-grams and their frequencies.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
