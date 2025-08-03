import pandas as pd
import re
import networkx as nx
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class PageRankAnalyzer(BaseAnalyzer):
    """
    Computes PageRank over chord transitions to identify central chords.
    """

    def __init__(self):
        self.results = None

    def extract_chord_transitions(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        # Remove section markers
        tokens = [t for t in chord_str.strip().split() if not re.match(r'^<.*>$', t)]
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        transitions = []
        for chords in df['chords']:
            transitions.extend(self.extract_chord_transitions(chords))

        G = nx.DiGraph()
        G.add_edges_from(transitions)

        pagerank_scores = nx.pagerank(G)

        df_scores = pd.DataFrame(pagerank_scores.items(), columns=['chord', 'pagerank'])
        df_scores.sort_values(by='pagerank', ascending=False, inplace=True)

        self.results = df_scores.head(20)
        return self.results.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        fig = px.bar(
            self.results,
            x='chord',
            y='pagerank',
            title='Top Chords by PageRank Score',
            labels={'pagerank': 'PageRank Score', 'chord': 'Chord'},
            color='pagerank',
            color_continuous_scale='Blues'
        )
        return fig

    def get_report(self):
        return {
            "summary": "Shows the most central chords based on PageRank over chord transitions.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
