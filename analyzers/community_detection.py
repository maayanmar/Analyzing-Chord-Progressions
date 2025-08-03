import pandas as pd
from analyzers.base_analyzer import BaseAnalyzer
import networkx as nx
import re
import plotly.express as px
from networkx.algorithms.community import greedy_modularity_communities

class CommunityDetection(BaseAnalyzer):
    """
    Detects communities of chords that frequently appear in progression networks.
    """

    def __init__(self):
        self.results = None

    def parse_transitions(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        chords = [token for token in tokens if not re.match(r'^<.*>$', token)]
        return list(zip(chords, chords[1:]))

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]

        if genres:
            df = df[df['genre'].isin(genres)]

        transitions = []
        for chords in df['chords']:
            transitions.extend(self.parse_transitions(chords))

        G = nx.Graph()
        for source, target in transitions:
            if G.has_edge(source, target):
                G[source][target]['weight'] += 1
            else:
                G.add_edge(source, target, weight=1)

        # זיהוי קהילות באמצעות greedy modularity
        communities = list(greedy_modularity_communities(G))
        flat_data = []
        for idx, community in enumerate(communities):
            for chord in community:
                flat_data.append({'chord': chord, 'community': idx})

        self.results = pd.DataFrame(flat_data)
        return self.results.to_dict(orient='records')

    def create_visualization(self):
        if self.results is None or self.results.empty:
            return px.bar(title="No Data Available")

        chord_counts = self.results['community'].value_counts().reset_index()
        chord_counts.columns = ['community', 'num_chords']

        fig = px.bar(
            chord_counts,
            x='community',
            y='num_chords',
            title='Chord Communities (Community Size)',
            labels={'community': 'Community ID', 'num_chords': 'Number of Chords'},
            color='num_chords',
            color_continuous_scale='Cividis'
        )
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Detects clusters of chords that tend to co-occur using modularity-based community detection.",
            "result_table": self.results.to_dict(orient='records') if self.results is not None else []
        }
