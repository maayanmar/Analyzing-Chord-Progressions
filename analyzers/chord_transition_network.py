import pandas as pd
import re
import networkx as nx
import plotly.graph_objects as go
import community as community_louvain
from analyzers.base_analyzer import BaseAnalyzer

class ChordTransitionNetwork(BaseAnalyzer):
    """
    Builds a chord transition network and applies PageRank and community detection.
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.positions = {}
        self.pagerank_scores = {}
        self.partition = {}

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

        self.G.clear()

        for chords in df['chords']:
            sequence = self.parse_chords_sequence(chords)
            for i in range(len(sequence) - 1):
                a, b = sequence[i], sequence[i + 1]
                if self.G.has_edge(a, b):
                    self.G[a][b]['weight'] += 1
                else:
                    self.G.add_edge(a, b, weight=1)

        # PageRank
        self.pagerank_scores = nx.pagerank(self.G, weight='weight')

        # Community detection (Louvain)
        undirected = self.G.to_undirected()
        self.partition = community_louvain.best_partition(undirected)

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "top_chords": sorted(self.pagerank_scores.items(), key=lambda x: -x[1])[:10]
        }

    def create_visualization(self):
        if not self.G:
            return go.Figure()

        pos = nx.spring_layout(self.G, seed=42, k=0.7)
        self.positions = pos

        node_x, node_y, node_size, node_color, node_text = [], [], [], [], []

        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_size.append(self.pagerank_scores.get(node, 0) * 2000 + 10)
            node_color.append(self.partition.get(node, 0))
            node_text.append(f"{node}<br>PageRank: {self.pagerank_scores.get(node, 0):.4f}")

        edge_x, edge_y = [], []
        for a, b in self.G.edges():
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition='top center',
            hoverinfo='text',
            text=[n for n in self.G.nodes()],
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Community',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=1
            ),
            textfont=dict(size=10)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Chord Transition Network with PageRank & Community Detection',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        return fig

    def get_report(self) -> dict:
        return {
            "summary": "Chord transition network using PageRank and community detection (Louvain).",
            "top_chords": sorted(self.pagerank_scores.items(), key=lambda x: -x[1])[:10]
        }
