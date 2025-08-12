import pandas as pd
import re
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class ChordTransitionHeatmapAnalyzer(BaseAnalyzer):
    """
    Analyzes chord transitions and visualizes them as a heatmap.
    """

    def __init__(self):
        self.transition_matrix = pd.DataFrame()

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

        transitions = {}

        for chord_str in df['chords']:
            chords = self.parse_chords_sequence(chord_str)
            for i in range(len(chords) - 1):
                a, b = chords[i], chords[i + 1]
                if a not in transitions:
                    transitions[a] = {}
                transitions[a][b] = transitions[a].get(b, 0) + 1

        all_chords = sorted(set(a for a in transitions) | set(b for v in transitions.values() for b in v))
        matrix = pd.DataFrame(0, index=all_chords, columns=all_chords)

        for a in transitions:
            for b in transitions[a]:
                matrix.loc[a, b] = transitions[a][b]

        self.transition_matrix = matrix

        return {
            "total_chords": len(matrix),
            "nonzero_transitions": int(matrix.values.sum()),
            "most_common_from": matrix.sum(axis=1).sort_values(ascending=False).head(5).to_dict(),
        }

    def create_visualization(self):
        if self.transition_matrix.empty:
            return px.imshow([[0]], text_auto=True)

        # אותו סדר בצירים + origin='lower'
        top_chords = self.transition_matrix.sum(axis=1).sort_values(ascending=False).head(20).index.tolist()
        trimmed = self.transition_matrix.loc[top_chords, top_chords]

        fig = px.imshow(
            trimmed,
            text_auto=True,
            color_continuous_scale='Viridis',
            labels=dict(x="To", y="From", color="Count"),
            title="Chord Transition Heatmap",
            origin="lower"
        )

        fig.update_layout(
            height=700,
            xaxis=dict(categoryorder="array", categoryarray=top_chords),
            yaxis=dict(categoryorder="array", categoryarray=top_chords)
        )
        return fig

    def get_report(self) -> dict:
        if self.transition_matrix.empty:
            return {"summary": "No data available."}
        top_chords = self.transition_matrix.sum(axis=1).sort_values(ascending=False).head(10)
        return {
            "summary": "Heatmap of the most common chord-to-chord transitions.",
            "top_transitions": top_chords.to_dict()
        }
