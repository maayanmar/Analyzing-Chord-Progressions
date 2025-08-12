import pandas as pd
import re
from collections import Counter
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class ChordLoopDetector(BaseAnalyzer):
    """
    Detects common repeating chord loops (e.g., I–V–vi–IV) in songs.
    """

    def __init__(self, min_loop_len=3, max_loop_len=6):
        self.min_loop_len = min_loop_len
        self.max_loop_len = max_loop_len
        self.loop_counter = Counter()

    def parse_chords_sequence(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [token for token in tokens if not re.match(r'^<.*>$', token)]

    def detect_loops_in_sequence(self, chords):
        loops = []
        for size in range(self.min_loop_len, self.max_loop_len + 1):
            for i in range(len(chords) - size * 2 + 1):
                first = chords[i:i+size]
                second = chords[i+size:i+2*size]
                if first == second:
                    loops.append(tuple(first))
        return loops

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        self.loop_counter.clear()

        for chords in df['chords']:
            sequence = self.parse_chords_sequence(chords)
            loops = self.detect_loops_in_sequence(sequence)
            self.loop_counter.update(loops)

        top_loops = self.loop_counter.most_common(10)

        return {
            "top_loops": [{"loop": " → ".join(loop), "count": count} for loop, count in top_loops],
            "total_loops": sum(self.loop_counter.values())
        }

    def create_visualization(self):
        top_items = self.loop_counter.most_common(10)
        if not top_items:
            return px.scatter(title="No loops found")

        df = pd.DataFrame({
            "Chord Loop": [" → ".join(loop) for loop, _ in top_items],
            "Count": [count for _, count in top_items]
        })

        fig = px.bar(df, x="Count", y="Chord Loop", orientation='h', title="Most Common Repeating Chord Loops")
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def get_report(self):
        top_loops = self.loop_counter.most_common(5)
        return {
            "summary": "Most common repeating chord sequences across selected songs.",
            "examples": [" → ".join(loop) for loop, _ in top_loops]
        }
