import pandas as pd
import re
from collections import Counter
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer

class NGramAnalyzer(BaseAnalyzer):
    """
    Analyzes chord progressions using N-grams.
    """

    def __init__(self, n=3):
        self.n = n
        self.ngram_counter = Counter()

    def parse_chords_sequence(self, chord_str):
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [token for token in tokens if not re.match(r'^<.*>$', token)]

    def generate_ngrams(self, chords):
        return [tuple(chords[i:i + self.n]) for i in range(len(chords) - self.n + 1)]

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df['year'].between(start_year, end_year)]
        if genres:
            df = df[df['genre'].isin(genres)]

        self.ngram_counter.clear()

        for chords in df['chords']:
            sequence = self.parse_chords_sequence(chords)
            ngrams = self.generate_ngrams(sequence)
            self.ngram_counter.update(ngrams)

        top_ngrams = self.ngram_counter.most_common(15)

        return {
            "top_ngrams": [{"sequence": " → ".join(ng), "count": count} for ng, count in top_ngrams],
            "total_unique_ngrams": len(self.ngram_counter)
        }

    def create_visualization(self):
        top_items = self.ngram_counter.most_common(15)
        df = pd.DataFrame({
            "Progression": [" → ".join(ng) for ng, _ in top_items],
            "Count": [count for _, count in top_items]
        })

        fig = px.treemap(df, path=['Progression'], values='Count',
                         title=f"Top {self.n}-gram Chord Progressions")

        return fig

    def get_report(self) -> dict:
        top_ngrams = self.ngram_counter.most_common(10)
        return {
            "summary": f"Top {self.n}-gram chord sequences across selected songs.",
            "examples": [" → ".join(ng) for ng, _ in top_ngrams]
        }
