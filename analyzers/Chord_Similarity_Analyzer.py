import pandas as pd
from analyzers.base_analyzer import BaseAnalyzer
import numpy as np
import streamlit as st
import plotly.express as px

class ChordSimilarityAnalyzer(BaseAnalyzer):
    """
    Analyzes similarity between chords based on their pitch content.
    """

    def __init__(self):
        self.similarity_matrix = pd.DataFrame()
        self.unique_chords = []
        self.chord_to_notes = {}

    def chord_to_pitch_classes(self, chord):
        """
        Converts a chord string (like 'Cmaj7') into a set of pitch classes.
        """
        import music21
        try:
            parsed = music21.harmony.ChordSymbol(chord)
            return set(p.midi % 12 for p in parsed.pitches)
        except Exception:
            return set()

    def build_similarity_matrix(self, df):
        chords_set = set()
        for chord_str in df["chords"]:
            chords = chord_str.strip().split()
            chords = [c for c in chords if not c.startswith("<")]
            chords_set.update(chords)

        self.unique_chords = sorted(chords_set)
        self.chord_to_notes = {chord: self.chord_to_pitch_classes(chord) for chord in self.unique_chords}

        size = len(self.unique_chords)
        matrix = np.zeros((size, size))

        for i, a in enumerate(self.unique_chords):
            for j, b in enumerate(self.unique_chords):
                if a == b:
                    matrix[i][j] = 1.0
                else:
                    set_a = self.chord_to_notes[a]
                    set_b = self.chord_to_notes[b]
                    if set_a or set_b:
                        intersection = len(set_a & set_b)
                        union = len(set_a | set_b)
                        matrix[i][j] = intersection / union if union > 0 else 0.0

        self.similarity_matrix = pd.DataFrame(matrix, index=self.unique_chords, columns=self.unique_chords)

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        self.build_similarity_matrix(df)

        top_similar = (
            self.similarity_matrix.stack()
            .sort_values(ascending=False)
            .loc[lambda s: s < 1.0]  # Exclude identity
            .head(5)
            .to_dict()
        )

        return {
            "num_unique_chords": len(self.unique_chords),
            "top_similarities": top_similar,
        }

    def create_visualization(self):
        if self.similarity_matrix.empty:
            return px.imshow([[0]], title="No data", labels={"x": "Chord", "y": "Chord"})

        fig = px.imshow(
            self.similarity_matrix,
            title="Chord Similarity Matrix",
            labels={"x": "Chord", "y": "Chord", "color": "Similarity"},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig

    def get_report(self):
        if self.similarity_matrix.empty:
            return {"summary": "No similarity data available."}
        top = (
            self.similarity_matrix.stack()
            .sort_values(ascending=False)
            .loc[lambda s: s < 1.0]
            .head(5)
        )
        return {
            "summary": "Top similar chords based on pitch-class overlap.",
            "examples": [f"{a} ↔ {b}: {s:.2f}" for (a, b), s in top.items()],
        }
