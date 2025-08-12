import pandas as pd
import numpy as np
from analyzers.base_analyzer import BaseAnalyzer
import plotly.express as px
import streamlit as st

class ChordEmbeddingAnalyzer(BaseAnalyzer):
    """
    Computes vector-based embeddings for chords and finds similar chords based on cosine similarity.
    """

    def __init__(self):
        self.embeddings = {}
        self.similarity_matrix = pd.DataFrame()

    def chord_to_vector(self, chord):
        """
        Embed a chord as a binary vector over 12 pitch classes.
        """
        import music21
        try:
            pitches = music21.harmony.ChordSymbol(chord).pitches
            vector = np.zeros(12)
            for p in pitches:
                vector[p.midi % 12] = 1
            return vector
        except Exception:
            return np.zeros(12)

    def compute_embeddings(self, chord_list):
        embeddings = {}
        for chord in chord_list:
            vector = self.chord_to_vector(chord)
            if vector.sum() > 0:
                embeddings[chord] = vector / np.linalg.norm(vector)
        return embeddings

    def cosine_similarity(self, vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def build_similarity_matrix(self, chord_list):
        self.embeddings = self.compute_embeddings(chord_list)
        chords = list(self.embeddings.keys())
        size = len(chords)
        matrix = np.zeros((size, size))

        for i, a in enumerate(chords):
            for j, b in enumerate(chords):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.cosine_similarity(self.embeddings[a], self.embeddings[b])

        self.similarity_matrix = pd.DataFrame(matrix, index=chords, columns=chords)

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        all_chords = set()
        for chord_str in df["chords"]:
            chords = [c for c in chord_str.strip().split() if not c.startswith("<")]
            all_chords.update(chords)

        self.build_similarity_matrix(sorted(all_chords))

        top_similar = (
            self.similarity_matrix.stack()
            .sort_values(ascending=False)
            .loc[lambda s: s < 1.0]
            .head(5)
            .to_dict()
        )

        return {
            "num_chords_embedded": len(self.embeddings),
            "top_similarities": top_similar,
        }

    def create_visualization(self):
        if self.similarity_matrix.empty:
            return px.imshow([[0]], title="No data", labels={"x": "Chord", "y": "Chord"})

        fig = px.imshow(
            self.similarity_matrix,
            title="Chord Embedding Similarity Matrix",
            labels={"x": "Chord", "y": "Chord", "color": "Cosine Similarity"},
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
            "summary": "Top similar chords based on cosine similarity of pitch-class embeddings.",
            "examples": [f"{a} ↔ {b}: {s:.2f}" for (a, b), s in top.items()],
        }
