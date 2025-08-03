import pandas as pd
import numpy as np
from analyzers.base_analyzer import BaseAnalyzer
import plotly.express as px

class ChordTensionAnalyzer(BaseAnalyzer):
    """
    Analyzes the harmonic 'tension' in chord progressions.
    Tension is approximated by counting dissonant or extended chords (7th, dim, aug, etc).
    """

    def __init__(self):
        self.tension_scores = {}

    def _calculate_tension(self, chord: str) -> float:
        """Assigns a tension score based on chord type."""
        chord = chord.lower()
        if any(x in chord for x in ["dim", "aug", "sus", "7", "9", "11", "13"]):
            return 2.0
        elif any(x in chord for x in ["m", "min"]):
            return 1.0
        else:
            return 0.5

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]
        df = df.dropna(subset=["chords", "genre"])

        genre_scores = {}

        for genre in df["genre"].unique():
            genre_df = df[df["genre"] == genre]
            scores = []
            for chord_prog in genre_df["chords"]:
                chords = [c for c in chord_prog.split() if not c.startswith("<")]
                tension = sum(self._calculate_tension(c) for c in chords) / max(len(chords), 1)
                scores.append(tension)
            avg_tension = np.mean(scores)
            genre_scores[genre] = avg_tension

        self.tension_scores = genre_scores
        return {
            "tension_scores": genre_scores,
            "num_genres": len(genre_scores),
            "num_songs": len(df),
        }

    def create_visualization(self):
        if not self.tension_scores:
            return px.bar(x=[], y=[], title="No tension data")

        genres = list(self.tension_scores.keys())
        scores = list(self.tension_scores.values())

        fig = px.bar(
            x=genres,
            y=scores,
            title="Average Chord Tension by Genre",
            labels={"x": "Genre", "y": "Avg. Tension"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    def get_report(self):
        if not self.tension_scores:
            return {"summary": "No tension scores calculated."}

        top = sorted(self.tension_scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "summary": "Genres ranked by average harmonic tension (based on extended/dissonant chords).",
            "ranked_genres": top,
        }
