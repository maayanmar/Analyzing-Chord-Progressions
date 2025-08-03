import pandas as pd
import numpy as np
from analyzers.base_analyzer import BaseAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import plotly.express as px

class GenreClassifierByChords(BaseAnalyzer):
    """
    Classifies genres based on chord sequences using Naive Bayes.
    """

    def __init__(self):
        self.pipeline = None
        self.vectorizer = None
        self.genres = []
        self.accuracy = None

    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]
        df = df.dropna(subset=["chords", "genre"])
        self.genres = df["genre"].unique().tolist()

        # Convert list of chords to space-separated strings
        df["chord_text"] = df["chords"].apply(lambda s: " ".join([c for c in s.split() if not c.startswith("<")]))

        # Train classifier pipeline
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer(analyzer="word")),
            ("classifier", MultinomialNB())
        ])
        self.pipeline.fit(df["chord_text"], df["genre"])

        # Cross-validation accuracy
        self.accuracy = np.mean(
            cross_val_score(self.pipeline, df["chord_text"], df["genre"], cv=5)
        )

        return {
            "accuracy": round(self.accuracy * 100, 2),
            "genre_count": dict(df["genre"].value_counts()),
            "num_songs": len(df),
        }

    def create_visualization(self):
        if not self.pipeline:
            return px.bar(x=[], y=[], title="No data")

        feature_names = self.pipeline.named_steps["vectorizer"].get_feature_names_out()
        class_labels = self.pipeline.named_steps["classifier"].classes_
        log_probs = self.pipeline.named_steps["classifier"].feature_log_prob_

        # Get top 5 chords for each genre
        top_chords = {}
        for i, genre in enumerate(class_labels):
            top = np.argsort(log_probs[i])[::-1][:5]
            top_chords[genre] = [feature_names[j] for j in top]

        # Flatten for display
        genre_list = []
        chord_list = []
        for genre, chords in top_chords.items():
            for chord in chords:
                genre_list.append(genre)
                chord_list.append(chord)

        fig = px.bar(
            x=chord_list,
            y=genre_list,
            orientation="h",
            title="Top Chords per Genre",
            labels={"x": "Chord", "y": "Genre"},
        )
        fig.update_layout(height=400)
        return fig

    def get_report(self):
        if not self.pipeline:
            return {"summary": "No genre classifier has been trained."}

        return {
            "summary": f"Classifier trained on chord progressions with ~{self.accuracy:.2%} accuracy.",
            "genres": self.genres,
        }
