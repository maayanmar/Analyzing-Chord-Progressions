from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import plotly.express as px
from analyzers.base_analyzer import BaseAnalyzer


class ChordSimilarityAnalyzer(BaseAnalyzer):
    """
    Analyzes similarity between chords based on their pitch content.
    """

    def __init__(self) -> None:
        """
        Initialize caches for the similarity matrix and chord metadata.
        """
        self.similarity_matrix: pd.DataFrame = pd.DataFrame()
        self.unique_chords: List[str] = []
        self.chord_to_notes: Dict[str, set[int]] = {}

    # -------- Parsing & pitch extraction --------

    def parse_chords_sequence(self, chord_str: str) -> List[str]:
        """
        Parse a chord sequence string into a list of chord tokens.
        Args:
            chord_str: Raw chord sequence.

        Returns:
            List[str]: Parsed chord tokens.
        """
        if not isinstance(chord_str, str):
            return []
        tokens = chord_str.strip().split()
        return [t for t in tokens if not re.match(r"^<.*>$", t)]

    def chord_to_pitch_classes(self, chord: str) -> set[int]:
        """
        Convert a chord symbol (e.g., 'Cmaj7') to a set of pitch classes (0–11).
        Uses music21's harmony parser; returns an empty set if parsing fails.
        """
        try:
            import music21  # local import to avoid hard dependency on import time
            parsed = music21.harmony.ChordSymbol(chord)
            return {p.midi % 12 for p in parsed.pitches}
        except Exception:
            return set()

    # -------- Core computation --------

    def build_similarity_matrix(self, df: pd.DataFrame) -> None:
        """
        Build a chord-by-chord similarity matrix over all unique chords
        found in the given dataframe (after filtering markup).
        Similarity is Jaccard over pitch sets.
        """
        chords_set: set[str] = set()
        for chord_str in df["chords"]:
            for c in self.parse_chords_sequence(chord_str):
                chords_set.add(c)

        self.unique_chords = sorted(chords_set)
        self.chord_to_notes = {
            chord: self.chord_to_pitch_classes(chord) for chord in self.unique_chords
        }

        n = len(self.unique_chords)
        matrix = np.zeros((n, n), dtype=float)

        for i, a in enumerate(self.unique_chords):
            set_a = self.chord_to_notes[a]
            for j, b in enumerate(self.unique_chords):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    set_b = self.chord_to_notes[b]
                    if set_a or set_b:
                        inter = len(set_a & set_b)
                        union = len(set_a | set_b)
                        matrix[i, j] = (inter / union) if union > 0 else 0.0
                    else:
                        matrix[i, j] = 0.0

        self.similarity_matrix = pd.DataFrame(
            matrix, index=self.unique_chords, columns=self.unique_chords
        )

    def _top_unique_pairs(self, k: int = 25) -> List[Dict[str, Any]]:
        """
        Extract top-25 most similar unique chord pairs (i<j), excluding identity.
        Returns a list of dicts: {'ChordA', 'ChordB', 'Similarity'} sorted desc.
        """
        if self.similarity_matrix.empty:
            return []

        chords = self.similarity_matrix.index.tolist()
        vals: List[Tuple[str, str, float]] = []

        # take upper triangle only (i < j) to avoid duplicates and exclude diagonal
        for i in range(len(chords)):
            for j in range(i + 1, len(chords)):
                a, b = chords[i], chords[j]
                s = float(self.similarity_matrix.iat[i, j])
                if s < 1.0:  # exclude perfect self-similarity (diagonal); i<j already excludes it
                    vals.append((a, b, s))

        vals.sort(key=lambda t: t[2], reverse=True)
        top = vals[:k]
        return [{"ChordA": a, "ChordB": b, "Similarity": round(s, 4)} for a, b, s in top]

    # -------- Public API --------

    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the similarity analysis:
          1) filter by years/genres
          2) build a chord similarity matrix (Jaccard over pitch classes)
          3) return top 25 most similar chord pairs (unique pairs only)

        Returns:
            Dict[str, Any]: {
                'num_unique_chords': int,
                'top_similarities': List[{ChordA, ChordB, Similarity}],
            }
        """
        df = data.copy()
        df = df[df["year"].between(start_year, end_year)]
        if genres:
            df = df[df["genre"].isin(genres)]

        # Build similarity matrix over chords actually present in the filtered data
        self.build_similarity_matrix(df)

        return {
            "num_unique_chords": int(len(self.unique_chords)),
            "top_similarities": self._top_unique_pairs(),  # default = 25
        }

    def create_visualization(self):
        """
        Build and return a heatmap of the chord similarity matrix.
        """
        if self.similarity_matrix.empty:
            return px.imshow([[0]], title="No data", labels={"x": "Chord", "y": "Chord"})

        order = self.similarity_matrix.index.tolist()
        fig = px.imshow(
            self.similarity_matrix.loc[order, order],
            title="Chord Similarity Matrix",
            labels={"x": "Chord", "y": "Chord", "color": "Similarity"},
            origin="lower"
        )
        fig.update_layout(
            xaxis=dict(categoryorder="array", categoryarray=order),
            yaxis=dict(categoryorder="array", categoryarray=order),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def get_report(self) -> Dict[str, Any]:
        """
        Return a serializable summary and a flat table of the most similar pairs.
        """
        if self.similarity_matrix.empty:
            return {"summary": "No similarity data available.", "result_table": []}

        top = self._top_unique_pairs()  # default = 25
        
        summary = "Top 25 similar chords based on pitch-class overlap."
        return {
            "summary": summary,
            "result_table": top,  # [{ChordA, ChordB, Similarity}, ...]
        }
