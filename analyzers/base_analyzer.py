from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go


class BaseAnalyzer(ABC):
    """
    Base class for all chord-analysis modules.
    Subclasses must implement: analyze, create_visualization, get_report.
    """

    # ---- Abstract API ----
    @abstractmethod
    def analyze(
        self,
        data: pd.DataFrame,
        genres: List[str],
        start_year: int,
        end_year: int
    ) -> Dict[str, Any]:
        """
        Run the analysis over the dataset and
        cache any results needed for visualization/report.

        Returns:
            Dict[str, Any]: The structured payload needed by the dashboard's panel.
        """
        pass

    @abstractmethod
    def create_visualization(self) -> go.Figure:
        """
        Build a Plotly figure from the analyzer's cached results.
        Called after `analyze`.
        """
        pass

    @abstractmethod
    def get_report(self) -> Dict[str, Any]:
        """
        Return a short, serializable summary of the analysis for export/logging.
        """
        pass
