from abc import ABC, abstractmethod
import pandas as pd
import plotly.graph_objects as go

class BaseAnalyzer(ABC):
    """
    Base class for all chord analysis modules.
    Each analyzer must implement the analyze, create_visualization, and get_report methods.
    """

    @abstractmethod
    def analyze(self, data: pd.DataFrame, genres: list, start_year: int, end_year: int) -> dict:
        """
        Perform the analysis.

        Parameters:
        - data: The full DataFrame containing the dataset.
        - genres: A list of genres selected by the user.
        - start_year: The starting year for filtering.
        - end_year: The ending year for filtering.

        Returns:
        - A dictionary or list of results to be displayed in the dashboard.
        """
        pass

    @abstractmethod
    def create_visualization(self) -> go.Figure:
        """
        Create a Plotly visualization of the results.

        Returns:
        - A plotly.graph_objects.Figure or plotly.express chart.
        """
        pass

    @abstractmethod
    def get_report(self) -> dict:
        """
        Generate a report summary of the analysis.

        Returns:
        - A dictionary containing summary information and raw results.
        """
        pass
