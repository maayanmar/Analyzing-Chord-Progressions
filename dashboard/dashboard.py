import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Page config
st.set_page_config(
    page_title="Chord Analysis Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        font-family: 'Georgia', serif;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .analyzer-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class DataMetadata:
    """Container for data metadata"""
    genres: List[str]
    min_year: int
    max_year: int


class MetadataLoader:
    """Loads metadata from data file"""

    def __init__(self, data_path: str = "data/processed.csv"):
        self.data_path = data_path
        self.metadata = None

    def load_metadata(self) -> Optional[DataMetadata]:
        """Load metadata from CSV"""
        try:
            df = pd.read_csv(self.data_path)

            # Extract unique genres
            genres = []
            if 'genre' in df.columns:
                genres = sorted(df['genre'].dropna().unique().tolist())

            # Extract year range
            min_year, max_year = 1950, 2024  # defaults
            if 'release_date' in df.columns:
                df['year'] = pd.to_datetime(df['release_date']).dt.year
                min_year = int(df['year'].min())
                max_year = int(df['year'].max())
            elif 'year' in df.columns:
                min_year = int(df['year'].min())
                max_year = int(df['year'].max())

            self.metadata = DataMetadata(genres, min_year, max_year)
            return self.metadata

        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return None


class AnalyzerManager:
    """Manages analyzer discovery and execution"""

    def __init__(self, analyzer_path: str = "analyzers"):
        self.analyzer_path = Path(analyzer_path)
        self.analyzers = {}
        self._discover_analyzers()

    def _discover_analyzers(self):
        """Dynamically discover available analyzers"""
        if not self.analyzer_path.exists():
            st.warning(f"Analyzers directory not found: {self.analyzer_path}")
            return

        # Add analyzers directory to Python path
        sys.path.insert(0, str(self.analyzer_path.parent))

        for file_path in self.analyzer_path.glob("*.py"):
            if file_path.name in ["__init__.py", "base_analyzer.py"]:
                continue

            analyzer_name = file_path.stem
            try:
                spec = importlib.util.spec_from_file_location(analyzer_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find analyzer class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                            hasattr(attr, 'analyze') and
                            hasattr(attr, 'create_visualization') and
                            hasattr(attr, 'get_report') and
                            attr_name != 'BaseAnalyzer'):
                        self.analyzers[analyzer_name] = {
                            'class': attr,
                            'name': analyzer_name.replace('_', ' ').title(),
                            'description': getattr(attr, '__doc__', 'No description available')
                        }
                        break
            except Exception as e:
                st.error(f"Error loading {analyzer_name}: {str(e)}")

    def run_analyzer(self, analyzer_id: str, genres: List[str], start_year: int, end_year: int) -> Tuple[go.Figure, Dict]:
        """Run a single analyzer"""
        if analyzer_id not in self.analyzers:
            raise ValueError(f"Analyzer {analyzer_id} not found")

        analyzer_class = self.analyzers[analyzer_id]['class']
        analyzer = analyzer_class()

        # Load data and run analysis
        data = pd.read_csv("data/processed.csv")
        results = analyzer.analyze(data, genres, start_year, end_year)

        # Get visualization and report
        fig = analyzer.create_visualization()
        report = analyzer.get_report()

        return fig, report


class TabManager:
    """Manages analysis tabs"""

    def __init__(self):
        if 'analysis_tabs' not in st.session_state:
            st.session_state.analysis_tabs = {}
        if 'tab_counter' not in st.session_state:
            st.session_state.tab_counter = 0

    def create_tab_id(self) -> str:
        """Create unique tab ID"""
        st.session_state.tab_counter += 1
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{st.session_state.tab_counter}"

    def add_tab(self, tab_id: str, results: Dict[str, Tuple[go.Figure, Dict]], metadata: Dict):
        """Add new analysis tab"""
        st.session_state.analysis_tabs[tab_id] = {
            'results': results,
            'metadata': metadata,
            'timestamp': datetime.now()
        }

    def render_tabs(self):
        """Render all analysis tabs"""
        if not st.session_state.analysis_tabs:
            return

        st.divider()
        st.header("📊 Analysis Results")

        # Create tabs
        tab_ids = list(st.session_state.analysis_tabs.keys())
        tab_names = [f"Analysis {tid}" for tid in tab_ids]
        tabs = st.tabs(tab_names)

        # Render each tab
        for tab, tab_id in zip(tabs, tab_ids):
            with tab:
                tab_data = st.session_state.analysis_tabs[tab_id]
                self._render_tab_content(tab_data)

    def _render_tab_content(self, tab_data: Dict):
        """Render content of a single tab"""
        # Show metadata
        metadata = tab_data['metadata']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Year Range", f"{metadata['start_year']} - {metadata['end_year']}")
        with col2:
            st.metric("Genres", len(metadata['genres']))
        with col3:
            st.metric("Time", tab_data['timestamp'].strftime("%H:%M:%S"))

        # Show results for each analyzer
        for analyzer_name, (fig, report) in tab_data['results'].items():
            st.subheader(analyzer_name.replace('_', ' ').title())

            # Display visualization with unique key
            if fig:
                # Create unique key using tab timestamp and analyzer name
                unique_key = f"chart_{tab_data['timestamp'].strftime('%H%M%S')}_{analyzer_name}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)

            # Display report
            if report:
                with st.expander("View Report"):
                    for key, value in report.items():
                        st.write(f"**{key}:** {value}")


class DashboardUI:
    """Main dashboard UI class"""

    def __init__(self):
        self.metadata_loader = MetadataLoader()
        self.analyzer_manager = AnalyzerManager()
        self.tab_manager = TabManager()
        self._init_session_state()
        self._load_metadata()

    def _init_session_state(self):
        """Initialize session state variables"""
        if 'selected_analyzers' not in st.session_state:
            st.session_state.selected_analyzers = []
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None

    def _load_metadata(self):
        """Load metadata on startup"""
        if st.session_state.metadata is None:
            st.session_state.metadata = self.metadata_loader.load_metadata()

    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🎵 Chord Analysis Dashboard 🎵</h1>', unsafe_allow_html=True)

    def render_sidebar(self) -> Dict:
        """Render sidebar with parameters"""
        with st.sidebar:
            st.header("⚙️ Analysis Parameters")

            if st.session_state.metadata is None:
                st.error("Failed to load metadata")
                return {}

            metadata = st.session_state.metadata

            # Year range slider
            st.subheader("📅 Time Range")
            year_range = st.slider(
                "Select Year Range",
                min_value=metadata.min_year,
                max_value=metadata.max_year,
                value=(metadata.min_year, metadata.max_year),
                format="%d"
            )

            # Genre selection
            st.subheader("🎸 Genres")

            # Select all button
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All"):
                    st.session_state.selected_genres = metadata.genres.copy()
            with col2:
                if st.button("Clear All"):
                    st.session_state.selected_genres = []

            # Initialize selected genres
            if 'selected_genres' not in st.session_state:
                st.session_state.selected_genres = []

            # Genre checkboxes - 3 per row
            for i in range(0, len(metadata.genres), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(metadata.genres):
                        genre = metadata.genres[i + j]
                        with cols[j]:
                            if st.checkbox(genre, value=genre in st.session_state.selected_genres, key=f"genre_{genre}"):
                                if genre not in st.session_state.selected_genres:
                                    st.session_state.selected_genres.append(genre)
                            else:
                                if genre in st.session_state.selected_genres:
                                    st.session_state.selected_genres.remove(genre)

            # Analyzer selection
            st.divider()
            st.subheader("🔍 Select Analyzers")
            self._render_analyzer_selection()

            return {
                'start_year': year_range[0],
                'end_year': year_range[1],
                'genres': st.session_state.selected_genres
            }

    def _render_analyzer_selection(self):
        """Render analyzer selection UI"""
        if not self.analyzer_manager.analyzers:
            st.warning("No analyzers found")
            return

        for analyzer_id, analyzer_info in self.analyzer_manager.analyzers.items():
            with st.container():
                st.markdown('<div class="analyzer-card">', unsafe_allow_html=True)
                if st.checkbox(analyzer_info['name'], key=f"analyzer_{analyzer_id}"):
                    if analyzer_id not in st.session_state.selected_analyzers:
                        st.session_state.selected_analyzers.append(analyzer_id)
                else:
                    if analyzer_id in st.session_state.selected_analyzers:
                        st.session_state.selected_analyzers.remove(analyzer_id)
                st.caption(analyzer_info['description'])
                st.markdown('</div>', unsafe_allow_html=True)

    def render_main_content(self, params: Dict):
        """Render main content area"""
        if not params:
            return

        # Analysis button
        if st.session_state.selected_analyzers and params['genres']:
            if st.button("🚀 Run Analysis", type="primary"):
                self._run_analysis(params)
        elif not params['genres']:
            st.warning("Please select at least one genre")
        elif not st.session_state.selected_analyzers:
            st.info("Please select at least one analyzer")

        # Display tabs
        self.tab_manager.render_tabs()

    def _run_analysis(self, params: Dict):
        """Run selected analyzers"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = {}
        total_analyzers = len(st.session_state.selected_analyzers)

        for i, analyzer_id in enumerate(st.session_state.selected_analyzers):
            status_text.text(f"Running {analyzer_id}...")
            progress_bar.progress((i + 1) / total_analyzers)

            try:
                # Run analyzer
                fig, report = self.analyzer_manager.run_analyzer(
                    analyzer_id,
                    params['genres'],
                    params['start_year'],
                    params['end_year']
                )
                results[analyzer_id] = (fig, report)
            except Exception as e:
                st.error(f"Error in {analyzer_id}: {str(e)}")

        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")

        # Create new tab
        tab_id = self.tab_manager.create_tab_id()
        self.tab_manager.add_tab(tab_id, results, params)
        st.success(f"✅ Analysis complete!")

    def run(self):
        """Main dashboard entry point"""
        self.render_header()
        params = self.render_sidebar()
        self.render_main_content(params)


# Run the dashboard
if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()