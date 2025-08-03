import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import glob
import importlib.util
from datetime import datetime
from typing import Dict, List, Any
import sys
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Chord Analysis Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for musical theme
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if 'selected_analyzers' not in st.session_state:
    st.session_state.selected_analyzers = []

# Helper functions
def discover_analyzers():
    """Dynamically discover available analyzers"""
    analyzers = {}
    analyzer_path = Path("../analyzers")

    # Add analyzers directory to Python path
    if analyzer_path.exists():
        sys.path.insert(0, str(analyzer_path.parent))

        for file_path in analyzer_path.glob("*.py"):
            if file_path.name != "__init__.py" and file_path.name != "base_analyzer.py":
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
                                attr_name != 'BaseAnalyzer'):
                            analyzers[analyzer_name] = {
                                'class': attr,
                                'name': analyzer_name.replace('_', ' ').title(),
                                'description': getattr(attr, '__doc__', 'No description available')
                            }
                            break
                except Exception as e:
                    st.error(f"Error loading {analyzer_name}: {str(e)}")

    return analyzers

def load_data(file_path):
    """Load chord data from CSV"""
    try:
        # Simulate loading with sample data structure
        data = pd.DataFrame({
            'artist': ['Artist1', 'Artist2', 'Artist3'] * 100,
            'song': ['Song1', 'Song2', 'Song3'] * 100,
            'release_date': pd.date_range('2000-01-01', periods=300, freq='M'),
            'genre': ['pop', 'rock', 'jazz'] * 100,
            'chords': ['C G Am F', 'G D Em C', 'Cmaj7 Dm7 G7 Cmaj7'] * 100
        })
        data['year'] = data['release_date'].dt.year
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def save_analysis(analysis_id, results, metadata):
    """Save analysis results"""
    if 'saved_analyses' not in st.session_state:
        st.session_state.saved_analyses = {}

    st.session_state.saved_analyses[analysis_id] = {
        'results': results,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }

# Main app
st.markdown('<h1 class="main-header">🎵 Chord Analysis Dashboard 🎵</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Analysis Parameters")

    # Year selection
    st.subheader("📅 Time Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1950, max_value=2024, value=2000)
    with col2:
        end_year = st.number_input("End Year", min_value=1950, max_value=2024, value=2024)

    # Genre selection
    st.subheader("🎸 Genres")
    available_genres = ['pop', 'rock', 'jazz', 'blues', 'country', 'electronic', 'hip-hop', 'classical']
    selected_genres = st.multiselect("Select Genres", available_genres, default=['pop', 'rock'])

    # Load data button
    if st.button("🔄 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading data..."):
            data = load_data("chordonomicon.csv")
            if data is not None:
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")

    # Analyzer selection
    if st.session_state.data_loaded:
        st.divider()
        st.subheader("🔍 Select Analyzers")

        analyzers = discover_analyzers()

        if not analyzers:
            st.info("No analyzers found. Using demo analyzers.")
            # Demo analyzers for testing
            demo_analyzers = {
                'chord_progression': {
                    'name': 'Chord Progression Analysis',
                    'description': 'Analyzes common chord progressions'
                },
                'complexity_analysis': {
                    'name': 'Complexity Analysis',
                    'description': 'Measures chord complexity over time'
                },
                'genre_patterns': {
                    'name': 'Genre Pattern Detection',
                    'description': 'Identifies genre-specific chord patterns'
                }
            }

            for analyzer_id, analyzer_info in demo_analyzers.items():
                with st.container():
                    st.markdown(f'<div class="analyzer-card">', unsafe_allow_html=True)
                    if st.checkbox(analyzer_info['name'], key=f"check_{analyzer_id}"):
                        if analyzer_id not in st.session_state.selected_analyzers:
                            st.session_state.selected_analyzers.append(analyzer_id)
                    else:
                        if analyzer_id in st.session_state.selected_analyzers:
                            st.session_state.selected_analyzers.remove(analyzer_id)
                    st.caption(analyzer_info['description'])
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            for analyzer_id, analyzer_info in analyzers.items():
                with st.container():
                    st.markdown(f'<div class="analyzer-card">', unsafe_allow_html=True)
                    if st.checkbox(analyzer_info['name'], key=f"check_{analyzer_id}"):
                        if analyzer_id not in st.session_state.selected_analyzers:
                            st.session_state.selected_analyzers.append(analyzer_id)
                    else:
                        if analyzer_id in st.session_state.selected_analyzers:
                            st.session_state.selected_analyzers.remove(analyzer_id)
                    st.caption(analyzer_info['description'])
                    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if not st.session_state.data_loaded:
    st.info("👈 Please load data from the sidebar to begin analysis")
else:
    # Analysis button
    if st.session_state.selected_analyzers:
        if st.button("🚀 Run Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = {}
            for i, analyzer_id in enumerate(st.session_state.selected_analyzers):
                status_text.text(f"Running {analyzer_id}...")
                progress_bar.progress((i + 1) / len(st.session_state.selected_analyzers))

                # Simulate analysis results
                if analyzer_id == 'chord_progression':
                    results[analyzer_id] = {
                        'name': 'Chord Progression Analysis',
                        'data': pd.DataFrame({
                            'progression': ['I-V-vi-IV', 'I-IV-V', 'ii-V-I', 'I-vi-IV-V'],
                            'count': [245, 189, 156, 134],
                            'percentage': [33.5, 25.9, 21.4, 18.3]
                        }),
                        'chart_type': 'bar'
                    }
                elif analyzer_id == 'complexity_analysis':
                    results[analyzer_id] = {
                        'name': 'Complexity Over Time',
                        'data': pd.DataFrame({
                            'year': range(2000, 2025),
                            'complexity_score': [3.2 + i*0.05 + (i%3)*0.1 for i in range(25)]
                        }),
                        'chart_type': 'line'
                    }
                elif analyzer_id == 'genre_patterns':
                    results[analyzer_id] = {
                        'name': 'Genre Chord Patterns',
                        'data': pd.DataFrame({
                            'genre': ['Pop', 'Rock', 'Jazz', 'Blues'],
                            'avg_chords_per_song': [4.2, 5.1, 7.8, 5.5],
                            'unique_chords': [12, 18, 32, 16]
                        }),
                        'chart_type': 'grouped_bar'
                    }

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            # Save results
            analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_analysis(analysis_id, results, {
                'start_year': start_year,
                'end_year': end_year,
                'genres': selected_genres,
                'analyzers': st.session_state.selected_analyzers
            })

            st.session_state.analyses[analysis_id] = results
            st.success(f"✅ Analysis complete! ID: {analysis_id}")

    # Display results
    if st.session_state.analyses:
        st.divider()
        st.header("📊 Analysis Results")

        # Tabs for different analyses
        tab_names = [f"Analysis {aid}" for aid in st.session_state.analyses.keys()]
        tabs = st.tabs(tab_names)

        for tab, (analysis_id, results) in zip(tabs, st.session_state.analyses.items()):
            with tab:
                for analyzer_id, result in results.items():
                    st.subheader(result['name'])

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Create visualization based on chart type
                        if result['chart_type'] == 'bar':
                            fig = px.bar(result['data'], x='progression', y='count',
                                         title=result['name'],
                                         color='count',
                                         color_continuous_scale='Viridis')
                        elif result['chart_type'] == 'line':
                            fig = px.line(result['data'], x='year', y='complexity_score',
                                          title=result['name'],
                                          markers=True)
                        elif result['chart_type'] == 'grouped_bar':
                            fig = go.Figure()
                            fig.add_trace(go.Bar(name='Avg Chords', x=result['data']['genre'],
                                                 y=result['data']['avg_chords_per_song']))
                            fig.add_trace(go.Bar(name='Unique Chords', x=result['data']['genre'],
                                                 y=result['data']['unique_chords']))
                            fig.update_layout(barmode='group', title=result['name'])

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Total Records", len(result['data']))
                        if 'percentage' in result['data'].columns:
                            st.metric("Top Result", f"{result['data']['percentage'].iloc[0]:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Show data table
                    with st.expander("View Raw Data"):
                        st.dataframe(result['data'], use_container_width=True)

                # Export button
                if st.button(f"💾 Export Analysis {analysis_id}", key=f"export_{analysis_id}"):
                    st.info("Export functionality coming soon!")

    # Comparison mode
    if len(st.session_state.analyses) > 1:
        st.divider()
        st.header("🔄 Compare Analyses")

        col1, col2 = st.columns(2)
        with col1:
            analysis1 = st.selectbox("Select First Analysis", list(st.session_state.analyses.keys()))
        with col2:
            analysis2 = st.selectbox("Select Second Analysis",
                                     [a for a in st.session_state.analyses.keys() if a != analysis1])

        if st.button("Compare"):
            st.info("Comparison view coming soon!")