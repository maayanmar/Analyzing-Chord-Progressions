"""
Component for loading and processing Chordonomicon data
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import time
from datetime import datetime

class DataLoader:
    """Class for loading and processing Chordonomicon data"""

    def __init__(self):
        self.cache = {}
        self.data_path = Path(__file__).parent.parent.parent / "data"

    @st.cache_data
    def load_chordonomicon_data(
            self,
            start_year: Optional[int] = None,
            end_year: Optional[int] = None,
            genres: Optional[List[str]] = None,
            sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load Chordonomicon data with filters

        Args:
            start_year: Start year for filtering
            end_year: End year for filtering
            genres: List of genres for filtering
            sample_size: Sample size (None = all data)

        Returns:
            DataFrame with loaded and filtered data
        """

        # Check if file exists
        data_file = self._find_data_file()

        if data_file is None:
            # If no data file, create sample data
            st.warning("⚠️ No Chordonomicon data file found. Creating sample data...")
            return self._generate_sample_data(start_year, end_year, genres, sample_size)

        try:
            # Load data
            with st.spinner("Loading Chordonomicon data..."):
                df = self._load_data_file(data_file)

                # Filter data
                df = self._apply_filters(df, start_year, end_year, genres)

                # Sample if needed
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)

                st.success(f"✅ Loaded {len(df):,} records from Chordonomicon")

                return df

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("🔄 Creating sample data instead...")
            return self._generate_sample_data(start_year, end_year, genres, sample_size)

    def _find_data_file(self) -> Optional[Path]:
        """Search for Chordonomicon data file"""

        possible_files = [
            "chordonomicon.csv",
            "chordonomicon.json",
            "chordonomicon.xlsx",
            "chord_data.csv",
            "music_data.csv"
        ]

        for filename in possible_files:
            file_path = self.data_path / filename
            if file_path.exists():
                return file_path

        return None

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """Load data file by file type"""

        suffix = file_path.suffix.lower()

        if suffix == '.csv':
            return pd.read_csv(file_path, encoding='utf-8')
        elif suffix == '.json':
            return pd.read_json(file_path, encoding='utf-8')
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _apply_filters(
            self,
            df: pd.DataFrame,
            start_year: Optional[int],
            end_year: Optional[int],
            genres: Optional[List[str]]
    ) -> pd.DataFrame:
        """Apply filters to data"""

        filtered_df = df.copy()

        # Filter by years
        if 'year' in filtered_df.columns:
            if start_year is not None:
                filtered_df = filtered_df[filtered_df['year'] >= start_year]
            if end_year is not None:
                filtered_df = filtered_df[filtered_df['year'] <= end_year]

        # Filter by genres
        if genres and 'genre' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['genre'].isin(genres)]

        return filtered_df

    def _generate_sample_data(
            self,
            start_year: Optional[int] = None,
            end_year: Optional[int] = None,
            genres: Optional[List[str]] = None,
            sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Create sample data for development and testing purposes"""

        # Set defaults
        start_year = start_year or 1950
        end_year = end_year or 2023
        genres = genres or ["Rock", "Pop", "Jazz", "Blues", "Country", "Folk"]
        sample_size = sample_size or 50000

        np.random.seed(42)  # For consistent results

        # Common chords
        major_chords = ["C", "G", "D", "A", "E", "B", "F#", "F", "Bb", "Eb", "Ab", "Db"]
        minor_chords = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"]
        seventh_chords = ["C7", "G7", "D7", "A7", "E7", "B7", "F7", "Bb7", "Eb7", "Ab7"]

        all_chords = major_chords + minor_chords + seventh_chords

        # Create data
        data = []

        for i in range(sample_size):
            # Choose random year
            year = np.random.randint(start_year, end_year + 1)

            # Choose random genre with weights
            genre_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]  # Rock, Pop, Jazz, Blues, Country, Folk
            genre = np.random.choice(genres, p=genre_weights[:len(genres)])

            # Choose chord with weights by genre
            if genre == "Jazz":
                chord_weights = [0.15] * len(major_chords) + [0.15] * len(minor_chords) + [0.2] * len(seventh_chords)
                chord = np.random.choice(all_chords, p=chord_weights[:len(all_chords)])
            elif genre in ["Rock", "Pop"]:
                # Preference for simple chords
                simple_chords = major_chords[:8] + minor_chords[:6]
                chord = np.random.choice(simple_chords)
            else:
                chord = np.random.choice(all_chords)

            # Additional data
            artist = f"Artist_{np.random.randint(1, 1000)}"
            song = f"Song_{np.random.randint(1, 10000)}"
            position = np.random.randint(1, 200)  # Chord position in song
            duration = np.random.normal(2.5, 1.0)  # Chord duration in beats

            data.append({
                'year': year,
                'genre': genre,
                'artist': artist,
                'song': song,
                'chord': chord,
                'position': position,
                'duration': max(0.5, duration),
                'song_id': f"{artist}_{song}".replace(" ", "_"),
                'chord_sequence_id': f"{artist}_{song}_{position}",
            })

        df = pd.DataFrame(data)

        # Add additional information
        df['chord_type'] = df['chord'].apply(self._classify_chord_type)
        df['decade'] = (df['year'] // 10) * 10
        df['era'] = df['year'].apply(lambda x:
                                     '1950s-1960s' if x < 1970 else
                                     '1970s-1980s' if x < 1990 else
                                     '1990s-2000s' if x < 2010 else
                                     '2010s-2020s'
                                     )

        return df

    def _classify_chord_type(self, chord: str) -> str:
        """Classify chord type"""
        if 'm' in chord and '7' not in chord:
            return 'minor'
        elif '7' in chord:
            return 'seventh'
        elif 'dim' in chord:
            return 'diminished'
        elif 'aug' in chord:
            return 'augmented'
        elif 'sus' in chord:
            return 'suspended'
        else:
            return 'major'

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean data"""

        processed_df = df.copy()

        # Clean missing data
        processed_df = processed_df.dropna(subset=['chord', 'genre', 'year'])

        # Normalize and validate data
        processed_df['genre'] = processed_df['genre'].str.strip().str.title()
        processed_df['chord'] = processed_df['chord'].str.strip()

        # Add computed columns
        processed_df['is_major'] = processed_df['chord_type'] == 'major'
        processed_df['is_minor'] = processed_df['chord_type'] == 'minor'
        processed_df['is_complex'] = processed_df['chord_type'].isin(['seventh', 'diminished', 'augmented'])

        # Create unique identifiers
        processed_df['song_chord_id'] = processed_df['song_id'] + '_' + processed_df['chord']

        return processed_df

    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return statistics about the data"""

        stats = {
            'total_records': len(df),
            'unique_songs': df['song_id'].nunique() if 'song_id' in df.columns else 0,
            'unique_chords': df['chord'].nunique() if 'chord' in df.columns else 0,
            'unique_artists': df['artist'].nunique() if 'artist' in df.columns else 0,
            'genres_count': df['genre'].nunique() if 'genre' in df.columns else 0,
        }

        # Year range
        if 'year' in df.columns:
            stats['year_range'] = (df['year'].min(), df['year'].max())

        # Main genres
        if 'genre' in df.columns:
            stats['top_genres'] = df['genre'].value_counts().head(10).items()

        # Popular chords
        if 'chord' in df.columns:
            stats['top_chords'] = df['chord'].value_counts().head(15).items()

        # Chord type distribution
        if 'chord_type' in df.columns:
            stats['chord_type_distribution'] = df['chord_type'].value_counts().to_dict()

        # Statistics by decade
        if 'decade' in df.columns:
            stats['decade_distribution'] = df['decade'].value_counts().sort_index().to_dict()

        return stats

    def get_genre_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return detailed genre analysis"""

        if 'genre' not in df.columns:
            return {}

        genre_analysis = {}

        for genre in df['genre'].unique():
            genre_data = df[df['genre'] == genre]

            analysis = {
                'total_records': len(genre_data),
                'unique_songs': genre_data['song_id'].nunique() if 'song_id' in genre_data.columns else 0,
                'unique_chords': genre_data['chord'].nunique(),
                'avg_song_length': genre_data.groupby('song_id')['position'].max().mean() if 'position' in genre_data.columns else 0,
                'top_chords': genre_data['chord'].value_counts().head(10).to_dict(),
                'chord_type_dist': genre_data['chord_type'].value_counts().to_dict() if 'chord_type' in genre_data.columns else {},
                'year_range': (genre_data['year'].min(), genre_data['year'].max()) if 'year' in genre_data.columns else (0, 0)
            }

            genre_analysis[genre] = analysis

        return genre_analysis

    def get_chord_transitions(self, df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
        """Return chord transition matrix"""

        if 'song_id' not in df.columns or 'position' not in df.columns:
            return pd.DataFrame()

        # Sort by song and position
        sorted_df = df.sort_values(['song_id', 'position'])

        transitions = []

        # Group by song
        for song_id, song_data in sorted_df.groupby('song_id'):
            if len(song_data) < 2:
                continue

            chords = song_data['chord'].tolist()

            # Create transition pairs
            for i in range(len(chords) - 1):
                from_chord = chords[i]
                to_chord = chords[i + 1]

                transitions.append({
                    'from_chord': from_chord,
                    'to_chord': to_chord,
                    'song_id': song_id,
                    'genre': song_data.iloc[i]['genre'] if 'genre' in song_data.columns else 'Unknown'
                })

                # Limit number of transitions for performance
                if len(transitions) >= limit:
                    break

            if len(transitions) >= limit:
                break

        return pd.DataFrame(transitions)

    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save processed data to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.csv"

        output_path = self.data_path / "processed" / filename
        output_path.parent.mkdir(exist_ok=True)

        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            return str(output_path)
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            return ""

    def load_saved_data(self, filename: str) -> pd.DataFrame:
        """Load saved data"""

        file_path = self.data_path / "processed" / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        return pd.read_csv(file_path, encoding='utf-8')

    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Return list of available datasets"""

        datasets = []

        # Main data files
        if self.data_path.exists():
            for file_path in self.data_path.glob("*.csv"):
                stats = file_path.stat()
                datasets.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size_mb': stats.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stats.st_mtime),
                    'type': 'raw'
                })

        # Processed data files
        processed_dir = self.data_path / "processed"
        if processed_dir.exists():
            for file_path in processed_dir.glob("*.csv"):
                stats = file_path.stat()
                datasets.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size_mb': stats.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stats.st_mtime),
                    'type': 'processed'
                })

        return sorted(datasets, key=lambda x: x['modified'], reverse=True)

    def validate_data_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data format validity"""

        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        required_columns = ['chord', 'song_id']
        recommended_columns = ['genre', 'year', 'artist']

        # Check required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_required}")

        # Check recommended columns
        missing_recommended = [col for col in recommended_columns if col not in df.columns]
        if missing_recommended:
            validation_result['warnings'].append(f"Missing recommended columns: {missing_recommended}")

        # Check missing data
        for col in df.columns:
            null_percentage = (df[col].isnull().sum() / len(df)) * 100
            if null_percentage > 50:
                validation_result['warnings'].append(f"Column {col} missing in {null_percentage:.1f}% of records")
            elif null_percentage > 10:
                validation_result['suggestions'].append(f"Column {col} missing in {null_percentage:.1f}% of records")

        # Check year range
        if 'year' in df.columns:
            min_year, max_year = df['year'].min(), df['year'].max()
            if min_year < 1900 or max_year > 2030:
                validation_result['warnings'].append(f"Unusual year range: {min_year}-{max_year}")

        # Check number of records
        if len(df) < 1000:
            validation_result['warnings'].append("Low number of records - results may not be meaningful")

        return validation_result

# Helper functions for interface
def show_data_upload_interface():
    """Display data upload interface"""

    st.markdown("### 📂 Upload Chordonomicon Data")

    uploaded_file = st.file_uploader(
        "Choose data file:",
        type=['csv', 'xlsx', 'json'],
        help="Upload file with chord data in CSV, Excel or JSON format"
    )

    if uploaded_file is not None:
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)

            st.success(f"✅ File loaded successfully! {len(df):,} records")

            # Data preview
            st.markdown("#### 👀 Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Data validation
            data_loader = DataLoader()
            validation = data_loader.validate_data_format(df)

            if validation['is_valid']:
                st.success("✅ Data format is valid")
            else:
                st.error("❌ Data format issues:")
                for error in validation['errors']:
                    st.error(f"• {error}")

            if validation['warnings']:
                st.warning("⚠️ Warnings:")
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")

            # Save data
            if st.button("💾 Save Data to System"):
                saved_path = data_loader.save_processed_data(df, uploaded_file.name)
                if saved_path:
                    st.success(f"✅ Data saved at: {saved_path}")
                    st.session_state.loaded_data = df

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def show_data_management_interface():
    """Display data management interface"""

    st.markdown("### 🗂️ Dataset Management")

    data_loader = DataLoader()
    datasets = data_loader.get_available_datasets()

    if not datasets:
        st.info("📁 No datasets available. Upload a data file to get started.")
        return

    # Dataset table
    df_datasets = pd.DataFrame(datasets)
    df_datasets['Size (MB)'] = df_datasets['size_mb'].round(2)
    df_datasets['Last Modified'] = df_datasets['modified'].dt.strftime('%d/%m/%Y %H:%M')
    df_datasets['Type'] = df_datasets['type'].map({'raw': 'Raw', 'processed': 'Processed'})

    display_df = df_datasets[['name', 'Size (MB)', 'Last Modified', 'Type']].rename(columns={'name': 'Filename'})

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Select dataset to load
    selected_dataset = st.selectbox(
        "Select dataset to load:",
        options=df_datasets['name'].tolist(),
        key="dataset_selector"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Load Selected Dataset"):
            try:
                selected_path = df_datasets[df_datasets['name'] == selected_dataset]['path'].iloc[0]
                df = data_loader.load_saved_data(Path(selected_path).name)
                st.session_state.loaded_data = df
                st.success(f"✅ Loaded dataset: {selected_dataset}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Loading error: {str(e)}")

    with col2:
        if st.button("🗑️ Delete Selected Dataset"):
            # Here we would add logic for file deletion
            st.warning("🚧 Delete functionality in development")

# Check if file is run directly
if __name__ == "__main__":
    st.title("📊 Testing Data Loader Component")

    data_loader = DataLoader()

    # Create sample data
    sample_data = data_loader.load_chordonomicon_data(
        start_year=2000,
        end_year=2020,
        genres=["Rock", "Pop"],
        sample_size=1000
    )

    st.write("Sample data:")
    st.dataframe(sample_data.head())

    st.write("Statistics:")
    stats = data_loader.get_data_statistics(sample_data)
    st.json(stats)