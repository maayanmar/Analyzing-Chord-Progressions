# this file is responsible the basic data processing task
# from raw data to processed data which contains only the necessary information
# song id, genre, release date, chords

# TODO: currently the code drops NA values, but it might be better to handle them differently

import pandas as pd


def process_data(file_path: str = '../data/raw.csv') -> pd.DataFrame:
    """
    Process the raw data file to extract necessary information.

    Args:
        file_path (str): Path to the raw data file.

    Returns:
        pd.DataFrame: Processed DataFrame containing song id, genre, year, and chords.
    """
    # Read the raw data
    raw_data = pd.read_csv(file_path)

    # rename main_genre to genre
    raw_data.rename(columns={'main_genre': 'genre'}, inplace=True)

    # Select relevant columns
    processed_data = raw_data[['id', 'genre', 'release_date', 'chords']]

    # Drop rows with missing values in essential columns
    processed_data.dropna(subset=['id', 'genre', 'release_date', 'chords'], inplace=True)

    return processed_data.reset_index(drop=True)


def save_processed_data(data: pd.DataFrame, output_path: str = '../data/processed.csv') -> None:
    """
    Save the processed data to a CSV file.

    Args:
        data (pd.DataFrame): Processed DataFrame to save.
        output_path (str): Path where the processed data will be saved.
    """
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Starting data processing...")
    processed_data = process_data()
    print("Data processing complete. Saving processed data...")
    save_processed_data(processed_data)
