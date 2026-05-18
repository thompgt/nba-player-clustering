import pandas as pd
import pytest
import os
from preprocess import preprocess_data

def test_preprocess_data():
    # Setup: Ensure we have a sample data file or use the existing one if small
    # For testing, we'll use the existing nba_stats.csv if it exists
    input_file = 'nba_stats.csv'
    output_file = 'processed_nba_stats.csv'
    
    if not os.path.exists(input_file):
        pytest.skip("Input file nba_stats.csv not found for testing.")
        
    df = preprocess_data(input_file)
    
    assert isinstance(df, pd.DataFrame)
    assert 'Cluster' in df.columns
    assert 'PC1' in df.columns
    assert 'PC2' in df.columns
    assert 'PC3' in df.columns
    assert df.isnull().sum().sum() == 0
    
    # Check if output file was created
    assert os.path.exists(output_file)

def test_cluster_names():
    # Ensure all clusters have names assigned in the app (or logic)
    df = pd.read_csv('processed_nba_stats.csv')
    clusters = df['Cluster'].unique()
    assert len(clusters) == 6 # As defined in preprocess.py
