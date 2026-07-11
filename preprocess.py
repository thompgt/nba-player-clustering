import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import os

import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['Player', 'Tm'] + config.CLUSTERING_FEATURES


def preprocess_data(file_path):
    # Load data with semicolon separator
    try:
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file '{file_path}' not found. See the README 'Data Source' section."
        ) from None

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input file '{file_path}' is missing required columns: {missing_columns}")

    # Handle duplicates: Keep 'TOT' for players who played for multiple teams
    # Drop rows where Tm != 'TOT' for players who have a 'TOT' row
    players_with_tot = df[df['Tm'] == 'TOT']['Player'].unique()
    df = df[~((df['Player'].isin(players_with_tot)) & (df['Tm'] != 'TOT'))]

    # Percentage columns (FG%/3P%/FT%) are NaN exactly when the player had
    # zero attempts of that shot type, so 0 is the correct value there. Fill
    # only the clustering feature columns rather than the whole dataframe,
    # so unrelated columns (e.g. Pos) aren't silently zeroed if they ever
    # contain gaps.
    df[config.CLUSTERING_FEATURES] = df[config.CLUSTERING_FEATURES].fillna(0)

    clustering_features = config.CLUSTERING_FEATURES

    X = df[clustering_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X_scaled)
    df['PC1'] = pca_results[:, 0]
    df['PC2'] = pca_results[:, 1]
    df['PC3'] = pca_results[:, 2]
    
    # Map clusters to names (Generic for now, can be improved)
    # 0: Defensive Bigs, 1: Scoring Guards, etc.
    # We can analyze cluster centers to give better names.
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=clustering_features)
    logger.info("Cluster centers:\n%s", centers_df)

    df.to_csv(config.OUTPUT_FILE, index=False)
    logger.info("Processed data saved to %s", config.OUTPUT_FILE)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    preprocess_data(config.INPUT_FILE)
