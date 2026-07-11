import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import os

logger = logging.getLogger(__name__)


def preprocess_data(file_path):
    # Load data with semicolon separator
    df = pd.read_csv(file_path, sep=';', encoding='latin1')
    
    # Handle duplicates: Keep 'TOT' for players who played for multiple teams
    # Drop rows where Tm != 'TOT' for players who have a 'TOT' row
    players_with_tot = df[df['Tm'] == 'TOT']['Player'].unique()
    df = df[~((df['Player'].isin(players_with_tot)) & (df['Tm'] != 'TOT'))]
    
    # Fill NaN values (mostly for % stats)
    df = df.fillna(0)
    
    # Features for clustering
    features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', '2P', 'FT', 'ORB', 'DRB']
    
    # Normalize by minutes played? Actually per game stats are already provided.
    # Let's use some efficiency stats too.
    # df['FG%'], df['3P%'], df['FT%'] are already there.
    
    clustering_features = features + ['FG%', '3P%', 'FT%']
    
    X = df[clustering_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    n_clusters = 6 # Standard number of archetypes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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

    df.to_csv('processed_nba_stats.csv', index=False)
    logger.info("Processed data saved to processed_nba_stats.csv")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    preprocess_data('nba_stats.csv')
