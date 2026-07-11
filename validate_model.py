import logging

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

logger = logging.getLogger(__name__)


def validate():
    if not os.path.exists('processed_nba_stats.csv'):
        logger.error("processed_nba_stats.csv not found.")
        return

    df = pd.read_csv('processed_nba_stats.csv')

    # Check columns
    expected_cols = ['Player', 'Cluster', 'PC1', 'PC2', 'PC3']
    for col in expected_cols:
        if col not in df.columns:
            logger.error("Missing column %s", col)
            return

    logger.info("Columns validation passed.")

    # Calculate Silhouette Score
    clustering_features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', '2P', 'FT', 'ORB', 'DRB', 'FG%', '3P%', 'FT%']
    X = df[clustering_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score = silhouette_score(X_scaled, df['Cluster'])
    logger.info("Silhouette Score: %.4f", score)

    # Basic Cluster Stats
    logger.info("Cluster Distribution:\n%s", df['Cluster'].value_counts().sort_index())

    if score > 0.1: # Threshold for "something reasonable" in high-dimensional player stats
        logger.info("Validation Successful!")
    else:
        logger.warning("Validation Warning: Low Silhouette Score. Model might need tuning.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    validate()
