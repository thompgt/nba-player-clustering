import logging

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

import config

logger = logging.getLogger(__name__)


def validate():
    if not os.path.exists(config.OUTPUT_FILE):
        logger.error("%s not found.", config.OUTPUT_FILE)
        return

    df = pd.read_csv(config.OUTPUT_FILE)

    # Check columns
    expected_cols = ['Player', 'Cluster', 'PC1', 'PC2', 'PC3']
    for col in expected_cols:
        if col not in df.columns:
            logger.error("Missing column %s", col)
            return

    logger.info("Columns validation passed.")

    # Calculate Silhouette Score
    X = df[config.CLUSTERING_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score = silhouette_score(X_scaled, df['Cluster'])
    logger.info("Silhouette Score: %.4f", score)

    # Basic Cluster Stats
    logger.info("Cluster Distribution:\n%s", df['Cluster'].value_counts().sort_index())

    if score > config.SILHOUETTE_THRESHOLD:
        logger.info("Validation Successful!")
    else:
        logger.warning("Validation Warning: Low Silhouette Score. Model might need tuning.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    validate()
