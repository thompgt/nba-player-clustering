import logging
import os
import sys

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import archetypes
import config

logger = logging.getLogger(__name__)


def validate() -> bool:
    """Return True if the processed data passes validation, False otherwise."""
    if not os.path.exists(config.OUTPUT_FILE):
        logger.error("%s not found. Run `python preprocess.py` first to generate it.", config.OUTPUT_FILE)
        return False

    df = pd.read_csv(config.OUTPUT_FILE)

    # Check columns
    expected_cols = ['Player', 'Cluster', 'Archetype', 'PC1', 'PC2', 'PC3']
    for col in expected_cols:
        if col not in df.columns:
            logger.error("Missing column %s", col)
            return False

    logger.info("Columns validation passed.")

    # Score only the rows the model was actually fit on. Unranked players sit
    # outside every cluster; including them would score a partition nobody built.
    ranked = df[df['Cluster'] != archetypes.UNRANKED_CLUSTER]
    logger.info(
        "%d ranked players, %d unranked.", len(ranked), len(df) - len(ranked)
    )
    if ranked['Cluster'].nunique() != config.N_CLUSTERS:
        logger.error(
            "Expected %d clusters, found %d.", config.N_CLUSTERS, ranked['Cluster'].nunique()
        )
        return False

    # Calculate Silhouette Score
    X = ranked[config.CLUSTERING_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    score = silhouette_score(X_scaled, ranked['Cluster'])
    logger.info("Silhouette Score: %.4f", score)

    # Basic Cluster Stats
    logger.info("Cluster Distribution:\n%s", df['Archetype'].value_counts())

    ok = True

    if score <= config.SILHOUETTE_THRESHOLD:
        logger.error(
            "Silhouette score %.4f is at or below the %.2f threshold. Model needs tuning.",
            score, config.SILHOUETTE_THRESHOLD,
        )
        ok = False

    if not _validate_archetypes(ranked):
        ok = False

    if ok:
        logger.info("Validation Successful!")
    return ok


def _validate_archetypes(df: pd.DataFrame) -> bool:
    """Confirm the stored archetype names still match the cluster profiles.

    Archetype names are assigned by matching each cluster's signature profile
    against the reference profiles in archetypes.py, so this re-derives the
    match and fails if either (a) a stored name disagrees with the profile it
    is attached to, or (b) some cluster no longer resembles any archetype
    closely enough to inherit its name.
    """
    try:
        names, distances = archetypes.assign_archetypes(df, df['Cluster'])
    except ValueError as exc:
        logger.error("Archetype assignment failed: %s", exc)
        return False

    ok = True
    for cid in sorted(names):
        stored = df.loc[df['Cluster'] == cid, 'Archetype'].unique()
        logger.info("Cluster %s -> %-30s (profile distance %.2f)", cid, names[cid], distances[cid])
        if list(stored) != [names[cid]]:
            logger.error(
                "Cluster %s is stored as %s but its profile matches %s. "
                "Regenerate with `python preprocess.py`.",
                cid, list(stored), names[cid],
            )
            ok = False

    within_tolerance, worst = archetypes.match_quality(distances)
    if not within_tolerance:
        logger.error(
            "Worst archetype profile distance %.2f exceeds the %.2f tolerance: at least one "
            "cluster no longer resembles the archetype it was named after. Review the cluster "
            "profiles (`python archetypes.py`) and rename or re-derive the archetypes.",
            worst, archetypes.MAX_PROFILE_DISTANCE,
        )
        ok = False
    else:
        logger.info(
            "Archetype match within tolerance (worst distance %.2f <= %.2f).",
            worst, archetypes.MAX_PROFILE_DISTANCE,
        )

    return ok

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(0 if validate() else 1)
