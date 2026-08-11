import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import archetypes
import config
import model_store

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

    # Score with the *training* transform, loaded from disk. Re-fitting a
    # fresh scaler here would silently score a slightly different space than
    # the one the model was built in.
    try:
        model = model_store.load()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return False

    X_scaled = model.transform(ranked)
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

    # The saved model and the saved CSV must describe the same fit.
    predicted = model.kmeans.predict(X_scaled)
    disagreements = int((predicted != ranked['Cluster'].to_numpy()).sum())
    if disagreements:
        logger.error(
            "%s and %s disagree: the saved model assigns %d of %d players to a different "
            "cluster than the CSV records. Regenerate both with `python preprocess.py`.",
            config.MODEL_FILE, config.OUTPUT_FILE, disagreements, len(ranked),
        )
        ok = False
    else:
        logger.info("Saved model reproduces every cluster assignment in %s.", config.OUTPUT_FILE)

    if not _validate_stability(X_scaled, ranked['Cluster']):
        ok = False

    if not _validate_cluster_sizes(ranked['Cluster']):
        ok = False

    if not _validate_archetypes(ranked):
        ok = False

    if ok:
        logger.info("Validation Successful!")
    return ok


def _validate_stability(X_scaled: np.ndarray, labels: pd.Series) -> bool:
    """Refit under other seeds and check the partition is reproducible.

    Silhouette alone says nothing about reproducibility: a clustering can score
    respectably and still shuffle its membership on the next seed. Since this
    project attaches human archetype names to clusters, an unstable partition
    means unstable names, which is the failure the naming rework exists to
    prevent. Measured as the mean adjusted Rand index against refits.
    """
    aris = []
    for seed in config.STABILITY_SEEDS:
        refit = KMeans(n_clusters=config.N_CLUSTERS, random_state=seed, n_init=10).fit(X_scaled)
        aris.append(adjusted_rand_score(labels, refit.labels_))

    mean_ari = float(np.mean(aris))
    logger.info(
        "Seed stability: mean ARI %.3f over seeds %s (worst %.3f)",
        mean_ari, list(config.STABILITY_SEEDS), min(aris),
    )
    if mean_ari < config.MIN_STABILITY_ARI:
        logger.error(
            "Clustering is not reproducible: mean ARI %.3f < %.2f. Cluster membership "
            "shifts between seeds, so archetype names cannot be trusted. Reconsider "
            "N_CLUSTERS (see model_selection.csv).",
            mean_ari, config.MIN_STABILITY_ARI,
        )
        return False
    return True


def _validate_cluster_sizes(labels: pd.Series) -> bool:
    """Reject a clustering that produced a cluster too small to be an archetype."""
    sizes = labels.value_counts()
    floor = max(1, int(config.MIN_CLUSTER_FRACTION * len(labels)))
    smallest = int(sizes.min())
    logger.info(
        "Smallest cluster: %d players (floor %d = %.0f%% of %d).",
        smallest, floor, config.MIN_CLUSTER_FRACTION * 100, len(labels),
    )
    if smallest < floor:
        logger.error(
            "Cluster %s holds only %d players, under the floor of %d. That is a sampling "
            "artifact, not an archetype.",
            sizes.idxmin(), smallest, floor,
        )
        return False
    return True


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
