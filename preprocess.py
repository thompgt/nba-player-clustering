import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import archetypes
import config

logger = logging.getLogger(__name__)

_SHOOTING_COLUMNS = [c for pair in config.SHOOTING_RATES.values() for c in pair]

REQUIRED_COLUMNS = sorted(
    set(
        ['Player', 'Tm', 'G']
        + config.CLUSTERING_FEATURES
        + archetypes.SIGNATURE_FEATURES
        + _SHOOTING_COLUMNS
    )
)


def shrink_shooting_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw shooting percentages with league-shrunk estimates.

    Basketball-Reference stores ``0.0``, not NaN, for a player who never
    attempted a given shot type -- 46 players in this dataset have
    ``3PA == 0`` and ``3P% == 0.0``. Standardised, that is the strongest
    possible negative signal: k-means reads "took no threes" as "worst
    three-point shooter alive". The mirror-image problem is just as bad: one
    made three in one attempt reads as a perfect shooter.

    Both are small-sample artifacts, so each percentage is shrunk toward the
    league rate in proportion to how many attempts back it::

        shrunk = (attempts * rate + k * league_rate) / (attempts + k)

    A player with zero attempts lands exactly on the league rate ("no data"),
    and a player with a full season of attempts is left essentially untouched.
    """
    df = df.copy()
    for pct, (makes, attempts) in config.SHOOTING_RATES.items():
        if makes not in df.columns or attempts not in df.columns:
            logger.warning(
                "Cannot shrink %s: missing %s/%s. Leaving the raw column in place.",
                pct, makes, attempts,
            )
            continue

        # Games-weighted league rate: season totals, not a mean of per-game rates.
        games = df['G'].astype(float)
        league_rate = float((df[makes] * games).sum() / (df[attempts] * games).sum())

        att = df[attempts].astype(float).fillna(0.0)
        observed = df[pct].astype(float).fillna(0.0)
        k = config.SHOOTING_PRIOR_ATTEMPTS
        df[pct] = (att * observed + k * league_rate) / (att + k)

        zero_attempt = int((att == 0).sum())
        logger.info(
            "%s: league rate %.3f, shrunk with k=%.1f attempts (%d players had zero attempts)",
            pct, league_rate, k, zero_attempt,
        )
    return df


def load_players(file_path: str) -> pd.DataFrame:
    """Read the raw CSV and apply every step that precedes model fitting.

    Shared by ``preprocess_data`` and ``select_k.py`` so the k-sweep is run
    over exactly the matrix the shipped model is fit on.
    """
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
    # .copy() so the later column assignments write to their own frame rather
    # than a view of the original (SettingWithCopyWarning).
    df = df[~((df['Player'].isin(players_with_tot)) & (df['Tm'] != 'TOT'))].copy()

    df = shrink_shooting_percentages(df)

    # Fill only the clustering feature columns rather than the whole dataframe,
    # so unrelated columns (e.g. Pos) aren't silently zeroed if they ever
    # contain gaps. Counting stats are genuinely 0 when absent; the percentage
    # columns have already been handled above.
    df[config.CLUSTERING_FEATURES] = df[config.CLUSTERING_FEATURES].fillna(0)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the clustering features. One definition, used everywhere."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[config.CLUSTERING_FEATURES]), scaler


def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_players(file_path)
    clustering_features = config.CLUSTERING_FEATURES
    X_scaled, scaler = build_feature_matrix(df)

    # K-Means clustering
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X_scaled)
    df['PC1'] = pca_results[:, 0]
    df['PC2'] = pca_results[:, 1]
    df['PC3'] = pca_results[:, 2]
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=clustering_features)
    logger.info("Cluster centers:\n%s", centers_df)

    # Attach archetype names by matching each cluster's statistical profile to
    # the reference profiles in archetypes.py. Cluster *indices* are
    # seed-dependent, so the name is derived from the centroid's shape, never
    # from its index. The distances are logged so a drifting cluster is
    # visible here as well as in validate_model.py, which gates on them.
    names, distances = archetypes.assign_archetypes(df, df['Cluster'])
    df['Archetype'] = df['Cluster'].map(names)
    for cid in sorted(names):
        logger.info(
            "Cluster %s -> %-30s (profile distance %.2f, n=%d)",
            cid, names[cid], distances[cid], int((df['Cluster'] == cid).sum()),
        )

    df.to_csv(config.OUTPUT_FILE, index=False)
    logger.info("Processed data saved to %s", config.OUTPUT_FILE)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    preprocess_data(config.INPUT_FILE)
