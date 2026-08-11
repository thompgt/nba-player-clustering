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

# The per-36 columns are derived, so the raw stats they come from are required
# instead. CLUSTERING_FEATURES is deliberately not in this list.
REQUIRED_COLUMNS = sorted(
    set(
        ['Player', 'Tm', 'G', 'MP']
        + config.RATE_STATS
        + config.SHOOTING_PERCENTAGES
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
    df = add_per_36_rates(df)

    # Fill only the clustering feature columns rather than the whole dataframe,
    # so unrelated columns (e.g. Pos) aren't silently zeroed if they ever
    # contain gaps. Counting stats are genuinely 0 when absent; the percentage
    # columns have already been handled above.
    df[config.CLUSTERING_FEATURES] = df[config.CLUSTERING_FEATURES].fillna(0)
    return df


def add_per_36_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-36-minutes column for each counting stat used in the model.

    Clustering raw per-game totals makes k-means a volume sorter: cluster mean
    minutes used to run 6.7 / 10.6 / 11.6 / 24.1 / 27.2 / 33.9, so the partition
    was mostly a minutes ladder and PC1 tracked overall production. Rates hold
    playing time constant, so what is left is style.

    Players below the minutes floor would produce wild rates from tiny
    denominators; they are dropped from the fit by ``eligible_mask``.
    """
    df = df.copy()
    minutes = df['MP'].astype(float)
    safe_minutes = minutes.where(minutes > 0)
    for stat in config.RATE_STATS:
        df[f"{stat}{config.PER_36_SUFFIX}"] = df[stat].astype(float) * 36.0 / safe_minutes
    return df


def eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with enough playing time for a per-36 rate to mean anything."""
    return (df['MP'].astype(float) >= config.MIN_MINUTES_PER_GAME) & (
        df['G'].astype(float) >= config.MIN_GAMES
    )


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the clustering features. One definition, used everywhere."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[config.CLUSTERING_FEATURES]), scaler


def preprocess_data(file_path: str) -> pd.DataFrame:
    df = load_players(file_path)
    clustering_features = config.CLUSTERING_FEATURES

    # Fit on eligible players only. Low-minute players are kept in the output
    # so the dashboard can still show them, but they are reported as unranked
    # rather than being given an archetype their sample cannot support.
    eligible = eligible_mask(df)
    logger.info(
        "%d of %d players are eligible (>= %.1f MPG and >= %d games); %d reported as %s.",
        int(eligible.sum()), len(df), config.MIN_MINUTES_PER_GAME, config.MIN_GAMES,
        int((~eligible).sum()), archetypes.UNRANKED,
    )
    if eligible.sum() < config.N_CLUSTERS:
        raise ValueError(
            f"Only {int(eligible.sum())} players clear the eligibility floors "
            f"({config.MIN_MINUTES_PER_GAME} MPG, {config.MIN_GAMES} games), which is fewer "
            f"than N_CLUSTERS={config.N_CLUSTERS}. Lower the floors in config.py."
        )

    fitted = df[eligible]
    X_scaled, scaler = build_feature_matrix(fitted)

    # K-Means clustering
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = archetypes.UNRANKED_CLUSTER
    df.loc[eligible, 'Cluster'] = labels

    # PCA for visualization. Fit on the same rows as the model, then project
    # the ineligible players into that space so they can still be plotted.
    pca = PCA(n_components=3)
    pca.fit(X_scaled)
    all_scaled = scaler.transform(df[clustering_features])
    pca_results = pca.transform(all_scaled)
    df['PC1'] = pca_results[:, 0]
    df['PC2'] = pca_results[:, 1]
    df['PC3'] = pca_results[:, 2]

    explained = pca.explained_variance_ratio_
    logger.info(
        "PCA explained variance: %s (cumulative %.1f%% of %d features)",
        ", ".join(f"PC{i + 1} {r:.1%}" for i, r in enumerate(explained)),
        explained.sum() * 100,
        len(clustering_features),
    )

    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=clustering_features)
    logger.info("Cluster centers:\n%s", centers_df)

    # Attach archetype names by matching each cluster's statistical profile to
    # the reference profiles in archetypes.py. Cluster *indices* are
    # seed-dependent, so the name is derived from the centroid's shape, never
    # from its index. The distances are logged so a drifting cluster is
    # visible here as well as in validate_model.py, which gates on them.
    names, distances = archetypes.assign_archetypes(fitted, labels)
    df['Archetype'] = df['Cluster'].map(names).fillna(archetypes.UNRANKED)
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
