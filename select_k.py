"""Re-derive the cluster count and write the evidence out as an artifact.

``N_CLUSTERS`` used to be a bare constant in ``config.py``. The only thing that
had ever *checked* it was a sweep buried in ``make_figures.py``, which runs
solely when regenerating PNGs -- so nothing re-derived or re-validated k as the
data or the feature engineering changed.

This script runs the sweep over the same matrix ``preprocess.py`` fits on,
writes ``model_selection.csv`` as a pipeline artifact, and warns when the
shipped k is no longer defensible: not at a local silhouette maximum, unstable
across seeds, or producing a cluster too small to be a real archetype.

    python select_k.py            # sweep, write the table, report
    python select_k.py --check    # additionally exit non-zero if k is indefensible

The metrics reported per k:

``silhouette``
    Mean silhouette on the fitted rows. Higher is better, but it is biased
    toward small k, so it is read for local structure rather than globally.
``inertia``
    Within-cluster sum of squares; the elbow curve.
``min_cluster_size``
    Size of the smallest cluster. A handful of players is a sampling artifact,
    not an archetype.
``stability``
    Mean adjusted Rand index between the shipped seed's partition and refits
    under other seeds. This is the metric that actually matters for a project
    that attaches human names to clusters: if the partition is not reproducible,
    the names cannot be either.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
from preprocess import build_feature_matrix, eligible_mask, load_players

logger = logging.getLogger(__name__)

#: Range of k to sweep.
K_RANGE = range(2, 13)

#: Smallest k that can plausibly be a set of *archetypes*.
#:
#: Silhouette is monotonically biased toward small k on continuum-shaped data
#: like per-game box scores, so the sweep's global maximum is always k=2 -- a
#: starters/bench split, which is not what this project is for. Comparing the
#: shipped k against the whole curve would therefore say "use 2" forever. The
#: local-maximum test is applied from here upward instead, and the choice of
#: this floor is the one genuinely editorial parameter in model selection.
INTERPRETABLE_K_MIN = 4

#: Seeds used to measure partition stability, alongside config.RANDOM_STATE.
STABILITY_SEEDS = (0, 1, 7, 2024)

#: A shipped k must clear these to be considered defensible.
MIN_STABILITY = 0.75
MIN_CLUSTER_FRACTION = 0.02

OUTPUT_FILE = "model_selection.csv"


def sweep(X: np.ndarray, k_range: range = K_RANGE) -> pd.DataFrame:
    """Fit K-Means across ``k_range`` and score each fit."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(X)
        aris = [
            adjusted_rand_score(
                km.labels_,
                KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X).labels_,
            )
            for seed in STABILITY_SEEDS
        ]
        rows.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(X, km.labels_)), 4),
                "inertia": round(float(km.inertia_), 2),
                "min_cluster_size": int(np.bincount(km.labels_).min()),
                "stability": round(float(np.mean(aris)), 4),
                "stability_worst": round(float(np.min(aris)), 4),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def interpretable(table: pd.DataFrame) -> pd.DataFrame:
    """The slice of the sweep that could plausibly be a set of archetypes."""
    return table[table.index >= INTERPRETABLE_K_MIN]


def is_local_maximum(table: pd.DataFrame, k: int) -> bool:
    """True if ``k`` is at least as good as its immediate neighbours."""
    if k not in table.index:
        return False
    score = table.at[k, "silhouette"]
    neighbours = [n for n in (k - 1, k + 1) if n in table.index]
    return all(score >= table.at[n, "silhouette"] for n in neighbours)


def report(table: pd.DataFrame, k: int, n_players: int) -> list[str]:
    """Return a list of human-readable problems with the shipped ``k``."""
    problems = []

    if k not in table.index:
        return [f"k={k} is outside the swept range {table.index.min()}-{table.index.max()}."]

    row = table.loc[k]

    if k < INTERPRETABLE_K_MIN:
        problems.append(
            f"k={k} is below INTERPRETABLE_K_MIN={INTERPRETABLE_K_MIN}: too coarse to be archetypes."
        )
    elif not is_local_maximum(interpretable(table), k):
        # Compared within the interpretable slice: k=3 always outscores k=4 on
        # continuum data, which would veto every archetype-sized k forever.
        candidates = interpretable(table)["silhouette"]
        better = candidates.idxmax()
        problems.append(
            f"k={k} is not at a local silhouette maximum "
            f"({row['silhouette']:.4f}); within the interpretable range k={better} "
            f"scores {candidates.max():.4f}."
        )

    if row["stability"] < MIN_STABILITY:
        problems.append(
            f"k={k} is unstable across seeds (mean ARI {row['stability']:.2f} < {MIN_STABILITY}). "
            "Cluster membership is not reproducible, so archetype names cannot be either."
        )

    floor = max(1, int(MIN_CLUSTER_FRACTION * n_players))
    if row["min_cluster_size"] < floor:
        problems.append(
            f"k={k} produces a cluster of only {int(row['min_cluster_size'])} players "
            f"(floor {floor} = {MIN_CLUSTER_FRACTION:.0%} of {n_players}); "
            "that is a sampling artifact, not an archetype."
        )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the shipped N_CLUSTERS is no longer defensible",
    )
    args = parser.parse_args(argv)

    # Eligible players only: the sweep must see exactly the rows the shipped
    # model is fit on, or it scores a partition nothing ever ships.
    df = load_players(config.INPUT_FILE)
    df = df[eligible_mask(df)]
    X, _ = build_feature_matrix(df)

    table = sweep(X)
    table.to_csv(OUTPUT_FILE)
    logger.info("Wrote %s\n%s", OUTPUT_FILE, table.to_string())

    problems = report(table, config.N_CLUSTERS, len(df))
    if not problems:
        logger.info("Shipped k=%s is still defensible.", config.N_CLUSTERS)
        return 0

    for problem in problems:
        logger.warning("%s", problem)
    logger.warning(
        "Review %s and either move N_CLUSTERS or record why the current value is kept.",
        OUTPUT_FILE,
    )
    return 1 if args.check else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
