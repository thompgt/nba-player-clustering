"""Stable, human-readable names for K-Means clusters.

K-Means cluster *indices* are an implementation detail: they depend on the
random seed, on the sklearn version's initialisation, and on the data. Naming
archetypes by index (``{2: "Star Players"}``) therefore silently mislabels the
whole dashboard the first time any of those change. Empirically, refitting the
shipped model with seeds 0/1/7/2024 moves the top scorer's cluster index to
3/0/0/0 while producing essentially the same partition (ARI 0.89-0.96).

So names are not attached to indices. Each archetype owns a *reference
profile*: the mean of a fixed set of per-game "signature" statistics, expressed
in standard deviations from the league average, that a human looked at once and
named. After every refit each fitted cluster is described in the same terms and
matched to the reference profiles by minimum total distance (a bijection, via
the Hungarian algorithm), so an archetype's name follows its statistical shape
rather than its index.

The signature statistics are deliberately *not* the clustering features: they
are plain per-game box-score columns that exist in the raw data regardless of
how the feature engineering evolves, so changing ``CLUSTERING_FEATURES`` does
not invalidate the naming scheme.

``match_quality`` reports how far each cluster sat from the archetype it was
given; ``validate_model.py`` gates on it, so a genuinely new kind of cluster
fails loudly instead of inheriting a stale name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Per-game box-score columns used to describe a cluster for naming purposes.
# Stable across changes to the clustering feature set.
SIGNATURE_FEATURES: list[str] = ["MP", "PTS", "TRB", "AST", "STL", "BLK", "3P"]

# A cluster whose signature profile sits further than this (Euclidean distance
# in standard-deviation space, over SIGNATURE_FEATURES) from the archetype it
# was matched to is not really that archetype any more.
MAX_PROFILE_DISTANCE: float = 1.5


@dataclass(frozen=True)
class Archetype:
    """A named player archetype and the statistical shape that defines it."""

    name: str
    description: str
    color: str
    #: Mean of each SIGNATURE_FEATURES column, in standard deviations from the
    #: league average, for the cluster a human originally gave this name to.
    profile: dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([self.profile[f] for f in SIGNATURE_FEATURES], dtype=float)


# Reference profiles. Regenerate with `python archetypes.py --profiles` after a
# deliberate model change, then re-read the printed profiles and confirm the
# names still describe them before pasting them back in.
ARCHETYPES: list[Archetype] = [
    Archetype(
        name="Star Players",
        description=(
            "High-usage players who anchor their team's offense across scoring, "
            "playmaking, and rebounding."
        ),
        color="#E45756",
        profile={"MP": 1.54, "PTS": 2.08, "TRB": 0.85, "AST": 1.83, "STL": 1.16, "BLK": 0.27, "3P": 1.52},
    ),
    Archetype(
        name="Starting Bigs",
        description="Starting-caliber frontcourt players who dominate the paint on both ends.",
        color="#54A24B",
        profile={"MP": 0.86, "PTS": 0.66, "TRB": 2.13, "AST": 0.15, "STL": 0.59, "BLK": 2.07, "3P": -0.50},
    ),
    Archetype(
        name="Role Players / Shooters",
        description="Efficient perimeter shooters who space the floor in a complementary role.",
        color="#4C78A8",
        profile={"MP": 0.55, "PTS": 0.24, "TRB": 0.21, "AST": 0.24, "STL": 0.55, "BLK": 0.12, "3P": 0.53},
    ),
    Archetype(
        name="Bench Guards/Wings",
        description=(
            "Reserve backcourt/wing players providing depth, ball-handling, and "
            "secondary scoring."
        ),
        color="#F58518",
        profile={"MP": -0.72, "PTS": -0.65, "TRB": -0.70, "AST": -0.52, "STL": -0.59, "BLK": -0.54, "3P": -0.40},
    ),
    Archetype(
        name="Reserve Bigs",
        description="Bench frontcourt players providing rebounding and interior depth.",
        color="#B279A2",
        profile={"MP": -0.81, "PTS": -0.69, "TRB": -0.13, "AST": -0.69, "STL": -0.77, "BLK": 0.17, "3P": -1.04},
    ),
    Archetype(
        name="Limited Minutes / Specialists",
        description="Players with a narrow, specialized role and limited playing time.",
        color="#72B7B2",
        profile={"MP": -1.21, "PTS": -1.05, "TRB": -0.97, "AST": -0.75, "STL": -1.00, "BLK": -0.72, "3P": -0.92},
    ),
]

ARCHETYPE_NAMES: list[str] = [a.name for a in ARCHETYPES]
BY_NAME: dict[str, Archetype] = {a.name: a for a in ARCHETYPES}


def cluster_profiles(df: pd.DataFrame, labels: pd.Series | np.ndarray) -> pd.DataFrame:
    """Describe each cluster by its mean signature stats, in league z-units.

    Standardising against the *player* population (not the cluster population)
    keeps profiles comparable across refits with different cluster counts.
    """
    missing = [f for f in SIGNATURE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Cannot profile clusters: missing signature columns {missing}")

    stats = df[SIGNATURE_FEATURES].astype(float)
    z = (stats - stats.mean()) / stats.std(ddof=0)
    z = z.assign(_cluster=np.asarray(labels))
    return z.groupby("_cluster")[SIGNATURE_FEATURES].mean()


def assign_archetypes(
    df: pd.DataFrame, labels: pd.Series | np.ndarray
) -> tuple[dict[int, str], dict[int, float]]:
    """Map each cluster index to an archetype name by profile similarity.

    Returns ``(names, distances)``: the chosen name per cluster index, and how
    far that cluster's profile sat from the archetype's reference profile.

    The mapping is a bijection - two clusters can never collapse onto the same
    name - found by minimising total assignment cost with the Hungarian
    algorithm rather than greedy nearest-neighbour, which would be
    order-dependent.
    """
    profiles = cluster_profiles(df, labels)
    if len(profiles) != len(ARCHETYPES):
        raise ValueError(
            f"Found {len(profiles)} clusters but {len(ARCHETYPES)} archetypes are defined. "
            "Every cluster must get exactly one name; update ARCHETYPES in archetypes.py."
        )

    reference = np.vstack([a.vector() for a in ARCHETYPES])
    observed = profiles[SIGNATURE_FEATURES].to_numpy()
    cost = np.linalg.norm(observed[:, None, :] - reference[None, :, :], axis=2)

    rows, cols = linear_sum_assignment(cost)
    cluster_ids = [int(c) for c in profiles.index]
    names = {cluster_ids[r]: ARCHETYPES[c].name for r, c in zip(rows, cols)}
    distances = {cluster_ids[r]: float(cost[r, c]) for r, c in zip(rows, cols)}
    return names, distances


def match_quality(distances: dict[int, float]) -> tuple[bool, float]:
    """Return ``(ok, worst_distance)`` for a set of archetype match distances."""
    worst = max(distances.values()) if distances else 0.0
    return worst <= MAX_PROFILE_DISTANCE, worst


def _print_profiles() -> None:
    """Print reference-profile literals for the currently processed data."""
    import config

    df = pd.read_csv(config.OUTPUT_FILE)
    profiles = cluster_profiles(df, df["Cluster"])
    names, distances = assign_archetypes(df, df["Cluster"])
    print(profiles.round(2).to_string())
    print()
    for cid in profiles.index:
        row = profiles.loc[cid]
        body = ", ".join(f'"{f}": {row[f]:.2f}' for f in SIGNATURE_FEATURES)
        print(f"# cluster {cid} -> {names[cid]} (distance {distances[cid]:.2f})")
        print(f"profile={{{body}}},")


if __name__ == "__main__":
    _print_profiles()
