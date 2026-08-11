"""Centralized configuration for the NBA player clustering pipeline.

Single source of truth for parameters shared across preprocess.py,
validate_model.py, app.py, and the test suite, so they can't drift out
of sync (e.g. app.py's cluster labels vs. preprocess.py's cluster count).
"""

INPUT_FILE: str = "nba_stats.csv"
OUTPUT_FILE: str = "processed_nba_stats.csv"

# Re-derived by select_k.py, whose table is committed as model_selection.csv.
# Within the interpretable range k=4 is the silhouette local maximum (0.1605)
# and by far the most stable (mean ARI 0.98 across five seeds, smallest cluster
# 75 players).
N_CLUSTERS: int = 4
RANDOM_STATE: int = 42

# Minimum silhouette score for validate_model.py to consider the
# clustering "reasonable" for this high-dimensional, noisy stat space.
SILHOUETTE_THRESHOLD: float = 0.1

# --- Eligibility -----------------------------------------------------------
# A player who logged 6.7 minutes across 8.7 games does not have a playing
# style, only a small sample. Clustering them alongside rotation players used
# to manufacture a whole archetype out of sampling noise, so they are excluded
# from the fit and reported as unranked instead.
MIN_MINUTES_PER_GAME: float = 10.0
MIN_GAMES: int = 20

# --- Features --------------------------------------------------------------
# Counting stats are converted to per-36-minutes rates before scaling. On raw
# per-game totals k-means is mostly a minutes sorter: it separates opportunity,
# not style. Rates hold playing time constant so the clusters describe how a
# player plays rather than how much. MP stays a reported column.
RATE_STATS: list[str] = ["2P", "3P", "FT", "ORB", "DRB", "AST", "STL", "BLK"]
PER_36_SUFFIX: str = "/36"
PER_36_FEATURES: list[str] = [f"{stat}{PER_36_SUFFIX}" for stat in RATE_STATS]

# Note the aggregates are deliberately absent: PTS is 2*2P + 3*3P + FT and TRB
# is ORB + DRB, so including both the components and their sums weighted
# scoring roughly 4x and rebounding 2x against steals and blocks in the
# Euclidean distance. The components are kept; the sums are reported, not fit.

# Shooting percentage -> (makes column, attempts column). The source CSV stores
# 0.0, not NaN, for a player who never attempted that shot type, which reads as
# "worst shooter in the league" once scaled. preprocess.py uses the makes and
# attempts columns to shrink each percentage toward the league rate instead.
SHOOTING_RATES: dict[str, tuple[str, str]] = {
    "FG%": ("FG", "FGA"),
    "3P%": ("3P", "3PA"),
    "FT%": ("FT", "FTA"),
}
SHOOTING_PERCENTAGES: list[str] = list(SHOOTING_RATES)

# Strength of the shrinkage prior, in attempts per game. A player with this
# many attempts per game lands halfway between their own rate and the league
# rate; a player with none sits exactly on the league rate.
SHOOTING_PRIOR_ATTEMPTS: float = 2.0

CLUSTERING_FEATURES: list[str] = PER_36_FEATURES + SHOOTING_PERCENTAGES

RADAR_FEATURES: list[str] = ["PTS", "TRB", "AST", "STL", "BLK", "3P"]

# Human-readable archetype names, descriptions and colors live in
# archetypes.py, keyed by statistical profile rather than by cluster index --
# K-Means indices are seed-dependent and must never carry meaning.
