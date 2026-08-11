"""Centralized configuration for the NBA player clustering pipeline.

Single source of truth for parameters shared across preprocess.py,
validate_model.py, app.py, and the test suite, so they can't drift out
of sync (e.g. app.py's cluster labels vs. preprocess.py's cluster count).
"""

INPUT_FILE: str = "nba_stats.csv"
OUTPUT_FILE: str = "processed_nba_stats.csv"

N_CLUSTERS: int = 6
RANDOM_STATE: int = 42

# Minimum silhouette score for validate_model.py to consider the
# clustering "reasonable" for this high-dimensional, noisy stat space.
SILHOUETTE_THRESHOLD: float = 0.1

COUNTING_STATS: list[str] = ["PTS", "TRB", "AST", "STL", "BLK", "3P", "2P", "FT", "ORB", "DRB"]

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

CLUSTERING_FEATURES: list[str] = COUNTING_STATS + SHOOTING_PERCENTAGES

RADAR_FEATURES: list[str] = ["PTS", "TRB", "AST", "STL", "BLK", "3P"]

# Human-readable archetype names, descriptions and colors live in
# archetypes.py, keyed by statistical profile rather than by cluster index --
# K-Means indices are seed-dependent and must never carry meaning.
