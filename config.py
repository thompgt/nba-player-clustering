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
SHOOTING_PERCENTAGES: list[str] = ["FG%", "3P%", "FT%"]
CLUSTERING_FEATURES: list[str] = COUNTING_STATS + SHOOTING_PERCENTAGES

RADAR_FEATURES: list[str] = ["PTS", "TRB", "AST", "STL", "BLK", "3P"]

CLUSTER_NAMES: dict[int, str] = {
    0: "Role Players / Shooters",
    1: "Bench Guards/Wings",
    2: "Star Players",
    3: "Limited Minutes / Specialists",
    4: "Starting Bigs",
    5: "Reserve Bigs",
}

assert len(CLUSTER_NAMES) == N_CLUSTERS, "CLUSTER_NAMES must have one entry per cluster in N_CLUSTERS"
