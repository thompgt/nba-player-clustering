# NBA Player Clustering Dashboard

An interactive web application built with [Solara](https://solara.dev/) and [Plotly](https://plotly.com/) to cluster NBA players into archetypes based on their performance metrics and visualize their profiles using radar charts.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![K--Means Clustering](https://img.shields.io/badge/K--Means%20Clustering-0B7261?style=for-the-badge)
![PCA](https://img.shields.io/badge/PCA-6A4C93?style=for-the-badge)
![Silhouette Analysis](https://img.shields.io/badge/Silhouette%20Analysis-B23A48?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

![The Solara dashboard running locally: PCA scatter colored by cluster, a radar chart comparing the selected player to their cluster average, and the cluster/player data tables below.](docs/images/dashboard.png)

*The dashboard at `http://localhost:8765` — pick a player in the sidebar and the PCA view, radar chart, and tables update reactively.*

## Features
- **Player Archetype Clustering**: Uses K-Means to group players based on normalized stats.
- **Interactive PCA Visualization**: Explore the player space in 2D, with the selected player (and an optional comparison player) starred.
- **Radar Charts**: Compare individual player profiles against each other and against the cluster average.
- **Similarity Search**: Find the players closest to the selected one in scaled stat space.
- **Cluster Explorer**: Per-archetype descriptions, colors, and average stat lines.
- **Player Explorer**: Name search, team filter, and CSV download of the filtered table.
- **Reactive Dashboard**: Real-time player selection using Solara.

## Architecture

```mermaid
flowchart TD
    subgraph src["Source data (committed to the repo)"]
        RAW["nba_stats.csv<br/><i>semicolon-delimited, latin1</i>"]
    end

    subgraph cfg["Configuration"]
        CONFIG["config.py<br/><i>file paths, N_CLUSTERS, RANDOM_STATE,<br/>feature lists, cluster names</i>"]
    end

    subgraph pipe["Pipeline — run_pipeline.py"]
        PRE["preprocess.py<br/>schema check → dedupe TOT rows →<br/>fillna → StandardScaler →<br/>KMeans(k=6) → PCA(3)"]
        VAL["validate_model.py<br/>column + silhouette gate<br/><i>exits non-zero on failure</i>"]
        TEST["test_preprocess.py<br/><i>pytest</i>"]
    end

    PROC[("processed_nba_stats.csv<br/><i>stats + Cluster + PC1/PC2/PC3</i>")]

    subgraph serve["Serving"]
        APP["app.py<br/><i>Solara components + Plotly figures</i>"]
        BROWSER(["Browser — localhost:8765"])
    end

    subgraph ops["Build & CI"]
        DOCKER["Dockerfile<br/><i>installs deps, runs preprocess.py<br/>at build time, serves on 8765</i>"]
        CI["GitHub Actions — .github/workflows/ci.yml<br/><i>ruff → mypy → pytest</i>"]
    end

    RAW --> PRE
    CONFIG -.-> PRE
    CONFIG -.-> VAL
    CONFIG -.-> APP
    PRE --> PROC
    PROC --> VAL
    PRE --> TEST
    PROC --> APP
    APP --> BROWSER
    RAW --> DOCKER
    DOCKER --> APP
    CI --> TEST
```

### How it fits together

`config.py` is the single source of truth — feature lists, `N_CLUSTERS`, `RANDOM_STATE`, and the human-readable cluster names all live there, so `preprocess.py`, `validate_model.py`, `app.py`, and the tests can't drift apart.

`preprocess.py` reads the committed `nba_stats.csv`, checks the required columns are present, collapses traded players onto their `TOT` (season-total) row, fills the shooting-percentage NaNs that mean "zero attempts", then standardizes the 13 clustering features and fits `KMeans(k=6)`. It also fits a 3-component PCA purely for visualization, and writes stats + `Cluster` + `PC1/PC2/PC3` out to `processed_nba_stats.csv`.

`validate_model.py` is a gate, not a report: it re-scales the processed features, computes the silhouette score, and exits non-zero if the score falls below `SILHOUETTE_THRESHOLD`. `run_pipeline.py` chains preprocess → validate → pytest and stops at the first failure.

`app.py` never re-fits anything. It loads `processed_nba_stats.csv` at import time and renders it — so serving is decoupled from training, and the dashboard starts instantly. The `Dockerfile` exploits this by running `preprocess.py` at *build* time, baking the processed CSV into the image so the container is ready to serve on start.

### Module responsibilities

| File | Responsibility |
|------|----------------|
| `config.py` | Shared configuration: input/output paths, `N_CLUSTERS`, `RANDOM_STATE`, silhouette threshold, clustering/radar feature lists, cluster display names. |
| `nba_stats.csv` | Committed raw input — per-game player stats (semicolon-delimited, `latin1`). |
| `preprocess.py` | Load → validate schema → dedupe traded players → scale → K-Means → PCA → write `processed_nba_stats.csv`. |
| `processed_nba_stats.csv` | Generated artifact consumed by both the validator and the app. Not the source of truth; regenerate it. |
| `validate_model.py` | Quality gate on the generated artifact: required columns present, silhouette score above threshold. Exit code drives CI/pipeline. |
| `run_pipeline.py` | Orchestrator — runs preprocess, validate, and tests in order, aborting on the first non-zero exit. |
| `app.py` | Solara dashboard: sidebar player/compare selectors, KPI strip, Plotly PCA scatter, radar chart, similarity search, cluster explorer, filterable player table with CSV export. |
| `make_figures.py` | Regenerates the analytical figures in `docs/images/` from the processed data. |
| `test_preprocess.py` | Pytest suite over `preprocess_data` — output shape, no NaNs, cluster count, `TOT` collapsing, and error handling. |
| `Dockerfile` | Container build: install deps, run preprocessing at build time, serve on port 8765. |
| `.github/workflows/ci.yml` | CI on push/PR to `main`: `ruff check` → `mypy` → `pytest`. |

## Model & clusters

These figures are generated by `make_figures.py` from the repo's own `processed_nba_stats.csv`.

### Where the archetypes sit in PCA space

![Six small-multiple scatter plots of PC1 vs PC2, one per cluster; the cluster's members are highlighted in blue against all other players in gray, with the highest-scoring player in each cluster labeled.](docs/images/pca_clusters.png)

*Each panel highlights one K-Means cluster against the full player population. PC1 tracks overall production and PC2 separates perimeter from interior play, so the archetypes occupy visibly distinct regions rather than overlapping blobs. Faceting rather than six colors keeps the clusters distinguishable for colorblind readers.*

### Why k = 6

![Two line charts: inertia falling smoothly from k=2 to k=12, and mean silhouette score peaking at k=2 then flattening around 0.22-0.24 between k=4 and k=7 before dropping, with k=6 marked on both.](docs/images/model_selection.png)

*The elbow is soft — per-game box-score stats are a continuum, not naturally separated groups. Silhouette is highest at k=2, but that just splits starters from bench. Within the interpretable range, k=6 sits at a local silhouette maximum (~0.24) and yields six archetypes a basketball fan would recognize. Every k shown clears the 0.1 threshold `validate_model.py` enforces.*

### What separates each archetype

![Heatmap of the six clusters against the thirteen clustering features, colored blue (below league average) to red (above), with the standard-deviation value printed in each cell.](docs/images/cluster_profiles.png)

*Cluster means in standard deviations from the league average. The structure is legible: Star Players are elevated across every counting stat; Starting Bigs spike on rebounding and blocks while sitting below average on 3P; Reserve Bigs share the interior shape at lower volume; and Limited Minutes / Specialists are uniformly negative, with FT% and FG% dragged down by tiny attempt counts.*

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/thompgt/nba-player-clustering.git
   cd nba-player-clustering
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess Data
If you have new data or want to re-run the clustering:
```bash
python preprocess.py
```

### 2. Validate Model
Run the validation script to check the silhouette score and cluster distribution:
```bash
python validate_model.py
```

### 3. Run Tests
Run the unit tests:
```bash
pytest
```

### 4. Run the Dashboard
```bash
solara run app.py
```
Then open http://localhost:8765.

### All of the above
`run_pipeline.py` runs preprocessing, validation, and the test suite in sequence, stopping at the first failure:
```bash
python run_pipeline.py
```

### Regenerate the README figures
Requires `matplotlib` (not in `requirements.txt` — the app itself doesn't need it):
```bash
pip install matplotlib
python make_figures.py
```

## Development

Install the dev dependencies and run the same checks CI runs:
```bash
pip install -r requirements-dev.txt
ruff check .
mypy preprocess.py validate_model.py run_pipeline.py config.py
pytest -v
```

## Running with Docker
Build and run the dashboard in a container (preprocessing runs at build time, so the image is ready to serve immediately):
```bash
docker build -t nba-player-clustering .
docker run -p 8765:8765 nba-player-clustering
```
Then open http://localhost:8765.

## Data Source
The project uses NBA player per-game stats sourced from [Basketball-Reference](https://www.basketball-reference.com/), committed to this repo as `nba_stats.csv` so the pipeline runs end-to-end from a fresh clone with no manual data-fetch step.

The file is semicolon-delimited (`;`) with `latin1` encoding and includes, per player-season row: identity/context columns (`Rk`, `Player`, `Pos`, `Age`, `Tm`, `G`, `GS`, `MP`), traditional counting stats (`PTS`, `TRB`, `AST`, `STL`, `BLK`, `ORB`, `DRB`, `TOV`, `PF`), and shooting stats with makes/attempts/percentages (`FG`/`FGA`/`FG%`, `3P`/`3PA`/`3P%`, `2P`/`2PA`/`2P%`, `eFG%`, `FT`/`FTA`/`FT%`). Players traded mid-season have a `Tm == 'TOT'` row aggregating their full-season totals, which `preprocess.py` uses in preference to the per-team split rows.

To refresh with a newer season, replace `nba_stats.csv` with an equivalent export in the same format and re-run `python preprocess.py`.

## License
This project is licensed under the [MIT License](LICENSE).
