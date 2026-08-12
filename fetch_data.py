"""Fetch the season's per-game stats and write nba_stats.csv.

There was no ingestion path at all: the README simply said to replace the CSV
with "an equivalent export in the same format", so reproducing the dataset
depended on a human scraping the right table and matching the schema by hand.

This script does it reproducibly:

* **Caches the raw response** under ``.cache/`` and reuses it, so re-running is
  free and does not hammer the source. ``--refresh`` forces a new request.
* **Retries with exponential backoff** on the transient failures a scraper
  actually hits -- 429, 5xx, connection resets -- and gives up loudly rather
  than writing a truncated file.
* **Writes the exact expected schema**, normalising the column names the source
  has changed over the years (``Team`` -> ``Tm``, ``2TM``/``3TM`` -> ``TOT``),
  and refuses to overwrite the CSV if a required column is missing.

Usage::

    python fetch_data.py                    # the configured season
    python fetch_data.py --season 2023      # a different one (end year)
    python fetch_data.py --refresh          # ignore the cache
    python fetch_data.py --output out.csv   # write somewhere else

Parsing is deliberately separated from fetching (``parse_per_game`` takes HTML
text) so the schema handling is testable without network access.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from io import StringIO

import pandas as pd

import config

logger = logging.getLogger(__name__)

CACHE_DIR = ".cache"

#: Basketball-Reference blocks the default urllib agent outright.
USER_AGENT = (
    "Mozilla/5.0 (compatible; nba-player-clustering/1.0; "
    "+https://github.com/thompgt/nba-player-clustering)"
)

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Columns the pipeline requires, in the order the committed CSV uses.
EXPECTED_COLUMNS = [
    "Rk", "Player", "Pos", "Age", "Tm", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
]

#: Source column names that have changed over the years.
COLUMN_ALIASES = {"Team": "Tm", "Tm.": "Tm", "Player-additional": None, "Awards": None}

#: The source has used several labels for a traded player's season-total row.
TOTAL_TEAM_LABELS = {"TOT", "2TM", "3TM", "4TM", "5TM"}


def season_url(season_end_year: int) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}_per_game.html"


def cache_path(season_end_year: int) -> str:
    return os.path.join(CACHE_DIR, f"NBA_{season_end_year}_per_game.html")


def fetch_html(url: str, cache_file: str, refresh: bool = False) -> str:
    """Return the page HTML, from cache when possible, retrying on failure."""
    if not refresh and os.path.exists(cache_file):
        logger.info("Using cached response %s (pass --refresh to re-download)", cache_file)
        with open(cache_file, encoding="utf-8") as fh:
            return fh.read()

    html = _get_with_retries(url)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("Cached raw response to %s (%d bytes)", cache_file, len(html))
    return html


def _get_with_retries(url: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            logger.info("GET %s (attempt %d/%d)", url, attempt, MAX_ATTEMPTS)
            with urllib.request.urlopen(request, timeout=30) as response:
                return str(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in RETRY_STATUS:
                raise RuntimeError(
                    f"{url} returned HTTP {exc.code} ({exc.reason}). This is not a transient "
                    "error, so retrying will not help."
                ) from exc
            logger.warning("HTTP %s from %s", exc.code, url)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            logger.warning("Request failed: %s", exc)

        if attempt < MAX_ATTEMPTS:
            # Exponential backoff with jitter, so parallel runs don't sync up.
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)) * (1 + random.random() * 0.3)
            logger.info("Retrying in %.1fs", delay)
            time.sleep(delay)

    raise RuntimeError(
        f"Gave up fetching {url} after {MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )


def parse_per_game(html: str) -> pd.DataFrame:
    """Extract and normalise the per-game table from a season page."""
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError("No tables found on the page. The source layout may have changed.")

    # The per-game table is the one carrying the stat columns.
    candidates = [t for t in tables if "PTS" in t.columns and "Player" in t.columns]
    if not candidates:
        raise ValueError(
            "No table with 'Player' and 'PTS' columns found. The source layout may have changed."
        )
    df = max(candidates, key=len).copy()

    # Rename or drop the columns the source has changed over the years.
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if v})
    df = df.drop(columns=[k for k, v in COLUMN_ALIASES.items() if v is None], errors="ignore")

    # Long tables repeat the header every 20 rows; those rows say Player == "Player".
    df = df[df["Player"] != "Player"]

    # Normalise the season-total row label for traded players. preprocess.py
    # keys its de-duplication on 'TOT'.
    if "Tm" in df.columns:
        df["Tm"] = df["Tm"].replace({label: "TOT" for label in TOTAL_TEAM_LABELS})

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parsed table is missing required columns {missing}. The source layout may have "
            "changed; the pipeline's schema is defined by EXPECTED_COLUMNS in fetch_data.py."
        )

    df = df[EXPECTED_COLUMNS]
    numeric = [c for c in EXPECTED_COLUMNS if c not in ("Player", "Pos", "Tm")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write in the exact dialect preprocess.py reads: ';' separated, latin1."""
    if df.empty:
        raise ValueError("Refusing to write an empty table.")
    df.to_csv(path, sep=";", index=False, encoding="latin1")
    logger.info("Wrote %s (%d rows, %d columns)", path, len(df), len(df.columns))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, default=config.SEASON_END_YEAR,
        help="season end year, e.g. 2024 for 2023-24 (default: %(default)s)",
    )
    parser.add_argument("--output", default=config.INPUT_FILE, help="output CSV path")
    parser.add_argument("--refresh", action="store_true", help="ignore the cached response")
    args = parser.parse_args(argv)

    url = season_url(args.season)
    try:
        html = fetch_html(url, cache_path(args.season), refresh=args.refresh)
        df = parse_per_game(html)
        write_csv(df, args.output)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.season != config.SEASON_END_YEAR:
        logger.warning(
            "Fetched season %s but config.SEASON_END_YEAR is %s. Update config.SEASON and "
            "config.SEASON_END_YEAR, then re-run the pipeline.",
            args.season, config.SEASON_END_YEAR,
        )
    logger.info("Next: python run_pipeline.py")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
