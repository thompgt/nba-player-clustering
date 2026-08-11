import os
import shutil

import pandas as pd
import pytest

import config
from preprocess import preprocess_data

INPUT_FILE = config.INPUT_FILE
OUTPUT_FILE = config.OUTPUT_FILE
EXPECTED_N_CLUSTERS = config.N_CLUSTERS


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Run preprocess_data once in an isolated tmp dir so tests don't
    depend on run order and don't clobber the real processed_nba_stats.csv."""
    tmp_dir = tmp_path_factory.mktemp("preprocess")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(repo_root, INPUT_FILE), tmp_dir / INPUT_FILE)

    cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        df = preprocess_data(INPUT_FILE)
    finally:
        os.chdir(cwd)

    return df, tmp_dir / OUTPUT_FILE


def test_preprocess_data_shape_and_columns(processed):
    df, _ = processed
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ("Cluster", "PC1", "PC2", "PC3"):
        assert col in df.columns


def test_preprocess_data_no_missing_values(processed):
    df, _ = processed
    assert df.isnull().sum().sum() == 0


def test_preprocess_data_writes_output_file(processed):
    _, output_path = processed
    assert output_path.exists()


def test_cluster_count_matches_configured_n_clusters(processed):
    df, _ = processed
    assert df["Cluster"].nunique() == EXPECTED_N_CLUSTERS


def test_multi_team_players_collapse_to_tot_row(processed):
    df, _ = processed
    raw = pd.read_csv(INPUT_FILE, sep=";", encoding="latin1")
    traded_players = raw[raw["Tm"] != "TOT"]["Player"]
    traded_players = traded_players[traded_players.duplicated(keep=False)].unique()
    assert len(traded_players) > 0  # sanity check: fixture data has traded players

    for player in traded_players:
        rows = df[df["Player"] == player]
        assert len(rows) == 1
        assert rows.iloc[0]["Tm"] == "TOT"


def test_zero_attempt_shooters_are_not_treated_as_the_worst_in_the_league(processed):
    """The source stores 0.0, not NaN, for a player who never took the shot."""
    df, _ = processed
    for pct, (_, attempts) in config.SHOOTING_RATES.items():
        never_attempted = df[df[attempts] == 0]
        assert len(never_attempted) > 0, f"fixture data should have players with no {attempts}"
        # They land on the league rate, not on 0 -- and so are never the minimum.
        assert (never_attempted[pct] > 0).all()
        assert never_attempted[pct].nunique() == 1
        assert never_attempted[pct].iloc[0] > df[pct].min()


def test_shooting_percentages_are_shrunk_toward_the_league_rate(processed):
    """A tiny sample must not produce a 0% or 100% shooter."""
    df, _ = processed
    raw = pd.read_csv(INPUT_FILE, sep=";", encoding="latin1")
    for pct in config.SHOOTING_RATES:
        assert df[pct].min() > raw[pct].min() or raw[pct].min() > 0
        assert df[pct].max() < 1.0


def test_high_volume_shooters_are_barely_moved(processed):
    """Shrinkage is proportional to sample size, so regulars keep their rate."""
    df, _ = processed
    raw = pd.read_csv(INPUT_FILE, sep=";", encoding="latin1")
    merged = df[["Player", "Tm", "FGA", "FG%"]].merge(
        raw[["Player", "Tm", "FG%"]], on=["Player", "Tm"], suffixes=("_shrunk", "_raw")
    )
    high_volume = merged[merged["FGA"] > 15]
    assert len(high_volume) > 0
    shift = (high_volume["FG%_shrunk"] - high_volume["FG%_raw"]).abs()
    # k=2 prior attempts against >15 attempts/game moves a rate by <2 points.
    assert shift.max() < 0.02


def test_preprocess_does_not_chain_assign_on_a_view(tmp_path):
    """The boolean filter is copied, so later column writes aren't on a view."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(repo_root, INPUT_FILE), tmp_path / INPUT_FILE)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # "raise" turns what would be a SettingWithCopyWarning into an error.
        with pd.option_context("mode.chained_assignment", "raise"):
            preprocess_data(INPUT_FILE)
    finally:
        os.chdir(cwd)


def test_missing_required_column_raises_clear_error(tmp_path):
    raw = pd.read_csv(INPUT_FILE, sep=";", encoding="latin1")
    bad_file = tmp_path / "missing_column.csv"
    raw.drop(columns=["PTS"]).to_csv(bad_file, sep=";", index=False, encoding="latin1")

    with pytest.raises(ValueError, match="PTS"):
        preprocess_data(str(bad_file))


def test_missing_input_file_raises_clear_error(tmp_path):
    missing_file = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="not found"):
        preprocess_data(str(missing_file))
