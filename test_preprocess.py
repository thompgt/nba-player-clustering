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
