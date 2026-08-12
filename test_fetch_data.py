"""Tests for the ingestion script.

Parsing is separated from fetching so the schema handling -- which is the part
that actually breaks when the source changes -- is testable without touching
the network. Nothing here makes a request.
"""

import os

import pandas as pd
import pytest

import config
import fetch_data


def _row(rk, player, tm, **overrides):
    values = {
        "Rk": rk, "Player": player, "Pos": "SF", "Age": 25, "Team": tm, "G": 70, "GS": 70,
        "MP": 30.0, "FG": 7.0, "FGA": 15.0, "FG%": 0.467, "3P": 2.0, "3PA": 5.0, "3P%": 0.400,
        "2P": 5.0, "2PA": 10.0, "2P%": 0.500, "eFG%": 0.533, "FT": 3.0, "FTA": 4.0, "FT%": 0.750,
        "ORB": 1.0, "DRB": 5.0, "TRB": 6.0, "AST": 4.0, "STL": 1.0, "BLK": 0.5, "TOV": 2.0,
        "PF": 2.0, "PTS": 19.0, "Awards": "", "Player-additional": "abcd01",
    }
    values.update(overrides)
    return values


def _html(rows) -> str:
    """Render rows as a Basketball-Reference-shaped HTML table."""
    frame = pd.DataFrame(rows)
    return f"<html><body><h1>Ignore me</h1><table>{frame.to_html(index=False)}</table></body></html>"


@pytest.fixture
def sample_html():
    return _html(
        [
            _row(1, "Alpha Player", "BOS"),
            _row(2, "Beta Player", "2TM", G=60),
            _row(3, "Gamma Player", "LAL", PTS=25.0),
        ]
    )


def test_parses_the_expected_schema(sample_html):
    df = fetch_data.parse_per_game(sample_html)
    assert list(df.columns) == fetch_data.EXPECTED_COLUMNS
    assert len(df) == 3


def test_renames_the_team_column(sample_html):
    """The source renamed 'Tm' to 'Team'; the pipeline still expects 'Tm'."""
    df = fetch_data.parse_per_game(sample_html)
    assert "Tm" in df.columns
    assert "Team" not in df.columns


def test_normalises_traded_player_team_labels(sample_html):
    """'2TM'/'3TM' are the newer spellings of 'TOT', which preprocess.py keys on."""
    df = fetch_data.parse_per_game(sample_html)
    assert df.loc[df["Player"] == "Beta Player", "Tm"].iloc[0] == "TOT"


def test_drops_columns_the_pipeline_does_not_use(sample_html):
    df = fetch_data.parse_per_game(sample_html)
    assert "Awards" not in df.columns
    assert "Player-additional" not in df.columns


def test_drops_repeated_header_rows():
    """Long tables repeat the header every 20 rows."""
    rows = [_row(1, "Alpha Player", "BOS"), _row(2, "Player", "Tm"), _row(3, "Beta Player", "LAL")]
    df = fetch_data.parse_per_game(_html(rows))
    assert "Player" not in df["Player"].tolist()
    assert len(df) == 2


def test_numeric_columns_are_numeric(sample_html):
    df = fetch_data.parse_per_game(sample_html)
    for column in ("PTS", "MP", "G", "FG%"):
        assert pd.api.types.is_numeric_dtype(df[column])


def test_missing_required_column_is_reported():
    rows = [_row(1, "Alpha Player", "BOS")]
    del rows[0]["BLK"]
    with pytest.raises(ValueError, match="missing required columns"):
        fetch_data.parse_per_game(_html(rows))


def test_a_page_without_a_stat_table_is_reported():
    with pytest.raises(ValueError, match="layout may have changed"):
        fetch_data.parse_per_game("<html><body><table><tr><th>x</th></tr></table></body></html>")


def test_output_round_trips_through_preprocess_dialect(sample_html, tmp_path):
    """The written file must be readable by preprocess.py exactly as-is."""
    df = fetch_data.parse_per_game(sample_html)
    path = tmp_path / "out.csv"
    fetch_data.write_csv(df, str(path))

    reread = pd.read_csv(path, sep=";", encoding="latin1")
    assert list(reread.columns) == fetch_data.EXPECTED_COLUMNS
    assert len(reread) == len(df)


def test_written_schema_covers_what_preprocess_requires(sample_html):
    import preprocess

    df = fetch_data.parse_per_game(sample_html)
    missing = [c for c in preprocess.REQUIRED_COLUMNS if c not in df.columns]
    assert missing == [], f"fetch_data would produce a CSV preprocess.py rejects: {missing}"


def test_expected_schema_matches_the_committed_csv():
    """The fetcher must reproduce the file it is meant to replace."""
    committed = pd.read_csv(config.INPUT_FILE, sep=";", encoding="latin1")
    assert list(committed.columns) == fetch_data.EXPECTED_COLUMNS


def test_refuses_to_write_an_empty_table(tmp_path):
    with pytest.raises(ValueError, match="empty table"):
        fetch_data.write_csv(pd.DataFrame(), str(tmp_path / "empty.csv"))


def test_cache_is_reused_without_a_request(tmp_path):
    cache_file = tmp_path / "cached.html"
    cache_file.write_text("<html>cached</html>", encoding="utf-8")
    # No network stub needed: a request would fail, so returning proves the
    # cache short-circuits it.
    assert fetch_data.fetch_html("http://invalid.invalid", str(cache_file)) == "<html>cached</html>"


def test_url_matches_the_configured_season():
    assert fetch_data.season_url(config.SEASON_END_YEAR) == config.SOURCE_URL


def test_cache_path_is_season_specific():
    assert fetch_data.cache_path(2024) != fetch_data.cache_path(2023)
    assert "2024" in os.path.basename(fetch_data.cache_path(2024))


def test_non_transient_http_errors_are_not_retried(monkeypatch):
    import urllib.error

    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="not a transient error"):
        fetch_data._get_with_retries("http://x")
    assert len(calls) == 1


def test_transient_http_errors_are_retried_then_reported(monkeypatch):
    import urllib.error

    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", boom)
    monkeypatch.setattr(fetch_data.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Gave up fetching"):
        fetch_data._get_with_retries("http://x")
    assert len(calls) == fetch_data.MAX_ATTEMPTS


def test_a_transient_failure_that_recovers_succeeds(monkeypatch):
    import urllib.error

    state = {"n": 0}

    class Response:
        def read(self):
            return b"<html>ok</html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return Response()

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch_data.time, "sleep", lambda _: None)
    assert fetch_data._get_with_retries("http://x") == "<html>ok</html>"
    assert state["n"] == 3
