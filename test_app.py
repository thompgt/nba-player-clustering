"""Smoke tests for the dashboard module.

``app.py`` is the largest file in the repo and used to be entirely untested and
unchecked: module-level I/O meant importing it required a generated CSV, so
CI's mypy step skipped it and no test could touch it. Now that loading is
deferred behind ``load_data()``, it can be imported and exercised like any
other module.
"""

import numpy as np
import pandas as pd
import pytest

import app
import archetypes
import config


@pytest.fixture(scope="module")
def data():
    return app.load_data()


def test_module_imports_without_reading_anything():
    """Importing must not require a generated artifact to exist."""
    import importlib

    importlib.reload(app)  # no exception even though nothing is loaded yet


def test_load_data_is_cached():
    assert app.load_data() is app.load_data()


def test_load_data_shape(data):
    assert len(data.df) > 0
    assert data.features.shape == (len(data.df), len(config.CLUSTERING_FEATURES))
    assert "Archetype" in data.df.columns


def test_every_archetype_has_a_color(data):
    for name in data.df["Archetype"].unique():
        assert name in app.ARCHETYPE_COLOR_MAP
    assert archetypes.UNRANKED in app.ARCHETYPE_ORDER


def test_player_position_is_positional_not_a_label():
    """The bug: an index *label* was used to index a positional NumPy array.

    With a default RangeIndex the two coincide, which is why it worked. Filter
    the frame and they diverge -- the old code would then have silently
    compared the wrong player's stats.
    """
    frame = pd.DataFrame({"Player": ["A", "B", "C", "D"], "Tm": ["X"] * 4})
    filtered = frame[frame["Player"].isin(["C", "D"])]

    assert list(filtered.index) == [2, 3]  # labels, not positions
    assert app.player_position(filtered, "C") == 0
    assert app.player_position(filtered, "D") == 1


def test_player_position_rejects_an_unknown_name(data):
    with pytest.raises(KeyError, match="No player named"):
        app.player_position(data.df, "Nobody At All")


def test_player_position_rejects_a_duplicate_name():
    frame = pd.DataFrame({"Player": ["A", "A"], "Tm": ["X", "Y"]})
    with pytest.raises(KeyError, match="matches 2 rows"):
        app.player_position(frame, "A")


def test_player_names_are_unique(data):
    """Lookups are by name, so duplicates would resolve silently."""
    assert not data.df["Player"].duplicated().any()


def test_find_similar_players_excludes_the_query(data):
    name = data.df["Player"].iloc[0]
    similar = app.find_similar_players(data, name, n=5)
    assert len(similar) == 5
    assert name not in similar["Player"].tolist()


def test_find_similar_players_is_ordered_by_similarity(data):
    similar = app.find_similar_players(data, "Stephen Curry", n=8)
    assert similar["Similarity"].is_monotonic_decreasing


def test_similar_players_mostly_share_an_archetype(data):
    """Sanity check that the distance metric is measuring something real."""
    agreed = 0
    sample = data.df[data.df["Archetype"] != archetypes.UNRANKED]["Player"].head(40)
    for name in sample:
        own = data.df.loc[data.df["Player"] == name, "Archetype"].iloc[0]
        neighbours = app.find_similar_players(data, name, n=5)["Archetype"]
        agreed += (neighbours == own).mean()
    assert agreed / len(sample) > 0.6


def test_percentile_in_cluster_is_bounded(data):
    row = data.df.iloc[0]
    for stat in config.RADAR_FEATURES:
        pct = app.percentile_in_cluster(data, row, stat)
        assert 0.0 <= pct <= 100.0


def test_similarity_uses_the_training_transform(data):
    """Not a freshly fit scaler over the processed CSV."""
    from sklearn.preprocessing import StandardScaler

    naive = StandardScaler().fit_transform(data.df[config.CLUSTERING_FEATURES])
    assert not np.allclose(naive, data.features)
