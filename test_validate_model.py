"""Tests for the quality gates.

The point of these is that the gates *can fail*. The silhouette threshold used
to be set at 0.1 while every k from 2 to 12 scored above it, so validation was
a tautology: it could not reject anything, including a clustering that had
degenerated completely.
"""

import numpy as np
import pandas as pd
import pytest

import config
import validate_model


@pytest.fixture(scope="module")
def processed():
    return pd.read_csv(config.OUTPUT_FILE)


def test_the_shipped_model_passes(processed, monkeypatch, tmp_path):
    assert validate_model.validate() is True


def test_silhouette_threshold_is_below_but_near_the_achieved_score(processed):
    """A gate set below every attainable value is not a gate."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    import archetypes

    ranked = processed[processed["Cluster"] != archetypes.UNRANKED_CLUSTER]
    X = StandardScaler().fit_transform(ranked[config.CLUSTERING_FEATURES])
    score = silhouette_score(X, ranked["Cluster"])

    assert score > config.SILHOUETTE_THRESHOLD, "shipped model must clear its own gate"
    # ...but only just. A threshold miles below the achieved score can never
    # catch a regression.
    assert score - config.SILHOUETTE_THRESHOLD < 0.05


def test_missing_output_file_fails_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_FILE", str(tmp_path / "nope.csv"))
    assert validate_model.validate() is False


def test_missing_column_fails_validation(processed, tmp_path, monkeypatch):
    path = tmp_path / "no_archetype.csv"
    processed.drop(columns=["Archetype"]).to_csv(path, index=False)
    monkeypatch.setattr(config, "OUTPUT_FILE", str(path))
    assert validate_model.validate() is False


def test_wrong_cluster_count_fails_validation(processed, tmp_path, monkeypatch):
    import archetypes

    df = processed.copy()
    ranked = df["Cluster"] != archetypes.UNRANKED_CLUSTER
    df.loc[ranked & (df["Cluster"] == df["Cluster"].max()), "Cluster"] = 0
    path = tmp_path / "collapsed.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(config, "OUTPUT_FILE", str(path))
    assert validate_model.validate() is False


def test_a_shuffled_clustering_fails_validation(processed, tmp_path, monkeypatch):
    """Random labels have near-zero silhouette; the gate must reject them."""
    import archetypes

    df = processed.copy()
    ranked = df["Cluster"] != archetypes.UNRANKED_CLUSTER
    rng = np.random.default_rng(0)
    df.loc[ranked, "Cluster"] = rng.integers(0, config.N_CLUSTERS, size=int(ranked.sum()))
    path = tmp_path / "shuffled.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(config, "OUTPUT_FILE", str(path))
    assert validate_model.validate() is False


def test_stability_gate_rejects_an_unreproducible_partition(processed):
    from sklearn.preprocessing import StandardScaler

    import archetypes

    ranked = processed[processed["Cluster"] != archetypes.UNRANKED_CLUSTER]
    X = StandardScaler().fit_transform(ranked[config.CLUSTERING_FEATURES])

    assert validate_model._validate_stability(X, ranked["Cluster"]) is True

    rng = np.random.default_rng(1)
    noise = pd.Series(rng.integers(0, config.N_CLUSTERS, size=len(ranked)), index=ranked.index)
    assert validate_model._validate_stability(X, noise) is False


def test_cluster_size_gate_rejects_a_degenerate_cluster():
    healthy = pd.Series([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100)
    assert validate_model._validate_cluster_sizes(healthy) is True

    degenerate = pd.Series([0] * 199 + [1] * 199 + [2] * 199 + [3] * 3)
    assert validate_model._validate_cluster_sizes(degenerate) is False


def test_archetype_gate_rejects_a_stale_name(processed, tmp_path, monkeypatch):
    """Swapping two archetype labels must be caught, not silently served."""
    import archetypes

    df = processed.copy()
    names = archetypes.ARCHETYPE_NAMES
    swap = {names[0]: names[1], names[1]: names[0]}
    df["Archetype"] = df["Archetype"].replace(swap)
    path = tmp_path / "swapped.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(config, "OUTPUT_FILE", str(path))
    assert validate_model.validate() is False
