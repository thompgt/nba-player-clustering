"""Tests for the persisted fitted model.

The bug this guards against: the scaler and K-Means used to be discarded after
preprocessing and independently re-fit in the validator and the dashboard.
Nothing checked that the three agreed, so the dashboard's similarity search
silently depended on reconstructing the training transform by coincidence.
"""

import numpy as np
import pandas as pd
import pytest

import config
import model_store


@pytest.fixture(scope="module")
def model():
    return model_store.load()


@pytest.fixture(scope="module")
def processed():
    return pd.read_csv(config.OUTPUT_FILE)


def test_model_is_saved_alongside_the_data(model):
    assert model.features == config.CLUSTERING_FEATURES
    assert model.kmeans.n_clusters == config.N_CLUSTERS
    assert len(model.archetype_names) == config.N_CLUSTERS


def test_saved_model_reproduces_the_stored_cluster_assignments(model, processed):
    """The whole point: one fit, used everywhere."""
    import archetypes

    ranked = processed[processed["Cluster"] != archetypes.UNRANKED_CLUSTER]
    predicted = model.kmeans.predict(model.transform(ranked))
    assert (predicted == ranked["Cluster"].to_numpy()).all()


def test_saved_model_reproduces_the_stored_archetype_names(model, processed):
    import archetypes

    ranked = processed[processed["Cluster"] != archetypes.UNRANKED_CLUSTER]
    for cluster_id, name in ranked[["Cluster", "Archetype"]].drop_duplicates().to_numpy():
        assert model.archetype_of(int(cluster_id)) == name


def test_training_scaler_differs_from_a_naive_refit(model, processed):
    """A fresh scaler over the processed CSV is *not* the training transform.

    preprocess.py fits on ranked players only, so re-fitting over every row --
    which is what app.py and validate_model.py used to do -- centres the space
    somewhere else. This test documents that the difference is real, not
    theoretical.
    """
    from sklearn.preprocessing import StandardScaler

    naive = StandardScaler().fit_transform(processed[config.CLUSTERING_FEATURES])
    trained = model.transform(processed)
    assert not np.allclose(naive, trained)


def test_transform_rejects_missing_columns(model, processed):
    stripped = processed.drop(columns=[config.CLUSTERING_FEATURES[0]])
    with pytest.raises(ValueError, match="missing feature columns"):
        model.transform(stripped)


def test_unknown_cluster_id_maps_to_unranked(model):
    import archetypes

    assert model.archetype_of(archetypes.UNRANKED_CLUSTER) == archetypes.UNRANKED


def test_missing_model_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `python preprocess.py`"):
        model_store.load(str(tmp_path / "absent.joblib"))


def test_a_stale_feature_list_is_rejected(model, tmp_path, monkeypatch):
    """Changing CLUSTERING_FEATURES must invalidate a model fit on the old set."""
    path = str(tmp_path / "stale.joblib")
    stale = model_store.FittedModel(
        scaler=model.scaler,
        kmeans=model.kmeans,
        pca=model.pca,
        features=["PTS", "TRB"],
        archetype_names=model.archetype_names,
    )
    model_store.save(stale, path)
    with pytest.raises(ValueError, match="was fit on"):
        model_store.load(path)


def test_a_stale_format_version_is_rejected(model, tmp_path):
    path = str(tmp_path / "old_format.joblib")
    stale = model_store.FittedModel(
        scaler=model.scaler,
        kmeans=model.kmeans,
        pca=model.pca,
        features=model.features,
        archetype_names=model.archetype_names,
        version=model_store.FORMAT_VERSION - 1,
    )
    model_store.save(stale, path)
    with pytest.raises(ValueError, match="format version"):
        model_store.load(path)


def test_a_file_containing_something_else_is_rejected(tmp_path):
    import joblib

    path = str(tmp_path / "not_a_model.joblib")
    joblib.dump({"scaler": None}, path)
    with pytest.raises(ValueError, match="does not contain a FittedModel"):
        model_store.load(path)
