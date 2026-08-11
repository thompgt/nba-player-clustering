"""Tests for profile-based archetype naming.

The property that matters: an archetype's name must follow the *statistical
shape* of a cluster, not its K-Means index. Refitting with a different random
seed renumbers the clusters, so these tests refit and check the names stay put.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import archetypes
import config
from preprocess import preprocess_data

SEEDS = [0, 1, 7, 2024]


def _run(tmp_path, seed, monkeypatch):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(repo_root, config.INPUT_FILE), tmp_path / config.INPUT_FILE)
    monkeypatch.setattr(config, "RANDOM_STATE", seed)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return preprocess_data(config.INPUT_FILE)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("baseline")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(repo_root, config.INPUT_FILE), tmp_dir / config.INPUT_FILE)
    cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        return preprocess_data(config.INPUT_FILE)
    finally:
        os.chdir(cwd)


def _ranked(df):
    return df[df["Cluster"] != archetypes.UNRANKED_CLUSTER]


def test_every_row_gets_an_archetype(baseline):
    assert baseline["Archetype"].notna().all()
    assert set(baseline["Archetype"]) <= set(archetypes.ARCHETYPE_NAMES) | {archetypes.UNRANKED}


def test_archetype_names_are_a_bijection_over_clusters(baseline):
    """Two clusters must never collapse onto the same name."""
    pairs = _ranked(baseline)[["Cluster", "Archetype"]].drop_duplicates()
    assert len(pairs) == config.N_CLUSTERS
    assert pairs["Archetype"].nunique() == config.N_CLUSTERS


def test_archetype_count_matches_cluster_count():
    assert len(archetypes.ARCHETYPES) == config.N_CLUSTERS


@pytest.mark.parametrize("seed", SEEDS)
def test_names_follow_the_profile_regardless_of_seed(tmp_path, monkeypatch, seed):
    """The regression test for the seed-dependent-label bug.

    With names keyed to cluster indices, the league's leading scorer landed in
    index 2 at seed 42 but 3/0/0/0 at seeds 0/1/7/2024 -- so a fixed
    ``{2: "Star Players"}`` map mislabelled the entire dashboard. Naming by
    profile has to be immune to that: whatever the indices come out as, the
    highest-usage group is still "Primary Creators", the rim-protecting group
    is still "Interior Bigs", and the three-point group is still "Floor
    Spacers".
    """
    df = _ranked(_run(tmp_path, seed, monkeypatch))
    means = df.groupby("Archetype")[["PTS", "AST", "BLK", "TRB", "3P"]].mean()

    assert means["PTS"].idxmax() == "Primary Creators"
    assert means["AST"].idxmax() == "Primary Creators"
    assert means["BLK"].idxmax() == "Interior Bigs"
    assert means["TRB"].idxmax() == "Interior Bigs"
    assert means["3P"].idxmax() in {"Primary Creators", "Floor Spacers"}
    assert means["3P"].idxmin() == "Interior Bigs"


@pytest.mark.parametrize("seed", SEEDS)
def test_archetype_membership_is_stable_across_seeds(tmp_path, monkeypatch, baseline, seed):
    """Most players keep their archetype when the model is refit."""
    df = _run(tmp_path, seed, monkeypatch)
    merged = _ranked(baseline)[["Player", "Archetype"]].merge(
        _ranked(df)[["Player", "Archetype"]], on="Player", suffixes=("_base", "_seed")
    )
    agreement = (merged["Archetype_base"] == merged["Archetype_seed"]).mean()
    assert agreement > 0.9, f"only {agreement:.1%} of players kept their archetype at seed {seed}"


def test_profiles_are_measured_against_the_player_population(baseline):
    """Cluster profiles are player-population z-scores, so they average to ~0."""
    ranked = _ranked(baseline)
    profiles = archetypes.cluster_profiles(ranked, ranked["Cluster"])
    weights = ranked["Cluster"].value_counts().sort_index().to_numpy()
    weighted_mean = np.average(profiles.to_numpy(), axis=0, weights=weights)
    assert np.allclose(weighted_mean, 0.0, atol=1e-6)


def test_assignment_rejects_a_mismatched_cluster_count(baseline):
    ranked = _ranked(baseline)
    trimmed = ranked[ranked["Cluster"] != ranked["Cluster"].max()]
    with pytest.raises(ValueError, match="archetypes are defined"):
        archetypes.assign_archetypes(trimmed, trimmed["Cluster"])


def test_missing_signature_column_is_reported(baseline):
    stripped = baseline.drop(columns=["BLK"])
    with pytest.raises(ValueError, match="BLK"):
        archetypes.cluster_profiles(stripped, stripped["Cluster"])


def test_match_quality_flags_a_drifting_cluster():
    ok, worst = archetypes.match_quality({0: 0.2, 1: 0.4})
    assert ok and worst == pytest.approx(0.4)

    ok, worst = archetypes.match_quality({0: 0.2, 1: archetypes.MAX_PROFILE_DISTANCE + 0.1})
    assert not ok


def test_reference_profiles_are_distinct():
    """Archetypes must be far enough apart that matching is not a coin flip."""
    vectors = np.vstack([a.vector() for a in archetypes.ARCHETYPES])
    dists = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    assert dists.min() > 0.5


def test_processed_output_carries_the_archetype_column(baseline, tmp_path_factory):
    """The dashboard reads names off the CSV, so they must be persisted."""
    tmp_dir = tmp_path_factory.mktemp("persisted")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(os.path.join(repo_root, config.INPUT_FILE), tmp_dir / config.INPUT_FILE)
    cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        preprocess_data(config.INPUT_FILE)
        written = pd.read_csv(config.OUTPUT_FILE)
    finally:
        os.chdir(cwd)
    assert "Archetype" in written.columns
    assert written["Archetype"].notna().all()
