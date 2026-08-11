"""Tests for the k-selection sweep and its defensibility checks."""

import numpy as np
import pandas as pd
import pytest

import select_k


@pytest.fixture(scope="module")
def blobs():
    """Three well-separated blobs: k=3 should win clearly."""
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return np.vstack([c + rng.normal(scale=0.5, size=(60, 2)) for c in centres])


def _table(**columns):
    return pd.DataFrame(columns).set_index("k")


def test_sweep_reports_one_row_per_k(blobs):
    table = select_k.sweep(blobs, range(2, 6))
    assert list(table.index) == [2, 3, 4, 5]
    for col in ("silhouette", "inertia", "min_cluster_size", "stability", "stability_worst"):
        assert col in table.columns


def test_sweep_finds_the_true_cluster_count(blobs):
    table = select_k.sweep(blobs, range(2, 6))
    assert table["silhouette"].idxmax() == 3


def test_sweep_inertia_decreases_with_k(blobs):
    table = select_k.sweep(blobs, range(2, 6))
    assert table["inertia"].is_monotonic_decreasing


def test_local_maximum_detection():
    table = _table(k=[4, 5, 6, 7], silhouette=[0.20, 0.30, 0.25, 0.10])
    assert select_k.is_local_maximum(table, 5)
    assert not select_k.is_local_maximum(table, 6)
    assert not select_k.is_local_maximum(table, 99)


def test_report_is_silent_for_a_defensible_k():
    table = _table(
        k=[4, 5, 6],
        silhouette=[0.20, 0.30, 0.25],
        min_cluster_size=[50, 50, 50],
        stability=[0.9, 0.9, 0.9],
    )
    assert select_k.report(table, 5, n_players=500) == []


def test_report_flags_a_k_that_is_not_a_local_maximum():
    table = _table(
        k=[4, 5, 6],
        silhouette=[0.20, 0.30, 0.25],
        min_cluster_size=[50, 50, 50],
        stability=[0.9, 0.9, 0.9],
    )
    problems = select_k.report(table, 6, n_players=500)
    assert any("local silhouette maximum" in p for p in problems)


def test_report_flags_an_unstable_k():
    table = _table(
        k=[4, 5, 6],
        silhouette=[0.20, 0.30, 0.25],
        min_cluster_size=[50, 50, 50],
        stability=[0.9, 0.4, 0.9],
    )
    problems = select_k.report(table, 5, n_players=500)
    assert any("unstable across seeds" in p for p in problems)


def test_report_flags_a_degenerate_cluster():
    table = _table(
        k=[4, 5, 6],
        silhouette=[0.20, 0.30, 0.25],
        min_cluster_size=[50, 3, 50],
        stability=[0.9, 0.9, 0.9],
    )
    problems = select_k.report(table, 5, n_players=500)
    assert any("sampling artifact" in p for p in problems)


def test_report_flags_a_k_below_the_interpretable_floor():
    table = _table(
        k=[2, 3, 4],
        silhouette=[0.50, 0.40, 0.30],
        min_cluster_size=[50, 50, 50],
        stability=[0.9, 0.9, 0.9],
    )
    problems = select_k.report(table, 2, n_players=500)
    assert any("too coarse" in p for p in problems)


def test_report_flags_a_k_outside_the_swept_range():
    table = _table(k=[4, 5], silhouette=[0.2, 0.3], min_cluster_size=[50, 50], stability=[0.9, 0.9])
    problems = select_k.report(table, 20, n_players=500)
    assert any("outside the swept range" in p for p in problems)


def test_interpretable_slice_drops_the_coarse_end():
    table = _table(k=[2, 3, 4, 5], silhouette=[0.5, 0.4, 0.3, 0.2])
    assert list(select_k.interpretable(table).index) == [4, 5]
