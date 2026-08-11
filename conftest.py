"""Shared pytest setup.

Some tests validate the *shipped* artifact rather than a throwaway fit, so
``processed_nba_stats.csv`` has to exist. It is generated (and gitignored)
rather than committed, so build it once per session if it is missing -- that
keeps `pytest` working from a fresh clone and in CI without a manual step.
"""

import os

import pytest

import config


@pytest.fixture(scope="session", autouse=True)
def ensure_processed_data():
    if not os.path.exists(config.OUTPUT_FILE):
        from preprocess import preprocess_data

        preprocess_data(config.INPUT_FILE)
