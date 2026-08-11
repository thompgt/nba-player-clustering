"""Persist the fitted model so nothing has to re-derive it.

The scaler, K-Means and PCA used to be thrown away as soon as
``preprocess.py`` finished, and then independently re-fit by
``validate_model.py`` and ``app.py``. Four ``fit_transform`` calls over the
same data, with nothing enforcing that they agreed: the dashboard's similarity
search silently depended on reconstructing the exact training transform from a
CSV, and would have drifted the moment the two diverged (a different row
filter, a column reordering, a changed feature list).

They are now fit once, saved here, and loaded everywhere else.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)

#: Bumped when the payload's shape changes, so a stale file is rejected with a
#: clear message rather than an AttributeError deep in a transform.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class FittedModel:
    """Everything needed to place a new player in the shipped model."""

    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA
    features: list[str]
    archetype_names: dict[int, str]
    version: int = FORMAT_VERSION

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardise rows with the *training* scaler, not a fresh one."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Cannot transform: missing feature columns {missing}")
        return np.asarray(self.scaler.transform(df[self.features]))

    def archetype_of(self, cluster_id: int) -> str:
        import archetypes

        return self.archetype_names.get(int(cluster_id), archetypes.UNRANKED)


def save(model: FittedModel, path: str | None = None) -> str:
    path = path or config.MODEL_FILE
    joblib.dump(model, path)
    logger.info("Saved fitted model to %s", path)
    return path


def load(path: str | None = None) -> FittedModel:
    path = path or config.MODEL_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fitted model '{path}' not found. Run `python preprocess.py` first to generate it."
        )

    model = joblib.load(path)
    if not isinstance(model, FittedModel):
        raise ValueError(f"'{path}' does not contain a FittedModel. Regenerate it.")
    if model.version != FORMAT_VERSION:
        raise ValueError(
            f"'{path}' was written by format version {model.version}, this code expects "
            f"{FORMAT_VERSION}. Regenerate it with `python preprocess.py`."
        )
    if model.features != config.CLUSTERING_FEATURES:
        raise ValueError(
            f"'{path}' was fit on {model.features}, but config.CLUSTERING_FEATURES is now "
            f"{config.CLUSTERING_FEATURES}. Regenerate it with `python preprocess.py`."
        )
    return model
