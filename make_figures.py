"""Generate the analytical figures embedded in README.md.

Runs the repo's own pipeline artifacts (processed_nba_stats.csv, produced by
preprocess.py) plus a k-sweep over the same feature matrix, and writes PNGs to
docs/images/. Regenerate with:

    python preprocess.py && python make_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import archetypes
import config

plt.switch_backend("Agg")

OUT_DIR = os.path.join("docs", "images")

# Design tokens: single accent hue, neutral chrome. Categorical color is
# deliberately not used for the 6 clusters -- at 6 series in a scatter no
# hue set stays separable under color-vision deficiency, so cluster identity
# is carried by facet position and labels instead.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#2a78d6"
ACCENT_DARK = "#184f95"
CONTEXT = "#dcdbd5"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red",
    ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a9a8", "#e34948", "#8f2020"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": "#52514e",
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(config.OUTPUT_FILE)
    X = StandardScaler().fit_transform(df[config.CLUSTERING_FEATURES])
    return df, X


def fig_pca_facets(df: pd.DataFrame) -> str:
    """One small multiple per cluster: the chosen cluster against all players."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    for arch, ax in zip(archetypes.ARCHETYPES, axes.ravel()):
        sub = df[df["Archetype"] == arch.name]
        ax.scatter(df["PC1"], df["PC2"], s=9, c=CONTEXT, linewidths=0, zorder=1)
        ax.scatter(
            sub["PC1"], sub["PC2"], s=16, c=ACCENT, linewidths=0.4,
            edgecolors=SURFACE, zorder=2,
        )
        ax.set_title(f"{arch.name}  ({len(sub)} players)", loc="left", color=INK)
        ax.grid(True, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Label the most productive player in the cluster as an anchor.
        top = sub.loc[sub["PTS"].idxmax()]
        right_half = top["PC1"] > df["PC1"].median()
        ax.annotate(
            top["Player"], (top["PC1"], top["PC2"]), textcoords="offset points",
            xytext=(-6 if right_half else 6, 5), fontsize=8, color="#52514e",
            ha="right" if right_half else "left",
        )
    for ax in axes[-1]:
        ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    fig.suptitle(
        "NBA player archetypes in PCA space — each panel highlights one K-Means cluster",
        x=0.01, ha="left", fontsize=12, color=INK,
    )
    fig.tight_layout()
    return save(fig, "pca_clusters.png")


def fig_model_selection(X: np.ndarray) -> str:
    """Elbow + silhouette sweep over k, on the same scaled feature matrix."""
    ks = list(range(2, 13))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    chosen = config.N_CLUSTERS

    for ax, ys, title, ylab in (
        (ax1, inertia, "Elbow — within-cluster sum of squares", "Inertia"),
        (ax2, sil, "Silhouette score by k", "Mean silhouette"),
    ):
        ax.plot(ks, ys, linewidth=2, color=ACCENT, zorder=2)
        ax.scatter(ks, ys, s=30, color=ACCENT, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        i = ks.index(chosen)
        ax.scatter([chosen], [ys[i]], s=90, color=ACCENT_DARK, edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.annotate(
            f"k = {chosen} (shipped)", (chosen, ys[i]), textcoords="offset points",
            xytext=(10, 10), fontsize=9, color=INK,
        )
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylab)
        ax.set_xticks(ks)
        ax.grid(True, linewidth=0.6, axis="y")
        ax.set_axisbelow(True)

    ax2.axhline(
        config.SILHOUETTE_THRESHOLD, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1
    )
    ax2.annotate(
        f"validate_model.py threshold ({config.SILHOUETTE_THRESHOLD})",
        (ks[0], config.SILHOUETTE_THRESHOLD), textcoords="offset points",
        xytext=(2, 5), fontsize=8, color=INK_MUTED,
    )
    fig.tight_layout()
    return save(fig, "model_selection.png")


def fig_cluster_profiles(df: pd.DataFrame) -> str:
    """Heatmap of cluster mean per feature, in standard deviations from league mean."""
    feats = config.CLUSTERING_FEATURES
    z = (df[feats] - df[feats].mean()) / df[feats].std()
    z["Archetype"] = df["Archetype"]
    order = [a.name for a in archetypes.ARCHETYPES]
    mat = z.groupby("Archetype")[feats].mean().reindex(order)
    labels = [f"{name}  (n={int((df['Archetype'] == name).sum())})" for name in mat.index]

    lim = float(np.abs(mat.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(mat.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(feats)), feats)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(np.arange(len(feats) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Direct value labels: identity/magnitude never rests on color alone.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            ax.text(
                j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if abs(v) > lim * 0.55 else INK,
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Std. deviations from league average")
    cbar.outline.set_visible(False)
    ax.set_title(
        "Cluster statistical profiles — what separates each archetype",
        loc="left", color=INK, pad=10,
    )
    fig.tight_layout()
    return save(fig, "cluster_profiles.png")


def save(fig: plt.Figure, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path):,} bytes)")
    return path


def main() -> None:
    df, X = load()
    fig_pca_facets(df)
    fig_model_selection(X)
    fig_cluster_profiles(df)


if __name__ == "__main__":
    main()
