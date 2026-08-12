import functools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import solara

import archetypes
import config
import model_store

logger = logging.getLogger(__name__)

RADAR_FEATURES = config.RADAR_FEATURES
# Archetype order and color come from archetypes.py, so charts stay consistent
# with each other and survive a refit that renumbers the cluster indices.
# Unranked players (below the eligibility floors) trail the named archetypes.
ARCHETYPE_ORDER = [a.name for a in archetypes.ARCHETYPES] + [archetypes.UNRANKED]
ARCHETYPE_COLOR_MAP = {a.name: a.color for a in archetypes.ARCHETYPES}
ARCHETYPE_COLOR_MAP[archetypes.UNRANKED] = archetypes.UNRANKED_COLOR


@dataclass(frozen=True)
class Dashboard:
    """The data the page renders, loaded once and reused."""

    df: pd.DataFrame
    model: model_store.FittedModel
    #: Standardized feature matrix, used to find statistically similar players.
    #: Built with the *training* scaler rather than a fresh fit, so similarity
    #: is measured in the space the model was actually built in.
    features: np.ndarray

    @property
    def teams(self) -> list[str]:
        return sorted(self.df["Tm"].unique().tolist())

    @property
    def players(self) -> list[str]:
        return self.df["Player"].tolist()


@functools.cache
def load_data() -> Dashboard:
    """Read the pipeline artifacts. Cached, so the page renders don't re-read.

    This deliberately does *not* run at import time. Module-level I/O made
    ``app.py`` impossible to import without a generated CSV, which is why CI's
    mypy step skipped the largest file in the repo and no test ever touched it.
    """
    try:
        df = pd.read_csv(config.OUTPUT_FILE)
    except FileNotFoundError:
        logger.error(
            "%s not found. Run `python preprocess.py` first to generate it (see README).",
            config.OUTPUT_FILE,
        )
        raise

    if "Archetype" not in df.columns:
        raise ValueError(
            f"{config.OUTPUT_FILE} has no 'Archetype' column. Regenerate it with "
            "`python preprocess.py` (archetype names are assigned during preprocessing)."
        )

    model = model_store.load()
    return Dashboard(df=df, model=model, features=model.transform(df))


def player_position(frame: pd.DataFrame, player_name: str) -> int:
    """Positional index of a player, for indexing the NumPy feature matrix.

    ``frame.index[...]`` returns an index *label*, which only happens to equal
    the positional offset while the frame carries a default RangeIndex. Mixing
    that with ``.iloc`` and a positional NumPy array worked by coincidence and
    would have broken silently the first time the frame was filtered or sorted.
    ``flatnonzero`` is positional by construction.
    """
    positions = np.flatnonzero(frame["Player"].to_numpy() == player_name)
    if len(positions) == 0:
        raise KeyError(f"No player named {player_name!r} in the dataset.")
    if len(positions) > 1:
        # Two players sharing a name would otherwise resolve to whichever came
        # first, with no indication that the other exists.
        teams = ", ".join(frame.iloc[positions]["Tm"].astype(str))
        raise KeyError(
            f"{player_name!r} matches {len(positions)} rows ({teams}). "
            "Player names must be unique; check the TOT de-duplication in preprocess.py."
        )
    return int(positions[0])


def find_similar_players(data: Dashboard, player_name: str, n: int = 5) -> pd.DataFrame:
    idx = player_position(data.df, player_name)
    distances = np.linalg.norm(data.features - data.features[idx], axis=1)
    order = np.argsort(distances)
    order = order[order != idx][:n]
    result = data.df.iloc[order][["Player", "Tm", "Archetype"]].copy()
    result["Similarity"] = (100 / (1 + distances[order])).round(1)
    return result.reset_index(drop=True)


def percentile_in_cluster(data: Dashboard, player_row: pd.Series, stat: str) -> float:
    cluster_vals = data.df[data.df["Cluster"] == player_row["Cluster"]][stat]
    return float((cluster_vals < player_row[stat]).mean() * 100)


CUSTOM_CSS = """
.v-application { font-family: "Inter", "Segoe UI", sans-serif !important; }
.solara-card { border-radius: 10px !important; }
.app-header { padding: 4px 0 16px 0; }
.app-header h1 { margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }
"""


@solara.component
def ArchetypeBadge(name: str):
    color = ARCHETYPE_COLOR_MAP[name]
    solara.HTML(
        tag="span",
        unsafe_innerHTML=name,
        style=(
            f"background:{color};color:white;padding:4px 12px;border-radius:12px;"
            "font-weight:600;font-size:13px;display:inline-block;"
        ),
    )


@solara.component
def MetricCard(label: str, value: str, caption: str = ""):
    with solara.Column(gap="0px", style={"align-items": "center", "min-width": "100px"}):
        solara.Text(value, classes=["metric-value"])
        solara.Text(label, classes=["metric-label"])
        if caption:
            solara.Text(caption, style={"font-size": "12px", "color": "#aaa"})


@solara.component
def Page():
    solara.Title(f"NBA Player Archetypes — {config.SEASON}")
    solara.Style(CUSTOM_CSS)

    data = load_data()
    df = data.df

    selected_player, set_selected_player = solara.use_state(df["Player"].iloc[0])
    compare_player, set_compare_player = solara.use_state("None")
    team_filter, set_team_filter = solara.use_state([])
    name_search, set_name_search = solara.use_state("")

    player_row = df[df["Player"] == selected_player].iloc[0]
    archetype = str(player_row["Archetype"])

    with solara.Sidebar():
        solara.Markdown(
            f"**{config.SEASON} season** · per-game stats from "
            f"[{config.SOURCE_NAME}]({config.SOURCE_URL})",
            style={"font-size": "12px", "color": "#888"},
        )
        solara.Markdown("## Player")
        solara.Select(
            label="Select Player",
            value=selected_player,
            values=df["Player"].tolist(),
            on_value=set_selected_player,
        )
        solara.Select(
            label="Compare with (optional)",
            value=compare_player,
            values=["None"] + [p for p in df["Player"].tolist() if p != selected_player],
            on_value=set_compare_player,
        )

        solara.Markdown(f"### {player_row['Player']}")
        solara.Markdown(f"**Team:** {player_row['Tm']}")
        ArchetypeBadge(archetype)

        solara.Markdown("&nbsp;")
        with solara.Row(gap="4px"):
            MetricCard("PTS", f"{player_row['PTS']:.1f}")
            MetricCard("TRB", f"{player_row['TRB']:.1f}")
            MetricCard("AST", f"{player_row['AST']:.1f}")

    # --- KPI strip -----------------------------------------------------
    with solara.Card():
        with solara.Row(justify="space-around"):
            for stat in ["PTS", "TRB", "AST", "STL", "BLK"]:
                pct = percentile_in_cluster(data, player_row, stat)
                MetricCard(stat, f"{player_row[stat]:.1f}", f"{pct:.0f}th pct. in cluster")

    with solara.Columns([1, 1]):
        with solara.Card("Player Archetypes (PCA)"):
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="Archetype",
                category_orders={"Archetype": ARCHETYPE_ORDER},
                color_discrete_map=ARCHETYPE_COLOR_MAP,
                hover_data=["Player", "PTS", "TRB", "AST"],
                title="NBA Player Archetypes (PCA)",
                opacity=0.65,
            )
            fig.update_traces(marker=dict(size=7))

            highlight_players = [selected_player] + ([compare_player] if compare_player != "None" else [])
            highlight_df = df[df["Player"].isin(highlight_players)]
            fig.add_trace(
                go.Scatter(
                    x=highlight_df["PC1"],
                    y=highlight_df["PC2"],
                    mode="markers+text",
                    text=highlight_df["Player"],
                    textposition="top center",
                    marker=dict(size=16, symbol="star", color="black", line=dict(width=1, color="white")),
                    name="Selected",
                    showlegend=False,
                )
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            solara.FigurePlotly(fig)

        with solara.Card("Player Profile (Radar Chart)"):
            max_stats = df[RADAR_FEATURES].max()

            def normalized(name: str) -> pd.Series:
                return df[df["Player"] == name][RADAR_FEATURES].iloc[0] / max_stats

            cluster_avg = df[df["Archetype"] == archetype][RADAR_FEATURES].mean() / max_stats

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=normalized(selected_player).values,
                    theta=RADAR_FEATURES,
                    fill="toself",
                    name=selected_player,
                    line_color="#4C78A8",
                )
            )
            if compare_player != "None":
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=normalized(compare_player).values,
                        theta=RADAR_FEATURES,
                        fill="toself",
                        name=compare_player,
                        line_color="#F58518",
                    )
                )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=cluster_avg.values,
                    theta=RADAR_FEATURES,
                    fill="toself",
                    name=f"{archetype} Average",
                    line_color="rgba(150, 150, 150, 0.7)",
                )
            )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                title="Stats Profile",
                margin=dict(l=50, r=60, t=50, b=90),
                height=480,
            )
            solara.FigurePlotly(fig_radar)

    with solara.Card("Similar Players"):
        solara.Markdown(f"Players most statistically similar to **{selected_player}**, by scaled per-game stats.")
        solara.DataFrame(find_similar_players(data, selected_player))

    with solara.Card("Archetype Explorer"):
        blurbs = [(a.name, a.description) for a in archetypes.ARCHETYPES]
        blurbs.append(
            (
                archetypes.UNRANKED,
                f"Below the eligibility floors ({config.MIN_MINUTES_PER_GAME:.0f} minutes per "
                f"game and {config.MIN_GAMES} games), so they are reported but not clustered — "
                "a handful of minutes is a sample size, not a playing style.",
            )
        )
        for name, description in blurbs:
            members = df[df["Archetype"] == name]
            with solara.Details(summary=f"{name}  ({len(members)} players)"):
                with solara.Column(gap="8px"):
                    ArchetypeBadge(name)
                    solara.Markdown(description)
                    avg = members[RADAR_FEATURES].mean().round(1)
                    solara.Markdown(
                        " · ".join(f"**{stat}:** {avg[stat]}" for stat in RADAR_FEATURES)
                    )

    with solara.Card("Player Explorer"):
        with solara.Row():
            solara.InputText("Search player name", value=name_search, on_value=set_name_search)
            solara.SelectMultiple(
                label="Filter by team",
                values=team_filter,
                all_values=data.teams,
                on_value=set_team_filter,
            )

        filtered = df
        if name_search:
            filtered = filtered[filtered["Player"].str.contains(name_search, case=False, na=False)]
        if team_filter:
            filtered = filtered[filtered["Tm"].isin(team_filter)]

        solara.Markdown(f"{len(filtered)} of {len(df)} players")
        solara.DataFrame(filtered[["Player", "Tm", "PTS", "TRB", "AST", "STL", "BLK", "Archetype"]])
        solara.FileDownload(
            lambda: filtered.to_csv(index=False),
            filename="nba_player_clusters_filtered.csv",
            label="Download filtered data (CSV)",
        )


@solara.component
def Layout(children):
    return solara.AppLayout(
        children=children,
        title=f"NBA Player Archetypes · {config.SEASON}",
    )
