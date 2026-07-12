import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import solara

import config

logger = logging.getLogger(__name__)

# Load data
try:
    df = pd.read_csv(config.OUTPUT_FILE)
except FileNotFoundError:
    logger.error(
        "%s not found. Run `python preprocess.py` first to generate it (see README).",
        config.OUTPUT_FILE,
    )
    raise

df['Cluster Name'] = df['Cluster'].map(config.CLUSTER_NAMES)

# Features for radar chart
radar_features = config.RADAR_FEATURES

@solara.component
def Page():
    selected_player, set_selected_player = solara.use_state(df['Player'].iloc[0])
    
    with solara.Sidebar():
        solara.Markdown("## Filters")
        solara.Select(
            label="Select Player",
            value=selected_player,
            values=df['Player'].tolist(),
            on_value=set_selected_player,
        )
        
        player_data = df[df['Player'] == selected_player].iloc[0]
        solara.Markdown(f"### {player_data['Player']}")
        solara.Markdown(f"**Team:** {player_data['Tm']}")
        solara.Markdown(f"**Cluster:** {player_data['Cluster Name']}")

    with solara.Columns([1, 1]):
        with solara.Card("PCA Visualization"):
            # 2D PCA Plot
            fig = px.scatter(
                df, x='PC1', y='PC2', color='Cluster Name',
                hover_data=['Player', 'PTS', 'TRB', 'AST'],
                title="NBA Player Archetypes (PCA)"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            solara.FigurePlotly(fig)
            
        with solara.Card("Player Profile (Radar Chart)"):
            player_stats = df[df['Player'] == selected_player][radar_features].iloc[0]
            
            # Normalize for radar chart visualization (simple max scaling)
            max_stats = df[radar_features].max()
            normalized_player_stats = player_stats / max_stats
            
            # Also get cluster average
            cluster_id = df[df['Player'] == selected_player]['Cluster'].iloc[0]
            cluster_avg = df[df['Cluster'] == cluster_id][radar_features].mean() / max_stats

            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=normalized_player_stats.values,
                theta=radar_features,
                fill='toself',
                name=selected_player
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_avg.values,
                theta=radar_features,
                fill='toself',
                name=f"{config.CLUSTER_NAMES[cluster_id]} Average",
                line_color='rgba(255, 0, 0, 0.5)'
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title=f"Stats Profile: {selected_player}"
            )
            solara.FigurePlotly(fig_radar)

    with solara.Card("Cluster Profiles (Averages)"):
        cluster_summary = df.groupby('Cluster Name')[radar_features].mean().reset_index()
        solara.DataFrame(cluster_summary)

    with solara.Card("Cluster Data"):
        solara.DataFrame(df[['Player', 'Tm', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'Cluster Name']])

@solara.component
def Layout(children):
    return solara.AppLayout(children=children)
