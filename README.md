# NBA Player Clustering Dashboard

An interactive web application built with [Solara](https://solara.dev/) and [Plotly](https://plotly.com/) to cluster NBA players into archetypes based on their performance metrics and visualize their profiles using radar charts.

## Features
- **Player Archetype Clustering**: Uses K-Means to group players based on normalized stats.
- **Interactive PCA Visualization**: Explore the player space in 2D/3D.
- **Radar Charts**: Compare individual player profiles against cluster averages.
- **Reactive Dashboard**: Real-time filtering and selection using Solara.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/thompgt/nba-player-clustering.git
   cd nba-player-clustering
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   solara run app.py
   ```
