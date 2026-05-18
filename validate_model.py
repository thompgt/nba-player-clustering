import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

def validate():
    if not os.path.exists('processed_nba_stats.csv'):
        print("Error: processed_nba_stats.csv not found.")
        return
    
    df = pd.read_csv('processed_nba_stats.csv')
    
    # Check columns
    expected_cols = ['Player', 'Cluster', 'PC1', 'PC2', 'PC3']
    for col in expected_cols:
        if col not in df.columns:
            print(f"Error: Missing column {col}")
            return
    
    print("Columns validation passed.")
    
    # Calculate Silhouette Score
    clustering_features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', '3P', '2P', 'FT', 'ORB', 'DRB', 'FG%', '3P%', 'FT%']
    X = df[clustering_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    score = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score: {score:.4f}")
    
    # Basic Cluster Stats
    print("\nCluster Distribution:")
    print(df['Cluster'].value_counts().sort_index())
    
    if score > 0.1: # Threshold for "something reasonable" in high-dimensional player stats
        print("\nValidation Successful!")
    else:
        print("\nValidation Warning: Low Silhouette Score. Model might need tuning.")

if __name__ == "__main__":
    validate()
