"""
Build heterogeneous graphs for each game with richer features.
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import pickle
import os
from collections import defaultdict

def compute_pitcher_stats(raw_df):
    """Compute rolling stats for pitchers (similar to batters)."""
    print("Computing pitcher rolling stats...")
    
    # Compute per-game wOBA allowed for each pitcher
    pitcher_stats = raw_df.groupby(['game_pk', 'game_date', 'pitcher']).agg({
        'woba_value': 'sum',
        'woba_denom': 'sum',
    }).reset_index()
    
    pitcher_stats['woba_allowed'] = pitcher_stats['woba_value'] / pitcher_stats['woba_denom'].replace(0, 1)
    
    # Sort by pitcher and date
    pitcher_stats = pitcher_stats.sort_values(['pitcher', 'game_date'])
    
    # Compute rolling average (10 game window)
    pitcher_stats['rolling_woba_allowed'] = pitcher_stats.groupby('pitcher')['woba_allowed'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
    )
    
    pitcher_stats['rolling_pa_faced'] = pitcher_stats.groupby('pitcher')['woba_denom'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
    )
    
    # Fill NaN with league average
    pitcher_stats['rolling_woba_allowed'].fillna(0.320, inplace=True)
    pitcher_stats['rolling_pa_faced'].fillna(0, inplace=True)
    
    return pitcher_stats

def compute_matchup_history(raw_df):
    """Compute historical matchup stats between batters and pitchers."""
    print("Computing matchup history...")
    
    matchup_stats = defaultdict(lambda: {'pa': 0, 'woba_sum': 0, 'woba_denom': 0})
    
    # Sort by date to compute historical matchups
    raw_df_sorted = raw_df.sort_values('game_date')
    
    for _, row in raw_df_sorted.iterrows():
        batter = row['batter']
        pitcher = row['pitcher']
        game_pk = row['game_pk']
        
        # Key is (game, batter, pitcher)
        key = (game_pk, batter, pitcher)
        
        # Store historical matchup stats (before this game)
        matchup_key = (batter, pitcher)
        matchup_stats[key] = {
            'historical_pa': matchup_stats[matchup_key]['pa'],
            'historical_woba': (matchup_stats[matchup_key]['woba_sum'] / 
                              matchup_stats[matchup_key]['woba_denom'] 
                              if matchup_stats[matchup_key]['woba_denom'] > 0 else 0.320)
        }
        
        # Update matchup stats
        if pd.notna(row.get('woba_value', 0)):
            matchup_stats[matchup_key]['pa'] += 1
            matchup_stats[matchup_key]['woba_sum'] += row.get('woba_value', 0)
            matchup_stats[matchup_key]['woba_denom'] += row.get('woba_denom', 1)
    
    return matchup_stats

def build_game_graph(game_pk, game_stats, raw_df, pitcher_stats_df, matchup_stats):
    """Build a single game graph with rich features."""
    
    graph = HeteroData()
    
    # Get data for this game
    game_raw = raw_df[raw_df['game_pk'] == game_pk]
    game_batters = game_stats[game_stats['game_pk'] == game_pk]
    
    if len(game_batters) == 0 or len(game_raw) == 0:
        return None
    
    # === NODE CONSTRUCTION ===
    
    # Batters with richer features
    batters = game_batters['batter'].unique().tolist()
    batter_to_idx = {b: i for i, b in enumerate(batters)}
    
    batter_features = []
    batter_labels = []
    for batter in batters:
        stats = game_batters[game_batters['batter'] == batter].iloc[0]
        
        # Enhanced batter features
        features = [
            stats['rolling_woba'],  # Rolling wOBA
            stats.get('rolling_pa', 0) / 100.0,  # Normalized rolling PAs
            stats.get('games_played', 0) / 100.0,  # Normalized games played
            stats.get('woba_value', 0) / 10.0,  # Recent wOBA numerator (normalized)
        ]
        
        batter_features.append(features)
        batter_labels.append(stats['woba'])
    
    graph['batter'].x = torch.tensor(batter_features, dtype=torch.float)
    graph['batter'].y = torch.tensor(batter_labels, dtype=torch.float)
    
    # Pitchers with computed features
    pitchers = game_raw['pitcher'].unique().tolist()
    pitcher_to_idx = {p: i for i, p in enumerate(pitchers)}
    
    pitcher_features = []
    for pitcher in pitchers:
        # Get pitcher's rolling stats
        pitcher_game_stats = pitcher_stats_df[
            (pitcher_stats_df['pitcher'] == pitcher) & 
            (pitcher_stats_df['game_pk'] == game_pk)
        ]
        
        if len(pitcher_game_stats) > 0:
            p_stats = pitcher_game_stats.iloc[0]
            features = [
                p_stats['rolling_woba_allowed'],
                p_stats['rolling_pa_faced'] / 100.0,
            ]
        else:
            # Default to league average
            features = [0.320, 0.0]
        
        pitcher_features.append(features)
    
    graph['pitcher'].x = torch.tensor(pitcher_features, dtype=torch.float)
    
    # Teams (still simple for now)
    teams = [game_raw['home_team'].iloc[0], game_raw['away_team'].iloc[0]]
    team_to_idx = {t: i for i, t in enumerate(teams)}
    
    graph['team'].x = torch.zeros(len(teams), 2, dtype=torch.float)
    
    # === EDGE CONSTRUCTION ===
    
    # 1. Batter-Pitcher matchups with edge features
    matchup_edges = []
    matchup_features = []
    
    for _, row in game_raw.iterrows():
        if row['batter'] in batter_to_idx and row['pitcher'] in pitcher_to_idx:
            matchup_edges.append([
                batter_to_idx[row['batter']], 
                pitcher_to_idx[row['pitcher']]
            ])
            
            # Edge features: historical matchup
            key = (game_pk, row['batter'], row['pitcher'])
            hist_pa = matchup_stats.get(key, {}).get('historical_pa', 0)
            hist_woba = matchup_stats.get(key, {}).get('historical_woba', 0.320)
            
            matchup_features.append([
                hist_pa / 10.0,  # Normalized historical PAs
                hist_woba,  # Historical wOBA in matchup
            ])
    
    if matchup_edges:
        graph['batter', 'faces', 'pitcher'].edge_index = torch.tensor(
            matchup_edges, dtype=torch.long
        ).t().contiguous()
        graph['batter', 'faces', 'pitcher'].edge_attr = torch.tensor(
            matchup_features, dtype=torch.float
        )
    
    # 2. Batter-Batter (teammates)
    teammate_edges = []
    
    team_col = None
    for col in ['bat_team', 'batting_team']:
        if col in game_batters.columns:
            team_col = col
            break
    
    if team_col:
        for team in game_batters[team_col].unique():
            team_batters = game_batters[game_batters[team_col] == team]['batter'].tolist()
            for i, b1 in enumerate(team_batters):
                for b2 in team_batters[i+1:]:
                    if b1 in batter_to_idx and b2 in batter_to_idx:
                        teammate_edges.append([batter_to_idx[b1], batter_to_idx[b2]])
                        teammate_edges.append([batter_to_idx[b2], batter_to_idx[b1]])
    
    if teammate_edges:
        graph['batter', 'teammates_with', 'batter'].edge_index = torch.tensor(
            teammate_edges, dtype=torch.long
        ).t().contiguous()
    
    # 3. Batter-Team membership
    batter_team_edges = []
    
    if team_col:
        for _, row in game_batters.iterrows():
            batter = row['batter']
            team = row[team_col]
            if batter in batter_to_idx and team in team_to_idx:
                batter_team_edges.append([batter_to_idx[batter], team_to_idx[team]])
    
    if batter_team_edges:
        graph['batter', 'plays_for', 'team'].edge_index = torch.tensor(
            batter_team_edges, dtype=torch.long
        ).t().contiguous()
    
    # 4. Pitcher-Team membership
    pitcher_team_edges = []
    
    for pitcher in pitchers:
        pitcher_rows = game_raw[game_raw['pitcher'] == pitcher]
        if len(pitcher_rows) > 0:
            if 'inning_topbot' in game_raw.columns:
                inning_topbot = pitcher_rows['inning_topbot'].iloc[0]
                pitcher_team = pitcher_rows['away_team'].iloc[0] if inning_topbot == 'Top' else pitcher_rows['home_team'].iloc[0]
            else:
                pitcher_team = teams[pitchers.index(pitcher) % 2]
            
            if pitcher_team in team_to_idx:
                pitcher_team_edges.append([pitcher_to_idx[pitcher], team_to_idx[pitcher_team]])
    
    if pitcher_team_edges:
        graph['pitcher', 'plays_for', 'team'].edge_index = torch.tensor(
            pitcher_team_edges, dtype=torch.long
        ).t().contiguous()
    
    return graph

def build_all_graphs(processed_file='data/processed_stats.csv', 
                     raw_file='data/statcast_2024.csv',
                     output_file='data/graphs.pkl'):
    
    print("Loading data...")
    game_stats = pd.read_csv(processed_file)
    raw_df = pd.read_csv(raw_file)
    
    print(f"Loaded {len(game_stats)} batter-game records")
    print(f"Loaded {len(raw_df)} pitch records")
    
    # Compute additional stats
    pitcher_stats = compute_pitcher_stats(raw_df)
    matchup_stats = compute_matchup_history(raw_df)
    
    games = game_stats['game_pk'].unique()
    print(f"\nBuilding graphs for {len(games)} games...")
    
    graphs = []
    edge_type_counts = {}
    
    for i, game_pk in enumerate(games):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(games)}")
        
        try:
            graph = build_game_graph(game_pk, game_stats, raw_df, pitcher_stats, matchup_stats)
            if graph is not None:
                graphs.append(graph)
                
                for et in graph.edge_types:
                    edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        except Exception as e:
            print(f"  Error on game {game_pk}: {e}")
    
    print(f"\nBuilt {len(graphs)} graphs successfully")
    
    print("\nEdge type coverage:")
    for et, count in edge_type_counts.items():
        print(f"  {et}: {count}/{len(graphs)} graphs ({100*count/len(graphs):.1f}%)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(graphs, f)
    
    print(f"\nSaved to {output_file}")
    
    if graphs:
        print("\n=== Sample Graph ===")
        g = graphs[0]
        print(f"Batter nodes: {g['batter'].x.shape}")
        print(f"Pitcher nodes: {g['pitcher'].x.shape}")
        print(f"Team nodes: {g['team'].x.shape}")
        print("Edge types:", g.edge_types)
    
    return graphs

if __name__ == '__main__':
    build_all_graphs()