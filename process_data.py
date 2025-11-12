"""
Process Statcast data to compute per-game wOBA and rolling averages.
"""

import pandas as pd
import numpy as np
import os

def process_data(input_file='data/statcast_2024.csv', window=10):
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()[:20]}")  # Print first 20 columns
    
    # Check for required columns
    if 'woba_value' not in df.columns or 'woba_denom' not in df.columns:
        print("Warning: woba_value or woba_denom not found. Computing from events...")
        # Compute wOBA values from events if not present
        woba_weights = {
            'walk': 0.69, 'single': 0.88, 'double': 1.24, 
            'triple': 1.56, 'home_run': 1.95
        }
        df['woba_value'] = df['events'].map(woba_weights).fillna(0)
        df['woba_denom'] = df['events'].notna().astype(int)
    
    # Compute per-game wOBA for each batter
    agg_dict = {
        'woba_value': 'sum',
        'woba_denom': 'sum',
    }
    
    # Add batting_team if it exists
    if 'batting_team' in df.columns:
        agg_dict['batting_team'] = 'first'
    elif 'bat_team' in df.columns:
        agg_dict['bat_team'] = 'first'
    
    game_stats = df.groupby(['game_pk', 'game_date', 'batter']).agg(agg_dict).reset_index()
    
    game_stats['woba'] = game_stats['woba_value'] / game_stats['woba_denom'].replace(0, 1)
    
    # Compute rolling averages (10 game window)
    game_stats = game_stats.sort_values(['batter', 'game_date'])
    
    game_stats['rolling_woba'] = game_stats.groupby('batter')['woba'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    
    game_stats['games_played'] = game_stats.groupby('batter').cumcount()
    
    # Filter cold start (skip players with <10 games)
    game_stats = game_stats[game_stats['games_played'] >= 10]
    
    # Fill NaN rolling_woba with league average
    game_stats['rolling_woba'].fillna(0.320, inplace=True)
    
    # Save
    game_stats.to_csv('data/processed_stats.csv', index=False)
    print(f"Processed {len(game_stats)} batter-game records")
    print(f"Games: {game_stats['game_pk'].nunique()}")
    print(f"Batters: {game_stats['batter'].nunique()}")
    
    return game_stats

if __name__ == '__main__':
    process_data()