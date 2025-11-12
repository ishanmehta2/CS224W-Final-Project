"""
Fetch MLB Statcast data using PyBaseball.
"""

from pybaseball import statcast
import pandas as pd
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Fetch 2024 season data
print("Fetching 2024 season data...")
df = statcast(start_dt='2024-04-01', end_dt='2024-10-31')

# Save
df.to_csv('data/statcast_2024.csv', index=False)
print(f"Saved {len(df)} pitches to data/statcast_2024.csv")
print(f"Games: {df['game_pk'].nunique()}")
print(f"Batters: {df['batter'].nunique()}")
