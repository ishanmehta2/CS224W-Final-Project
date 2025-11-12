"""
Baseline models for predicting batter wOBA.
Baselines: 
1. Player rolling average (last 10 games)
2. Season average
3. XGBoost with basic features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

def load_data(processed_file='data/processed_stats.csv'):
    df = pd.read_csv(processed_file)
    return df

def baseline_rolling_average(df):
    """
    Baseline 1: Predict using rolling average (already computed).
    This is the simplest baseline - just use the rolling_woba as prediction.
    """
    # Filter out rows where rolling_woba is NaN
    df_valid = df[df['rolling_woba'].notna()].copy()
    
    predictions = df_valid['rolling_woba'].values
    actuals = df_valid['woba'].values
    
    mae = mean_absolute_error(actuals, predictions)
    
    print("=== Baseline 1: Rolling Average (10 games) ===")
    print(f"MAE: {mae:.4f}")
    print(f"Predictions on {len(predictions)} batter-games")
    
    return mae

def baseline_season_average(df):
    """
    Baseline 2: Predict using each player's season average up to that game.
    """
    df = df.sort_values(['batter', 'game_date'])
    
    # Compute expanding mean (season average up to each game)
    df['season_avg'] = df.groupby('batter')['woba'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Fill NaN with league average
    df['season_avg'].fillna(0.320, inplace=True)
    
    predictions = df['season_avg'].values
    actuals = df['woba'].values
    
    mae = mean_absolute_error(actuals, predictions)
    
    print("\n=== Baseline 2: Season Average ===")
    print(f"MAE: {mae:.4f}")
    
    return mae

def baseline_xgboost(df):
    """
    Baseline 3: XGBoost with basic features.
    Features: rolling_woba, rolling_pa, games_played
    """
    # Prepare features
    feature_cols = ['rolling_woba', 'rolling_pa', 'games_played']
    
    # Check which columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    
    if 'rolling_pa' not in df.columns:
        # Compute rolling PA if not present
        df = df.sort_values(['batter', 'game_date'])
        df['rolling_pa'] = df.groupby('batter')['woba_denom'].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
        )
        available_features.append('rolling_pa')
    
    X = df[available_features].fillna(0).values
    y = df['woba'].values
    
    # Split data randomly (same distribution)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Train XGBoost
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n=== Baseline 3: XGBoost ===")
    print(f"Features: {available_features}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Val MAE: {val_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save model
    with open('data/xgboost_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return test_mae

def run_all_baselines(processed_file='data/processed_stats.csv'):

    print("Loading data...")
    df = load_data(processed_file)
    print(f"Loaded {len(df)} batter-game records\n")
    
    # Run baselines
    mae_rolling = baseline_rolling_average(df)
    mae_season = baseline_season_average(df)
    mae_xgboost = baseline_xgboost(df)
    
    print("\n" + "="*50)
    print("BASELINE SUMMARY")
    print("="*50)
    print(f"Rolling Average (10 games):  {mae_rolling:.4f}")
    print(f"Season Average:              {mae_season:.4f}")
    print(f"XGBoost:                     {mae_xgboost:.4f}")
    print("="*50)
    
    return {
        'rolling': mae_rolling,
        'season': mae_season,
        'xgboost': mae_xgboost
    }

if __name__ == '__main__':
    results = run_all_baselines()
