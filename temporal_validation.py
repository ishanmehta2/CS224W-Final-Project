import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# MODEL DEFINITION 
# ==============================================================================

class ImprovedHeteroGraphSAGE(nn.Module):
    
    def __init__(self, batter_in=4, pitcher_in=2, team_in=2, 
                 hidden_channels=128, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.batter_lin = nn.Sequential(
            nn.Linear(batter_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.pitcher_lin = nn.Sequential(
            nn.Linear(pitcher_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.team_lin = nn.Sequential(
            nn.Linear(team_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                ('batter', 'faces', 'pitcher'): SAGEConv(hidden_channels, hidden_channels),
                ('batter', 'teammates_with', 'batter'): SAGEConv(hidden_channels, hidden_channels),
                ('batter', 'plays_for', 'team'): SAGEConv(hidden_channels, hidden_channels),
                ('pitcher', 'plays_for', 'team'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
            
            bn_dict = nn.ModuleDict({
                'batter': nn.BatchNorm1d(hidden_channels),
                'pitcher': nn.BatchNorm1d(hidden_channels),
                'team': nn.BatchNorm1d(hidden_channels)
            })
            self.bns.append(bn_dict)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'batter': self.batter_lin(x_dict['batter']),
            'pitcher': self.pitcher_lin(x_dict['pitcher']),
            'team': self.team_lin(x_dict['team'])
        }
        
        for i, conv in enumerate(self.convs):
            filtered_edge_dict = {
                edge_type: edge_index 
                for edge_type, edge_index in edge_index_dict.items()
                if edge_index.size(1) > 0
            }
            
            if filtered_edge_dict:
                x_dict_new = conv(x_dict, filtered_edge_dict)
                
                for key in x_dict_new.keys():
                    x_dict[key] = self.bns[i][key](x_dict_new[key])
                    x_dict[key] = F.relu(x_dict[key])
                    x_dict[key] = F.dropout(x_dict[key], p=0.1, training=self.training)
        
        out = self.predictor(x_dict['batter'])
        return out.squeeze(-1)


# ==============================================================================
# SPLITTING FUNCTIONS
# ==============================================================================

def random_split(graphs, train_size=0.7, val_size=0.15, random_state=42):
    n = len(graphs)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    
    return (
        [graphs[i] for i in indices[:train_end]],
        [graphs[i] for i in indices[train_end:val_end]],
        [graphs[i] for i in indices[val_end:]]
    )


def temporal_split(graphs, graph_dates, train_size=0.7, val_size=0.15):
    """
    Temporal split: train on earlier games, test on later games.
    
    This is another way to evaluate time-series predictions because
    it simulates real-world usage where we predict future performance.
    """
    # Sort by date
    sorted_indices = np.argsort(graph_dates)
    sorted_graphs = [graphs[i] for i in sorted_indices]
    sorted_dates = [graph_dates[i] for i in sorted_indices]
    
    n = len(sorted_graphs)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    
    train_graphs = sorted_graphs[:train_end]
    val_graphs = sorted_graphs[train_end:val_end]
    test_graphs = sorted_graphs[val_end:]
    
    print(f"\nTemporal Split Summary:")
    print(f"  Train: games 1-{train_end} (dates: {sorted_dates[0]} to {sorted_dates[train_end-1]})")
    print(f"  Val:   games {train_end+1}-{val_end} (dates: {sorted_dates[train_end]} to {sorted_dates[val_end-1]})")
    print(f"  Test:  games {val_end+1}-{n} (dates: {sorted_dates[val_end]} to {sorted_dates[-1]})")
    
    return train_graphs, val_graphs, test_graphs


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_epoch(model, graphs, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for graph in graphs:
        graph = graph.to(device)
        optimizer.zero_grad()
        
        pred = model(graph.x_dict, graph.edge_index_dict)
        target = graph['batter'].y
        
        loss = F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, graphs, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    for graph in graphs:
        graph = graph.to(device)
        pred = model(graph.x_dict, graph.edge_index_dict)
        target = graph['batter'].y
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def train_and_evaluate(train_graphs, val_graphs, test_graphs, 
                       batter_in, pitcher_in, team_in,
                       epochs=200, hidden_channels=128, num_layers=3, 
                       lr=0.001, patience=50, verbose=True):
    """Train model and return test metrics."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedHeteroGraphSAGE(
        batter_in=batter_in,
        pitcher_in=pitcher_in,
        team_in=team_in,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        val_preds, val_targets = evaluate(model, val_graphs, device)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    test_preds, test_targets = evaluate(model, test_graphs, device)
    
    return compute_metrics(test_targets, test_preds), test_preds, test_targets


# ==============================================================================
# BASELINE: Rolling Average
# ==============================================================================

def baseline_rolling_average(test_graphs):
    """
    Baseline using rolling wOBA (first feature of batter nodes).
    """
    all_preds = []
    all_targets = []
    
    for graph in test_graphs:
        # Assuming rolling_woba is first feature
        rolling_woba = graph['batter'].x[:, 0].numpy()
        targets = graph['batter'].y.numpy()
        
        all_preds.extend(rolling_woba)
        all_targets.extend(targets)
    
    return compute_metrics(np.array(all_targets), np.array(all_preds))


# ==============================================================================
# MAIN VALIDATION SCRIPT
# ==============================================================================

def extract_dates_from_graphs(graphs):

    dates = []
    for i, graph in enumerate(graphs):
        # Try different ways to get the date
        if hasattr(graph, 'date'):
            dates.append(graph.date)
        elif hasattr(graph, 'game_date'):
            dates.append(graph.game_date)
        elif hasattr(graph, 'metadata') and 'date' in graph.metadata:
            dates.append(graph.metadata['date'])
        else:
            dates.append(i)
    
    return dates


def run_validation(graphs_file='data/graphs.pkl'):
    """
    Run validation comparing random vs temporal splits.
    """
    print("="*70)
    print("TEMPORAL VALIDATION: Checking for Data Leakage")
    print("="*70)
    
    # Load graphs
    print("\nLoading graphs...")
    try:
        with open(graphs_file, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
    except FileNotFoundError:
        print(f"ERROR: Could not find {graphs_file}")
        return None
    
    # Get feature dimensions
    sample = graphs[0]
    batter_in = sample['batter'].x.shape[1]
    pitcher_in = sample['pitcher'].x.shape[1]
    team_in = sample['team'].x.shape[1]
    
    print(f"\nFeature dimensions:")
    print(f"  Batter:  {batter_in}")
    print(f"  Pitcher: {pitcher_in}")
    print(f"  Team:    {team_in}")
    
    results = {}
    
    # ==== RANDOM SPLIT ====
    print("\n" + "="*70)
    print("="*70)
    
    train_r, val_r, test_r = random_split(graphs)
    print(f"Split sizes - Train: {len(train_r)}, Val: {len(val_r)}, Test: {len(test_r)}")
    
    print("\nTraining GNN with random split...")
    gnn_random, _, _ = train_and_evaluate(
        train_r, val_r, test_r,
        batter_in, pitcher_in, team_in,
        epochs=200, patience=50
    )
    
    baseline_random = baseline_rolling_average(test_r)
    
    results['random_split'] = {
        'gnn': gnn_random,
        'baseline': baseline_random
    }
    
    print(f"\nRandom Split Results:")
    print(f"  GNN MAE:      {gnn_random['mae']:.4f}")
    print(f"  Baseline MAE: {baseline_random['mae']:.4f}")
    print(f"  Improvement:  {((baseline_random['mae'] - gnn_random['mae']) / baseline_random['mae'] * 100):.1f}%")
    
    # ==== TEMPORAL SPLIT ====
    print("\n" + "="*70)
    print("EXPERIMENT 2: Temporal Split (Proper Time-Series Validation)")
    print("="*70)
    
    # Extract dates
    dates = extract_dates_from_graphs(graphs)
    
    # Check if we got real dates
    if isinstance(dates[0], int):
        print("\nWARNING: Could not extract dates from graphs.")
    
    train_t, val_t, test_t = temporal_split(graphs, dates)
    print(f"Split sizes - Train: {len(train_t)}, Val: {len(val_t)}, Test: {len(test_t)}")
    
    print("\nTraining GNN with temporal split...")
    gnn_temporal, _, _ = train_and_evaluate(
        train_t, val_t, test_t,
        batter_in, pitcher_in, team_in,
        epochs=200, patience=50
    )
    
    baseline_temporal = baseline_rolling_average(test_t)
    
    results['temporal_split'] = {
        'gnn': gnn_temporal,
        'baseline': baseline_temporal
    }
    
    print(f"\nTemporal Split Results:")
    print(f"  GNN MAE:      {gnn_temporal['mae']:.4f}")
    print(f"  Baseline MAE: {baseline_temporal['mae']:.4f}")
    print(f"  Improvement:  {((baseline_temporal['mae'] - gnn_temporal['mae']) / baseline_temporal['mae'] * 100):.1f}%")
    
    # ==== COMPARISON ====
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Random Split':<15} {'Temporal Split':<15} {'Difference':<15}")
    print("-"*70)
    
    gnn_diff = gnn_temporal['mae'] - gnn_random['mae']
    baseline_diff = baseline_temporal['mae'] - baseline_random['mae']
    
    print(f"{'GNN MAE':<25} {gnn_random['mae']:<15.4f} {gnn_temporal['mae']:<15.4f} {gnn_diff:+.4f}")
    print(f"{'Baseline MAE':<25} {baseline_random['mae']:<15.4f} {baseline_temporal['mae']:<15.4f} {baseline_diff:+.4f}")
    
    random_improvement = (baseline_random['mae'] - gnn_random['mae']) / baseline_random['mae'] * 100
    temporal_improvement = (baseline_temporal['mae'] - gnn_temporal['mae']) / baseline_temporal['mae'] * 100
    
    print(f"{'GNN vs Baseline (%)':<25} {random_improvement:<15.1f}% {temporal_improvement:<15.1f}%")
    
    # ==== INTERPRETATION ====
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    mae_increase = (gnn_temporal['mae'] - gnn_random['mae']) / gnn_random['mae'] * 100
    
    if mae_increase > 50:
        print(f""" POTENTIAL DATA LEAKAGE DETECTED""")

    elif mae_increase > 20:
        print(f"""
    MODERATE PERFORMANCE DROP """)
    else:
        print(f"""
    RESULTS VALIDATED
""")
    
    # Save results
    with open('data/temporal_validation_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for split_name, split_results in results.items():
            serializable_results[split_name] = {}
            for model_name, metrics in split_results.items():
                serializable_results[split_name][model_name] = {
                    k: float(v) for k, v in metrics.items()
                }
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to data/temporal_validation_results.json")
    
    return results


# ==============================================================================
# ADDITIONAL VALIDATION: Check Feature Computation
# ==============================================================================

def check_feature_leakage(processed_file='data/processed_stats.csv'):
    """
    Additional check: verify rolling statistics are computed correctly.
    """
    print("\n" + "="*70)
    print("FEATURE LEAKAGE CHECK")
    print("="*70)
    
    try:
        df = pd.read_csv(processed_file)
    except FileNotFoundError:
        print(f"Could not find {processed_file}. Skipping feature check.")
        return
    
    print(f"\nLoaded {len(df)} records from {processed_file}")
    
    # Check if rolling_woba ever equals woba (would indicate no shift)
    if 'rolling_woba' in df.columns and 'woba' in df.columns:
        exact_matches = (df['rolling_woba'] == df['woba']).sum()
        match_rate = exact_matches / len(df) * 100
        
        print(f"\nRolling wOBA == Current wOBA: {exact_matches} times ({match_rate:.2f}%)")
        
        if match_rate > 5:
            print(" WARNING: High match rate suggests rolling stats may include current game!")
        else:
            print(" Rolling stats appear to be properly lagged.")
    
    # Check correlation between rolling and actual
    if 'rolling_woba' in df.columns and 'woba' in df.columns:
        corr = df['rolling_woba'].corr(df['woba'])
        print(f"\nCorrelation (rolling_woba, woba): {corr:.4f}")
        
        if corr > 0.5:
            print(" WARNING: Unusually high correlation. Verify feature computation.")
        else:
            print(" Correlation is in expected range for predictive features.")
    
    # Check for any future-looking features
    print("\nChecking for potential future-leaking columns...")
    suspicious_cols = [col for col in df.columns if any(x in col.lower() for x in ['total', 'season', 'final', 'cumulative'])]
    
    if suspicious_cols:
        print(f"  Found potentially suspicious columns: {suspicious_cols}")
        print("   Verify these don't use end-of-season or future data.")
    else:
        print("  No obviously suspicious column names found.")


if __name__ == '__main__':
    # Run main validation
    results = run_validation()
    
    # Run feature leakage check
    check_feature_leakage()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)