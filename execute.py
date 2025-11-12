"""
Training script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import pickle
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred, name="Model"):
    """Compute multiple evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    df_results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    low_mask = df_results['actual'] < 0.3
    mae_low = mean_absolute_error(
        df_results[low_mask]['actual'], 
        df_results[low_mask]['pred']
    ) if low_mask.sum() > 0 else np.nan
    
    avg_mask = (df_results['actual'] >= 0.3) & (df_results['actual'] < 0.4)
    mae_avg = mean_absolute_error(
        df_results[avg_mask]['actual'], 
        df_results[avg_mask]['pred']
    ) if avg_mask.sum() > 0 else np.nan
    
    high_mask = df_results['actual'] >= 0.4
    mae_high = mean_absolute_error(
        df_results[high_mask]['actual'], 
        df_results[high_mask]['pred']
    ) if high_mask.sum() > 0 else np.nan
    
    print(f"\n{name} Results:")
    print(f"  Overall MAE:  {mae:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  R²:           {r2:.4f}")
    print(f"  MAE (wOBA < 0.3):     {mae_low:.4f} (n={low_mask.sum()})")
    print(f"  MAE (0.3 ≤ wOBA < 0.4): {mae_avg:.4f} (n={avg_mask.sum()})")
    print(f"  MAE (wOBA ≥ 0.4):     {mae_high:.4f} (n={high_mask.sum()})")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_low': mae_low,
        'mae_avg': mae_avg,
        'mae_high': mae_high
    }


class ImprovedHeteroGraphSAGE(nn.Module):
    
    def __init__(self, batter_in=4, pitcher_in=2, team_in=2, 
                 hidden_channels=128, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Initial embeddings
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
        
        # Graph convolutions
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
        
        # Prediction head
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
        # Initial embeddings
        x_dict = {
            'batter': self.batter_lin(x_dict['batter']),
            'pitcher': self.pitcher_lin(x_dict['pitcher']),
            'team': self.team_lin(x_dict['team'])
        }
        
        # Message passing
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


def split_graphs(graphs, train_size=0.7, val_size=0.15, random_state=42):
    """Split graphs maintaining distribution."""
    n = len(graphs)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (
        [graphs[i] for i in train_idx],
        [graphs[i] for i in val_idx],
        [graphs[i] for i in test_idx]
    )


def train_epoch(model, graphs, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for graph in graphs:
        graph = graph.to(device)
        
        optimizer.zero_grad()
        
        pred = model(graph.x_dict, graph.edge_index_dict)
        target = graph['batter'].y
        
        # Combined loss
        loss = F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, graphs, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for graph in graphs:
        graph = graph.to(device)
        
        pred = model(graph.x_dict, graph.edge_index_dict)
        target = graph['batter'].y
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return all_preds, all_targets


def plot_training_history(history, save_path='data/training_curves.png'):
    """Plot training curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Training Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Validation MAE
    axes[1].plot(history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    axes[1].axhline(y=min(history['val_mae']), color='red', linestyle='--', 
                    label=f'Best: {min(history["val_mae"]):.4f}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Validation MAE', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Learning Rate
    axes[2].plot(history['lr'], label='Learning Rate', color='green', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")
    plt.close()


def train_model(graphs_file='data/graphs.pkl', 
                epochs=300,  # Increased for longer training
                hidden_channels=128,
                num_layers=3,
                lr=0.001):
    
    print("="*60)
    print("TRAINING IMPROVED HETEROGENEOUS GRAPHSAGE")
    print("="*60)
    
    print("\nLoading graphs...")
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs")
    
    sample = graphs[0]
    print(f"\nFeature dimensions:")
    print(f"  Batter: {sample['batter'].x.shape[1]}")
    print(f"  Pitcher: {sample['pitcher'].x.shape[1]}")
    print(f"  Team: {sample['team'].x.shape[1]}")
    
    # Split data
    print("\nSplitting data...")
    train_graphs, val_graphs, test_graphs = split_graphs(graphs)
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = ImprovedHeteroGraphSAGE(
        batter_in=sample['batter'].x.shape[1],
        pitcher_in=sample['pitcher'].x.shape[1],
        team_in=sample['team'].x.shape[1],
        hidden_channels=hidden_channels,
        num_layers=num_layers
    )
    model = model.to(device)
    
    print(f"\nModel architecture:")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Num layers: {num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    print(f"\nTraining for up to {epochs} epochs...")
    print("-"*60)
    
    best_val_mae = float('inf')
    patience = 100  # Increased patience
    patience_counter = 0
    
    # Track training history
    history = {
        'train_loss': [],
        'val_mae': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        
        val_preds, val_targets = evaluate(model, val_graphs, device)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'data/best_gnn_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f} | LR: {current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    plot_training_history(history)
    
    # Save history
    with open('data/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    model.load_state_dict(torch.load('data/best_gnn_model.pt', weights_only=True))
    test_preds, test_targets = evaluate(model, test_graphs, device)
    
    gnn_metrics = compute_metrics(test_targets, test_preds, "Improved GNN")
    
    test_results = pd.DataFrame({
        'actual': test_targets,
        'predicted': test_preds
    })
    test_results.to_csv('data/gnn_predictions.csv', index=False)
    
    return model, gnn_metrics, history


def compare_with_baselines(gnn_metrics):
    """Compare with baselines."""
    
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINES")
    print("="*60)
    
    try:
        with open('data/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
    except FileNotFoundError:
        print("Baseline results not found. Run baselines.py first.")
        return
    
    print(f"\n{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*60)
    
    for name, metrics in baseline_results.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")
    
    print(f"{'Improved GNN':<25} {gnn_metrics['mae']:<10.4f} {gnn_metrics['rmse']:<10.4f} {gnn_metrics['r2']:<10.4f}")
    
    print("="*60)
    
    best_baseline_mae = min([m['mae'] for m in baseline_results.values()])
    improvement = ((best_baseline_mae - gnn_metrics['mae']) / best_baseline_mae) * 100
    print(f"\nImprovement over best baseline: {improvement:+.2f}%")
    
    # Stratified comparison
    print("\nStratified MAE comparison:")
    print(f"{'Model':<25} {'Low (<0.3)':<12} {'Avg (0.3-0.4)':<12} {'High (≥0.4)':<12}")
    print("-"*60)
    for name, metrics in baseline_results.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {metrics['mae_low']:<12.4f} {metrics['mae_avg']:<12.4f} {metrics['mae_high']:<12.4f}")
    print(f"{'Improved GNN':<25} {gnn_metrics['mae_low']:<12.4f} {gnn_metrics['mae_avg']:<12.4f} {gnn_metrics['mae_high']:<12.4f}")
    
    all_results = baseline_results.copy()
    all_results['improved_gnn'] = gnn_metrics
    
    with open('data/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nAll results saved to data/all_results.json")


if __name__ == '__main__':
    model, gnn_metrics, history = train_model(
        epochs=300,
        hidden_channels=128,
        num_layers=3,
        lr=0.001
    )
    
    compare_with_baselines(gnn_metrics)
