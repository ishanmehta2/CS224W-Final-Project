import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import json
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# ==============================================================================
# MODEL VARIANTS
# ==============================================================================

class AblationHeteroGraphSAGE(nn.Module):
    """
    Flexible GraphSAGE that allows disabling specific edge types.
    """
    
    def __init__(self, batter_in=4, pitcher_in=2, team_in=2, 
                 hidden_channels=128, num_layers=3,
                 use_batter_pitcher=True,
                 use_batter_batter=True, 
                 use_batter_team=True,
                 use_pitcher_team=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_batter_pitcher = use_batter_pitcher
        self.use_batter_batter = use_batter_batter
        self.use_batter_team = use_batter_team
        self.use_pitcher_team = use_pitcher_team
        
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
        
        # Build convolutions based on enabled edge types
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            edge_types = {}
            if use_batter_pitcher:
                edge_types[('batter', 'faces', 'pitcher')] = SAGEConv(hidden_channels, hidden_channels)
            if use_batter_batter:
                edge_types[('batter', 'teammates_with', 'batter')] = SAGEConv(hidden_channels, hidden_channels)
            if use_batter_team:
                edge_types[('batter', 'plays_for', 'team')] = SAGEConv(hidden_channels, hidden_channels)
            if use_pitcher_team:
                edge_types[('pitcher', 'plays_for', 'team')] = SAGEConv(hidden_channels, hidden_channels)
            
            if edge_types:
                conv = HeteroConv(edge_types, aggr='mean')
            else:
                conv = None
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
        x_dict = {
            'batter': self.batter_lin(x_dict['batter']),
            'pitcher': self.pitcher_lin(x_dict['pitcher']),
            'team': self.team_lin(x_dict['team'])
        }
        
        for i, conv in enumerate(self.convs):
            if conv is None:
                continue
                
            filtered_edge_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                if edge_index.size(1) == 0:
                    continue
                src, rel, dst = edge_type
                if rel == 'faces' and self.use_batter_pitcher:
                    filtered_edge_dict[edge_type] = edge_index
                elif rel == 'teammates_with' and self.use_batter_batter:
                    filtered_edge_dict[edge_type] = edge_index
                elif rel == 'plays_for' and src == 'batter' and self.use_batter_team:
                    filtered_edge_dict[edge_type] = edge_index
                elif rel == 'plays_for' and src == 'pitcher' and self.use_pitcher_team:
                    filtered_edge_dict[edge_type] = edge_index
            
            if filtered_edge_dict:
                x_dict_new = conv(x_dict, filtered_edge_dict)
                
                for key in x_dict_new.keys():
                    x_dict[key] = self.bns[i][key](x_dict_new[key])
                    x_dict[key] = F.relu(x_dict[key])
                    x_dict[key] = F.dropout(x_dict[key], p=0.1, training=self.training)
        
        out = self.predictor(x_dict['batter'])
        return out.squeeze(-1)


class MLPBaseline(nn.Module):
    """MLP that ignores graph structure - uses only batter features."""
    
    def __init__(self, batter_in=4, hidden_channels=128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(batter_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x_dict, edge_index_dict):
        out = self.mlp(x_dict['batter'])
        return out.squeeze(-1)


class HeteroGAT(nn.Module):
    """GAT variant for comparison - uses attention instead of mean aggregation."""
    
    def __init__(self, batter_in=4, pitcher_in=2, team_in=2, 
                 hidden_channels=128, num_layers=3, heads=4):
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
        
        for i in range(num_layers):
            if i < num_layers - 1:
                conv = HeteroConv({
                    ('batter', 'faces', 'pitcher'): GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                    ('batter', 'teammates_with', 'batter'): GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                    ('batter', 'plays_for', 'team'): GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                    ('pitcher', 'plays_for', 'team'): GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                }, aggr='mean')
            else:
                conv = HeteroConv({
                    ('batter', 'faces', 'pitcher'): GATConv(hidden_channels, hidden_channels, heads=1, concat=False, add_self_loops=False),
                    ('batter', 'teammates_with', 'batter'): GATConv(hidden_channels, hidden_channels, heads=1, concat=False, add_self_loops=False),
                    ('batter', 'plays_for', 'team'): GATConv(hidden_channels, hidden_channels, heads=1, concat=False, add_self_loops=False),
                    ('pitcher', 'plays_for', 'team'): GATConv(hidden_channels, hidden_channels, heads=1, concat=False, add_self_loops=False),
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
# TEMPORAL SPLIT (Validated as leak-free)
# ==============================================================================

def temporal_split(graphs, train_size=0.7, val_size=0.15):
    """
    Split graphs chronologically - train on past, test on future.
    Assumes graphs are already sorted by date.
    """
    # Check if graphs have dates and are sorted
    if hasattr(graphs[0], 'game_date') and graphs[0].game_date:
        dates = [g.game_date for g in graphs]
        if dates != sorted(dates):
            print("  Warning: Graphs not sorted by date, sorting now...")
            sorted_pairs = sorted(zip(graphs, dates), key=lambda x: x[1])
            graphs = [g for g, _ in sorted_pairs]
            dates = [d for _, d in sorted_pairs]
    
    n = len(graphs)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    
    return (
        graphs[:train_end],
        graphs[train_end:val_end],
        graphs[val_end:]
    )


# ==============================================================================
# TRAINING UTILITIES
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


def train_model(model, train_graphs, val_graphs, device, 
                epochs=150, lr=0.001, patience=40, verbose=False):
    """Train a model and return best validation MAE."""
    
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
            print(f"    Epoch {epoch+1:3d} | Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f}")
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    return best_val_mae, model


def evaluate_on_test(model, test_graphs, device):
    """Evaluate model on test set."""
    test_preds, test_targets = evaluate(model, test_graphs, device)
    return mean_absolute_error(test_targets, test_preds)


def baseline_rolling_average(test_graphs):
    """Baseline: predict using rolling wOBA (first batter feature)."""
    all_preds = []
    all_targets = []
    
    for graph in test_graphs:
        rolling_woba = graph['batter'].x[:, 0].numpy()
        targets = graph['batter'].y.numpy()
        all_preds.extend(rolling_woba)
        all_targets.extend(targets)
    
    return mean_absolute_error(all_targets, all_preds)


# ==============================================================================
# ABLATION EXPERIMENTS
# ==============================================================================

def ablation_edge_types(train_graphs, val_graphs, test_graphs, 
                        batter_in, pitcher_in, team_in, device):
    """Ablation 1: Which edge types contribute to performance?"""
    
    print("\n" + "="*70)
    print("ABLATION 1: Edge Type Importance")
    print("="*70)
    print("Testing which relationships (edges) contribute to predictions...")
    
    configurations = [
        ("Full Model", True, True, True, True),
        ("- Batter-Pitcher", False, True, True, True),
        ("- Teammates", True, False, True, True),
        ("- Batter-Team", True, True, False, True),
        ("- Pitcher-Team", True, True, True, False),
        ("Only Batter-Pitcher", True, False, False, False),
        ("Only Teammates", False, True, False, False),
        ("No Graph (MLP)", False, False, False, False),
    ]
    
    results = {}
    
    for name, bp, bb, bt, pt in configurations:
        print(f"\n  {name}...")
        
        if name == "No Graph (MLP)":
            model = MLPBaseline(batter_in=batter_in).to(device)
        else:
            model = AblationHeteroGraphSAGE(
                batter_in=batter_in,
                pitcher_in=pitcher_in,
                team_in=team_in,
                use_batter_pitcher=bp,
                use_batter_batter=bb,
                use_batter_team=bt,
                use_pitcher_team=pt
            ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        val_mae, model = train_model(model, train_graphs, val_graphs, device)
        test_mae = evaluate_on_test(model, test_graphs, device)
        
        results[name] = {
            'val_mae': float(val_mae), 
            'test_mae': float(test_mae),
            'n_params': n_params
        }
        print(f"    Test MAE: {test_mae:.4f}")
    
    return results


def ablation_num_layers(train_graphs, val_graphs, test_graphs,
                        batter_in, pitcher_in, team_in, device):
    """Ablation 2: How many message passing layers are optimal?"""
    
    print("\n" + "="*70)
    print("ABLATION 2: Number of GNN Layers")
    print("="*70)
    print("Testing depth of message passing...")
    
    results = {}
    
    for num_layers in [1, 2, 3, 4, 5]:
        print(f"\n  {num_layers} layer(s)...")
        
        model = AblationHeteroGraphSAGE(
            batter_in=batter_in,
            pitcher_in=pitcher_in,
            team_in=team_in,
            num_layers=num_layers
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        val_mae, model = train_model(model, train_graphs, val_graphs, device)
        test_mae = evaluate_on_test(model, test_graphs, device)
        
        results[f"{num_layers} layers"] = {
            'val_mae': float(val_mae), 
            'test_mae': float(test_mae),
            'n_params': n_params
        }
        print(f"    Test MAE: {test_mae:.4f} | Params: {n_params:,}")
    
    return results


def ablation_hidden_dim(train_graphs, val_graphs, test_graphs,
                        batter_in, pitcher_in, team_in, device):
    """Ablation 3: Model capacity analysis."""
    
    print("\n" + "="*70)
    print("ABLATION 3: Hidden Dimension Size")
    print("="*70)
    print("Testing model capacity...")
    
    results = {}
    
    for hidden_dim in [32, 64, 128, 256]:
        print(f"\n  Hidden dim = {hidden_dim}...")
        
        model = AblationHeteroGraphSAGE(
            batter_in=batter_in,
            pitcher_in=pitcher_in,
            team_in=team_in,
            hidden_channels=hidden_dim
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        val_mae, model = train_model(model, train_graphs, val_graphs, device)
        test_mae = evaluate_on_test(model, test_graphs, device)
        
        results[f"dim={hidden_dim}"] = {
            'val_mae': float(val_mae), 
            'test_mae': float(test_mae),
            'n_params': n_params
        }
        print(f"    Test MAE: {test_mae:.4f} | Params: {n_params:,}")
    
    return results


def ablation_architecture(train_graphs, val_graphs, test_graphs,
                          batter_in, pitcher_in, team_in, device):
    """Ablation 4: GraphSAGE vs GAT comparison."""
    
    print("\n" + "="*70)
    print("ABLATION 4: Architecture Comparison")
    print("="*70)
    print("Comparing aggregation methods (mean vs attention)...")
    
    results = {}
    
    # GraphSAGE
    print("\n  GraphSAGE (mean aggregation)...")
    model_sage = AblationHeteroGraphSAGE(
        batter_in=batter_in,
        pitcher_in=pitcher_in,
        team_in=team_in,
    ).to(device)
    
    n_params_sage = sum(p.numel() for p in model_sage.parameters())
    val_mae, model_sage = train_model(model_sage, train_graphs, val_graphs, device)
    test_mae = evaluate_on_test(model_sage, test_graphs, device)
    results['GraphSAGE'] = {
        'val_mae': float(val_mae), 
        'test_mae': float(test_mae),
        'n_params': n_params_sage
    }
    print(f"    Test MAE: {test_mae:.4f}")
    
    # GAT
    print("\n  GAT (attention aggregation)...")
    model_gat = HeteroGAT(
        batter_in=batter_in,
        pitcher_in=pitcher_in,
        team_in=team_in,
    ).to(device)
    
    n_params_gat = sum(p.numel() for p in model_gat.parameters())
    val_mae, model_gat = train_model(model_gat, train_graphs, val_graphs, device)
    test_mae = evaluate_on_test(model_gat, test_graphs, device)
    results['GAT'] = {
        'val_mae': float(val_mae), 
        'test_mae': float(test_mae),
        'n_params': n_params_gat
    }
    print(f"    Test MAE: {test_mae:.4f}")
    
    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_ablation_results(all_results, baseline_mae, save_path='data/ablation_results.png'):
    """Create publication-ready visualization of ablation results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Color scheme
    colors = {
        'best': '#2ecc71',      # Green
        'good': '#3498db',      # Blue  
        'neutral': '#95a5a6',   # Gray
        'worse': '#e74c3c',     # Red
        'baseline': '#e67e22'   # Orange
    }
    
    # ===== Plot 1: Edge Type Ablation =====
    ax1 = axes[0, 0]
    edge_results = all_results['edge_types']
    names = list(edge_results.keys())
    maes = [edge_results[n]['test_mae'] for n in names]
    
    full_mae = edge_results['Full Model']['test_mae']
    bar_colors = []
    for n, mae in zip(names, maes):
        if n == 'Full Model':
            bar_colors.append(colors['best'])
        elif mae <= full_mae * 1.02:
            bar_colors.append(colors['good'])
        elif mae <= full_mae * 1.1:
            bar_colors.append(colors['neutral'])
        else:
            bar_colors.append(colors['worse'])
    
    bars = ax1.barh(range(len(names)), maes, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=full_mae, color=colors['best'], linestyle='--', alpha=0.7, linewidth=2, label='Full Model')
    ax1.axvline(x=baseline_mae, color=colors['baseline'], linestyle=':', alpha=0.7, linewidth=2, label='Rolling Avg Baseline')
    
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Test MAE (lower is better)', fontsize=11)
    ax1.set_title('Edge Type Ablation', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, max(maes) * 1.15)
    
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        diff = ((mae - full_mae) / full_mae) * 100
        sign = '+' if diff > 0 else ''
        ax1.text(mae + 0.003, i, f'{mae:.4f} ({sign}{diff:.1f}%)', va='center', fontsize=9)
    
    # ===== Plot 2: Layer Depth =====
    ax2 = axes[0, 1]
    layer_results = all_results['num_layers']
    layers = [int(n.split()[0]) for n in layer_results.keys()]
    maes = [layer_results[n]['test_mae'] for n in layer_results.keys()]
    
    ax2.plot(layers, maes, 'o-', markersize=12, linewidth=2.5, color=colors['good'], markeredgecolor='black')
    ax2.axhline(y=baseline_mae, color=colors['baseline'], linestyle=':', linewidth=2, label='Rolling Avg Baseline')
    
    best_idx = np.argmin(maes)
    ax2.scatter([layers[best_idx]], [maes[best_idx]], color=colors['best'], s=200, 
                zorder=5, edgecolor='black', linewidth=2, label=f'Best: {layers[best_idx]} layers')
    
    ax2.set_xlabel('Number of GNN Layers', fontsize=11)
    ax2.set_ylabel('Test MAE', fontsize=11)
    ax2.set_title('GNN Depth Analysis', fontsize=13, fontweight='bold')
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    for i, (l, mae) in enumerate(zip(layers, maes)):
        ax2.annotate(f'{mae:.4f}', (l, mae), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # ===== Plot 3: Hidden Dimension =====
    ax3 = axes[1, 0]
    dim_results = all_results['hidden_dim']
    dims = [int(n.split('=')[1]) for n in dim_results.keys()]
    maes = [dim_results[n]['test_mae'] for n in dim_results.keys()]
    params = [dim_results[n]['n_params'] / 1000 for n in dim_results.keys()]
    
    ax3.plot(dims, maes, 'o-', markersize=12, linewidth=2.5, color=colors['good'], 
             markeredgecolor='black', label='Test MAE')
    ax3.axhline(y=baseline_mae, color=colors['baseline'], linestyle=':', linewidth=2, label='Rolling Avg Baseline')
    
    ax3.set_xlabel('Hidden Dimensions', fontsize=11)
    ax3.set_ylabel('Test MAE', fontsize=11, color=colors['good'])
    ax3.tick_params(axis='y', labelcolor=colors['good'])
    ax3.set_title('Model Capacity Analysis', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(dims)
    ax3.set_xticklabels(dims)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(dims, params, 's--', markersize=8, linewidth=1.5, 
                  color='#9b59b6', alpha=0.7, label='Params (K)')
    ax3_twin.set_ylabel('Parameters (thousands)', fontsize=11, color='#9b59b6')
    ax3_twin.tick_params(axis='y', labelcolor='#9b59b6')
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    # ===== Plot 4: Architecture Comparison =====
    ax4 = axes[1, 1]
    arch_results = all_results['architecture']
    names = list(arch_results.keys())
    maes = [arch_results[n]['test_mae'] for n in names]
    params = [arch_results[n]['n_params'] / 1000 for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, maes, width, label='Test MAE', color=colors['good'], edgecolor='black')
    ax4.axhline(y=baseline_mae, color=colors['baseline'], linestyle=':', linewidth=2, label='Rolling Avg Baseline')
    
    ax4.set_ylabel('Test MAE', fontsize=11)
    ax4.set_title('Architecture Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=11)
    ax4.legend(fontsize=9)
    
    for bar, mae, param in zip(bars1, maes, params):
        ax4.text(bar.get_x() + bar.get_width()/2, mae + 0.003, 
                f'{mae:.4f}\n({param:.0f}K params)', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Ablation plots saved to {save_path}")
    plt.close()


def create_latex_table(all_results, baseline_mae, save_path='data/ablation_table.tex'):
    """Generate LaTeX table for paper."""
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Ablation Study Results (Test MAE, Temporal Split)}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Test MAE} & \textbf{$\Delta$ vs Full} \\
\midrule
\multicolumn{3}{l}{\textit{Baselines}} \\
Rolling Average (10-game) & %.4f & -- \\
\midrule
\multicolumn{3}{l}{\textit{Edge Type Ablation}} \\
""" % baseline_mae
    
    edge_results = all_results['edge_types']
    full_mae = edge_results['Full Model']['test_mae']
    
    for name, metrics in edge_results.items():
        mae = metrics['test_mae']
        diff = ((mae - full_mae) / full_mae) * 100
        sign = '+' if diff > 0 else ''
        latex += f"{name} & {mae:.4f} & {sign}{diff:.1f}\\% \\\\\n"
    
    latex += r"""
\midrule
\multicolumn{3}{l}{\textit{Architecture Comparison}} \\
"""
    
    for name, metrics in all_results['architecture'].items():
        mae = metrics['test_mae']
        diff = ((mae - full_mae) / full_mae) * 100
        sign = '+' if diff > 0 else ''
        latex += f"{name} & {mae:.4f} & {sign}{diff:.1f}\\% \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f" LaTeX table saved to {save_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def run_ablation_study(graphs_file='data/graphs.pkl'):
    """Run complete ablation study with temporal split."""
    
    print("="*70)
    print("ABLATION STUDY - CS224W Final Project")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n Loading graphs...")
    try:
        with open(graphs_file, 'rb') as f:
            graphs = pickle.load(f)
        print(f"   Loaded {len(graphs)} graphs")
    except FileNotFoundError:
        print(f"ERROR: Could not find {graphs_file}")
        return None
    
    # Get dimensions
    sample = graphs[0]
    batter_in = sample['batter'].x.shape[1]
    pitcher_in = sample['pitcher'].x.shape[1]
    team_in = sample['team'].x.shape[1]
    
    print(f"   Features - Batter: {batter_in}, Pitcher: {pitcher_in}, Team: {team_in}")
    
    # Temporal split
    print("\n Using TEMPORAL split (validated as leak-free)")
    train_graphs, val_graphs, test_graphs = temporal_split(graphs)
    print(f"   Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    if hasattr(graphs[0], 'game_date'):
        print(f"   Train dates: {train_graphs[0].game_date} → {train_graphs[-1].game_date}")
        print(f"   Test dates:  {test_graphs[0].game_date} → {test_graphs[-1].game_date}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Compute baseline
    baseline_mae = baseline_rolling_average(test_graphs)
    print(f"\n Rolling Average Baseline MAE: {baseline_mae:.4f}")
    
    # Run ablations
    all_results = {}
    
    all_results['edge_types'] = ablation_edge_types(
        train_graphs, val_graphs, test_graphs,
        batter_in, pitcher_in, team_in, device
    )
    
    all_results['num_layers'] = ablation_num_layers(
        train_graphs, val_graphs, test_graphs,
        batter_in, pitcher_in, team_in, device
    )
    
    all_results['hidden_dim'] = ablation_hidden_dim(
        train_graphs, val_graphs, test_graphs,
        batter_in, pitcher_in, team_in, device
    )
    
    all_results['architecture'] = ablation_architecture(
        train_graphs, val_graphs, test_graphs,
        batter_in, pitcher_in, team_in, device
    )
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    print(f"\n Baseline (Rolling Average): {baseline_mae:.4f}")
    
    print("\n  EDGE TYPE IMPORTANCE:")
    full_mae = all_results['edge_types']['Full Model']['test_mae']
    for name, metrics in sorted(all_results['edge_types'].items(), key=lambda x: x[1]['test_mae']):
        diff = ((metrics['test_mae'] - full_mae) / full_mae) * 100
        print(f" {name:<35} MAE: {metrics['test_mae']:.4f} ({diff:+.1f}%)")
    
    print("\n  OPTIMAL DEPTH:")
    best_layers = min(all_results['num_layers'].items(), key=lambda x: x[1]['test_mae'])
    print(f"  Best: {best_layers[0]} (MAE: {best_layers[1]['test_mae']:.4f})")
    
    print("\n OPTIMAL CAPACITY:")
    best_dim = min(all_results['hidden_dim'].items(), key=lambda x: x[1]['test_mae'])
    print(f"  Best: {best_dim[0]} (MAE: {best_dim[1]['test_mae']:.4f})")
    
    print("\n  ARCHITECTURE WINNER:")
    best_arch = min(all_results['architecture'].items(), key=lambda x: x[1]['test_mae'])
    print(f"   Winner: {best_arch[0]} (MAE: {best_arch[1]['test_mae']:.4f})")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS FOR FINAL REPORT")
    print("="*70)
    
    mlp_mae = all_results['edge_types']['No Graph (MLP)']['test_mae']
    graph_improvement = ((mlp_mae - full_mae) / mlp_mae) * 100
    baseline_improvement = ((baseline_mae - full_mae) / baseline_mae) * 100
    
    
    # Save results
    all_results['baseline_mae'] = float(baseline_mae)
    
    with open('data/ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n Results saved to data/ablation_results.json")
    
    # Generate visualizations
    plot_ablation_results(all_results, baseline_mae)
    create_latex_table(all_results, baseline_mae)
    
    print(f"\n Ablation study complete! ({datetime.now().strftime('%H:%M:%S')})")
    
    return all_results


if __name__ == '__main__':
    results = run_ablation_study()