import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# OPTIMAL MODEL (Based on Ablation Study)
# ==============================================================================

class OptimalHeteroGAT(nn.Module):

    def __init__(self, batter_in=4, pitcher_in=2, team_in=2, 
                 hidden_channels=32, num_layers=3, heads=4):
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
                    # Keep batter-pitcher matchups
                    ('batter', 'faces', 'pitcher'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, concat=True, add_self_loops=False
                    ),
                    # Keep teammate connections
                    ('batter', 'teammates_with', 'batter'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, concat=True, add_self_loops=False
                    ),
                    # REMOVED: batter-team (ablation showed it hurts)
                    # Keep pitcher-team
                    ('pitcher', 'plays_for', 'team'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, concat=True, add_self_loops=False
                    ),
                }, aggr='mean')
            else:
                conv = HeteroConv({
                    ('batter', 'faces', 'pitcher'): GATConv(
                        hidden_channels, hidden_channels, 
                        heads=1, concat=False, add_self_loops=False
                    ),
                    ('batter', 'teammates_with', 'batter'): GATConv(
                        hidden_channels, hidden_channels, 
                        heads=1, concat=False, add_self_loops=False
                    ),
                    ('pitcher', 'plays_for', 'team'): GATConv(
                        hidden_channels, hidden_channels, 
                        heads=1, concat=False, add_self_loops=False
                    ),
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
    
    def forward(self, x_dict, edge_index_dict, return_embeddings=False):
        x_dict = {
            'batter': self.batter_lin(x_dict['batter']),
            'pitcher': self.pitcher_lin(x_dict['pitcher']),
            'team': self.team_lin(x_dict['team'])
        }
        
        for i, conv in enumerate(self.convs):
            # Filter to only use the edge types we want
            filtered_edge_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                if edge_index.size(1) == 0:
                    continue
                src, rel, dst = edge_type
                # Skip batter-team edges (ablation showed removing helps)
                if rel == 'plays_for' and src == 'batter':
                    continue
                filtered_edge_dict[edge_type] = edge_index
            
            if filtered_edge_dict:
                x_dict_new = conv(x_dict, filtered_edge_dict)
                
                for key in x_dict_new.keys():
                    x_dict[key] = self.bns[i][key](x_dict_new[key])
                    x_dict[key] = F.relu(x_dict[key])
                    x_dict[key] = F.dropout(x_dict[key], p=0.1, training=self.training)
        
        if return_embeddings:
            return x_dict['batter']
        
        out = self.predictor(x_dict['batter'])
        return out.squeeze(-1)


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def temporal_split(graphs, train_size=0.7, val_size=0.15):
    n = len(graphs)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    return graphs[:train_end], graphs[train_end:val_end], graphs[val_end:]


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
def evaluate(model, graphs, device, return_details=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_embeddings = []
    all_features = []
    all_game_dates = []
    
    for graph in graphs:
        graph = graph.to(device)
        
        pred = model(graph.x_dict, graph.edge_index_dict)
        embeddings = model(graph.x_dict, graph.edge_index_dict, return_embeddings=True)
        
        target = graph['batter'].y
        features = graph['batter'].x
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())
        all_features.append(features.cpu().numpy())
        
        if hasattr(graph, 'game_date'):
            all_game_dates.extend([graph.game_date] * len(target))
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_embeddings = np.concatenate(all_embeddings)
    all_features = np.concatenate(all_features)
    
    if return_details:
        return all_preds, all_targets, all_embeddings, all_features, all_game_dates
    return all_preds, all_targets


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Stratified metrics
    low_mask = y_true < 0.3
    avg_mask = (y_true >= 0.3) & (y_true < 0.4)
    high_mask = y_true >= 0.4
    
    mae_low = mean_absolute_error(y_true[low_mask], y_pred[low_mask]) if low_mask.sum() > 0 else np.nan
    mae_avg = mean_absolute_error(y_true[avg_mask], y_pred[avg_mask]) if avg_mask.sum() > 0 else np.nan
    mae_high = mean_absolute_error(y_true[high_mask], y_pred[high_mask]) if high_mask.sum() > 0 else np.nan
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'mae_low': mae_low, 'mae_avg': mae_avg, 'mae_high': mae_high,
        'n_low': int(low_mask.sum()), 'n_avg': int(avg_mask.sum()), 'n_high': int(high_mask.sum())
    }


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_embedding_tsne(embeddings, targets, save_path='data/embedding_tsne.png'):
    """Visualize learned embeddings using t-SNE."""
    
    print("\n Generating t-SNE visualization...")
    
    # Sample if too many points
    n_samples = min(5000, len(embeddings))
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        targets = targets[idx]
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by actual wOBA
    scatter1 = axes[0].scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=targets, cmap='RdYlGn', alpha=0.6, s=20
    )
    axes[0].set_title('Embeddings Colored by Actual wOBA', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='wOBA')
    
    # Plot 2: Colored by performance tier
    tiers = np.zeros(len(targets))
    tiers[targets < 0.3] = 0  # Low
    tiers[(targets >= 0.3) & (targets < 0.4)] = 1  # Average
    tiers[targets >= 0.4] = 2  # High
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    labels = ['Low (<0.3)', 'Average (0.3-0.4)', 'High (≥0.4)']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = tiers == i
        axes[1].scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=color, label=label, alpha=0.6, s=20
        )
    
    axes[1].set_title('Embeddings by Performance Tier', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved to {save_path}")
    plt.close()


def plot_prediction_analysis(y_true, y_pred, save_path='data/prediction_analysis.png'):
    """Analyze prediction quality."""
    
    print("\n Generating prediction analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual wOBA', fontsize=11)
    ax1.set_ylabel('Predicted wOBA', fontsize=11)
    ax1.set_title('Actual vs Predicted wOBA', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Plot 2: Error distribution
    ax2 = axes[0, 1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax2.set_xlabel('Prediction Error (Pred - Actual)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    
    # Plot 3: Error by actual wOBA range
    ax3 = axes[1, 0]
    bins = [0, 0.2, 0.3, 0.4, 0.5, 1.0]
    bin_labels = ['0-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+']
    
    mae_by_bin = []
    counts_by_bin = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            mae_by_bin.append(mean_absolute_error(y_true[mask], y_pred[mask]))
            counts_by_bin.append(mask.sum())
        else:
            mae_by_bin.append(0)
            counts_by_bin.append(0)
    
    bars = ax3.bar(bin_labels, mae_by_bin, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Actual wOBA Range', fontsize=11)
    ax3.set_ylabel('MAE', fontsize=11)
    ax3.set_title('MAE by Performance Level', fontsize=13, fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, counts_by_bin):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'n={count}', ha='center', fontsize=9)
    
    # Plot 4: Residuals vs predicted
    ax4 = axes[1, 1]
    ax4.scatter(y_pred, errors, alpha=0.3, s=10)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted wOBA', fontsize=11)
    ax4.set_ylabel('Residual (Pred - Actual)', fontsize=11)
    ax4.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved to {save_path}")
    plt.close()


def plot_model_comparison(results, save_path='data/model_comparison.png'):
    """Create summary comparison figure."""
    
    print("\n Generating model comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    maes = [results[m]['mae'] for m in models]
    
    # Sort by MAE
    sorted_pairs = sorted(zip(models, maes), key=lambda x: x[1])
    models = [p[0] for p in sorted_pairs]
    maes = [p[1] for p in sorted_pairs]
    
    # Color scheme
    colors = []
    for m in models:
        if 'Optimal' in m:
            colors.append('#2ecc71')  # Green for our model
        elif 'Baseline' in m:
            colors.append('#e67e22')  # Orange for baseline
        else:
            colors.append('#3498db')  # Blue for others
    
    bars = ax.barh(range(len(models)), maes, color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Test MAE (lower is better)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        ax.text(mae + 0.005, i, f'{mae:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved to {save_path}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def train_final_model(graphs_file='data/graphs.pkl'):
    """Train optimal model and generate all visualizations."""
    
    print("="*70)
    print("FINAL MODEL TRAINING - CS224W Project")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n Loading graphs...")
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    print(f"   Loaded {len(graphs)} graphs")
    
    # Get dimensions
    sample = graphs[0]
    batter_in = sample['batter'].x.shape[1]
    pitcher_in = sample['pitcher'].x.shape[1]
    team_in = sample['team'].x.shape[1]
    
    # Split
    train_graphs, val_graphs, test_graphs = temporal_split(graphs)
    print(f"\n Temporal split:")
    print(f"   Train: {len(train_graphs)} ({train_graphs[0].game_date} → {train_graphs[-1].game_date})")
    print(f"   Val:   {len(val_graphs)}")
    print(f"   Test:  {len(test_graphs)} ({test_graphs[0].game_date} → {test_graphs[-1].game_date})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Initialize optimal model
    print("\n  Building Optimal Model:")
    print("   Architecture: GAT (attention aggregation)")
    print("   Layers: 3")
    print("   Hidden dim: 32")
    print("   Edge types: batter-pitcher, teammates, pitcher-team (NO batter-team)")
    
    model = OptimalHeteroGAT(
        batter_in=batter_in,
        pitcher_in=pitcher_in,
        team_in=team_in,
        hidden_channels=32,
        num_layers=3,
        heads=4
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Training
    print("\n Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    best_val_mae = float('inf')
    patience = 50
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'val_mae': []}
    
    for epoch in range(300):
        train_loss = train_epoch(model, train_graphs, optimizer, device)
        val_preds, val_targets = evaluate(model, val_graphs, device)
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        
        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            print(f"   Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    torch.save(best_state, 'data/optimal_model.pt')
    print(f"\n Model saved to data/optimal_model.pt")
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    test_preds, test_targets, embeddings, features, game_dates = evaluate(
        model, test_graphs, device, return_details=True
    )
    
    metrics = compute_metrics(test_targets, test_preds)
    
    # Baseline comparison
    baseline_mae = mean_absolute_error(test_targets, features[:, 0])  # Rolling wOBA
    
    print(f"\n📊 Test Set Results:")
    print(f"   Overall MAE:  {metrics['mae']:.4f}")
    print(f"   RMSE:         {metrics['rmse']:.4f}")
    print(f"   R²:           {metrics['r2']:.4f}")
    print(f"\n   Stratified MAE:")
    print(f"   Low wOBA (<0.3):      {metrics['mae_low']:.4f} (n={metrics['n_low']})")
    print(f"   Avg wOBA (0.3-0.4):   {metrics['mae_avg']:.4f} (n={metrics['n_avg']})")
    print(f"   High wOBA (≥0.4):     {metrics['mae_high']:.4f} (n={metrics['n_high']})")
    print(f"\n   vs Baseline (Rolling Avg): {baseline_mae:.4f}")
    print(f"   Improvement: {((baseline_mae - metrics['mae']) / baseline_mae * 100):.1f}%")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_embedding_tsne(embeddings, test_targets)
    plot_prediction_analysis(test_targets, test_preds)
    
    # Comparison with other models
    comparison_results = {
        'Rolling Avg Baseline': {'mae': baseline_mae},
        'Optimal GAT (Ours)': {'mae': metrics['mae']},
    }
    
    # Load ablation results if available
    try:
        with open('data/ablation_results.json', 'r') as f:
            ablation = json.load(f)
        comparison_results['Full GraphSAGE'] = {'mae': ablation['edge_types']['Full Model']['test_mae']}
        comparison_results['MLP (No Graph)'] = {'mae': ablation['edge_types']['No Graph (MLP)']['test_mae']}
    except:
        pass
    
    plot_model_comparison(comparison_results)
    
    # Save all results
    final_results = {
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics.items()},
        'baseline_mae': float(baseline_mae),
        'improvement_pct': float((baseline_mae - metrics['mae']) / baseline_mae * 100),
        'model_config': {
            'architecture': 'GAT',
            'layers': 3,
            'hidden_dim': 32,
            'heads': 4,
            'edge_types': ['batter-pitcher', 'teammates', 'pitcher-team'],
            'n_params': n_params
        },
        'training': {
            'epochs': len(history['train_loss']),
            'best_val_mae': float(best_val_mae)
        }
    }
    
    with open('data/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'actual': test_targets,
        'predicted': test_preds,
        'error': test_preds - test_targets,
        'abs_error': np.abs(test_preds - test_targets),
        'rolling_woba': features[:, 0]
    })
    predictions_df.to_csv('data/final_predictions.csv', index=False)
    
    print("\n" + "="*70)
    print("SUMMARY FOR FINAL REPORT")
    print("="*70)
    print(f"""
     OPTIMAL MODEL PERFORMANCE:
    
    • Architecture: Heterogeneous GAT
    • Configuration: 3 layers, 32 hidden dims, 4 attention heads
    • Edge types used: batter-pitcher, teammates, pitcher-team
    • Edge types removed: batter-team (improved performance)
    
     KEY METRICS (Temporal Split):
    
    • Test MAE: {metrics['mae']:.4f}
    • Baseline MAE: {baseline_mae:.4f}
    • Improvement: {((baseline_mae - metrics['mae']) / baseline_mae * 100):.1f}%
    
     FILES GENERATED:
    
    • data/optimal_model.pt - Trained model weights
    • data/final_results.json - All metrics and config
    • data/final_predictions.csv - Predictions for analysis
    • data/embedding_tsne.png - t-SNE visualization
    • data/prediction_analysis.png - Error analysis
    • data/model_comparison.png - Model comparison chart
    """)
    
    print(f"\n Complete! ({datetime.now().strftime('%H:%M:%S')})")
    
    return model, final_results


if __name__ == '__main__':
    model, results = train_final_model()
