"""
Create training curves visualization from hardcoded data.
There was an error in creating the graph from execute.py so this simply copies the results from that output run and builds a visualization for it. 
"""

import matplotlib.pyplot as plt
import numpy as np

# Hardcoded training data
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
          160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]

train_loss = [0.0903, 0.0878, 0.0857, 0.0841, 0.0832, 0.0879, 0.0877, 0.0874, 
              0.0858, 0.0857, 0.0846, 0.0836, 0.0826, 0.0826, 0.0821, 0.0882,
              0.0877, 0.0875, 0.0873, 0.0869, 0.0862, 0.0859, 0.0859, 0.0857,
              0.0850, 0.0843, 0.0843, 0.0839, 0.0830, 0.0832]

val_mae = [0.0652, 0.0655, 0.0626, 0.0613, 0.0616, 0.0654, 0.0621, 0.0642,
           0.0634, 0.0646, 0.0619, 0.0613, 0.0629, 0.0621, 0.0616, 0.0618,
           0.0636, 0.0605, 0.0637, 0.0620, 0.0611, 0.0641, 0.0672, 0.0636,
           0.0636, 0.0675, 0.0633, 0.0635, 0.0629, 0.0635]

learning_rate = [0.000922, 0.000684, 0.000376, 0.000116, 0.000002, 0.000980,
                 0.000914, 0.000807, 0.000670, 0.000516, 0.000361, 0.000220,
                 0.000106, 0.000031, 0.000001, 0.000995, 0.000978, 0.000949,
                 0.000909, 0.000859, 0.000800, 0.000734, 0.000662, 0.000586,
                 0.000508, 0.000430, 0.000354, 0.000281, 0.000213, 0.000153]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Training Loss
axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2.5, color='#2E86AB', marker='o', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(fontsize=11)
axes[0].set_xlim(0, 310)

# Plot 2: Validation MAE
best_mae = min(val_mae)
best_epoch = epochs[val_mae.index(best_mae)]

axes[1].plot(epochs, val_mae, label='Validation MAE', color='#A23B72', linewidth=2.5, marker='o', markersize=4)
axes[1].axhline(y=best_mae, color='#F18F01', linestyle='--', linewidth=2,
                label=f'Best: {best_mae:.4f} (Epoch {best_epoch})')
axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('MAE', fontsize=13, fontweight='bold')
axes[1].set_title('Validation MAE Over Time', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(fontsize=11)
axes[1].set_xlim(0, 310)

# Plot 3: Learning Rate
axes[2].plot(epochs, learning_rate, label='Learning Rate', color='#06A77D', linewidth=2.5, marker='o', markersize=4)
axes[2].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
axes[2].set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].set_yscale('log')
axes[2].legend(fontsize=11)
axes[2].set_xlim(0, 310)

# Add annotations
# Annotate learning rate restarts
axes[2].annotate('Warm Restart', xy=(60, learning_rate[5]), xytext=(80, 0.0008),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')
axes[2].annotate('Warm Restart', xy=(160, learning_rate[15]), xytext=(180, 0.0008),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')

# Add best validation point
axes[1].scatter([best_epoch], [best_mae], color='#F18F01', s=200, zorder=5, 
               edgecolors='black', linewidths=2, label='Best Model')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Training curves saved to training_curves.png")

# Also create a summary statistics plot
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

summary_text = f"""
Training Summary (300 Epochs)
{'='*50}

Best Validation MAE: {best_mae:.4f} (Epoch {best_epoch})
Final Training Loss: {train_loss[-1]:.4f}
Initial Training Loss: {train_loss[0]:.4f}
Loss Reduction: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.1f}%

Learning Rate Schedule: Cosine Annealing with Warm Restarts
  - Initial LR: 0.001
  - Min LR: 1e-6
  - Warm Restarts at: Epoch 60, 160
  - Final LR: {learning_rate[-1]:.6f}

Model Architecture:
  - Heterogeneous GraphSAGE
  - 3 layers, 128 hidden dimensions
  - 424,065 total parameters
  
Regularization:
  - Batch Normalization
  - Dropout (0.1-0.3)
  - Gradient Clipping (max_norm=1.0)
  - Weight Decay (1e-4)
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax.axis('off')

plt.tight_layout()
plt.savefig('training_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Training summary saved to training_summary.png")

plt.show()
