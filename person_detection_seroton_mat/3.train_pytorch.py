"""
Simple PyTorch Training Script
Input: augmented_data.csv
Output: Trained model (best_model.pth)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "="*60)
print("📥 Loading normalized data...")
print("="*60)

# Load CSV
# df = pd.read_csv('normalized_data.csv')
df = pd.read_csv('augmented_data.csv')
print(f"✓ Loaded {len(df)} rows")
print(f"✓ Columns: {list(df.columns)}")

# Extract features and labels
X = df[['Float1', 'Float2', 'Float3', 'Float4']].values
y = df['Label'].values

print(f"✓ Features shape: {X.shape}")
print(f"✓ Labels: {np.unique(y)} (0=Empty, 1=Occupied)")
print(f"  - Empty samples: {np.sum(y == 0)}")
print(f"  - Occupied samples: {np.sum(y == 1)}")

# Split into train and test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# Create data loaders
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"✓ Batch size: {batch_size}")

# ============================================================================
# 2. DEFINE MODEL
# ============================================================================

print("\n" + "="*60)
print("🧠 Building Model...")
print("="*60)

class SimpleNN(nn.Module):
    """Simple neural network for binary classification"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)      # Input: 4 features
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()      # Binary classification
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN().to(device)
print(model)
print(f"✓ Model moved to {device}")

# ============================================================================
# 3. TRAINING SETUP
# ============================================================================

print("\n" + "="*60)
print("⚙️  Training Setup...")
print("="*60)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
best_loss = float('inf')
train_losses = []
test_losses = []
accuracies = []

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

print("\n" + "="*60)
print("🚀 Training Model...")
print("="*60)

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        # Test loss
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
        
        # Test accuracy
        predictions = (test_outputs > 0.5).float()
        accuracy = (predictions == y_test_tensor).float().mean().item()
        accuracies.append(accuracy)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save best model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pth')

print("\n✅ Training complete!")
print(f"✓ Best model saved as: best_model.pth")

# ============================================================================
# 5. EVALUATION
# ============================================================================

print("\n" + "="*60)
print("📊 Final Evaluation")
print("="*60)

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def evaluate_model(model, X, y, set_name="Dataset"):
    """Evaluate model on a given dataset"""
    with torch.no_grad():
        outputs = model(X)
        predictions = (outputs > 0.5).float()

        accuracy = (predictions == y).float().mean().item()

        tp = ((predictions == 1) & (y == 1)).sum().item()
        tn = ((predictions == 0) & (y == 0)).sum().item()
        fp = ((predictions == 1) & (y == 0)).sum().item()
        fn = ((predictions == 0) & (y == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{set_name} ({len(y)} samples):")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return accuracy, precision, recall, f1, tn, fp, fn, tp

# Evaluate on all three sets
train_acc, train_prec, train_rec, train_f1, _, _, _, _ = evaluate_model(
    model, X_train_tensor, y_train_tensor, "✓ TRAINING SET"
)
test_acc, test_prec, test_rec, test_f1, tn, fp, fn, tp = evaluate_model(
    model, X_test_tensor, y_test_tensor, "✓ TEST SET"
)

# Create full dataset tensor for additional evaluation
X_full_tensor = torch.FloatTensor(X).to(device)
y_full_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)
full_acc, full_prec, full_rec, full_f1, _, _, _, _ = evaluate_model(
    model, X_full_tensor, y_full_tensor, "✓ FULL DATASET"
)

# Check for overfitting
print("\n" + "="*60)
print("⚠️  OVERFITTING ANALYSIS:")
print("="*60)
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Full Accuracy:  {full_acc:.4f}")
print(f"  Train-Test Diff: {abs(train_acc - test_acc):.4f}")
if abs(train_acc - test_acc) < 0.05:
    print("  ✓ Good generalization - no overfitting detected")
else:
    print("  ⚠️  WARNING: Model overfits! Train >> Test performance")

# Use test set metrics for final results
accuracy, precision, recall, f1 = test_acc, test_prec, test_rec, test_f1

print(f"\n📊 Test Set Confusion Matrix ({len(y_test)} samples):")
print(f"                Predicted")
print(f"              Empty  Occupied")
print(f"        Empty    {tn:3d}      {fp:3d}")
print(f"Actual")
print(f"     Occupied    {fn:3d}      {tp:3d}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("💾 Saving Results...")
print("="*60)

# Save training history
results = {
    'epoch': list(range(1, epochs+1)),
    'train_loss': train_losses,
    'test_loss': test_losses,
    'accuracy': accuracies
}
results_df = pd.DataFrame(results)
results_df.to_csv('training_results.csv', index=False)
print(f"✓ Training results saved: training_results.csv")

# Save model info
model_info = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'best_loss': best_loss,
    'device': str(device)
}
print(f"\n✅ Model Training Complete!")
print(f"   Model: best_model.pth")
print(f"   Results: training_results.csv")

# ============================================================================
# 7. PLOT RESULTS (optional)
# ============================================================================

print("\n" + "="*60)
print("📈 Creating plots...")
print("="*60)

# Create a figure with 2x2 subplots
fig = plt.figure(figsize=(14, 10))

# Plot 1: Loss (top-left)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(train_losses, label='Train Loss', linewidth=2, marker='o')
ax1.plot(test_losses, label='Test Loss', linewidth=2, marker='s')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training vs Test Loss', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy (top-right)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(accuracies, label='Test Accuracy', linewidth=2, color='green', marker='o')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Test Accuracy Over Time', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.axhline(y=accuracy, color='r', linestyle='--', label=f'Final: {accuracy:.4f}')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap (bottom-left) - TEST SET
ax3 = plt.subplot(2, 2, 3)
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax3, annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax3.set_title(f'Test Set Confusion Matrix ({len(y_test)} samples)', fontsize=12, fontweight='bold')

# Plot 4: Evaluation Metrics (bottom-right)
ax4 = plt.subplot(2, 2, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax4.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Test Set Evaluation Metrics', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.1])
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Comprehensive plots saved: training_plots.png (Test Set: {len(y_test)} samples)")

# Create comprehensive confusion matrix comparison (Train vs Test vs Full)
fig_cm_compare = plt.figure(figsize=(16, 5))

# Get confusion matrices for all datasets
with torch.no_grad():
    # Training set CM
    train_out = model(X_train_tensor)
    train_pred = (train_out > 0.5).float()
    train_tp = ((train_pred == 1) & (y_train_tensor == 1)).sum().item()
    train_tn = ((train_pred == 0) & (y_train_tensor == 0)).sum().item()
    train_fp = ((train_pred == 1) & (y_train_tensor == 0)).sum().item()
    train_fn = ((train_pred == 0) & (y_train_tensor == 1)).sum().item()

    # Full dataset CM
    full_out = model(X_full_tensor)
    full_pred = (full_out > 0.5).float()
    full_tp = ((full_pred == 1) & (y_full_tensor == 1)).sum().item()
    full_tn = ((full_pred == 0) & (y_full_tensor == 0)).sum().item()
    full_fp = ((full_pred == 1) & (y_full_tensor == 0)).sum().item()
    full_fn = ((full_pred == 0) & (y_full_tensor == 1)).sum().item()

# Plot 1: Training Set CM
ax_train = plt.subplot(1, 3, 1)
cm_train = np.array([[train_tn, train_fp], [train_fn, train_tp]])
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax_train, annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax_train.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax_train.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax_train.set_title(f'Training Set\n({len(y_train)} samples)', fontsize=12, fontweight='bold')

# Plot 2: Test Set CM
ax_test = plt.subplot(1, 3, 2)
cm_test = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax_test, annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax_test.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax_test.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax_test.set_title(f'Test Set\n({len(y_test)} samples)', fontsize=12, fontweight='bold')

# Plot 3: Full Dataset CM
ax_full = plt.subplot(1, 3, 3)
cm_full = np.array([[full_tn, full_fp], [full_fn, full_tp]])
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax_full, annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax_full.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax_full.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax_full.set_title(f'Full Dataset\n({len(y)} samples)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Comparison plots saved: confusion_matrices_comparison.png")
print(f"  - Training Set: {len(y_train)} samples")
print(f"  - Test Set: {len(y_test)} samples")
print(f"  - Full Dataset: {len(y)} samples")

# Save individual detailed confusion matrix plot (Full Dataset)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax_cm, annot_kws={'size': 16, 'weight': 'bold'},
            cbar_kws={'label': 'Count'})
ax_cm.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax_cm.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
ax_cm.set_title(f'Confusion Matrix - Full Dataset ({len(y)} samples)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_full.png', dpi=150, bbox_inches='tight')
print(f"✓ Full dataset confusion matrix saved: confusion_matrix_full.png")