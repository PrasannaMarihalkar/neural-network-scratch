# ══════════════════════════════════════════
# FILE: 04_compare_visualize.py
# WHAT THIS FILE DOES: Generates head-to-head comparison visualizations
#   of the NumPy NN vs PyTorch CNN, including accuracy bars, side-by-side
#   confusion matrices, error analysis, and a printed summary table.
# DEPENDS ON: 02_numpy_train.py AND 03_pytorch_cnn.py must be run first
# RUN WITH: python 04_compare_visualize.py
# EXPECTED OUTPUT: 3 comparison plots + printed table with key insights
# ══════════════════════════════════════════

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Add project root to path for src/ imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from src.network import NeuralNetwork

# ── CREATE OUTPUT DIRECTORIES ──
for directory in ['outputs/plots', 'outputs/results']:
    os.makedirs(directory, exist_ok=True)

# ══════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════

INPUT_SIZE  = 784
HIDDEN_1    = 128
HIDDEN_2    = 64
OUTPUT_SIZE = 10
NUM_CLASSES = 10

HUMAN_ACCURACY = 98.0  # Human-level MNIST accuracy (well-studied baseline)

# ══════════════════════════════════════════
# LOAD SAVED RESULTS
# ══════════════════════════════════════════

def load_results(filepath):
    """
    WHAT: Parses a results .txt file into a dictionary.
    INPUT:  filepath — path to results file
    OUTPUT: dict with keys matching file contents
    """
    if not os.path.exists(filepath):
        print(f"❌ Missing file: {filepath}")
        print("   Run the corresponding training script first!")
        return None

    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line and '=' not in line:
                key, val = line.strip().split(':', 1)
                results[key.strip()] = val.strip()
    return results

# Load both results files
numpy_results  = load_results('outputs/results/numpy_results.txt')
pytorch_results = load_results('outputs/results/pytorch_results.txt')

if numpy_results is None or pytorch_results is None:
    print("\nCannot generate comparison without both models trained.")
    print("Please run:")
    print("  python 02_numpy_train.py")
    print("  python 03_pytorch_cnn.py")
    sys.exit(1)

# Parse accuracy values
numpy_acc  = float(numpy_results.get('Test Accuracy', '0').replace('%', ''))
pytorch_acc = float(pytorch_results.get('Test Accuracy', '0').replace('%', ''))
numpy_params  = int(numpy_results.get('Total Parameters', '0').replace(',', ''))
pytorch_params = int(pytorch_results.get('Total Parameters', '0').replace(',', ''))
numpy_time   = float(numpy_results.get('Training Time', '0').replace(' seconds', ''))
pytorch_time  = float(pytorch_results.get('Training Time', '0').replace(' seconds', ''))

print("=" * 60)
print("COMPARISON: NumPy NN vs PyTorch CNN")
print("=" * 60)
print(f"\n  NumPy NN  test accuracy: {numpy_acc:.2f}%")
print(f"  PyTorch CNN test accuracy: {pytorch_acc:.2f}%")

# ══════════════════════════════════════════
# LOAD TEST DATA AND RUN INFERENCE FOR BOTH MODELS
# ══════════════════════════════════════════

print("\nLoading test data...")

# ── NUMPY MODEL PREDICTIONS ──
RESULTS_DIR = 'outputs/results'
REQUIRED = [f'{RESULTS_DIR}/X_test.npy', f'{RESULTS_DIR}/y_test.npy',
            f'{RESULTS_DIR}/y_test_onehot.npy']
for f in REQUIRED:
    if not os.path.exists(f):
        print(f"❌ Missing: {f} — run 01_data_explore.py first")
        sys.exit(1)

X_test       = np.load(f'{RESULTS_DIR}/X_test.npy')        # (10000, 784)
y_test       = np.load(f'{RESULTS_DIR}/y_test.npy')        # (10000,)
y_test_oh    = np.load(f'{RESULTS_DIR}/y_test_onehot.npy') # (10000, 10)

# Rebuild and load NumPy model
numpy_model = NeuralNetwork([
    {'in': INPUT_SIZE, 'out': HIDDEN_1,    'act': 'relu'},
    {'in': HIDDEN_1,   'out': HIDDEN_2,    'act': 'relu'},
    {'in': HIDDEN_2,   'out': OUTPUT_SIZE, 'act': 'softmax'},
])

NUMPY_WEIGHTS_PATH = 'outputs/models/numpy_nn_weights.npz'
if not os.path.exists(NUMPY_WEIGHTS_PATH):
    print(f"❌ Missing: {NUMPY_WEIGHTS_PATH} — run 02_numpy_train.py first")
    sys.exit(1)
numpy_model.load_weights(NUMPY_WEIGHTS_PATH)

numpy_preds = numpy_model.predict(X_test)  # (10000,)
print(f"✓ NumPy NN predictions computed ({len(numpy_preds):,} samples)")

# ── PYTORCH MODEL PREDICTIONS ──
# Import CNN class (defined in 03_pytorch_cnn.py, we'll reconstruct it)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define CNN inline for this script (can't import from 03_ directly)
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout2d = nn.Dropout2d(0.25)
        self.fc1   = nn.Linear(64 * 14 * 14, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B,32,28,28)
        x = self.relu(self.bn2(self.conv2(x)))   # (B,64,28,28)
        x = self.pool(x)                          # (B,64,14,14)
        x = self.dropout2d(x)
        x = x.view(x.size(0), -1)                # (B,12544)
        x = self.relu(self.fc1(x))               # (B,256)
        x = self.dropout(x)
        return self.fc2(x)                        # (B,10)

CNN_WEIGHTS_PATH = 'outputs/models/pytorch_cnn.pth'
if not os.path.exists(CNN_WEIGHTS_PATH):
    print(f"❌ Missing: {CNN_WEIGHTS_PATH} — run 03_pytorch_cnn.py first")
    sys.exit(1)

cnn_model = CNN().to(DEVICE)
cnn_model.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE))
cnn_model.eval()

# Load MNIST test set with transforms
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = torchvision.datasets.MNIST(
    root='data/', train=False, download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=512, shuffle=False, num_workers=0)

pytorch_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(DEVICE)
        logits = cnn_model(images)
        pytorch_preds.extend(logits.argmax(dim=1).cpu().numpy())
pytorch_preds = np.array(pytorch_preds)  # (10000,)
print(f"✓ PyTorch CNN predictions computed ({len(pytorch_preds):,} samples)")

# ── RAW IMAGES FOR DISPLAY ──
test_images_raw = torchvision.datasets.MNIST(
    root='data/', train=False, download=False).data.numpy()  # (10000, 28, 28)

# ══════════════════════════════════════════
# PLOT 1: HEAD-TO-HEAD ACCURACY BAR CHART
# ══════════════════════════════════════════

print("\nGenerating comparison bar chart...")

fig, ax = plt.subplots(figsize=(9, 6))

MODELS     = ['NumPy NN\n(Dense Only)', 'PyTorch CNN\n(Conv + Dense)']
ACCURACIES = [numpy_acc, pytorch_acc]
COLORS     = ['#4878CF', '#6ACC65']

bars = ax.bar(MODELS, ACCURACIES, color=COLORS, width=0.4,
              edgecolor='black', linewidth=1.2, zorder=3)

# Add value labels on top of each bar
for bar, acc in zip(bars, ACCURACIES):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=14, fontweight='bold')

# Human baseline line
ax.axhline(y=HUMAN_ACCURACY, color='red', linestyle='--', linewidth=2,
           zorder=4, label=f'Human Accuracy ≈ {HUMAN_ACCURACY:.0f}%')

ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Model Comparison: NumPy NN vs PyTorch CNN on MNIST',
             fontsize=13, fontweight='bold')
ax.set_ylim(90, 100.5)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=11)

# Add parameter count annotation
for i, (bar, params) in enumerate(zip(bars, [numpy_params, pytorch_params])):
    ax.text(bar.get_x() + bar.get_width() / 2,
            92.0,
            f'{params:,} params',
            ha='center', va='bottom', fontsize=9, color='white',
            fontweight='bold')

plt.tight_layout()
COMPARISON_PLOT_PATH = 'outputs/plots/15_model_comparison.png'
plt.savefig(COMPARISON_PLOT_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {COMPARISON_PLOT_PATH}")

# ══════════════════════════════════════════
# PLOT 2: SIDE-BY-SIDE CONFUSION MATRICES
# ══════════════════════════════════════════

print("Generating side-by-side confusion matrices...")

cm_numpy   = confusion_matrix(y_test, numpy_preds)
cm_pytorch = confusion_matrix(y_test, pytorch_preds)

# Use same max value for IDENTICAL color scale on both matrices
vmax = max(cm_numpy.max(), cm_pytorch.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Confusion Matrix Comparison: NumPy NN vs PyTorch CNN\n'
             '(Same color scale for fair comparison)',
             fontsize=13, fontweight='bold')

for ax, cm, title, acc in [
    (axes[0], cm_numpy,   f'NumPy NN ({numpy_acc:.2f}%)',   numpy_acc),
    (axes[1], cm_pytorch, f'PyTorch CNN ({pytorch_acc:.2f}%)', pytorch_acc)
]:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(10)),
                yticklabels=list(range(10)),
                vmax=vmax, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for i in range(10):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='gold', linewidth=2))

plt.tight_layout()
CM_COMPARE_PATH = 'outputs/plots/16_confusion_matrix_comparison.png'
plt.savefig(CM_COMPARE_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {CM_COMPARE_PATH}")

# ══════════════════════════════════════════
# PLOT 3: ERROR ANALYSIS
# ══════════════════════════════════════════

print("Generating error analysis...")

# Find which (true_class, predicted_class) pairs cause the most errors
def top_confusions(cm, n=5):
    """Returns list of (true, predicted, count) for worst off-diagonal cells."""
    errors = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                errors.append((i, j, cm[i, j]))
    errors.sort(key=lambda x: x[2], reverse=True)
    return errors[:n]

numpy_errors   = top_confusions(cm_numpy)
pytorch_errors = top_confusions(cm_pytorch)

print("\n  NumPy NN — Top Confusions:")
for true, pred, count in numpy_errors:
    print(f"    True={true} → Predicted={pred}: {count} errors")

print("\n  PyTorch CNN — Top Confusions:")
for true, pred, count in pytorch_errors:
    print(f"    True={true} → Predicted={pred}: {count} errors")

# Visualize hardest misclassifications for NumPy NN
num_show = 10  # Show top-10 hardest errors

# Find the actual misclassified samples for the top confusion pair
top_true, top_pred, _ = numpy_errors[0]
numpy_wrong_idx = np.where(
    (numpy_preds != y_test) & (y_test == top_true) & (numpy_preds == top_pred)
)[0]

fig, axes = plt.subplots(2, num_show, figsize=(14, 5))
fig.suptitle(f'Error Analysis: Where Each Model Fails\n'
             f'Top NumPy Error: True={top_true} → Predicted as {top_pred}  |  '
             f'PyTorch CNN rare errors',
             fontsize=11, fontweight='bold')

# Top row: NumPy errors for top confusion pair
shown = 0
for idx in numpy_wrong_idx[:num_show]:
    ax = axes[0][shown]
    ax.imshow(test_images_raw[idx], cmap='gray', interpolation='nearest')
    ax.set_title(f'T:{y_test[idx]}→P:{numpy_preds[idx]}', fontsize=7,
                 color='red', fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(2); spine.set_visible(True)
    shown += 1

# Pad if not enough samples
for i in range(shown, num_show):
    axes[0][i].axis('off')

# Bottom row: PyTorch CNN errors (all wrong predictions, hardest ones)
pytorch_wrong_idx = np.where(pytorch_preds != y_test)[0]
shown = 0
for idx in pytorch_wrong_idx[:num_show]:
    ax = axes[1][shown]
    ax.imshow(test_images_raw[idx], cmap='gray', interpolation='nearest')
    ax.set_title(f'T:{y_test[idx]}→P:{pytorch_preds[idx]}', fontsize=7,
                 color='red', fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(2); spine.set_visible(True)
    shown += 1

for i in range(shown, num_show):
    axes[1][i].axis('off')

# Add model labels
axes[0][0].set_ylabel('NumPy NN\nErrors', fontsize=9, rotation=90, labelpad=5)
axes[1][0].set_ylabel('PyTorch CNN\nErrors', fontsize=9, rotation=90, labelpad=5)

plt.tight_layout()
ERROR_PATH = 'outputs/plots/17_error_analysis.png'
plt.savefig(ERROR_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {ERROR_PATH}")

# ══════════════════════════════════════════
# FINAL COMPARISON TABLE (printed)
# ══════════════════════════════════════════

print("\n" + "=" * 65)
print("FINAL COMPARISON TABLE")
print("=" * 65)
print(f"{'Metric':<25} {'NumPy NN':<18} {'PyTorch CNN':<18}")
print("-" * 65)
print(f"{'Test Accuracy':<25} {numpy_acc:<17.2f}% {pytorch_acc:<17.2f}%")
print(f"{'Parameters':<25} {numpy_params:<18,} {pytorch_params:<18,}")
print(f"{'Training Time':<25} {numpy_time/60:<17.1f}m {pytorch_time/60:<17.1f}m")
print(f"{'Architecture':<25} {'Dense only':<18} {'Conv + Dense':<18}")
print(f"{'Conv Layers':<25} {'0':<18} {'2':<18}")
print(f"{'Regularization':<25} {'Early stopping':<18} {'Dropout + BN':<18}")
print("=" * 65)

# ── KEY INSIGHT ──
accuracy_gap = pytorch_acc - numpy_acc
print(f"\n📊 KEY INSIGHT: CNN outperforms Dense by {accuracy_gap:.2f}% accuracy")
print()
print("WHY DOES CNN WIN? (despite potentially fewer or similar parameters)")
print()
print("1. PARAMETER SHARING:")
print("   A 3×3 conv filter (9 weights) scans the ENTIRE 28×28 image.")
print("   A dense layer needs separate weights for each pixel position.")
print("   Same pattern detected ANYWHERE with just 9 weights.")
print()
print("2. SPATIAL INVARIANCE:")
print("   A CNN detecting a loop at position (5,5) automatically")
print("   detects it at (20,10) too — same filter, same weights.")
print("   Dense networks have no such spatial sharing.")
print()
print("3. LOCAL CONNECTIVITY:")
print("   Each conv neuron sees only a 3×3 neighborhood.")
print("   This matches the LOCAL structure of image features:")
print("   edges, corners, curves — all defined by neighboring pixels.")
print()
print("4. HIERARCHICAL FEATURES:")
print("   Layer 1: edges and gradients")
print("   Layer 2: curves and junctions (built from Layer 1 edges)")
print("   FC Layer: digit-specific patterns (built from curves)")
print()
print("Dense networks treat pixel (0,0) and pixel (27,27) as equally")
print("related. CNNs know they're in opposite corners of the image.")

print("\n" + "=" * 65)
print("ALL 4 FILES COMPLETE — PROJECT DONE!")
print("=" * 65)
print()
print("Output files created:")
print("  outputs/plots/  — 17 PNG visualizations")
print("  outputs/models/ — trained model weights")
print("  outputs/results/ — accuracy logs")
