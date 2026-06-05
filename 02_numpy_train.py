# ══════════════════════════════════════════
# FILE: 02_numpy_train.py
# WHAT THIS FILE DOES: Trains a neural network built entirely in NumPy
#   (zero PyTorch, zero sklearn for training) and evaluates it on MNIST.
#   This is the file that proves mathematical understanding.
# DEPENDS ON: 01_data_explore.py must be run first (generates .npy files)
# RUN WITH: python 02_numpy_train.py
# EXPECTED OUTPUT: ~95-96% test accuracy, 6 saved plots, saved weights
# ══════════════════════════════════════════

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to Python path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom neural network from src/
from src.network import NeuralNetwork

# ── ENSURE OUTPUT DIRECTORIES EXIST ──
for directory in ['outputs/plots', 'outputs/models', 'outputs/results']:
    os.makedirs(directory, exist_ok=True)

# ══════════════════════════════════════════
# HYPERPARAMETERS — ALL NAMED CONSTANTS, NO MAGIC NUMBERS
# ══════════════════════════════════════════

# ── ARCHITECTURE ──
INPUT_SIZE  = 784   # 28 × 28 = 784 pixels per image (flattened)
HIDDEN_1    = 128   # First hidden layer: 128 neurons
HIDDEN_2    = 64    # Second hidden layer: 64 neurons
OUTPUT_SIZE = 10    # Output layer: 10 neurons (one per digit class)

# ── TRAINING ──
LEARNING_RATE       = 0.01   # Step size for gradient descent
BATCH_SIZE          = 64     # Samples per gradient update
EPOCHS              = 50     # Maximum number of full passes through data
EARLY_STOP_PATIENCE = 10     # Stop if val loss doesn't improve for this many epochs
VAL_SPLIT           = 0.10   # Use 10% of training data for validation

# ── WHY THESE VALUES? ──
# LEARNING_RATE = 0.01: Common starting point for SGD. Too large → loss diverges.
#   Too small → training takes forever. 0.01 works well with normalized MNIST.
# BATCH_SIZE = 64: Powers of 2 are preferred (hardware alignment). 64 balances
#   gradient quality (larger batch = better estimate) vs. speed (smaller = faster).
# HIDDEN_1 = 128, HIDDEN_2 = 64: Bottleneck architecture — each layer compresses
#   information, forcing the network to learn efficient representations.

print("=" * 60)
print("NUMPY NEURAL NETWORK — TRAINING")
print("=" * 60)
print(f"\nHyperparameters:")
print(f"  Architecture:   {INPUT_SIZE} → {HIDDEN_1} → {HIDDEN_2} → {OUTPUT_SIZE}")
print(f"  Learning rate:  {LEARNING_RATE}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Max epochs:     {EPOCHS}")
print(f"  Val split:      {VAL_SPLIT:.0%}")
print(f"  Early stopping: {EARLY_STOP_PATIENCE} epochs patience")

# ══════════════════════════════════════════
# LOAD PREPROCESSED DATA
# ══════════════════════════════════════════

RESULTS_DIR = 'outputs/results'
REQUIRED_FILES = [
    f'{RESULTS_DIR}/X_train.npy',
    f'{RESULTS_DIR}/X_test.npy',
    f'{RESULTS_DIR}/y_train.npy',
    f'{RESULTS_DIR}/y_test.npy',
    f'{RESULTS_DIR}/y_train_onehot.npy',
    f'{RESULTS_DIR}/y_test_onehot.npy',
]

for filepath in REQUIRED_FILES:
    if not os.path.exists(filepath):
        print(f"\n❌ Missing file: {filepath}")
        print("Please run 01_data_explore.py first:")
        print("  python 01_data_explore.py")
        sys.exit(1)

print("\nLoading preprocessed data...")
X_train_full = np.load(f'{RESULTS_DIR}/X_train.npy')       # (60000, 784)
X_test       = np.load(f'{RESULTS_DIR}/X_test.npy')        # (10000, 784)
y_train_full = np.load(f'{RESULTS_DIR}/y_train.npy')       # (60000,)
y_test       = np.load(f'{RESULTS_DIR}/y_test.npy')        # (10000,)
y_train_oh_full = np.load(f'{RESULTS_DIR}/y_train_onehot.npy')  # (60000, 10)
y_test_oh    = np.load(f'{RESULTS_DIR}/y_test_onehot.npy') # (10000, 10)

print(f"  X_train shape:        {X_train_full.shape}")
print(f"  X_test shape:         {X_test.shape}")
print(f"  y_train shape:        {y_train_full.shape}")
print(f"  y_test shape:         {y_test.shape}")
print(f"  y_train_onehot shape: {y_train_oh_full.shape}")
print(f"  y_test_onehot shape:  {y_test_oh.shape}")

# ── SPLIT TRAINING DATA INTO TRAIN + VALIDATION ──
NUM_TRAIN_FULL = X_train_full.shape[0]  # 60000
NUM_VAL = int(NUM_TRAIN_FULL * VAL_SPLIT)  # 6000
NUM_TRAIN = NUM_TRAIN_FULL - NUM_VAL       # 54000

# Shuffle before splitting (ensure validation isn't biased)
np.random.seed(42)  # Reproducibility
shuffle_idx = np.random.permutation(NUM_TRAIN_FULL)
X_all_shuffled    = X_train_full[shuffle_idx]        # (60000, 784)
y_all_shuffled    = y_train_oh_full[shuffle_idx]     # (60000, 10)
y_int_shuffled    = y_train_full[shuffle_idx]        # (60000,)

X_train    = X_all_shuffled[:NUM_TRAIN]          # (54000, 784)
y_train_oh = y_all_shuffled[:NUM_TRAIN]          # (54000, 10)
X_val      = X_all_shuffled[NUM_TRAIN:]          # (6000, 784)
y_val_oh   = y_all_shuffled[NUM_TRAIN:]          # (6000, 10)
y_val_int  = y_int_shuffled[NUM_TRAIN:]          # (6000,)

print(f"\nData split:")
print(f"  Training:   {X_train.shape[0]:,} samples")
print(f"  Validation: {X_val.shape[0]:,} samples")
print(f"  Test:       {X_test.shape[0]:,} samples (held out until final eval)")

# ══════════════════════════════════════════
# BUILD THE NEURAL NETWORK
# ══════════════════════════════════════════

print("\nBuilding network architecture:")

# Each dict: {'in': input_neurons, 'out': output_neurons, 'act': activation}
layer_configs = [
    {'in': INPUT_SIZE, 'out': HIDDEN_1,    'act': 'relu'},
    {'in': HIDDEN_1,   'out': HIDDEN_2,    'act': 'relu'},
    {'in': HIDDEN_2,   'out': OUTPUT_SIZE, 'act': 'softmax'},
]

model = NeuralNetwork(layer_configs)

# ── COUNT AND EXPLAIN PARAMETERS ──
# How to count parameters manually:
# Layer 1: W(784×128) + b(128)   = 784×128 + 128 = 100,352 + 128 = 100,480
# Layer 2: W(128×64)  + b(64)    = 128×64  + 64  =   8,192 + 64  =   8,256
# Layer 3: W(64×10)   + b(10)    = 64×10   + 10  =     640 + 10  =     650
# Total: 100,480 + 8,256 + 650 = 109,386 parameters
layer1_params = INPUT_SIZE * HIDDEN_1    + HIDDEN_1
layer2_params = HIDDEN_1   * HIDDEN_2   + HIDDEN_2
layer3_params = HIDDEN_2   * OUTPUT_SIZE + OUTPUT_SIZE
total_params  = layer1_params + layer2_params + layer3_params

print(f"\nParameter count verification:")
print(f"  Layer 1: {INPUT_SIZE}×{HIDDEN_1} + {HIDDEN_1} = {layer1_params:,}")
print(f"  Layer 2: {HIDDEN_1}×{HIDDEN_2} + {HIDDEN_2}   = {layer2_params:,}")
print(f"  Layer 3: {HIDDEN_2}×{OUTPUT_SIZE} + {OUTPUT_SIZE}  = {layer3_params:,}")
print(f"  TOTAL:   {total_params:,} trainable parameters")

# ══════════════════════════════════════════
# TRAIN THE NETWORK
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

start_time = time.time()

history = model.train(
    X_train=X_train,
    y_train=y_train_oh,
    X_val=X_val,
    y_val=y_val_oh,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
)

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")

# ══════════════════════════════════════════
# PLOT TRAINING CURVES
# ══════════════════════════════════════════

best_epoch = history.get('best_epoch', len(history['train_loss']))
actual_epochs = len(history['train_loss'])
epoch_range = range(1, actual_epochs + 1)

print("\nGenerating training curve plots...")

# ── PLOT 5: LOSS CURVES ──
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epoch_range, history['train_loss'],
        label='Training Loss', color='royalblue', linewidth=2)
ax.plot(epoch_range, history['val_loss'],
        label='Validation Loss', color='darkorange', linewidth=2, linestyle='--')

# Mark where early stopping triggered (the best epoch)
if best_epoch <= actual_epochs:
    best_val = history['val_loss'][best_epoch - 1]
    ax.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2,
               label=f'Best model (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val], color='green', s=100, zorder=5)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title('NumPy Neural Network — Training vs Validation Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, actual_epochs)

plt.tight_layout()
PLOT5_PATH = 'outputs/plots/05_numpy_training_loss.png'
plt.savefig(PLOT5_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT5_PATH}")

# ── PLOT 6: ACCURACY CURVES ──
TARGET_ACCURACY = 0.95  # Draw a line at 95% to show target

fig, ax = plt.subplots(figsize=(10, 6))

train_acc_pct = [a * 100 for a in history['train_acc']]
val_acc_pct   = [a * 100 for a in history['val_acc']]

ax.plot(epoch_range, train_acc_pct,
        label='Training Accuracy', color='royalblue', linewidth=2)
ax.plot(epoch_range, val_acc_pct,
        label='Validation Accuracy', color='darkorange', linewidth=2, linestyle='--')

# Draw 95% target line
ax.axhline(y=TARGET_ACCURACY * 100, color='green', linestyle='-.', linewidth=1.5,
           label=f'Target: {TARGET_ACCURACY:.0%}')

if best_epoch <= actual_epochs:
    ax.axvline(x=best_epoch, color='purple', linestyle=':', linewidth=2,
               label=f'Best model (epoch {best_epoch})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('NumPy Neural Network — Training vs Validation Accuracy', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, actual_epochs)
ax.set_ylim(0, 101)

plt.tight_layout()
PLOT6_PATH = 'outputs/plots/06_numpy_training_accuracy.png'
plt.savefig(PLOT6_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT6_PATH}")

# ══════════════════════════════════════════
# FINAL EVALUATION ON TEST SET
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_acc, test_loss = model.evaluate(X_test, y_test_oh)
print(f"\n  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Test Loss:     {test_loss:.4f}")

# Get predictions for all test samples
y_pred_int = model.predict(X_test)
# y_pred_int shape: (10000,) — integer predictions 0-9

# ── CLASSIFICATION REPORT ──
print("\nPer-class Performance (Classification Report):")
print(classification_report(y_test, y_pred_int,
                             target_names=[f'Digit {i}' for i in range(10)]))

# ── PLOT 7: CONFUSION MATRIX ──
print("Generating Plot 7: Confusion matrix...")

# WHAT IS A CONFUSION MATRIX?
# A 10×10 grid. Row i, Column j = number of times the true digit was i
# but we predicted j. Perfect model has a bright diagonal and zeros elsewhere.
# The off-diagonal entries reveal which digits get confused with which others.
# Common confusions: 4 vs 9, 3 vs 5, 7 vs 1

cm = confusion_matrix(y_test, y_pred_int)
# cm shape: (10, 10)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,         # Print numbers in each cell
    fmt='d',            # Integer format
    cmap='Blues',
    xticklabels=list(range(10)),
    yticklabels=list(range(10)),
    linewidths=0.5,
    ax=ax
)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'NumPy Neural Network — Confusion Matrix\n(Test Accuracy: {test_acc*100:.2f}%)',
             fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Highlight correct predictions (diagonal) in a different color
for i in range(10):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='gold', linewidth=2))

plt.tight_layout()
PLOT7_PATH = 'outputs/plots/07_numpy_confusion_matrix.png'
plt.savefig(PLOT7_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT7_PATH}")

# ── PLOT 8: SAMPLE PREDICTIONS WITH GREEN/RED BORDERS ──
print("Generating Plot 8: Sample predictions with color-coded borders...")

# Load original 2D images for display (not flattened)
import torchvision
mnist_test_raw = torchvision.datasets.MNIST(
    root='data/', train=False, download=False)
X_test_images = mnist_test_raw.data.numpy()  # (10000, 28, 28) original images

NUM_DISPLAY = 20  # Show 20 samples
DISPLAY_ROWS = 4
DISPLAY_COLS = 5

fig, axes = plt.subplots(DISPLAY_ROWS, DISPLAY_COLS, figsize=(12, 10))
fig.suptitle('NumPy NN Predictions: Green=Correct, Red=Wrong',
             fontsize=13, fontweight='bold')

# Select 10 correct + 10 wrong predictions for a balanced display
correct_indices = np.where(y_pred_int == y_test)[0][:10]
wrong_indices   = np.where(y_pred_int != y_test)[0][:10]
display_indices = np.concatenate([correct_indices, wrong_indices])

axes_flat = axes.flatten()
for plot_idx, test_idx in enumerate(display_indices[:NUM_DISPLAY]):
    ax = axes_flat[plot_idx]
    is_correct = (y_pred_int[test_idx] == y_test[test_idx])
    border_color = 'green' if is_correct else 'red'

    # Show the image
    ax.imshow(X_test_images[test_idx], cmap='gray', interpolation='nearest')
    ax.set_title(f'True: {y_test[test_idx]}\nPred: {y_pred_int[test_idx]}',
                 fontsize=8,
                 color='green' if is_correct else 'red',
                 fontweight='bold')
    ax.axis('off')

    # Add colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
PLOT8_PATH = 'outputs/plots/08_numpy_predictions.png'
plt.savefig(PLOT8_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT8_PATH}")

# ══════════════════════════════════════════
# SAVE WEIGHTS AND RESULTS
# ══════════════════════════════════════════

WEIGHTS_PATH = 'outputs/models/numpy_nn_weights.npz'
model.save_weights(WEIGHTS_PATH)

# Save text results for the comparison file
RESULTS_PATH = 'outputs/results/numpy_results.txt'
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    f.write(f"NumPy Neural Network Results\n")
    f.write(f"{'='*40}\n")
    f.write(f"Architecture: {INPUT_SIZE} → {HIDDEN_1} → {HIDDEN_2} → {OUTPUT_SIZE}\n")
    f.write(f"Total Parameters: {total_params:,}\n")
    f.write(f"Test Accuracy: {test_acc * 100:.4f}%\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")
    f.write(f"Training Time: {training_time:.1f} seconds\n")
    f.write(f"Epochs Trained: {actual_epochs}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
print(f"✓ Results saved to: {RESULTS_PATH}")

# ══════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("NUMPY TRAINING COMPLETE!")
print("=" * 60)
print(f"\n  Test Accuracy:   {test_acc * 100:.2f}%")
print(f"  Test Loss:       {test_loss:.4f}")
print(f"  Epochs trained:  {actual_epochs}/{EPOCHS}")
print(f"  Training time:   {training_time:.1f}s ({training_time/60:.1f} min)")
print(f"  Parameters:      {total_params:,}")
print(f"\n  Saved files:")
print(f"    {WEIGHTS_PATH}")
print(f"    {RESULTS_PATH}")
print(f"    outputs/plots/05_numpy_training_loss.png")
print(f"    outputs/plots/06_numpy_training_accuracy.png")
print(f"    outputs/plots/07_numpy_confusion_matrix.png")
print(f"    outputs/plots/08_numpy_predictions.png")
print(f"\nRun NEXT: python 03_pytorch_cnn.py")
print("=" * 60)
