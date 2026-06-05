# ══════════════════════════════════════════
# FILE: 03_pytorch_cnn.py
# WHAT THIS FILE DOES: Builds, trains, and evaluates a Convolutional Neural
#   Network in PyTorch on MNIST. Includes feature map and filter visualization
#   to show WHAT the CNN is learning. Achieves ~99%+ accuracy.
# DEPENDS ON: 01_data_explore.py (for data/ folder with MNIST cache)
# RUN WITH: python 03_pytorch_cnn.py
# EXPECTED OUTPUT: ~99-99.3% test accuracy, 7 saved plots, saved model
# ══════════════════════════════════════════

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ── CREATE OUTPUT DIRECTORIES ──
for directory in ['outputs/plots', 'outputs/models', 'outputs/results', 'data']:
    os.makedirs(directory, exist_ok=True)

# ══════════════════════════════════════════
# HYPERPARAMETERS — ALL NAMED CONSTANTS
# ══════════════════════════════════════════

BATCH_SIZE    = 64     # Samples per gradient update
EPOCHS        = 15     # Number of full passes through training data
LEARNING_RATE = 0.001  # Adam optimizer step size (smaller than SGD)
LR_STEP_SIZE  = 5      # Reduce learning rate every N epochs
LR_GAMMA      = 0.5    # Multiply LR by this factor at each step
DROPOUT_RATE  = 0.5    # Fraction of neurons to randomly zero during training
NUM_WORKERS   = 0      # DataLoader worker processes (0 avoids Windows issues)

# ── ARCHITECTURE CONSTANTS ──
NUM_INPUT_CHANNELS  = 1    # Grayscale images have 1 channel
IMAGE_SIZE          = 28   # MNIST images are 28×28 pixels
CONV1_FILTERS       = 32   # Number of filters in first conv layer
CONV2_FILTERS       = 64   # Number of filters in second conv layer
CONV_KERNEL_SIZE    = 3    # 3×3 convolution kernels
CONV_PADDING        = 1    # Padding=1 keeps spatial size unchanged after conv
POOL_SIZE           = 2    # 2×2 max pooling halves spatial dimensions
FC1_SIZE            = 256  # First fully-connected layer neurons
NUM_CLASSES         = 10   # Output classes (digits 0-9)

# ── DEVICE DETECTION WITH GRACEFUL FALLBACK ──
# Priority: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("✓ Apple Silicon GPU (MPS) detected — using Metal acceleration")
else:
    DEVICE = torch.device('cpu')
    print("ℹ Training on CPU (no GPU detected)")
    print("  Expected training time: ~3-5 minutes for 15 epochs")
    print("  (Install CUDA or use Google Colab for GPU acceleration)")

print(f"\nTraining on: {DEVICE}")

# ══════════════════════════════════════════
# DATA LOADING WITH TRANSFORMS
# ══════════════════════════════════════════

# WHAT ARE TRANSFORMS?
# Before feeding images to the network, we apply preprocessing.
# PyTorch's transforms API chains these cleanly.

# WHY NORMALIZE WITH (0.1307, 0.3081)?
# These are MNIST's precomputed dataset-wide mean and standard deviation.
# After: Normalize((mean,), (std,)) applies: x_norm = (x - mean) / std
# Result: pixel values centered around 0 with unit variance.
# WHY THIS HELPS:
# 1. The model sees inputs in a consistent range regardless of image brightness
# 2. Gradient magnitudes stay balanced across all input dimensions
# 3. Speeds up convergence significantly
# These exact values (0.1307 mean, 0.3081 std) are computed over ALL
# 60,000 training images and are universally used for MNIST.

MNIST_MEAN = (0.1307,)  # Dataset mean (computed over 60k training images)
MNIST_STD  = (0.3081,)  # Dataset std  (computed over 60k training images)

train_transform = transforms.Compose([
    transforms.ToTensor(),          # PIL image [0,255] → tensor [0.0,1.0]
    transforms.Normalize(MNIST_MEAN, MNIST_STD),  # standardize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

print("\nLoading MNIST dataset...")

# Download with retry logic
MAX_RETRIES = 3
for attempt in range(1, MAX_RETRIES + 1):
    try:
        train_dataset = torchvision.datasets.MNIST(
            root='data/', train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(
            root='data/', train=False, download=True, transform=test_transform)
        print(f"✓ MNIST loaded (attempt {attempt})")
        break
    except Exception as e:
        print(f"  Attempt {attempt} failed: {e}")
        if attempt == MAX_RETRIES:
            print("❌ Cannot download MNIST. Check internet connection.")
            sys.exit(1)
        import time; time.sleep(5 * attempt)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,           # Shuffle each epoch for better generalization
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == 'cuda'),  # Faster GPU transfer if available
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,  # Can use larger batch for eval (no grad)
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == 'cuda'),
)

print(f"  Training samples:   {len(train_dataset):,}")
print(f"  Test samples:       {len(test_dataset):,}")
print(f"  Training batches:   {len(train_loader)}")
print(f"  Test batches:       {len(test_loader)}")

# ══════════════════════════════════════════
# CNN ARCHITECTURE
# ══════════════════════════════════════════

# WHAT IS A CNN (Convolutional Neural Network)?
# ─────────────────────────────────────────────
# In a dense (fully-connected) layer, every output is connected to
# EVERY input. For a 28×28 image, that's 784 connections per neuron.
# Dense layers IGNORE the spatial structure of images — pixel (0,0) is
# treated just like pixel (27,27), with no knowledge they're neighbors.
#
# A CNN instead uses FILTERS (small weight matrices, e.g., 3×3).
# Each filter SLIDES across the image, computing a dot product at each
# position. This is "convolution."
#
# WHY THIS IS BETTER FOR IMAGES:
# 1. PARAMETER SHARING: One 3×3 filter (9 weights) detects the same
#    pattern everywhere in the image. A dense layer would need different
#    weights for the pattern at each position — massively inefficient.
#
# 2. SPATIAL INVARIANCE: A filter that detects vertical edges works
#    whether the edge is in the top-left or bottom-right.
#
# 3. HIERARCHICAL FEATURES:
#    Layer 1 filters detect edges and corners
#    Layer 2 filters detect curves and shapes (combinations of edges)
#    Deeper layers detect complex digit-specific features
#
# ANALOGY: A CNN is like a flashlight scanning the image in a grid.
# The flashlight has a fixed pattern it's looking for (the filter).
# It lights up brightly wherever it finds that pattern.

class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    Input: (batch, 1, 28, 28) — grayscale images
    → Conv Block 1: (batch, 32, 28, 28)
    → Conv Block 2 + Pool: (batch, 64, 14, 14)
    → Flatten: (batch, 64*14*14) = (batch, 12544)
    → FC Layer: (batch, 256)
    → Output: (batch, 10)
    """

    def __init__(self):
        super(CNN, self).__init__()

        # ── CONVOLUTIONAL BLOCK 1 ──
        # Conv2d(in_channels, out_channels, kernel_size, padding):
        #   in_channels=1: our images have 1 grayscale channel
        #   out_channels=32: we learn 32 different filters
        #   kernel_size=3: each filter is 3×3 pixels
        #   padding=1: add 1 row/col of zeros around the border
        #              keeps output size same as input: 28×28 → 28×28
        #   After Conv1: (batch, 32, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels=NUM_INPUT_CHANNELS,
            out_channels=CONV1_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING
        )

        # BatchNorm2d: normalizes activations WITHIN each mini-batch
        # FORMULA: x̂ = (x - μ_batch) / sqrt(σ²_batch + ε)
        # After BN: each channel has mean≈0, std≈1 within the batch
        # BENEFITS:
        # 1. Reduces "internal covariate shift" — the distribution
        #    of each layer's inputs shifts as weights change. BN
        #    stabilizes this, allowing higher learning rates.
        # 2. Acts as regularization (slight noise from batch statistics)
        # 3. Reduces sensitivity to weight initialization
        self.bn1 = nn.BatchNorm2d(CONV1_FILTERS)

        # ── CONVOLUTIONAL BLOCK 2 ──
        # in_channels=32: accepts output of conv1 (32 feature maps)
        # out_channels=64: learns 64 more complex patterns
        # After Conv2: (batch, 64, 28, 28)
        self.conv2 = nn.Conv2d(
            in_channels=CONV1_FILTERS,
            out_channels=CONV2_FILTERS,
            kernel_size=CONV_KERNEL_SIZE,
            padding=CONV_PADDING
        )
        self.bn2 = nn.BatchNorm2d(CONV2_FILTERS)

        # MaxPool2d(kernel_size=2, stride=2): takes max value in each 2×2 window
        # Effect: halves spatial dimensions — (batch, 64, 28, 28) → (batch, 64, 14, 14)
        # WHY MAX POOLING?
        # 1. Reduces spatial size → fewer parameters in FC layers
        # 2. Creates translation invariance (small shifts of pattern still detected)
        # 3. Selects the STRONGEST activation in each region
        self.pool = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)

        # Dropout2d: zeros ENTIRE feature maps randomly during training
        # During eval: all feature maps active (no dropout)
        # Prevents overfitting by forcing the network not to rely on
        # any single feature map
        self.dropout2d = nn.Dropout2d(p=0.25)

        # ── FULLY-CONNECTED CLASSIFIER ──
        # After conv blocks + pooling:
        # Input was 28×28, after pool: 28/2 = 14
        # 64 feature maps, each 14×14 pixels → 64 × 14 × 14 = 12,544 values
        FLATTENED_SIZE = CONV2_FILTERS * (IMAGE_SIZE // POOL_SIZE) * (IMAGE_SIZE // POOL_SIZE)
        # FLATTENED_SIZE = 64 * 14 * 14 = 12,544

        self.fc1 = nn.Linear(FLATTENED_SIZE, FC1_SIZE)
        # fc1: (batch, 12544) → (batch, 256)

        # Regular Dropout: randomly zeros INDIVIDUAL neurons (not whole feature maps)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

        self.fc2 = nn.Linear(FC1_SIZE, NUM_CLASSES)
        # fc2: (batch, 256) → (batch, 10)

        # WHY NO SOFTMAX AT OUTPUT?
        # nn.CrossEntropyLoss combines LogSoftmax + NLLLoss internally.
        # PyTorch's implementation is numerically more stable this way.
        # We only add softmax when we want probabilities (at inference time).

        # ── ACTIVATION FUNCTION ──
        self.relu = nn.ReLU()

        # ── PRINT ARCHITECTURE SUMMARY ──
        print("\n" + "=" * 60)
        print("CNN ARCHITECTURE SUMMARY")
        print("=" * 60)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Input:    (batch, {NUM_INPUT_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})")
        print(f"  Conv1:    (batch, {CONV1_FILTERS}, {IMAGE_SIZE}, {IMAGE_SIZE})  [+BN, +ReLU]")
        print(f"  Conv2:    (batch, {CONV2_FILTERS}, {IMAGE_SIZE}, {IMAGE_SIZE})  [+BN, +ReLU]")
        print(f"  MaxPool:  (batch, {CONV2_FILTERS}, {IMAGE_SIZE//POOL_SIZE}, {IMAGE_SIZE//POOL_SIZE})")
        print(f"  Flatten:  (batch, {FLATTENED_SIZE})")
        print(f"  FC1:      (batch, {FC1_SIZE})       [+ReLU, +Dropout{DROPOUT_RATE}]")
        print(f"  Output:   (batch, {NUM_CLASSES})")
        print(f"  Total Trainable Parameters: {total_params:,}")
        print("=" * 60)

    def forward(self, x):
        """
        WHAT: Forward pass through the CNN.
        INPUT:  x — tensor of shape (batch, 1, 28, 28)
        OUTPUT: logits — tensor of shape (batch, 10)
                These are raw scores (not probabilities — no softmax here)
        """
        # ── CONV BLOCK 1 ──
        x = self.conv1(x)
        # x shape: (batch, 1, 28, 28) → (batch, 32, 28, 28)
        x = self.bn1(x)
        # x shape: (batch, 32, 28, 28) — normalized
        x = self.relu(x)
        # x shape: (batch, 32, 28, 28) — negative values zeroed

        # ── CONV BLOCK 2 ──
        x = self.conv2(x)
        # x shape: (batch, 32, 28, 28) → (batch, 64, 28, 28)
        x = self.bn2(x)
        # x shape: (batch, 64, 28, 28) — normalized
        x = self.relu(x)
        # x shape: (batch, 64, 28, 28) — negative values zeroed
        x = self.pool(x)
        # x shape: (batch, 64, 28, 28) → (batch, 64, 14, 14)  [halved!]
        x = self.dropout2d(x)
        # x shape: (batch, 64, 14, 14) — some feature maps zeroed

        # ── FLATTEN ──
        x = x.view(x.size(0), -1)
        # x.size(0) = batch size (e.g., 64)
        # -1 = infer remaining dimension automatically = 64*14*14 = 12,544
        # x shape: (batch, 64, 14, 14) → (batch, 12544)

        # ── FULLY-CONNECTED CLASSIFIER ──
        x = self.fc1(x)
        # x shape: (batch, 12544) → (batch, 256)
        x = self.relu(x)
        # x shape: (batch, 256)
        x = self.dropout(x)
        # x shape: (batch, 256) — some neurons zeroed during training

        x = self.fc2(x)
        # x shape: (batch, 256) → (batch, 10)

        return x
        # Return shape: (batch, 10) — raw logits, one per class


# ══════════════════════════════════════════
# TRAINING AND VALIDATION FUNCTIONS
# ══════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    """
    WHAT: Runs one complete pass through the training data (one epoch).
    INPUT:  model     — CNN instance
            loader    — DataLoader for training set
            optimizer — Adam optimizer
            criterion — CrossEntropyLoss function
            device    — cuda, mps, or cpu
    OUTPUT: (avg_loss, accuracy) for this epoch

    ADAM OPTIMIZER:
    Unlike plain SGD (W = W - lr * gradient), Adam adapts the learning rate
    for each parameter individually. It tracks:
    - m: exponential moving average of gradients (first moment)
    - v: exponential moving average of squared gradients (second moment)
    Then: W = W - lr * m / (sqrt(v) + ε)
    Effect: Parameters with consistently large gradients get smaller updates.
            Parameters with small/noisy gradients get relatively larger updates.
    This makes training faster and more robust than plain SGD.
    """
    model.train()  # Set to training mode (enables dropout, batch norm training behavior)
    total_loss = 0.0
    correct = 0
    total = 0

    if TQDM_AVAILABLE:
        loader_iter = tqdm(loader, desc="  Training", leave=False, unit="batch")
    else:
        loader_iter = loader

    for images, labels in loader_iter:
        # Move tensors to the correct device (GPU if available)
        images = images.to(device)  # shape: (batch, 1, 28, 28)
        labels = labels.to(device)  # shape: (batch,) — integer labels

        # ── GRADIENT RESET ──
        # CRITICAL: PyTorch ACCUMULATES gradients by default.
        # If we don't zero them before each step, they pile up from
        # previous batches. We always zero before the forward pass.
        optimizer.zero_grad()

        # ── FORWARD PASS ──
        logits = model(images)
        # logits shape: (batch, 10) — raw scores

        # ── COMPUTE LOSS ──
        # CrossEntropyLoss:
        # 1. Internally applies softmax to convert logits → probabilities
        # 2. Computes -log(probability of correct class) for each sample
        # 3. Returns the mean over the batch
        # This is equivalent to: NLLLoss(LogSoftmax(logits), labels)
        loss = criterion(logits, labels)

        # ── BACKWARD PASS ──
        # Computes gradients of loss w.r.t. ALL parameters using backprop
        # PyTorch's autograd handles the chain rule automatically
        loss.backward()

        # ── GRADIENT CLIPPING (optional but good practice) ──
        # Prevents gradient explosion in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── WEIGHT UPDATE ──
        # Uses the gradients computed in loss.backward()
        # to update all model parameters
        optimizer.step()

        # ── TRACK METRICS ──
        total_loss += loss.item() * images.size(0)  # Accumulate sum (not mean)
        _, predicted = logits.max(1)  # argmax along class dimension
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    WHAT: Evaluates the model on validation/test data without updating weights.
    INPUT:  model     — CNN instance
            loader    — DataLoader for val/test set
            criterion — CrossEntropyLoss function
            device    — device to run on
    OUTPUT: (avg_loss, accuracy)

    torch.no_grad():
    During evaluation, we don't need gradients (we're not updating weights).
    Disabling gradient tracking:
    1. Saves memory (no gradient tensors stored)
    2. Speeds up computation (~2x faster for inference)
    3. Prevents accidental weight updates

    ALWAYS wrap evaluation in torch.no_grad() — this is standard practice.
    """
    model.eval()  # Set to evaluation mode (disables dropout, uses running BN stats)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for speed + memory
        for images, labels in loader:
            images = images.to(device)  # (batch, 1, 28, 28)
            labels = labels.to(device)  # (batch,)

            logits = model(images)      # (batch, 10)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════

print(f"\n{'='*60}")
print("STARTING CNN TRAINING")
print(f"{'='*60}")

# Instantiate model and move to device
model = CNN().to(DEVICE)

# ── OPTIMIZER: Adam ──
# Adam works well with learning_rate=0.001 (default) for most tasks
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── LOSS FUNCTION: CrossEntropyLoss ──
criterion = nn.CrossEntropyLoss()

# ── LEARNING RATE SCHEDULER ──
# StepLR multiplies LR by gamma every step_size epochs
# After epoch 5:  lr = 0.001 × 0.5 = 0.0005
# After epoch 10: lr = 0.001 × 0.5² = 0.00025
# WHY: As training progresses, we want smaller steps to fine-tune
#      rather than overshooting the optimal weights
scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# ── TRAINING HISTORY ──
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [],  'val_acc': [],
    'lr': []
}

best_val_acc = 0.0
best_model_path = 'outputs/models/pytorch_cnn_best.pth'

print(f"\nHyperparameters:")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Epochs:         {EPOCHS}")
print(f"  Learning rate:  {LEARNING_RATE} (×{LR_GAMMA} every {LR_STEP_SIZE} epochs)")
print(f"  Dropout rate:   {DROPOUT_RATE}")
print()

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ONE EPOCH ──
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    # ── VALIDATE ──
    val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)

    # ── SCHEDULER STEP ──
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    # ── STORE HISTORY ──
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    # ── MODEL CHECKPOINTING ──
    # Save the best model weights seen during training.
    # WHY: The last epoch's weights might not be the best — if the model
    # starts overfitting near the end, we want the weights from when
    # validation accuracy was highest.
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, best_model_path)

    # ── PRINT EPOCH SUMMARY ──
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | "
          f"LR: {current_lr:.5f}"
          + (" ✓ BEST" if val_acc >= best_val_acc else ""))

training_time = time.time() - start_time
print(f"\nTraining complete in {training_time:.1f}s ({training_time/60:.1f} min)")
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

# Load best model for evaluation
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")

# ══════════════════════════════════════════
# PLOT TRAINING CURVES
# ══════════════════════════════════════════

epoch_range = range(1, EPOCHS + 1)

# ── PLOT 9: LOSS CURVES ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PyTorch CNN — Training Curves', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(epoch_range, history['train_loss'], label='Train Loss',
        color='royalblue', linewidth=2)
ax.plot(epoch_range, history['val_loss'],   label='Val Loss',
        color='darkorange', linewidth=2, linestyle='--')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax.set_title('Loss Curves', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Add LR change markers
for step_epoch in range(LR_STEP_SIZE, EPOCHS + 1, LR_STEP_SIZE):
    ax.axvline(x=step_epoch, color='gray', linestyle=':', alpha=0.5)
ax.text(0.98, 0.98, 'Gray lines = LR reduction steps',
        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

ax = axes[1]
ax.plot(epoch_range, [a*100 for a in history['train_acc']], label='Train Acc',
        color='royalblue', linewidth=2)
ax.plot(epoch_range, [a*100 for a in history['val_acc']], label='Val Acc',
        color='darkorange', linewidth=2, linestyle='--')
ax.axhline(y=99.0, color='green', linestyle='-.', linewidth=1.5, label='Target: 99%')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Accuracy Curves', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(90, 100.5)

plt.tight_layout()
PLOT9_PATH = 'outputs/plots/09_pytorch_training_loss.png'
plt.savefig(PLOT9_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: {PLOT9_PATH}")

# ══════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)

# Get all predictions
all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)  # Convert to probabilities for display
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)  # (10000, 10)

test_acc = np.mean(all_preds == all_labels)
print(f"\n  Test Accuracy: {test_acc * 100:.2f}%")

print("\nPer-class Performance:")
print(classification_report(all_labels, all_preds,
                             target_names=[f'Digit {i}' for i in range(10)]))

# ── PLOT 11: CONFUSION MATRIX ──
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(range(10)),
            yticklabels=list(range(10)),
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'PyTorch CNN — Confusion Matrix\n(Test Accuracy: {test_acc*100:.2f}%)',
             fontsize=13, fontweight='bold')
for i in range(10):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='gold', linewidth=2))

plt.tight_layout()
PLOT11_PATH = 'outputs/plots/11_pytorch_confusion_matrix.png'
plt.savefig(PLOT11_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT11_PATH}")

# ══════════════════════════════════════════
# FEATURE MAP VISUALIZATION
# ══════════════════════════════════════════

# WHAT ARE FEATURE MAPS?
# After passing an image through a conv layer, we get a 3D tensor:
# (num_filters, height, width). Each 2D "slice" is a feature map —
# it shows WHERE in the image a particular filter's pattern was activated.
# High values (bright spots) = "I found my pattern here!"
#
# For conv1 with 32 filters, we get 32 feature maps.
# Each map shows a different low-level feature: edges, gradients, corners.
# This is literally what the CNN "sees" at each layer.

print("\nGenerating feature map visualization...")

# Get one sample from test set (the first image)
sample_image, sample_label = test_dataset[0]
sample_image_tensor = sample_image.unsqueeze(0).to(DEVICE)
# sample_image_tensor shape: (1, 1, 28, 28) — add batch dimension

model.eval()
with torch.no_grad():
    # Get output of just the first conv layer + BN + ReLU
    # We need to manually stop the forward pass after conv1
    conv1_output = model.relu(model.bn1(model.conv1(sample_image_tensor)))
    # conv1_output shape: (1, 32, 28, 28)

# Convert to numpy for visualization
feature_maps = conv1_output.squeeze(0).cpu().numpy()
# feature_maps shape: (32, 28, 28)

# Get original image for comparison
original_image = sample_image.squeeze(0).numpy()
# original_image shape: (28, 28)

# Plot 4×8 grid of feature maps
fig = plt.figure(figsize=(16, 9))
fig.suptitle(f'Feature Maps After Conv Layer 1\n(Input: Digit "{sample_label}")',
             fontsize=13, fontweight='bold')

# Show original image first
ax_orig = fig.add_subplot(5, 8, 1)
ax_orig.imshow(original_image, cmap='gray')
ax_orig.set_title('Original', fontsize=8)
ax_orig.axis('off')

# Show all 32 feature maps
for i in range(CONV1_FILTERS):
    ax = fig.add_subplot(5, 8, i + 2)
    ax.imshow(feature_maps[i], cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i+1}', fontsize=7)
    ax.axis('off')

plt.tight_layout()
PLOT12_PATH = 'outputs/plots/12_pytorch_feature_maps.png'
plt.savefig(PLOT12_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT12_PATH}")
print("  (Each colored map shows which image regions activated that filter)")
print("  (Bright = strong activation = 'I found this pattern here!')")

# ══════════════════════════════════════════
# LEARNED FILTER VISUALIZATION
# ══════════════════════════════════════════

# The 32 filters in conv1 are the actual LEARNED patterns.
# Each filter is a 3×3 grid of weights.
# After training, these weights encode low-level image features:
# - Some filters detect horizontal edges
# - Some detect vertical edges
# - Some detect diagonal gradients
# - Some detect specific curves or corners
# This is what the network learned to look for in digit images!

print("Generating learned filter visualization...")

# Extract weights from conv1
conv1_weights = model.conv1.weight.data.cpu().numpy()
# conv1_weights shape: (32, 1, 3, 3)
# 32 filters, each with 1 input channel, 3×3 kernel

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle('Learned Conv Layer 1 Filters (3×3 Kernels)\n'
             'These are the patterns the CNN learned to detect',
             fontsize=12, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    if i < CONV1_FILTERS:
        # Squeeze out channel dimension: (1, 3, 3) → (3, 3)
        kernel = conv1_weights[i, 0, :, :]
        # Normalize to [0,1] for display
        kernel_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        ax.imshow(kernel_norm, cmap='RdBu_r', interpolation='nearest',
                  vmin=0, vmax=1)
        ax.set_title(f'F{i+1}', fontsize=7)
    ax.axis('off')

plt.tight_layout()
PLOT13_PATH = 'outputs/plots/13_pytorch_conv_filters.png'
plt.savefig(PLOT13_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT13_PATH}")

# ══════════════════════════════════════════
# SAMPLE PREDICTIONS
# ══════════════════════════════════════════

print("Generating prediction samples...")

NUM_DISPLAY  = 20
DISPLAY_ROWS = 4
DISPLAY_COLS = 5

# Get actual test images (un-normalized for display)
test_images_raw = torchvision.datasets.MNIST(
    root='data/', train=False, download=False).data.numpy()  # (10000, 28, 28)

correct_idx = np.where(all_preds == all_labels)[0][:10]
wrong_idx   = np.where(all_preds != all_labels)[0][:10]
display_idx = np.concatenate([correct_idx, wrong_idx])

fig, axes = plt.subplots(DISPLAY_ROWS, DISPLAY_COLS, figsize=(12, 10))
fig.suptitle('PyTorch CNN Predictions: Green=Correct, Red=Wrong',
             fontsize=13, fontweight='bold')

for plot_i, test_i in enumerate(display_idx[:NUM_DISPLAY]):
    ax = axes.flatten()[plot_i]
    is_correct = (all_preds[test_i] == all_labels[test_i])
    confidence = all_probs[test_i, all_preds[test_i]] * 100

    ax.imshow(test_images_raw[test_i], cmap='gray', interpolation='nearest')
    color = 'green' if is_correct else 'red'
    ax.set_title(f'True: {all_labels[test_i]}\n'
                 f'Pred: {all_preds[test_i]} ({confidence:.1f}%)',
                 fontsize=7.5, color=color, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
PLOT14_PATH = 'outputs/plots/14_pytorch_predictions.png'
plt.savefig(PLOT14_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT14_PATH}")

# ══════════════════════════════════════════
# SAVE FINAL MODEL AND RESULTS
# ══════════════════════════════════════════

# Save final model
FINAL_MODEL_PATH = 'outputs/models/pytorch_cnn.pth'
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"\n✓ Final model saved to: {FINAL_MODEL_PATH}")

# Save results for comparison file
total_cnn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
RESULTS_PATH = 'outputs/results/pytorch_results.txt'
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    f.write(f"PyTorch CNN Results\n")
    f.write(f"{'='*40}\n")
    f.write(f"Architecture: Conv(32)+BN → Conv(64)+BN+Pool → FC(256) → FC(10)\n")
    f.write(f"Total Parameters: {total_cnn_params:,}\n")
    f.write(f"Test Accuracy: {test_acc * 100:.4f}%\n")
    f.write(f"Training Time: {training_time:.1f} seconds\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Best Val Accuracy: {best_val_acc * 100:.4f}%\n")
print(f"✓ Results saved to: {RESULTS_PATH}")

# ══════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("PYTORCH CNN TRAINING COMPLETE!")
print("=" * 60)
print(f"\n  Test Accuracy:   {test_acc * 100:.2f}%")
print(f"  Parameters:      {total_cnn_params:,}")
print(f"  Training time:   {training_time:.1f}s ({training_time/60:.1f} min)")
print(f"  Device:          {DEVICE}")
print()
print("  Saved plots:")
for path in [PLOT9_PATH, PLOT11_PATH, PLOT12_PATH, PLOT13_PATH, PLOT14_PATH]:
    print(f"    {path}")
print()
print("Run NEXT: python 04_compare_visualize.py")
print("=" * 60)
