# ══════════════════════════════════════════
# FILE: 01_data_explore.py
# WHAT THIS FILE DOES: Downloads MNIST, creates 4 visualizations to
#   understand the data, and saves preprocessed numpy arrays for
#   subsequent training files.
# DEPENDS ON: Nothing (downloads data automatically)
# RUN WITH: python 01_data_explore.py
# EXPECTED OUTPUT: "Train: 60000, Test: 10000" + 4 saved plots + 6 .npy files
# ══════════════════════════════════════════

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── CREATE ALL OUTPUT DIRECTORIES ──
# exist_ok=True means: don't crash if the folder already exists
# This satisfies RULE 6: auto-create missing directories
for directory in ['outputs/plots', 'outputs/models', 'outputs/results', 'data']:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Directory ready: {directory}/")

print()

# ── DOWNLOAD MNIST WITH RETRY LOGIC ──
# WHAT IS MNIST?
# The Modified National Institute of Standards and Technology dataset.
# 70,000 handwritten digit images, collected from Census Bureau employees
# and high school students in the 1990s. It became THE standard benchmark
# for computer vision from 1998 (LeNet) through ~2012 (AlexNet on ImageNet).
# Every ML library includes it. It's the "Hello World" of deep learning.
#
# MNIST specs:
# - 60,000 training images + 10,000 test images
# - Each image: 28×28 pixels, grayscale (1 channel)
# - Pixel values: 0 (black) to 255 (white)
# - Labels: integer 0-9 representing the digit

MAX_DOWNLOAD_RETRIES = 3  # Number of times to retry if download fails

print("Downloading MNIST dataset...")
print("(This takes ~1 minute on first run, instant on subsequent runs)")
print()

for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
    try:
        import torchvision
        import torchvision.transforms as transforms
        import torch

        # torchvision handles downloading, caching, and loading MNIST
        # root='data/' → saves to ./data/MNIST/
        # download=True → download if not already present
        # train=True/False → training vs test split
        mnist_train = torchvision.datasets.MNIST(
            root='data/',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        mnist_test = torchvision.datasets.MNIST(
            root='data/',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        print(f"✓ MNIST downloaded successfully on attempt {attempt}")
        break  # Success — exit retry loop

    except Exception as e:
        print(f"  Attempt {attempt}/{MAX_DOWNLOAD_RETRIES} failed: {e}")
        if attempt < MAX_DOWNLOAD_RETRIES:
            wait_time = 5 * attempt  # Wait longer each retry
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print("\n❌ All download attempts failed!")
            print("Solutions to try:")
            print("  1. Check your internet connection")
            print("  2. Try running: pip install --upgrade torchvision")
            print("  3. Use a VPN if MNIST servers are blocked in your region")
            sys.exit(1)

# ══════════════════════════════════════════
# CONVERT TO NUMPY ARRAYS
# ══════════════════════════════════════════

print("\nConverting to numpy arrays...")

# torchvision gives us tensors — convert to numpy for exploration
# .data.numpy() extracts the raw pixel values
X_train_raw = mnist_train.data.numpy()   # shape: (60000, 28, 28)
y_train_raw = mnist_train.targets.numpy() # shape: (60000,)
X_test_raw  = mnist_test.data.numpy()    # shape: (10000, 28, 28)
y_test_raw  = mnist_test.targets.numpy() # shape: (10000,)

# ══════════════════════════════════════════
# PRINT DATASET STATISTICS
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("MNIST DATASET STATISTICS")
print("=" * 60)
print(f"Training set shape:   {X_train_raw.shape}")
print(f"  → {X_train_raw.shape[0]:,} images, each {X_train_raw.shape[1]}×{X_train_raw.shape[2]} pixels")
print(f"Training labels shape: {y_train_raw.shape}")
print(f"Test set shape:        {X_test_raw.shape}")
print(f"Test labels shape:     {y_test_raw.shape}")
print()
print(f"Pixel value range: {X_train_raw.min()} to {X_train_raw.max()}")
print(f"Data type: {X_train_raw.dtype}")

# Memory usage
train_bytes = X_train_raw.nbytes
print(f"Training data memory: {train_bytes / 1024 / 1024:.1f} MB")
print()

# Class distribution
print("Class Distribution (training set):")
NUM_CLASSES = 10
for digit in range(NUM_CLASSES):
    count = np.sum(y_train_raw == digit)
    bar = "█" * (count // 500)  # Each block = 500 samples
    print(f"  Digit {digit}: {count:,}  {bar}")

# Compare to fraud dataset
print()
print("CONTRAST WITH PROJECT 1 (Fraud Detection):")
print("  MNIST:   BALANCED — ~6,000 samples per class")
print("  Fraud:   IMBALANCED — ~0.17% fraud, ~99.83% legitimate")
print("  Impact:  With MNIST, accuracy is a fair metric.")
print("           With fraud, accuracy is misleading (predict all legit = 99.83%!)")

# ══════════════════════════════════════════
# PLOT 1: SAMPLE IMAGES GRID (5 per class × 10 classes)
# ══════════════════════════════════════════

print("\nGenerating Plot 1: Sample images grid...")

# Figure size: (column_width × num_cols, row_height × num_rows)
SAMPLES_PER_CLASS = 5
fig, axes = plt.subplots(
    NUM_CLASSES, SAMPLES_PER_CLASS,
    figsize=(SAMPLES_PER_CLASS * 1.5, NUM_CLASSES * 1.5)
)
fig.suptitle('MNIST Sample Images\n(5 samples per digit class)',
             fontsize=14, fontweight='bold', y=0.98)

for digit in range(NUM_CLASSES):
    # Find indices of this digit in training set
    digit_indices = np.where(y_train_raw == digit)[0]
    # Select first SAMPLES_PER_CLASS examples
    sample_indices = digit_indices[:SAMPLES_PER_CLASS]

    for col, idx in enumerate(sample_indices):
        ax = axes[digit][col]
        # Display the 28×28 image in grayscale
        ax.imshow(X_train_raw[idx], cmap='gray', interpolation='nearest')
        ax.set_title(f'Label: {digit}', fontsize=7, pad=2)
        ax.axis('off')  # Hide axes ticks and labels

plt.tight_layout()
PLOT1_PATH = 'outputs/plots/01_mnist_samples.png'
plt.savefig(PLOT1_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT1_PATH}")

# ══════════════════════════════════════════
# PLOT 2: CLASS DISTRIBUTION BAR CHART
# ══════════════════════════════════════════

print("Generating Plot 2: Class distribution...")

class_counts = [np.sum(y_train_raw == d) for d in range(NUM_CLASSES)]
test_class_counts = [np.sum(y_test_raw == d) for d in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MNIST Class Distribution', fontsize=14, fontweight='bold')

# Training set distribution
ax = axes[0]
bars = ax.bar(range(NUM_CLASSES), class_counts,
              color=plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES)),
              edgecolor='black', linewidth=0.5)
ax.set_xlabel('Digit Class', fontsize=11)
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_title('Training Set (60,000 images)', fontsize=12)
ax.set_xticks(range(NUM_CLASSES))
ax.set_ylim(0, max(class_counts) * 1.15)
# Add count labels on top of each bar
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{count:,}', ha='center', va='bottom', fontsize=8)

# Draw the mean line
mean_count = np.mean(class_counts)
ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {mean_count:.0f}')
ax.legend()

# Test set distribution
ax = axes[1]
bars = ax.bar(range(NUM_CLASSES), test_class_counts,
              color=plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES)),
              edgecolor='black', linewidth=0.5)
ax.set_xlabel('Digit Class', fontsize=11)
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_title('Test Set (10,000 images)', fontsize=12)
ax.set_xticks(range(NUM_CLASSES))
ax.set_ylim(0, max(test_class_counts) * 1.15)
for bar, count in zip(bars, test_class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f'{count:,}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
PLOT2_PATH = 'outputs/plots/02_class_distribution.png'
plt.savefig(PLOT2_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT2_PATH}")

# ══════════════════════════════════════════
# PLOT 3: PIXEL INTENSITY DISTRIBUTION
# ══════════════════════════════════════════

print("Generating Plot 3: Pixel intensity distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('MNIST Pixel Intensity Analysis', fontsize=14, fontweight='bold')

# Flatten all training images to a 1D array for histogram
all_pixels = X_train_raw.flatten()  # shape: (60000*28*28,) = (47,040,000,)

# Histogram of all pixel values
ax = axes[0]
ax.hist(all_pixels, bins=50, color='steelblue', edgecolor='navy', linewidth=0.3,
        range=(0, 255))
ax.set_xlabel('Pixel Value (0=black, 255=white)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of All Pixel Values\n(training set)', fontsize=12)
ax.axvline(x=all_pixels.mean(), color='red', linestyle='--',
           label=f'Mean: {all_pixels.mean():.1f}')
ax.axvline(x=np.median(all_pixels), color='orange', linestyle='-.',
           label=f'Median: {np.median(all_pixels):.0f}')
ax.legend()
# NOTE: The massive spike at 0 is because most pixels in handwritten digits
# are background (black = 0). The actual digit strokes are white.

# Per-class mean pixel value
ax = axes[1]
class_mean_pixels = [X_train_raw[y_train_raw == d].mean() for d in range(NUM_CLASSES)]
ax.bar(range(NUM_CLASSES), class_mean_pixels,
       color=plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES)),
       edgecolor='black', linewidth=0.5)
ax.set_xlabel('Digit Class', fontsize=11)
ax.set_ylabel('Mean Pixel Intensity', fontsize=11)
ax.set_title('Average Pixel Intensity per Digit', fontsize=12)
ax.set_xticks(range(NUM_CLASSES))

plt.tight_layout()
PLOT3_PATH = 'outputs/plots/03_pixel_distribution.png'
plt.savefig(PLOT3_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT3_PATH}")

# ══════════════════════════════════════════
# PLOT 4: MEAN IMAGE PER CLASS ("average digit")
# ══════════════════════════════════════════

print("Generating Plot 4: Mean digit images...")

# WHAT THIS SHOWS:
# For each digit class, we average all images of that class pixel-by-pixel.
# The result is the "average face" of each digit. You can see:
# - 0 and 8 are ring-shaped, with more dark space in the center
# - 1 is narrow and vertical
# - 4 shows a branching structure
# These average images reveal what patterns are common to each digit class.

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Mean Image per Digit Class\n(averaged over all training samples)',
             fontsize=13, fontweight='bold')

axes_flat = axes.flatten()  # Easier to iterate over

for digit in range(NUM_CLASSES):
    # Select all training images for this digit
    digit_images = X_train_raw[y_train_raw == digit]
    # digit_images shape: (~6000, 28, 28)

    # Compute pixel-wise mean
    mean_image = digit_images.mean(axis=0)
    # mean_image shape: (28, 28) — the "average" digit

    ax = axes_flat[digit]
    im = ax.imshow(mean_image, cmap='hot', interpolation='bilinear')
    ax.set_title(f'Digit {digit}\n(n={len(digit_images):,})', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
PLOT4_PATH = 'outputs/plots/04_mean_digits.png'
plt.savefig(PLOT4_PATH, dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {PLOT4_PATH}")

# ══════════════════════════════════════════
# PREPROCESS AND SAVE DATA ARRAYS
# ══════════════════════════════════════════

print("\nPreprocessing data for training...")

# ── NORMALIZE PIXEL VALUES FROM [0, 255] TO [0, 1] ──
# WHY NORMALIZE?
# 1. Gradient stability: If pixel values are 0-255, weights would need to
#    be very tiny to produce reasonable outputs. With 0-1, weights stay
#    in a reasonable range.
# 2. Comparable scale: All inputs are on the same scale, so gradients
#    are balanced across features.
# 3. Activation compatibility: sigmoid/softmax work best with inputs ~0-1

# Convert to float32 (saves memory vs float64, sufficient precision)
X_train_normalized = X_train_raw.astype(np.float32) / 255.0
X_test_normalized  = X_test_raw.astype(np.float32)  / 255.0
# Shapes: (60000, 28, 28) and (10000, 28, 28), values now in [0, 1]

# ── FLATTEN 28×28 IMAGES TO 784-DIM VECTORS ──
# Our dense neural network expects 1D input per sample, not 2D images.
# (We'll keep the 2D structure for the CNN in file 03)
NUM_TRAIN = X_train_normalized.shape[0]  # 60000
NUM_TEST  = X_test_normalized.shape[0]   # 10000
IMAGE_FLAT_SIZE = 28 * 28                # 784

X_train_flat = X_train_normalized.reshape(NUM_TRAIN, IMAGE_FLAT_SIZE)
X_test_flat  = X_test_normalized.reshape(NUM_TEST, IMAGE_FLAT_SIZE)
# X_train_flat shape: (60000, 784)
# X_test_flat shape:  (10000, 784)

print(f"  Flattened training shape: {X_train_flat.shape}")
print(f"  Flattened test shape:     {X_test_flat.shape}")

# ── ONE-HOT ENCODE LABELS ──
# WHY ONE-HOT ENCODING?
# Our network outputs 10 probabilities (one per digit).
# We need our labels in the same format to compute loss and gradients.
#
# Example: digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#          digit 0 → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#          digit 9 → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#
# np.eye(10) creates a 10×10 identity matrix:
#   [[1,0,0,...,0],   ← row 0 = one-hot for digit 0
#    [0,1,0,...,0],   ← row 1 = one-hot for digit 1
#    ...
#    [0,0,0,...,1]]   ← row 9 = one-hot for digit 9
# Then np.eye(10)[y_train_raw] selects the right row for each label.

y_train_onehot = np.eye(NUM_CLASSES)[y_train_raw]
y_test_onehot  = np.eye(NUM_CLASSES)[y_test_raw]
# y_train_onehot shape: (60000, 10)
# y_test_onehot shape:  (10000, 10)

print(f"\n  One-hot encoding example:")
print(f"  y_train_raw[0] = {y_train_raw[0]} (the digit five)")
print(f"  y_train_onehot[0] = {y_train_onehot[0].astype(int)}")
print(f"  (1 at index {y_train_raw[0]}, zeros elsewhere)")

# ── SAVE ALL ARRAYS ──
RESULTS_DIR = 'outputs/results'

np.save(f'{RESULTS_DIR}/X_train.npy', X_train_flat)
np.save(f'{RESULTS_DIR}/X_test.npy',  X_test_flat)
np.save(f'{RESULTS_DIR}/y_train.npy', y_train_raw)
np.save(f'{RESULTS_DIR}/y_test.npy',  y_test_raw)
np.save(f'{RESULTS_DIR}/y_train_onehot.npy', y_train_onehot)
np.save(f'{RESULTS_DIR}/y_test_onehot.npy',  y_test_onehot)

print(f"\n✓ Saved all arrays to {RESULTS_DIR}/:")
print(f"  X_train.npy         shape: {X_train_flat.shape}    dtype: {X_train_flat.dtype}")
print(f"  X_test.npy          shape: {X_test_flat.shape}     dtype: {X_test_flat.dtype}")
print(f"  y_train.npy         shape: {y_train_raw.shape}     dtype: {y_train_raw.dtype}")
print(f"  y_test.npy          shape: {y_test_raw.shape}      dtype: {y_test_raw.dtype}")
print(f"  y_train_onehot.npy  shape: {y_train_onehot.shape}  dtype: {y_train_onehot.dtype}")
print(f"  y_test_onehot.npy   shape: {y_test_onehot.shape}   dtype: {y_test_onehot.dtype}")

# ══════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════

print("\n" + "=" * 60)
print("5 KEY OBSERVATIONS FROM DATA EXPLORATION")
print("=" * 60)
print()
print("1. BALANCED DATASET: Each digit class has ~6,000 training samples.")
print(f"   Min: {min(class_counts):,} (digit {class_counts.index(min(class_counts))}),",
      f"Max: {max(class_counts):,} (digit {class_counts.index(max(class_counts))})")
print("   This means ACCURACY is a valid metric (unlike fraud detection!)")
print()
print("2. HEAVY BACKGROUND: ~80% of pixels are black (0).")
print(f"   Mean pixel value: {all_pixels.mean():.1f}/255 = {all_pixels.mean()/255:.1%}")
print("   The digit strokes occupy a small fraction of each image.")
print()
print("3. NORMALIZATION: Pixel values scaled from [0-255] → [0-1].")
print("   This prevents large gradient magnitudes and speeds training.")
print()
print("4. DIMENSIONALITY: Each image = 784 numbers after flattening.")
print("   For dense layers, spatial structure is DISCARDED (28×28 → 784).")
print("   CNNs (file 03) preserve the 2D structure — that's their advantage!")
print()
print("5. ONE-HOT ENCODING: Labels converted from integers to 10D vectors.")
print("   This is required for cross-entropy loss computation.")
print("   digit 3 → [0,0,0,1,0,0,0,0,0,0]")
print()
print("=" * 60)
print("Run NEXT: python 02_numpy_train.py")
print("=" * 60)
