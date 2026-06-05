# ══════════════════════════════════════════
# FILE: src/network.py
# WHAT THIS FILE DOES: Implements the complete NeuralNetwork class that
#   chains multiple DenseLayer objects and orchestrates the full training
#   loop including mini-batch gradient descent and early stopping.
# DEPENDS ON: src/layers.py, src/activations.py
# RUN WITH: python src/network.py    (runs a quick smoke test)
# EXPECTED OUTPUT: Architecture summary, a few training steps, no errors
# ══════════════════════════════════════════

import numpy as np
import os

# tqdm gives us the animated progress bar during training
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not installed. Progress bars won't show.")
    print("Install with: pip install tqdm")

from src.layers import DenseLayer


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network built entirely in NumPy.

    Supports:
    - Arbitrary depth (number of layers)
    - ReLU hidden layers + Softmax output layer
    - Mini-batch gradient descent
    - Early stopping
    - Weight save/load
    - Full training history tracking

    ANALOGY: Think of this class as the "manager" of the layers.
    Each DenseLayer knows how to do its own forward/backward math.
    NeuralNetwork's job is to orchestrate: pass data through all layers
    in order (forward), then pass gradients back in reverse (backward).
    """

    def __init__(self, layer_configs):
        """
        WHAT: Creates a neural network from a list of layer specifications.
        INPUT:  layer_configs — list of dicts, each describing one layer:
                [
                  {'in': 784, 'out': 128, 'act': 'relu'},
                  {'in': 128, 'out': 64,  'act': 'relu'},
                  {'in': 64,  'out': 10,  'act': 'softmax'}
                ]
        OUTPUT: A NeuralNetwork ready for training.

        NOTE: 'in' of each layer must match 'out' of previous layer!
        """
        self.layers = []

        print("\n" + "=" * 60)
        print("NEURAL NETWORK ARCHITECTURE")
        print("=" * 60)

        total_params = 0
        for i, config in enumerate(layer_configs):
            layer = DenseLayer(
                input_size=config['in'],
                output_size=config['out'],
                activation=config['act']
            )
            self.layers.append(layer)
            # Count parameters: W has (in × out) + b has (out)
            n_params = config['in'] * config['out'] + config['out']
            total_params += n_params

        print(f"  Total Layers:      {len(self.layers)}")
        print(f"  Total Parameters:  {total_params:,}")
        print("=" * 60)
        print(f"\nParameter breakdown:")
        for i, (layer, config) in enumerate(zip(self.layers, layer_configs)):
            n = config['in'] * config['out'] + config['out']
            print(f"  Layer {i+1}: W({config['in']}×{config['out']}) + "
                  f"b({config['out']}) = {n:,} params")
        print(f"  TOTAL: {total_params:,} trainable parameters\n")

        self.total_params = total_params

    def forward(self, X):
        """
        WHAT: Passes input through all layers sequentially (forward pass).
        INPUT:  X — numpy array of shape (batch_size, input_size)
                    e.g. (64, 784) for a batch of 64 flattened images
        OUTPUT: A — numpy array of shape (batch_size, output_size)
                    e.g. (64, 10) — probability distribution over 10 classes

        DATA FLOW:
        X (64,784) → Layer1 → (64,128) → Layer2 → (64,64) → Layer3 → (64,10)
        """
        A = X
        # A shape evolves: (64,784) → (64,128) → (64,64) → (64,10)

        for layer in self.layers:
            A = layer.forward(A)
            # Each layer transforms A: (batch, in) → (batch, out)

        # A shape after all layers: (batch_size, num_classes) e.g. (64, 10)
        return A

    def compute_loss(self, y_pred, y_true):
        """
        WHAT: Computes categorical cross-entropy loss.
        INPUT:  y_pred — numpy array of shape (N, 10), predicted probabilities
                         from softmax (values in (0,1), rows sum to 1)
                y_true — numpy array of shape (N, 10), one-hot encoded labels
                         e.g. digit 3 → [0,0,0,1,0,0,0,0,0,0]
        OUTPUT: loss — scalar float, the average cross-entropy loss

        MATH: L = -1/N × Σᵢ Σⱼ y_true[i,j] × log(y_pred[i,j])

        WHY CROSS-ENTROPY?
        Imagine the network assigns probability 0.01 to the correct class.
        - Squared error penalty: (1 - 0.01)² ≈ 0.98 (mild penalty)
        - Cross-entropy penalty: -log(0.01) ≈ 4.6  (severe penalty)
        Cross-entropy punishes confident wrong answers much more harshly,
        which gives stronger gradient signals and faster learning.

        WHY ADD epsilon = 1e-8?
        log(0) = -infinity → NaN in loss → training crashes.
        Adding a tiny constant prevents this: log(0 + 1e-8) = -18.4
        (a finite, large penalty for confident wrong predictions)
        """
        epsilon = 1e-8  # Prevent log(0) which gives -infinity
        N = y_pred.shape[0]
        # y_pred shape: (N, 10)
        # y_true shape: (N, 10)

        # Compute cross-entropy: -sum(y_true × log(y_pred)) / N
        # y_true × log(y_pred + ε): (N, 10) — elementwise multiply
        # .sum(): sum all elements
        # / N: average over batch
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / N
        # loss: scalar float

        # ── NaN DETECTION ──
        if np.isnan(loss):
            raise ValueError(
                "NaN detected in loss! This usually means:\n"
                "  1. Learning rate is too large (try 10x smaller)\n"
                "  2. Weight initialization exploded (check your layer sizes)\n"
                "  3. Input data not normalized (should be in [0,1] range)\n"
                f"  Debug info: y_pred min={y_pred.min():.4f}, "
                f"max={y_pred.max():.4f}, "
                f"y_true sum per row={y_true.sum(axis=1).mean():.1f}"
            )

        return float(loss)

    def backward(self, y_pred, y_true, learning_rate):
        """
        WHAT: Runs the backward pass (backpropagation) through all layers.
        INPUT:  y_pred        — shape (N, 10), softmax output probabilities
                y_true        — shape (N, 10), one-hot true labels
                learning_rate — float, gradient descent step size
        OUTPUT: None (each layer updates its own weights as a side effect)

        THE BEAUTIFUL SIMPLIFICATION:
        For a network ending in Softmax + Cross-Entropy loss, the combined
        gradient w.r.t. the output layer's pre-activation Z is:

            dL/dZ_output = y_pred - y_true

        PROOF SKETCH:
        Let A = softmax(Z), L = -Σ y_true × log(A)
        dL/dA_j = -y_true_j / A_j
        dA_j/dZ_i = A_j(δᵢⱼ - A_i)  [Softmax Jacobian]
        Combining: dL/dZ_i = A_i - y_true_i

        This is so clean because cross-entropy is the "natural" loss
        for softmax — they were designed for each other.

        HOW BACKPROP FLOWS:
        dL = y_pred - y_true        (initial gradient at output)
        ↓ Layer 3 backward: updates W3, b3, passes gradient back
        ↓ Layer 2 backward: updates W2, b2, passes gradient back
        ↓ Layer 1 backward: updates W1, b1 (done — no layer before)
        """
        # ── INITIAL GRADIENT: Combined Softmax + Cross-Entropy ──
        # This is the gradient dL/dA at the output layer
        # For output layer with softmax, we pass this directly as "dA"
        # The DenseLayer.backward() method knows to treat softmax layers
        # by using this value directly as dZ (no additional multiplication)
        dA = y_pred - y_true
        # dA shape: (batch_size, 10) — how wrong were our probabilities?

        # ── BACKPROPAGATE THROUGH LAYERS IN REVERSE ORDER ──
        # reversed() goes: Layer3 → Layer2 → Layer1
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)
            # dA shape shrinks: (N,10) → (N,64) → (N,128) → (N,784)
            # Each layer:
            #   1. Computes gradients for its own W and b
            #   2. Updates W and b using gradient descent
            #   3. Returns gradient for the PREVIOUS layer

    def train(self, X_train, y_train, X_val, y_val,
              epochs, learning_rate, batch_size):
        """
        WHAT: Full training loop with mini-batch gradient descent.
        INPUT:  X_train       — (60000, 784) normalized training images
                y_train       — (60000, 10)  one-hot training labels
                X_val         — (N_val, 784) validation images
                y_val         — (N_val, 10)  validation labels
                epochs        — int, max training epochs e.g. 50
                learning_rate — float e.g. 0.01
                batch_size    — int, samples per gradient step e.g. 64
        OUTPUT: history — dict with keys:
                  'train_loss', 'val_loss',
                  'train_acc', 'val_acc'
                  Each is a list of length (actual epochs trained)

        MINI-BATCH GRADIENT DESCENT:
        Instead of computing gradient on all 60,000 samples at once
        (slow, memory-heavy) or one sample at a time (noisy, slow),
        we process B=64 samples per step. This gives:
        - Noisy enough gradient to escape local minima
        - Efficient enough to run in seconds per epoch
        - Memory-efficient for any dataset size

        EARLY STOPPING:
        If validation loss doesn't improve for PATIENCE epochs, we stop.
        WHY: After the validation loss plateaus or worsens, continuing
        training just makes the model OVERFIT — it memorizes training data
        but gets worse on new data. Early stopping is free regularization.
        """
        EARLY_STOP_PATIENCE = 10  # stop if no improvement for this many epochs
        N = X_train.shape[0]  # number of training samples e.g. 60000

        # ── HISTORY TRACKING ──
        history = {
            'train_loss': [],
            'val_loss':   [],
            'train_acc':  [],
            'val_acc':    [],
        }

        # ── EARLY STOPPING STATE ──
        best_val_loss = float('inf')  # best validation loss seen so far
        patience_counter = 0          # epochs since last improvement
        best_weights = None           # save best weights to restore later
        best_epoch = 0

        print(f"\nStarting training:")
        print(f"  Epochs:        {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Training samples: {N:,}")
        print(f"  Validation samples: {X_val.shape[0]:,}")
        print(f"  Steps per epoch: {N // batch_size}")
        print(f"  Early stopping patience: {EARLY_STOP_PATIENCE}\n")

        for epoch in range(epochs):

            # ── SHUFFLE TRAINING DATA EACH EPOCH ──
            # WHY SHUFFLE? If we always see samples in the same order,
            # the network might learn to expect that order rather than
            # generalizing. Shuffling ensures every mini-batch is a
            # random sample of the full training set.
            shuffle_idx = np.random.permutation(N)
            X_shuffled = X_train[shuffle_idx]   # (60000, 784)
            y_shuffled = y_train[shuffle_idx]   # (60000, 10)

            # ── MINI-BATCH LOOP ──
            epoch_losses = []

            # Create batches: 0, 64, 128, ..., up to N
            batch_starts = range(0, N, batch_size)

            if TQDM_AVAILABLE:
                batch_iter = tqdm(
                    batch_starts,
                    desc=f"Epoch {epoch+1:3d}/{epochs}",
                    leave=False,
                    unit="batch"
                )
            else:
                batch_iter = batch_starts

            for start in batch_iter:
                end = min(start + batch_size, N)
                # Slice the batch
                X_batch = X_shuffled[start:end]  # (batch_size, 784)
                y_batch = y_shuffled[start:end]  # (batch_size, 10)

                # ── FORWARD PASS ──
                y_pred_batch = self.forward(X_batch)
                # y_pred_batch shape: (batch_size, 10)

                # ── COMPUTE LOSS ──
                batch_loss = self.compute_loss(y_pred_batch, y_batch)
                epoch_losses.append(batch_loss)

                # ── BACKWARD PASS + WEIGHT UPDATE ──
                self.backward(y_pred_batch, y_batch, learning_rate)

            # ── EPOCH-LEVEL METRICS ──
            train_loss = float(np.mean(epoch_losses))

            # Evaluate on full training set (use batches to save memory)
            train_acc, _ = self.evaluate(X_train, y_train)
            val_acc, val_loss = self.evaluate(X_val, y_val)

            # Store in history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # ── PRINT PROGRESS EVERY 5 EPOCHS ──
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc*100:.2f}% | "
                      f"Val Acc: {val_acc*100:.2f}%")

            # ── EARLY STOPPING CHECK ──
            if val_loss < best_val_loss - 1e-4:  # require meaningful improvement
                # Improvement found! Reset patience counter
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                # Save current best weights
                best_weights = [layer.get_params() for layer in self.layers]
            else:
                # No improvement
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n⚡ Early stopping triggered at epoch {epoch+1}!")
                    print(f"   Best validation loss was at epoch {best_epoch}: "
                          f"{best_val_loss:.4f}")
                    # Restore best weights
                    if best_weights is not None:
                        for layer, params in zip(self.layers, best_weights):
                            layer.load_params(params)
                        print(f"   Restored weights from epoch {best_epoch}")
                    break

        print(f"\nTraining complete! Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # Store best_epoch in history for plotting
        history['best_epoch'] = best_epoch

        return history

    def predict(self, X):
        """
        WHAT: Returns class predictions (integer 0-9) for each input.
        INPUT:  X — numpy array of shape (N, 784)
        OUTPUT: predictions — numpy array of shape (N,), integer class indices

        argmax picks the class with the highest probability.
        """
        probs = self.forward(X)
        # probs shape: (N, 10)
        predictions = np.argmax(probs, axis=1)
        # predictions shape: (N,) — integer in range [0, 9]
        return predictions

    def predict_proba(self, X):
        """
        WHAT: Returns probability distribution over all classes.
        INPUT:  X — numpy array of shape (N, 784)
        OUTPUT: probabilities — numpy array of shape (N, 10)
                Each row sums to 1.0
        """
        return self.forward(X)
        # Return shape: (N, 10)

    def evaluate(self, X, y_onehot, batch_size=512):
        """
        WHAT: Computes accuracy and loss on a dataset.
        INPUT:  X         — numpy array of shape (N, 784)
                y_onehot  — numpy array of shape (N, 10), one-hot labels
                batch_size — int, process in chunks to avoid memory issues
        OUTPUT: (accuracy, loss) — both floats

        We process in batches of 512 to avoid trying to allocate a
        (60000, 10) matrix all at once, which can run out of RAM.
        """
        N = X.shape[0]
        all_preds = []
        all_probs = []

        # Process in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X[start:end]    # (batch, 784)
            probs_batch = self.forward(X_batch)  # (batch, 10)
            all_preds.append(np.argmax(probs_batch, axis=1))
            all_probs.append(probs_batch)

        # Stack all batch results
        all_preds = np.concatenate(all_preds)   # (N,)
        all_probs = np.concatenate(all_probs)   # (N, 10)

        # Convert one-hot to integer labels for accuracy computation
        true_labels = np.argmax(y_onehot, axis=1)  # (N,)
        accuracy = float(np.mean(all_preds == true_labels))
        loss = self.compute_loss(all_probs, y_onehot)

        return accuracy, loss

    def save_weights(self, filepath):
        """
        WHAT: Saves all layer weights to a .npz file (numpy compressed format).
        INPUT:  filepath — string, path to save file e.g. 'outputs/models/nn.npz'
        OUTPUT: None (creates the file)

        .npz is numpy's built-in format for storing multiple arrays in one file.
        It's like a zip file of numpy arrays — efficient and portable.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {}
        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            save_dict[f'layer_{i}_W'] = params['W']
            save_dict[f'layer_{i}_b'] = params['b']

        np.savez(filepath, **save_dict)
        print(f"✓ Weights saved to: {filepath}")
        print(f"  ({len(self.layers)} layers, {self.total_params:,} parameters)")

    def load_weights(self, filepath):
        """
        WHAT: Loads all layer weights from a .npz file.
        INPUT:  filepath — string, path to the .npz file
        OUTPUT: None (updates all layer weights in place)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Weight file not found: {filepath}\n"
                f"Did you run 02_numpy_train.py first to generate weights?"
            )

        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            layer.load_params({
                'W': data[f'layer_{i}_W'],
                'b': data[f'layer_{i}_b'],
            })
        print(f"✓ Weights loaded from: {filepath}")


# ══════════════════════════════════════════
# TESTING BLOCK
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NeuralNetwork: ARCHITECTURE AND FORWARD PASS")
    print("=" * 60)

    np.random.seed(42)

    # ── BUILD MNIST ARCHITECTURE ──
    INPUT_SIZE  = 784
    HIDDEN_1    = 128
    HIDDEN_2    = 64
    OUTPUT_SIZE = 10

    layer_configs = [
        {'in': INPUT_SIZE, 'out': HIDDEN_1,    'act': 'relu'},
        {'in': HIDDEN_1,   'out': HIDDEN_2,    'act': 'relu'},
        {'in': HIDDEN_2,   'out': OUTPUT_SIZE, 'act': 'softmax'},
    ]

    model = NeuralNetwork(layer_configs)

    # ── TEST FORWARD PASS ──
    print("\nTesting forward pass...")
    BATCH_SIZE = 64
    X_fake = np.random.randn(BATCH_SIZE, INPUT_SIZE)  # (64, 784)
    y_probs = model.forward(X_fake)
    # Expected shape: (64, 10)
    print(f"Input shape:  {X_fake.shape}")
    print(f"Output shape: {y_probs.shape}")
    assert y_probs.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert np.allclose(y_probs.sum(axis=1), 1.0), "Probabilities should sum to 1!"
    print("✓ Forward pass shape correct")
    print("✓ Output probabilities sum to 1.0")

    # ── TEST LOSS COMPUTATION ──
    print("\nTesting loss computation...")
    y_fake_onehot = np.eye(OUTPUT_SIZE)[np.random.randint(0, OUTPUT_SIZE, BATCH_SIZE)]
    # y_fake_onehot shape: (64, 10)
    loss = model.compute_loss(y_probs, y_fake_onehot)
    print(f"Initial loss: {loss:.4f} (random init, should be ~log(10) ≈ 2.303)")
    # With random init, expected loss ≈ -log(1/10) = log(10) ≈ 2.303
    assert 1.0 < loss < 5.0, f"Loss {loss} seems wrong for random initialization!"
    print("✓ Loss is in expected range for random initialization")

    # ── TEST PREDICT ──
    preds = model.predict(X_fake)
    print(f"\nPredictions shape: {preds.shape}")
    assert preds.shape == (BATCH_SIZE,)
    assert preds.min() >= 0 and preds.max() <= 9
    print("✓ Predictions are integers in range [0, 9]")

    print("\n" + "=" * 60)
    print("ALL NeuralNetwork TESTS PASSED ✓")
    print("=" * 60)
    print("\nReady to train on MNIST! Run 02_numpy_train.py next.")
