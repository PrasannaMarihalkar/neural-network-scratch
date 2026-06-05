# ══════════════════════════════════════════
# FILE: src/layers.py
# WHAT THIS FILE DOES: Implements a single fully-connected (Dense) layer
#   with forward pass, backward pass (backpropagation), and weight updates.
#   This is the core mathematical building block of the neural network.
# DEPENDS ON: src/activations.py
# RUN WITH: python src/layers.py    (runs the __main__ test block)
# EXPECTED OUTPUT: Forward pass shapes verified, backward pass shapes verified
# ══════════════════════════════════════════

import numpy as np
from src.activations import relu, relu_derivative, softmax, sigmoid, sigmoid_derivative


# ──────────────────────────────────────────
# WHAT IS A "DENSE LAYER"?
#
# A dense (fully-connected) layer is where EVERY input neuron connects
# to EVERY output neuron. For a layer with 784 inputs and 128 outputs,
# that's 784 × 128 = 100,352 individual learned weights.
#
# The computation is: Z = X @ W + b
# Then we apply an activation: A = activation(Z)
#
# During TRAINING, we also need to run the backward pass to compute
# gradients (how much each weight contributed to the error) so we
# can update the weights to improve.
#
# ANALOGY: Imagine a voting system where 784 voters each vote on
# 128 different proposals. The weight W[i][j] is how much voter i
# influences proposal j. The bias b[j] is the baseline score for
# proposal j regardless of voting.
# ──────────────────────────────────────────


class DenseLayer:
    """
    A fully-connected neural network layer with:
    - Xavier weight initialization
    - Forward pass (compute output from input)
    - Backward pass (compute gradients and update weights)
    """

    def __init__(self, input_size, output_size, activation='relu'):
        """
        WHAT: Initializes the layer's weights, biases, and activation function.
        INPUT:  input_size  — int, number of input features
                              e.g. 784 for flattened 28×28 MNIST images
                output_size — int, number of neurons in this layer
                              e.g. 128 for a hidden layer with 128 neurons
                activation  — str, 'relu', 'softmax', or 'sigmoid'
        OUTPUT: A configured layer ready for forward/backward passes.

        XAVIER/GLOROT INITIALIZATION — WHY IT MATTERS:
        ────────────────────────────────────────────────
        If we initialize weights too large:
          → Pre-activation values Z become huge
          → Activations saturate (sigmoid→1, softmax→extreme)
          → Gradients vanish (sigmoid derivative near 0 at extremes)
          → Network can't learn

        If we initialize weights too small:
          → Signals get smaller with each layer
          → Eventually become too tiny to distinguish
          → Again, gradients vanish

        Xavier initialization sets the scale to sqrt(2/n_in).
        WHY sqrt(2/n_in)?
          If X has variance 1, and W ~ Normal(0, σ²), then:
          Var(X·W) = n_in × Var(X) × Var(W) = n_in × 1 × σ²
          We want Var(output) = 1, so: n_in × σ² = 1 → σ = 1/sqrt(n_in)
          The factor of 2 is the "He" correction for ReLU which kills
          half the neurons (negative → 0), so we need 2× the variance.

        Result: Variance stays consistent across all layers.
                Neither exploding nor vanishing signals.
        """
        self.input_size = input_size    # e.g. 784
        self.output_size = output_size  # e.g. 128
        self.activation = activation    # e.g. 'relu'

        # ── WEIGHT INITIALIZATION (Xavier/He) ──
        # W shape: (input_size, output_size), e.g. (784, 128)
        # Each weight drawn from Normal(0, sqrt(2/input_size))
        xavier_scale = np.sqrt(2.0 / input_size)
        self.W = np.random.randn(input_size, output_size) * xavier_scale
        # W shape: (input_size, output_size) e.g. (784, 128)

        # ── BIAS INITIALIZATION (zeros) ──
        # Biases initialized to 0 — this is standard practice.
        # (Initializing W to 0 would be wrong: all neurons would be
        #  identical and learn the same thing — "symmetry problem")
        self.b = np.zeros((1, output_size))
        # b shape: (1, output_size) e.g. (1, 128)
        # Using (1, output_size) instead of (output_size,) makes
        # broadcasting work cleanly with batch inputs

        # ── CACHES FOR BACKWARD PASS ──
        # During forward pass, we'll cache the inputs X and pre-activation Z
        # The backward pass NEEDS these to compute gradients
        self.X_cache = None  # will store input X during forward pass
        self.Z_cache = None  # will store pre-activation Z during forward pass

        print(f"  Layer created: {input_size} → {output_size} [{activation}] | "
              f"Params: {input_size * output_size + output_size:,}")

    def forward(self, X):
        """
        WHAT: Computes the layer output for a batch of inputs.
        INPUT:  X — numpy array of shape (batch_size, input_size)
                    e.g. (64, 784) for a batch of 64 flattened images
        OUTPUT: A — numpy array of shape (batch_size, output_size)
                    e.g. (64, 128) — the activated outputs of this layer

        MATH:
        Step 1: Z = X @ W + b        (linear combination)
        Step 2: A = activation(Z)    (nonlinear transformation)

        WHY CACHE X AND Z?
        During backward pass we need:
        - X_cache to compute dL/dW = X.T @ dZ / N
        - Z_cache to compute dZ = dA * activation'(Z)
        Without caching, we'd have to recompute them, which is wasteful.
        """
        # ── CACHE THE INPUT ──
        self.X_cache = X
        # X_cache shape: (batch_size, input_size) e.g. (64, 784)

        # ── LINEAR STEP ──
        # X:  (64, 784)
        # W:  (784, 128)
        # b:  (1, 128)
        # X @ W: (64, 128) — matrix multiply
        # + b: broadcasts b across all 64 rows
        # Z:  (64, 128)
        Z = X @ self.W + self.b
        # Z shape: (batch_size, output_size) e.g. (64, 128)

        # ── CACHE PRE-ACTIVATION Z ──
        self.Z_cache = Z
        # Z_cache shape: (batch_size, output_size)

        # ── ACTIVATION STEP ──
        if self.activation == 'relu':
            A = relu(Z)
        elif self.activation == 'softmax':
            A = softmax(Z)
        elif self.activation == 'sigmoid':
            A = sigmoid(Z)
        else:
            raise ValueError(f"Unknown activation '{self.activation}'. "
                             f"Choose from: 'relu', 'softmax', 'sigmoid'")
        # A shape: (batch_size, output_size) — same as Z

        return A
        # Return shape: (batch_size, output_size) e.g. (64, 128)

    def backward(self, dA, learning_rate):
        """
        WHAT: Computes gradients and updates weights via backpropagation.
        INPUT:  dA            — gradient of loss w.r.t. this layer's OUTPUT
                                shape: (batch_size, output_size) e.g. (64, 128)
                                This flows in from the NEXT layer (or from the
                                loss function for the output layer)
                learning_rate — float, step size for gradient descent e.g. 0.01
        OUTPUT: dA_prev — gradient of loss w.r.t. this layer's INPUT
                          shape: (batch_size, input_size) e.g. (64, 784)
                          This will be passed to the PREVIOUS layer.
        SIDE EFFECT: Updates self.W and self.b in place.

        CHAIN RULE DERIVATION — Step by step:
        ──────────────────────────────────────
        We want to compute dL/dW and dL/db to update weights.

        The computation graph is:
        X → [Z = X @ W + b] → [A = activation(Z)] → Loss

        By chain rule:
        dL/dZ = dL/dA × dA/dZ          ← dA (input) × activation derivative
        dL/dW = dL/dZ × dZ/dW          ← dZ × X (since dZ/dW = X)
              = X.T @ dZ / N           ← divide by N for mean gradient
        dL/db = mean(dL/dZ, axis=0)    ← mean over batch
        dL/dX = dL/dZ × dZ/dX          ← this goes to previous layer
              = dZ @ W.T

        For softmax output layer, dL/dZ is passed in directly as dA
        (because we combined softmax + cross-entropy gradient: A - y_true)

        For ReLU hidden layers, dA is the gradient from the next layer,
        and we multiply by relu_derivative(Z) to get dZ.
        """
        batch_size = dA.shape[0]
        # dA shape: (batch_size, output_size) e.g. (64, 128)

        # ── STEP 1: COMPUTE dZ (gradient w.r.t. pre-activation) ──
        # Chain rule: dL/dZ = dL/dA × dA/dZ
        # For the softmax output layer: dA is ALREADY dZ (passed in directly)
        # For hidden layers with ReLU/sigmoid: multiply by activation derivative

        if self.activation == 'softmax':
            # dA is already dL/dZ = A - y_true (the beautiful simplification)
            # We pass it through unchanged
            dZ = dA
            # dZ shape: (batch_size, output_size) e.g. (64, 10)
        elif self.activation == 'relu':
            # dZ = dA * relu'(Z) — gradient passes only where Z > 0
            dZ = dA * relu_derivative(self.Z_cache)
            # dA shape:        (64, 128)
            # relu_derivative: (64, 128)  ← uses cached Z
            # dZ shape:        (64, 128)  ← elementwise multiply
        elif self.activation == 'sigmoid':
            # dZ = dA * sigmoid'(Z)
            dZ = dA * sigmoid_derivative(self.Z_cache)
            # dZ shape: (batch_size, output_size)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # ── STEP 2: COMPUTE dW (gradient w.r.t. weights) ──
        # dL/dW = X.T @ dZ / N
        # Since Z = X @ W, the derivative dZ/dW = X
        # And dL/dW = (dL/dZ) × (dZ/dW) = (dL/dZ) × X
        # The transpose + division by N gives us the average gradient
        # over the batch (standard mini-batch gradient descent)
        #
        # X_cache.T shape: (input_size, batch_size)  e.g. (784, 64)
        # dZ shape:        (batch_size, output_size) e.g. (64, 128)
        # dW shape:        (input_size, output_size) e.g. (784, 128)
        dW = self.X_cache.T @ dZ / batch_size
        # dW shape: (input_size, output_size) — same as self.W ✓

        # ── STEP 3: COMPUTE db (gradient w.r.t. biases) ──
        # dL/db = mean(dZ, axis=0) — average over the batch
        # dZ shape: (64, 128) → sum over axis=0 → db shape: (1, 128)
        db = np.mean(dZ, axis=0, keepdims=True)
        # db shape: (1, output_size) — same as self.b ✓

        # ── STEP 4: COMPUTE dA_prev (gradient to pass to previous layer) ──
        # dL/dX = dZ @ W.T
        # This is the gradient w.r.t. the INPUT of this layer,
        # which becomes the gradient w.r.t. the OUTPUT of the PREVIOUS layer.
        # dZ shape:  (batch_size, output_size) e.g. (64, 128)
        # W.T shape: (output_size, input_size) e.g. (128, 784)
        # dA_prev shape: (batch_size, input_size) e.g. (64, 784)
        dA_prev = dZ @ self.W.T
        # dA_prev shape: (batch_size, input_size) — this goes backward ✓

        # ── STEP 5: UPDATE WEIGHTS (Gradient Descent) ──
        # W_new = W_old - learning_rate × dW
        # We SUBTRACT because we want to move in the direction of
        # DECREASING loss (gradient points toward steepest ascent,
        # so we go the opposite way to descend)
        self.W = self.W - learning_rate * dW
        # self.W shape unchanged: (input_size, output_size)

        self.b = self.b - learning_rate * db
        # self.b shape unchanged: (1, output_size)

        return dA_prev
        # Return shape: (batch_size, input_size) e.g. (64, 784)

    def get_params(self):
        """
        WHAT: Returns the layer's weights and biases as a dictionary.
        INPUT:  None
        OUTPUT: dict with keys 'W' and 'b'
        WHY:    Used by NeuralNetwork.save_weights() to persist the model.
        """
        return {
            'W': self.W,  # shape: (input_size, output_size)
            'b': self.b,  # shape: (1, output_size)
        }

    def load_params(self, params):
        """
        WHAT: Loads weights and biases from a dictionary.
        INPUT:  params — dict with keys 'W' and 'b' (numpy arrays)
        OUTPUT: None (updates self.W and self.b in place)
        WHY:    Used by NeuralNetwork.load_weights() to restore a saved model.
        """
        # Validate shapes before loading to catch size mismatches early
        expected_W_shape = (self.input_size, self.output_size)
        if params['W'].shape != expected_W_shape:
            raise ValueError(
                f"Weight shape mismatch! Expected {expected_W_shape}, "
                f"got {params['W'].shape}"
            )
        self.W = params['W']  # shape: (input_size, output_size)
        self.b = params['b']  # shape: (1, output_size)

    def __repr__(self):
        """String representation for debugging."""
        n_params = self.input_size * self.output_size + self.output_size
        return (f"DenseLayer({self.input_size} → {self.output_size}, "
                f"activation={self.activation}, params={n_params:,})")


# ══════════════════════════════════════════
# TESTING BLOCK
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DenseLayer: FORWARD AND BACKWARD PASS")
    print("=" * 60)

    np.random.seed(42)  # For reproducible tests

    # ── TEST 1: ReLU hidden layer ──
    print("\n--- Test 1: Dense Layer with ReLU (hidden layer) ---")
    INPUT_SIZE_TEST  = 784   # flattened 28x28 image
    OUTPUT_SIZE_TEST = 128   # 128 hidden neurons
    BATCH_SIZE_TEST  = 64    # 64 samples per batch

    layer_relu = DenseLayer(INPUT_SIZE_TEST, OUTPUT_SIZE_TEST, activation='relu')
    print(f"Layer: {layer_relu}")

    # Create fake input batch
    X_fake = np.random.randn(BATCH_SIZE_TEST, INPUT_SIZE_TEST)
    # X_fake shape: (64, 784)
    print(f"Input X shape: {X_fake.shape}")

    # Forward pass
    A_out = layer_relu.forward(X_fake)
    print(f"Forward output A shape: {A_out.shape}")
    # Expected: (64, 128)
    assert A_out.shape == (BATCH_SIZE_TEST, OUTPUT_SIZE_TEST), "Forward shape wrong!"
    assert np.all(A_out >= 0), "ReLU output should be non-negative!"
    print("✓ Forward pass: correct shape and non-negative outputs (ReLU)")

    # Backward pass
    dA_fake = np.random.randn(BATCH_SIZE_TEST, OUTPUT_SIZE_TEST)
    # dA_fake shape: (64, 128) — fake gradient from next layer
    dA_prev = layer_relu.backward(dA_fake, learning_rate=0.01)
    print(f"Backward dA_prev shape: {dA_prev.shape}")
    # Expected: (64, 784) — gradient flows back to the 784-dim input
    assert dA_prev.shape == (BATCH_SIZE_TEST, INPUT_SIZE_TEST), "Backward shape wrong!"
    print("✓ Backward pass: correct gradient shape returned")

    # ── TEST 2: Softmax output layer ──
    print("\n--- Test 2: Dense Layer with Softmax (output layer) ---")
    HIDDEN_SIZE_TEST = 64
    NUM_CLASSES_TEST = 10

    layer_softmax = DenseLayer(HIDDEN_SIZE_TEST, NUM_CLASSES_TEST, activation='softmax')
    X_hidden = np.random.randn(BATCH_SIZE_TEST, HIDDEN_SIZE_TEST)
    # X_hidden shape: (64, 64)

    A_probs = layer_softmax.forward(X_hidden)
    print(f"Softmax output shape: {A_probs.shape}")
    # Expected: (64, 10)
    assert A_probs.shape == (BATCH_SIZE_TEST, NUM_CLASSES_TEST)

    # Probabilities should sum to 1 per sample
    row_sums = A_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Softmax rows must sum to 1!"
    print("✓ Softmax output rows all sum to 1.0")

    # ── TEST 3: Parameter save/load ──
    print("\n--- Test 3: Parameter save and load ---")
    original_W = layer_relu.W.copy()
    params = layer_relu.get_params()

    # Corrupt the weights
    layer_relu.W = np.zeros_like(layer_relu.W)

    # Restore
    layer_relu.load_params(params)
    assert np.allclose(layer_relu.W, original_W), "Weights not restored correctly!"
    print("✓ save/load params works correctly")

    print("\n" + "=" * 60)
    print("ALL DenseLayer TESTS PASSED ✓")
    print("=" * 60)
