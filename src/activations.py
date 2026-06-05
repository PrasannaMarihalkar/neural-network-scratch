# ══════════════════════════════════════════
# FILE: src/activations.py
# WHAT THIS FILE DOES: Implements every activation function used in our
#   neural network — ReLU, Softmax, Sigmoid — with their derivatives.
#   These are the "decision functions" of each neuron.
# DEPENDS ON: Nothing (only numpy)
# RUN WITH: python src/activations.py    (runs the __main__ test block)
# EXPECTED OUTPUT: Printed test results confirming each function works
# ══════════════════════════════════════════

import numpy as np

# ──────────────────────────────────────────
# INTUITION BEFORE THE MATH:
# A neuron receives a weighted sum of inputs (called Z). Without an
# activation function, a neural network is just a linear regression —
# no matter how many layers you stack. Activation functions introduce
# NON-LINEARITY, which is what lets the network learn complex patterns.
#
# Analogy: A light dimmer (linear) vs. a light switch (nonlinear).
# Real-world problems — like recognizing handwritten digits — need
# the switch-like behavior that activation functions provide.
# ──────────────────────────────────────────


# ══════════════════════════════════════════
# RELU — Rectified Linear Unit
# ══════════════════════════════════════════

def relu(Z):
    """
    WHAT: Applies ReLU activation elementwise: max(0, Z)
    INPUT:  Z — numpy array of any shape, e.g. (64, 128)
            These are the pre-activation values (weighted sums)
    OUTPUT: A — same shape as Z, negative values replaced with 0
    WHY:    ReLU is the most popular hidden-layer activation because:
            1. It does NOT saturate for positive values (gradient = 1)
            2. It is computationally free (just a max operation)
            3. It creates sparse activations (many neurons output 0)

    MATH: A = max(0, Z)
          For each element Zᵢⱼ:
            if Zᵢⱼ > 0 → Aᵢⱼ = Zᵢⱼ  (pass through)
            if Zᵢⱼ ≤ 0 → Aᵢⱼ = 0    (kill the negative signal)

    ANALOGY: ReLU is like a half-wave rectifier in electronics.
             It lets the positive current through and blocks negative.
    """
    # np.maximum compares Z against 0 elementwise — very fast in numpy
    A = np.maximum(0, Z)
    # Z shape: (batch_size, layer_size) → A shape: same as Z
    return A


def relu_derivative(Z):
    """
    WHAT: Computes the derivative of ReLU for use in backpropagation
    INPUT:  Z — numpy array of any shape, the PRE-activation values
            (we use the original Z, not the activated A, because
             the derivative is defined in terms of Z)
    OUTPUT: dA_dZ — same shape as Z, values are 0 or 1

    MATH: d/dZ max(0,Z) = 1 if Z > 0 else 0
          This is the Heaviside step function.

    WHY THIS MATTERS IN BACKPROP:
    During backpropagation, we need dL/dZ = dL/dA * dA/dZ
    The dA/dZ part is exactly this function.
    Values where Z ≤ 0 contribute ZERO gradient — the neuron is
    "dead" for that input. This is called the "dead ReLU" problem.

    VANISHING GRADIENT COMPARISON:
    Sigmoid derivative: σ'(Z) = σ(Z)(1-σ(Z)) ≤ 0.25 always
    After 4 layers:     gradient shrinks by 0.25^4 = 0.004 (0.4%!)
    ReLU derivative:    1.0 for positive Z — gradient passes through INTACT
    After 4 layers:     gradient stays full size (no shrinkage)
    This is why deep networks use ReLU, not sigmoid, in hidden layers.
    """
    # Z shape: (batch_size, layer_size) → output shape: same as Z
    # (Z > 0) creates a boolean array; numpy treats True=1, False=0
    dA_dZ = (Z > 0).astype(float)
    return dA_dZ


# ══════════════════════════════════════════
# SOFTMAX — Probability Distribution Function
# ══════════════════════════════════════════

def softmax(Z):
    """
    WHAT: Converts raw scores (logits) into a probability distribution
          so all outputs sum to 1.0 and each is between 0 and 1.
    INPUT:  Z — numpy array of shape (batch_size, num_classes)
            e.g. (64, 10) — 64 samples, 10 digit classes
    OUTPUT: A — same shape as Z, each ROW sums to 1.0
            These are the predicted probabilities for each class.

    MATH: Softmax(Zᵢ) = exp(Zᵢ) / Σⱼ exp(Zⱼ)

    ⚠️  NUMERICAL STABILITY — THIS IS CRITICAL ⚠️
    Problem: exp(Z) can overflow to inf when Z is large.
             For example: np.exp(1000) = inf (overflow!)
             Then inf/inf = NaN → your entire model breaks.

    Fix (the standard trick):
    Softmax(Z) = exp(Z - max(Z)) / Σ exp(Z - max(Z))

    WHY IS THIS VALID? Let C = max(Z). Then:
    exp(Zᵢ - C) / Σ exp(Zⱼ - C)
    = [exp(Zᵢ)/exp(C)] / [Σ exp(Zⱼ)/exp(C)]
    = exp(Zᵢ) / Σ exp(Zⱼ)       ← the C cancels out!
    The result is mathematically IDENTICAL, but numerically safe.
    After subtracting max: the largest value becomes exp(0) = 1.0
    All other values are exp(negative number) < 1.0
    No overflow possible!

    ANALOGY: Softmax is like converting raw exam scores to class ranks.
             If everyone shifts up by the same amount, the ranking
             doesn't change — just like subtracting max doesn't
             change the probabilities.
    """
    # Z shape: (64, 10) — 64 samples, each with 10 raw scores

    # Step 1: Subtract max for numerical stability
    # keepdims=True preserves shape (64,1) so broadcasting works correctly
    # Without keepdims, shape would be (64,) and subtraction would fail
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    # Z_shifted shape: (64, 10) — same as input, but max per row is now 0

    # Step 2: Exponentiate — now safe because max value is exp(0)=1
    exp_Z = np.exp(Z_shifted)
    # exp_Z shape: (64, 10)

    # Step 3: Divide by row sums to normalize to probabilities
    # keepdims=True preserves shape (64,1) for broadcasting
    A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    # A shape: (64, 10) — each row sums to exactly 1.0

    return A


# WHY WE DON'T IMPLEMENT softmax_derivative SEPARATELY:
# ─────────────────────────────────────────────────────
# The derivative of softmax alone is a complex Jacobian matrix.
# But in practice, softmax is ALWAYS paired with cross-entropy loss.
# When you combine them, the derivative simplifies BEAUTIFULLY to:
#
#   dL/dZ = A - y_true
#
# where A is the softmax output and y_true is the one-hot label.
# This means: the gradient is just "how far off were your predictions?"
# We implement this combined gradient directly in network.py's backward().
# This is one of the most elegant results in deep learning math.


# ══════════════════════════════════════════
# SIGMOID — The Original Neural Network Activation
# ══════════════════════════════════════════

def sigmoid(Z):
    """
    WHAT: Squashes any real number into the range (0, 1)
    INPUT:  Z — numpy array of any shape
    OUTPUT: A — same shape as Z, all values in (0, 1)

    MATH: σ(Z) = 1 / (1 + exp(-Z))

    HISTORICAL NOTE: Sigmoid was THE activation function for decades.
    It was inspired by biological neurons: either "firing" (→1)
    or "not firing" (→0). It's still used in output layers for
    binary classification (e.g., is this email spam? yes/no).

    WHY WE DON'T USE IT IN HIDDEN LAYERS (vanishing gradient problem):
    σ'(Z) reaches maximum of 0.25 at Z=0, and approaches 0 for large |Z|.
    In a 5-layer network:   0.25^5 = 0.001 — gradient almost gone!
    In a 10-layer network:  0.25^10 = 0.0000009 — effectively dead!
    Weights in early layers NEVER get meaningful gradient signals.
    This is the vanishing gradient problem that plagued deep learning
    until ReLU was popularized around 2012.

    NUMERICAL STABILITY NOTE:
    np.exp(-Z) can overflow for large negative Z.
    The clipping ensures we don't get NaN values.
    """
    # Clip Z to prevent overflow in exp — values outside ±500 are
    # effectively saturated anyway (sigmoid output ≈ 0 or ≈ 1)
    Z_clipped = np.clip(Z, -500, 500)
    # Z_clipped shape: same as input Z

    A = 1.0 / (1.0 + np.exp(-Z_clipped))
    # A shape: same as Z, all values in (0, 1)
    return A


def sigmoid_derivative(Z):
    """
    WHAT: Computes the derivative of sigmoid for backpropagation
    INPUT:  Z — numpy array of any shape (pre-activation values)
    OUTPUT: dA_dZ — same shape as Z

    MATH: d/dZ σ(Z) = σ(Z) · (1 - σ(Z))

    DERIVATION (beautiful result):
    Let σ = 1/(1+e^(-Z))
    dσ/dZ = e^(-Z) / (1+e^(-Z))²
           = [1/(1+e^(-Z))] · [e^(-Z)/(1+e^(-Z))]
           = σ · (1 - σ)

    KEY OBSERVATION: Maximum value is 0.25 (at Z=0, σ=0.5)
    This is what causes the vanishing gradient problem.
    """
    # Compute sigmoid first
    s = sigmoid(Z)
    # s shape: same as Z

    # Derivative: σ(Z) * (1 - σ(Z))
    dA_dZ = s * (1.0 - s)
    # dA_dZ shape: same as Z, maximum value is 0.25
    return dA_dZ


# ══════════════════════════════════════════
# TESTING BLOCK — Run this file directly to verify everything works
# ══════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ALL ACTIVATION FUNCTIONS")
    print("=" * 60)

    # Small test array — mix of positive and negative values
    Z_test = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0],
                       [ 0.5, -0.5, 3.0,-3.0, 0.1]])
    # Z_test shape: (2, 5) — 2 samples, 5 features

    print("\n--- ReLU ---")
    print(f"Input Z:\n{Z_test}")
    A_relu = relu(Z_test)
    print(f"relu(Z):\n{A_relu}")
    # Expected: negative values become 0, positive values unchanged
    assert A_relu.shape == Z_test.shape, "ReLU output shape mismatch!"
    assert np.all(A_relu >= 0), "ReLU should never output negative!"
    print("✓ ReLU shape and non-negativity verified")

    print("\n--- ReLU Derivative ---")
    dZ_relu = relu_derivative(Z_test)
    print(f"relu_derivative(Z):\n{dZ_relu}")
    # Expected: 1 where Z > 0, 0 where Z <= 0
    assert set(np.unique(dZ_relu)).issubset({0.0, 1.0}), "Derivative should be 0 or 1!"
    print("✓ ReLU derivative only contains 0s and 1s")

    print("\n--- Softmax ---")
    Z_logits = np.array([[1.0, 2.0, 3.0, 0.0, -1.0,  0.5, 0.2, -0.3, 0.8, 1.5],
                         [0.1, 0.2, 0.1, 0.8,  2.0, -0.5, 0.3,  0.1, 0.4, 0.2]])
    # Z_logits shape: (2, 10) — 2 samples, 10 digit classes
    A_softmax = softmax(Z_logits)
    print(f"softmax output (2 samples, 10 classes):\n{A_softmax.round(4)}")
    # Each row must sum to 1.0
    row_sums = A_softmax.sum(axis=1)
    print(f"Row sums (should be [1.0, 1.0]): {row_sums}")
    assert np.allclose(row_sums, 1.0), "Softmax rows must sum to 1!"
    assert np.all(A_softmax >= 0), "Softmax must be non-negative!"
    print("✓ Softmax rows sum to 1.0 and all values non-negative")

    # Test numerical stability with very large values
    Z_large = np.array([[1000.0, 999.0, 998.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    A_large = softmax(Z_large)
    print(f"\nNumerical stability test (large inputs):")
    print(f"softmax([1000, 999, 998, 0, ...]) = {A_large.round(4)}")
    assert not np.any(np.isnan(A_large)), "NaN detected! Numerical instability!"
    print("✓ No NaN even with extreme inputs — numerical stability works!")

    print("\n--- Sigmoid ---")
    A_sigmoid = sigmoid(Z_test)
    print(f"sigmoid(Z):\n{A_sigmoid.round(4)}")
    assert np.all(A_sigmoid > 0) and np.all(A_sigmoid < 1), "Sigmoid must be in (0,1)!"
    print("✓ Sigmoid output confirmed in range (0, 1)")

    print("\n--- Sigmoid Derivative ---")
    dZ_sigmoid = sigmoid_derivative(Z_test)
    print(f"sigmoid_derivative(Z):\n{dZ_sigmoid.round(4)}")
    max_deriv = dZ_sigmoid.max()
    print(f"Maximum derivative value: {max_deriv:.4f} (should be ≤ 0.25)")
    assert max_deriv <= 0.25 + 1e-9, f"Sigmoid derivative exceeded 0.25: {max_deriv}"
    print("✓ Sigmoid derivative ≤ 0.25 confirmed (this causes vanishing gradients!)")

    print("\n" + "=" * 60)
    print("ALL ACTIVATION FUNCTION TESTS PASSED ✓")
    print("=" * 60)
    print("\nKEY INSIGHT: Notice how sigmoid's max derivative is 0.25.")
    print("In a 4-layer network: 0.25^4 = {:.6f}".format(0.25**4))
    print("In an 8-layer network: 0.25^8 = {:.8f}".format(0.25**8))
    print("This is why deep networks use ReLU — gradients stay at 1.0!")
