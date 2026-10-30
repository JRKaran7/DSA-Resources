"""
Neural Networks from Scratch - Reference Implementation
Aligned with "Neural Networks from Scratch" Slide Deck (Sessions 1 & 2)

Session 1:
- Perceptron & Linear Transformation (Slide 4, 9)
- Activation Functions: Sigmoid, ReLU, Tanh (Slide 5, 6)
- Forward Propagation with Cache (Slide 8, 9, 10)
- Loss Functions: MSE & Binary Cross-Entropy (Slide 11, 12)

Session 2:
- Gradient Descent & Parameter Update (Slide 19, 20)
- Backpropagation via Chain Rule (Slide 21, 22, 23)
- Training Loop (Slide 24, 25)
- Regularization: L2 & Dropout (Slide 26)
- Lab 2 Non-linear XOR Problem (Slide 28)
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. ACTIVATION FUNCTIONS (Slide 5 & 6)
# ==========================================

def sigmoid(z):
    """Maps any real value into (0, 1)."""
    # Clip z to prevent numerical overflow in exp
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_derivative(a):
    """Derivative of sigmoid given activation a = sigmoid(z)."""
    return a * (1.0 - a)

def relu(z):
    """Returns max of 0 and input."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: 1 for z > 0, 0 otherwise."""
    return (z > 0).astype(float)

def tanh(z):
    """Squashes values between -1 and 1."""
    return np.tanh(z)

def tanh_derivative(a):
    """Derivative of tanh given activation a = tanh(z)."""
    return 1.0 - np.square(a)


# ==========================================
# 2. PARAMETER INITIALIZATION (Slide 14 & 24)
# ==========================================

def initialize_parameters(n_x, n_h, n_y, seed=42):
    """
    Initialize network weights and biases.
    
    Parameters:
    - n_x: number of input features (e.g. 2 for 2D points)
    - n_h: number of hidden neurons (e.g. 4)
    - n_y: number of output neurons (e.g. 1 for binary classification)
    
    Returns:
    - W1: weight matrix of shape (n_h, n_x)
    - b1: bias vector of shape (n_h, 1)
    - W2: weight matrix of shape (n_y, n_h)
    - b2: bias vector of shape (n_y, 1)
    """
    if seed is not None:
        np.random.seed(seed)
        
    # He/Xavier-like scaling for stable training
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((n_y, 1))
    
    return W1, b1, W2, b2


# ==========================================
# 3. FORWARD PROPAGATION (Slide 9 & 10)
# ==========================================

def forward_prop(X, W1, b1, W2, b2, activation="relu"):
    """
    2-Layer Forward Propagation Pipeline.
    
    Mathematical Steps:
      Layer 1:
        Z1 = W1 . X + b1
        A1 = activation(Z1)
      Layer 2:
        Z2 = W2 . A1 + b2
        A2 = sigmoid(Z2)
        
    Stores intermediate variables in cache for backpropagation.
    """
    # Layer 1
    Z1 = np.dot(W1, X) + b1
    if activation == "relu":
        A1 = relu(Z1)
    elif activation == "tanh":
        A1 = tanh(Z1)
    elif activation == "sigmoid":
        A1 = sigmoid(Z1)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
        
    # Layer 2 (Output classification layer with Sigmoid)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # Store values for backprop
    cache = (Z1, A1, Z2, A2)
    return A2, cache


# ==========================================
# 4. LOSS FUNCTIONS (Slide 11 & 12)
# ==========================================

def mse_loss(Y_hat, Y):
    """
    Mean Squared Error (MSE) loss (Slide 12).
    Loss = (1 / (2 * m)) * sum((Y_hat - Y)^2)
    """
    m = Y.shape[1]
    error = (Y_hat - Y) ** 2
    loss = (1.0 / (2.0 * m)) * np.sum(error)
    return float(loss)

def binary_cross_entropy(Y_hat, Y, epsilon=1e-15):
    """
    Binary Cross-Entropy Loss (Slide 11).
    Loss = - (1 / m) * sum(Y * log(Y_hat) + (1 - Y) * log(1 - Y_hat))
    """
    m = Y.shape[1]
    Y_hat_safe = np.clip(Y_hat, epsilon, 1.0 - epsilon)
    loss = - (1.0 / m) * np.sum(Y * np.log(Y_hat_safe) + (1.0 - Y) * np.log(1.0 - Y_hat_safe))
    return float(loss)


# ==========================================
# 5. BACKPROPAGATION (Slide 22 & 23)
# ==========================================

def backward_prop(X, Y, cache, W2, W1=None, activation="relu", l2_lambda=0.0):
    """
    Calculates parameter gradients using the calculus chain rule.
    
    Formulas from Slide 23:
      dZ2 = A2 - Y
      dW2 = (1/m) * np.dot(dZ2, A1.T)
      db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
      dZ1 = np.dot(W2.T, dZ2) * g'(Z1)
      dW1 = (1/m) * np.dot(dZ1, X.T)
      db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    """
    Z1, A1, Z2, A2 = cache
    m = X.shape[1]
    
    # Layer 2 gradients
    dZ2 = A2 - Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    if l2_lambda > 0.0 and W2 is not None:
        dW2 += (l2_lambda / m) * W2
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Layer 1 gradients
    if activation == "relu":
        d_act = (Z1 > 0).astype(float)
    elif activation == "tanh":
        d_act = 1.0 - np.square(A1)
    elif activation == "sigmoid":
        d_act = A1 * (1.0 - A1)
    else:
        d_act = 1.0
        
    dZ1 = np.dot(W2.T, dZ2) * d_act
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    if l2_lambda > 0.0 and W1 is not None:
        dW1 += (l2_lambda / m) * W1
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2


# ==========================================
# 6. PARAMETER UPDATE (Slide 19 & 20)
# ==========================================

def update_params(W, b, dW, db, alpha):
    """
    Applies gradient descent update rule:
      W_new = W_old - (alpha * dW)
      b_new = b_old - (alpha * db)
    """
    W_new = W - alpha * dW
    b_new = b - alpha * db
    return W_new, b_new


# ==========================================
# 7. FULL TRAINING LOOP (Slide 24 & 25)
# ==========================================

def train_neural_network(X, Y, n_h=4, epochs=1000, lr=0.5, activation="relu", loss_fn="mse", l2_lambda=0.0, verbose=True):
    """
    Complete training loop combining forward pass, loss calculation,
    backward pass, and gradient descent update.
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Initialize
    W1, b1, W2, b2 = initialize_parameters(n_x, n_h, n_y, seed=42)
    history = {"loss": [], "epoch": []}
    
    for i in range(1, epochs + 1):
        # 1. Forward Pass
        A2, cache = forward_prop(X, W1, b1, W2, b2, activation=activation)
        
        # 2. Compute Loss
        if loss_fn == "mse":
            loss = mse_loss(A2, Y)
        else:
            loss = binary_cross_entropy(A2, Y)
            
        history["loss"].append(loss)
        history["epoch"].append(i)
        
        # 3. Backward Pass
        dW1, db1, dW2, db2 = backward_prop(X, Y, cache, W2, W1=W1, activation=activation, l2_lambda=l2_lambda)
        
        # 4. Update Weights
        W1, b1 = update_params(W1, b1, dW1, db1, lr)
        W2, b2 = update_params(W2, b2, dW2, db2, lr)
        
        if verbose and (i % 100 == 0 or i == 1 or i == epochs):
            predictions = (A2 > 0.5).astype(int)
            acc = np.mean(predictions == Y) * 100
            print(f"Epoch {i:4d}/{epochs} | Loss: {loss:.6f} | Accuracy: {acc:.1f}%")
            
    return (W1, b1, W2, b2), history


# ==========================================
# 8. DATASETS & VISUALIZATION (Lab 2 XOR Slide 28)
# ==========================================

def get_xor_dataset():
    """Non-linear XOR logic gate dataset from Slide 28."""
    X = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ], dtype=float)
    Y = np.array([
        [0, 1, 1, 0]
    ], dtype=float)
    return X, Y

def get_circle_dataset(m=200, noise=0.05, seed=42):
    """2D concentric circle dataset for visual non-linear classification."""
    np.random.seed(seed)
    n = m // 2
    # Inner circle (Class 1)
    r1 = np.random.uniform(0.0, 0.45, n)
    theta1 = np.random.uniform(0, 2 * np.pi, n)
    X1 = np.vstack([r1 * np.cos(theta1), r1 * np.sin(theta1)]) + np.random.randn(2, n) * noise
    Y1 = np.ones((1, n))
    
    # Outer circle (Class 0)
    r2 = np.random.uniform(0.65, 1.0, n)
    theta2 = np.random.uniform(0, 2 * np.pi, n)
    X2 = np.vstack([r2 * np.cos(theta2), r2 * np.sin(theta2)]) + np.random.randn(2, n) * noise
    Y2 = np.zeros((1, n))
    
    X = np.hstack([X1, X2])
    Y = np.hstack([Y1, Y2])
    return X, Y

def plot_decision_boundary(X, Y, params, activation="relu", title="Decision Boundary"):
    """Generates decision boundary contour plot in 2D space."""
    W1, b1, W2, b2 = params
    x_min, x_max = X[0, :].min() - 0.2, X[0, :].max() + 0.2
    y_min, y_max = X[1, :].min() - 0.2, X[1, :].max() + 0.2
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    
    preds, _ = forward_prop(grid_points, W1, b1, W2, b2, activation=activation)
    preds = preds.reshape(xx.shape)
    
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds, levels=25, cmap="RdBu", alpha=0.7)
    plt.colorbar(label="Predicted Probability P(Y=1)")
    
    # Plot data points
    plt.scatter(X[0, Y[0] == 0], X[1, Y[0] == 0], c="crimson", marker="s", edgecolors="k", s=80, label="Class 0")
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], c="dodgerblue", marker="o", edgecolors="k", s=80, label="Class 1")
    plt.contour(xx, yy, preds, levels=[0.5], colors="black", linewidths=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Feature x1", fontsize=11)
    plt.ylabel("Feature x2", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    print("Saved decision boundary plot to decision_boundary.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  NEURAL NETWORKS FROM SCRATCH - VERIFICATION RUN")
    print("=" * 60)
    
    # 1. Run Lab 2: XOR Problem
    print("\n--- Running Lab 2: XOR Gate (Slide 28) ---")
    X_xor, Y_xor = get_xor_dataset()
    print("Input X:\n", X_xor)
    print("Target Y:\n", Y_xor)
    
    params_xor, hist_xor = train_neural_network(
        X_xor, Y_xor, 
        n_h=4, 
        epochs=1000, 
        lr=0.8, 
        activation="relu", 
        loss_fn="mse",
        verbose=True
    )
    
    # Verify final predictions on XOR
    W1, b1, W2, b2 = params_xor
    A2_final, _ = forward_prop(X_xor, W1, b1, W2, b2, activation="relu")
    print("\nXOR Ground Truth: ", Y_xor[0])
    print("Model Output:      ", np.round(A2_final[0], 4))
    print("Class Predictions: ", (A2_final[0] > 0.5).astype(int))
    
    # 2. Run Circle Dataset with plot
    print("\n--- Running 2D Circle Dataset with Visualization ---")
    X_circ, Y_circ = get_circle_dataset(m=200, noise=0.04)
    params_circ, hist_circ = train_neural_network(
        X_circ, Y_circ,
        n_h=6,
        epochs=1200,
        lr=0.5,
        activation="tanh",
        loss_fn="mse",
        verbose=False
    )
    plot_decision_boundary(X_circ, Y_circ, params_circ, activation="tanh", title="Neural Network Decision Boundary (Circle)")
    print("\nTraining verification complete! Loss reduced successfully.")
