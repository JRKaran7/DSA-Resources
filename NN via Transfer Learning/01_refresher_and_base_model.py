"""
===============================================================================
Part 1 - 3 Demo: Standard NN Refresher & The Cost of Training from Scratch
===============================================================================
Covers:
  1. How Standard NNs Learn (Forward Pass -> Loss -> Backprop -> Optimisation)
  2. The Cost of Training from Scratch (10M+ parameters) vs. Base Model reuse
===============================================================================
"""

import numpy as np


def part1_standard_nn_refresher():
    """
    Illustrative example of a 2-input, 2-hidden-node, 1-output Neural Network
    learning a binary classification problem from random weights.
    """
    print("=" * 72)
    print("PART 1 REFRESHER: How Standard Neural Networks Learn")
    print("=" * 72)

    # Toy dataset (4 examples, 2 input features x1, x2 -> binary target y)
    X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
    y = np.array([[0.0], [0.0], [1.0], [1.0]])

    # Randomly initialize weights & biases (Training from Scratch)
    np.random.seed(42)
    W_hidden = np.random.randn(2, 2) * 0.5  # w11, w12, w21, w22
    b_hidden = np.zeros((1, 2))
    W_out = np.random.randn(2, 1) * 0.5     # w1k, w2k
    b_out = np.zeros((1, 1))

    learning_rate = 0.5

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for epoch in range(1, 201):
        # 1. Forward Pass: Data flows through the network to generate prediction
        h_raw = X @ W_hidden + b_hidden
        h = np.maximum(0, h_raw)            # ReLU activation on hidden layer
        y_pred = sigmoid(h @ W_out + b_out) # Sigmoid activation on output layer

        # 2. Loss Calculation: Binary Cross-Entropy Loss
        eps = 1e-8
        loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

        # 3. Backpropagation: Gradients calculated backwards
        d_out = (y_pred - y) / len(X)
        dW_out = h.T @ d_out
        db_out = np.sum(d_out, axis=0, keepdims=True)

        dh = d_out @ W_out.T
        dh_raw = dh * (h_raw > 0)           # Derivative of ReLU
        dW_hidden = X.T @ dh_raw
        db_hidden = np.sum(dh_raw, axis=0, keepdims=True)

        # 4. Optimisation: Gradient Descent updates weights to minimise future loss
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out
        W_hidden -= learning_rate * dW_hidden
        b_hidden -= learning_rate * db_hidden

        if epoch in (1, 10, 50, 200):
            preds_str = ", ".join(f"{p[0]:.2f}" for p in y_pred)
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Predictions: [{preds_str}] (Target: [0, 0, 1, 1])")

    print()


def part2_cost_of_training_from_scratch():
    """
    Compare parameter counts of foundational architectures
    (ResNet50, VGG16, BERT) when trained from scratch vs. Feature Extraction.
    """
    print("=" * 72)
    print("PART 2 & 3: Why Starting Over is Expensive (10M+ Parameters)")
    print("=" * 72)

    # Real parameter counts of popular Base Models (for 5 target classes)
    num_target_classes = 5
    models = [
        ("ResNet50 (CNN - Skip Connections)", 23_587_712, 2048 * num_target_classes + num_target_classes),
        ("VGG16 (CNN - 3x3 Convolutions)",    14_714_688, 512 * num_target_classes + num_target_classes),
        ("MobileNetV2 (Lightweight CNN)",      2_257_984, 1280 * num_target_classes + num_target_classes),
        ("BERT-Base (Transformer NLP)",      109_482_240, 768 * num_target_classes + num_target_classes),
    ]

    print(f"{'Base Architecture':<34} | {'Scratch Params':>15} | {'Transfer Head':>14} | {'Reduction':>10}")
    print("-" * 82)
    for name, base_params, head_params in models:
        total_scratch = base_params + head_params
        pct_trained = (head_params / total_scratch) * 100
        print(
            f"{name:<34} | {total_scratch:>15,d} | {head_params:>14,d} | "
            f"{pct_trained:>8.3f}% trained"
        )

    print("-" * 82)
    print("Key Takeaway:")
    print("  - Training from scratch updates 10M to 100M+ parameters (needs 1.2M+ images & weeks of GPUs).")
    print("  - Feature Extraction freezes the Base Model and trains <0.5% of the parameters in minutes!")
    print()


if __name__ == "__main__":
    part1_standard_nn_refresher()
    part2_cost_of_training_from_scratch()
