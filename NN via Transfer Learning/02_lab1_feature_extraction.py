"""
===============================================================================
Lab 1: Image Classification via Feature Extraction
===============================================================================
Core Concepts:
  - Freeze all weights of the pre-trained convolutional base:
        base_model.trainable = False   (or layer.trainable = False)
  - Replace the final classification head with a new Dense layer for the
    Target Domain.
  - Best when: Target dataset is small and similar to Source Domain.

Includes:
  1. Clean TensorFlow / Keras implementation
  2. Clean PyTorch equivalent snippet
  3. Instant classroom simulation so you can run this script live in class!
===============================================================================
"""

import numpy as np

# -----------------------------------------------------------------------------
# 1. TENSORFLOW / KERAS SNIPPET (layer.trainable = False)
# -----------------------------------------------------------------------------
KERAS_FEATURE_EXTRACTION_CODE = """
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 1: Load Pre-trained Base Model (Source Domain = ImageNet, 1.2M+ images)
# include_top=False strips away the original 1000-class ImageNet classification head
base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Step 2: FREEZE the Base Model (layer.trainable = False)
# Prevents weights from updating so we don't overfit on our small Target Dataset
base_model.trainable = False

# Step 3: Attach a new Classification Head for our Target Domain (e.g., 5 classes)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")  # Only these new layers are trained!
])

# Step 4: Compile & Train fast with minimal VRAM
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# history = model.fit(train_ds, validation_data=val_ds, epochs=10)
"""

# -----------------------------------------------------------------------------
# 2. PYTORCH EQUIVALENT SNIPPET
# -----------------------------------------------------------------------------
PYTORCH_FEATURE_EXTRACTION_CODE = """
import torch.nn as nn
from torchvision import models

# Step 1: Load Pre-trained ResNet50 Base Model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Step 2: Freeze all layers in the Base Model
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace the final fully-connected layer for our Target Domain (5 classes)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 5)  # requires_grad=True by default for newly created layers
)
"""


def run_live_classroom_demo():
    """
    Simulates Lab 1 Feature Extraction in pure NumPy so students can watch how
    freezing a rich feature extractor lets a tiny linear head converge in seconds
    on a small dataset (where training from scratch overfits).
    """
    print("=" * 76)
    print("LAB 1: IMAGE CLASSIFICATION VIA FEATURE EXTRACTION")
    print("=" * 76)
    print("\n--- TensorFlow / Keras Reference Code ---")
    print(KERAS_FEATURE_EXTRACTION_CODE.strip())

    print("\n" + "=" * 76)
    print("LIVE CLASSROOM SIMULATION: Small Target Dataset (only 60 labeled images)")
    print("=" * 76)

    np.random.seed(7)
    n_train, n_val = 60, 300
    n_raw_pixels = 784        # 28x28 raw image input (high-dimensional, noisy)
    n_extracted_features = 8  # Compact semantic feature vector from frozen Base Model
    n_classes = 3

    # Ground-truth underlying visual factors in the data (non-linear in raw pixels)
    true_factors_train = np.random.randn(n_train, n_extracted_features)
    true_factors_val = np.random.randn(n_val, n_extracted_features)

    true_W = np.random.randn(n_extracted_features, n_classes)
    y_train_idx = np.argmax(true_factors_train @ true_W, axis=1)
    y_val_idx = np.argmax(true_factors_val @ true_W, axis=1)
    Y_train = np.eye(n_classes)[y_train_idx]

    # Raw pixels contain weak signal + high background clutter (784 dims vs 60 samples -> overfits!)
    projection = np.random.randn(n_extracted_features, n_raw_pixels) * 0.15
    X_raw_train = true_factors_train @ projection + np.random.randn(n_train, n_raw_pixels) * 1.2
    X_raw_val = true_factors_val @ projection + np.random.randn(n_val, n_raw_pixels) * 1.2

    # Frozen pre-trained Base Model extracts clean features from raw pixels
    # (Simulates Early + Middle CNN layers already knowing edges, shapes, textures)
    frozen_base_features_train = true_factors_train + np.random.randn(n_train, n_extracted_features) * 0.15
    frozen_base_features_val = true_factors_val + np.random.randn(n_val, n_extracted_features) * 0.15

    # Train Model A (From Scratch on Raw Pixels) vs Model B (Feature Extraction on Frozen Base)
    W_scratch = np.random.randn(n_raw_pixels, n_classes) * 0.01
    W_transfer = np.random.randn(n_extracted_features, n_classes) * 0.01

    def softmax(logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    print(f"\n{'Epoch':>5} | {'Scratch Train Acc':>18} | {'Scratch Val Acc':>16} | {'Transfer Val Acc':>17}")
    print("-" * 68)

    for epoch in range(1, 101):
        # Update Scratch Model on raw pixels
        probs_s = softmax(X_raw_train @ W_scratch)
        grad_s = X_raw_train.T @ (probs_s - Y_train) / n_train
        W_scratch -= 0.15 * grad_s

        # Update Transfer Model (ONLY the small Dense head on frozen features)
        probs_t = softmax(frozen_base_features_train @ W_transfer)
        grad_t = frozen_base_features_train.T @ (probs_t - Y_train) / n_train + 0.05 * W_transfer
        W_transfer -= 0.8 * grad_t

        if epoch in (1, 5, 10, 25, 50, 100):
            scratch_train_acc = np.mean(np.argmax(X_raw_train @ W_scratch, axis=1) == y_train_idx) * 100
            scratch_val_acc = np.mean(np.argmax(X_raw_val @ W_scratch, axis=1) == y_val_idx) * 100
            transfer_val_acc = np.mean(np.argmax(frozen_base_features_val @ W_transfer, axis=1) == y_val_idx) * 100
            print(
                f"{epoch:5d} | {scratch_train_acc:17.1f}% | {scratch_val_acc:15.1f}% | {transfer_val_acc:16.1f}%"
            )

    print("-" * 68)
    print("Key Observations:")
    print("  1. Training from scratch memorizes the 60 training images (100% train acc)")
    print("     but fails to generalize on validation data (severe overfitting!).")
    print("  2. Feature Extraction (`layer.trainable = False`) prevents the base from")
    print("     memorizing noise and achieves high validation accuracy in just a few epochs.")
    print()


if __name__ == "__main__":
    run_live_classroom_demo()
