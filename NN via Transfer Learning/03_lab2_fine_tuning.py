"""
===============================================================================
Lab 2: End-to-End Fine-Tuning of a CNN & Mitigating Catastrophic Forgetting
===============================================================================
Core Concepts:
  - Fine-Tuning Best Practices:
      1. Warm-up Phase: Train new classifier head (Feature Extraction) FIRST
         before unfreezing any base layers.
      2. Top-Down Unfreezing: Unfreeze only the top few layers of the base;
         keep early/middle generic layers frozen.
      3. Micro Learning Rates: Use a tiny learning rate (e.g., 1e-5) so weight
         updates do not destroy pre-trained features.
  - The Danger: Catastrophic Forgetting
  - Mitigating Catastrophic Forgetting:
      - Gradual Unfreezing
      - Weight Decay (L2 Regularization)
      - Discriminative Learning Rates (e.g., 1e-3 for head, 1e-6 for base)
===============================================================================
"""

import numpy as np

# -----------------------------------------------------------------------------
# 1. TENSORFLOW / KERAS END-TO-END FINE-TUNING CODE
# -----------------------------------------------------------------------------
KERAS_FINE_TUNING_CODE = """
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# --- PHASE 1: WARM-UP PHASE (Feature Extraction First) ---
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze entire base during warm-up!

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    # Weight Decay (L2 Regularization) to penalize large weight changes
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
# Warm-up: Train ONLY the new head for 5 epochs until it stabilizes
# model.fit(train_ds, validation_data=val_ds, epochs=5)

# --- PHASE 2: TOP-DOWN UNFREEZING & MICRO LEARNING RATE ---
base_model.trainable = True

# Freeze all early & middle layers; only unfreeze the top 30 layers
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile with a MICRO LEARNING RATE (1e-5) + Weight Decay to prevent
# Catastrophic Forgetting
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# model.fit(train_ds, validation_data=val_ds, epochs=15)
"""

# -----------------------------------------------------------------------------
# 2. DISCRIMINATIVE LEARNING RATES SNIPPET (10^-3 head vs 10^-6 base)
# -----------------------------------------------------------------------------
DISCRIMINATIVE_LR_CODE = """
# Discriminative Learning Rates in PyTorch
# Apply different learning rates to different layers:
#   - Deeper/Early base layers: 1e-6 (preserve generic edge/shape detectors)
#   - Top base layers (layer4): 1e-5 (gently adapt high-level features)
#   - Newly added head (fc):    1e-3 (learn target classes quickly)

optimizer = torch.optim.AdamW([
    {"params": model.layer1.parameters(), "lr": 1e-6},
    {"params": model.layer2.parameters(), "lr": 1e-6},
    {"params": model.layer3.parameters(), "lr": 5e-6},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),     "lr": 1e-3},
], weight_decay=1e-4)
"""


def run_fine_tuning_vs_catastrophic_forgetting_demo():
    """
    Simulates what happens inside a pre-trained network during Lab 2 when:
      Case A (Bad): We unfreeze everything immediately with a large LR (1e-2)
                    -> Catastrophic Forgetting! Random head gradients destroy
                       the pre-trained base weights.
      Case B (Good): Warm-up Phase + Top-Down Unfreezing + Micro LR (1e-5)
                    -> Smoothly adapts high-level features to reach 96%+ accuracy.
    """
    print("=" * 78)
    print("LAB 2: END-TO-END FINE-TUNING & CATASTROPHIC FORGETTING")
    print("=" * 78)
    print("\n--- Phase 1 & 2 Keras Implementation ---")
    print(KERAS_FINE_TUNING_CODE.strip())
    print("\n--- Discriminative Learning Rates ---")
    print(DISCRIMINATIVE_LR_CODE.strip())

    print("\n" + "=" * 78)
    print("LIVE SIMULATION: Good Fine-Tuning (1e-5 + Warm-up) vs. Catastrophic Forgetting")
    print("=" * 78)

    # Simulate layer table of a 5-stage CNN during Top-Down Unfreezing
    layers_info = [
        ("Block 1 (Early: Edges & Color Blobs)",   "FROZEN (trainable=False)", "0.0 (Static)"),
        ("Block 2 (Early/Mid: Corners & Grids)",   "FROZEN (trainable=False)", "0.0 (Static)"),
        ("Block 3 (Middle: Shapes & Textures)",    "FROZEN (trainable=False)", "0.0 (Static)"),
        ("Block 4 (Deep: High-Level Parts)",       "UNFROZEN (Phase 2)",       "1e-5 (Micro LR)"),
        ("Dense Head (Target Domain Classifier)",  "TRAINABLE (Phase 1 & 2)",  "1e-3 -> 1e-5"),
    ]
    print("\nLayer-by-Layer Architecture Status (Top-Down Unfreezing):")
    print(f"  {'Layer Group':<38} | {'Status':<25} | {'Learning Rate'}")
    print("  " + "-" * 74)
    for layer_name, status, lr_str in layers_info:
        print(f"  {layer_name:<38} | {status:<25} | {lr_str}")

    # Numerical simulation of validation accuracy & base feature retention
    print("\nEpoch-by-Epoch Comparison:")
    print(
        f"{'Epoch':>5} | {'Phase':<16} | {'Best-Practice Val Acc':>21} | "
        f"{'Naive (LR=1e-2) Val Acc':>23} | {'Naive Base Retention':>20}"
    )
    print("-" * 96)

    for epoch in range(1, 101):
        phase = "1: Warm-up Head" if epoch <= 20 else "2: Fine-Tune Top"

        # Best practice curve: Warm-up reaches ~86%, then fine-tuning pushes to ~96.5%
        if epoch <= 20:
            good_acc = 68.0 + 18.0 * (1 - np.exp(-epoch / 7.0))
        else:
            good_acc = 85.0 + 11.5 * (1 - np.exp(-(epoch - 20) / 18.0))

        # Naive fine-tuning (no warm-up, aggressive LR=1e-2 on unfrozen base):
        # Immediately experiences gradient shock from randomly initialized head,
        # destroying pre-trained weights (Catastrophic Forgetting).
        if epoch == 1:
            bad_acc = 52.0
            retention = 64.0
        else:
            bad_acc = 41.0 + 18.0 * (1 - np.exp(-epoch / 38.0))
            retention = max(12.0, 64.0 * np.exp(-(epoch - 1) / 12.0))

        if epoch in (1, 10, 20, 25, 50, 75, 100):
            print(
                f"{epoch:5d} | {phase:<16} | {good_acc:20.1f}% | "
                f"{bad_acc:22.1f}% | {retention:19.1f}%"
            )

    print("-" * 96)
    print("Key Takeaway:")
    print("  - Notice Epoch 6 when Phase 2 starts: because the head was warmed up in")
    print("    Epochs 1-5 and we use a Micro LR (1e-5), accuracy jumps from 85% -> 96%.")
    print("  - Without warm-up and with a large LR (1e-2), gradients from the random")
    print("    head overwrite the pre-trained base ('Catastrophic Forgetting'), dropping")
    print("    base feature retention to ~12% and degrading into a slow scratch model!")
    print()


if __name__ == "__main__":
    run_fine_tuning_vs_catastrophic_forgetting_demo()
