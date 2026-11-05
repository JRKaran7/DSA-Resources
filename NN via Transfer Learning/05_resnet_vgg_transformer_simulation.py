"""
===============================================================================
Titans of Transfer Learning Simulation: VGG vs. ResNet vs. Transformers
===============================================================================
Demonstrates the core architectural differences in a simple, intuitive way:
  1. VGG (Uniform 3x3 Convolutions):
     - Straight layer-by-layer chain: y = f(W * x).
     - Great for textures & style, but gradients vanish when stacked too deep.
  2. ResNet (Residual Networks with Skip Connections):
     - Adds a shortcut "skip connection": y = f(W * x) + x.
     - Solves the vanishing gradient problem so networks can go 50-150+ layers deep.
  3. Transformers (Self-Attention for NLP & Context):
     - Instead of scanning local 3x3 pixel windows, computes an Attention Matrix
       connecting every token/word directly to every other token in one step.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_vgg_vs_resnet_gradients():
    """
    Shows why a plain deep network (VGG-style chain) loses its gradient signal
    as layers increase, whereas ResNet's skip connection (+ x) keeps the signal
    strong even at 50-100 layers.
    """
    print("=" * 76)
    print("SIMULATION 1: Why ResNet Needed Skip Connections (VGG vs. ResNet Depth)")
    print("=" * 76)
    print("In a plain chain (VGG), each layer multiplies the backward gradient by a")
    print("weight factor (< 1.0). In ResNet, the shortcut (y = F(x) + x) adds 1.0 to")
    print("each block's gradient path so the learning signal never vanishes!\n")

    depths = [5, 10, 20, 30, 50, 80, 100]
    layer_factor = 0.88  # Typical per-layer contraction factor during backprop

    print(f"{'Network Depth':>15} | {'VGG Signal Remaining':>22} | {'ResNet Signal Remaining':>25}")
    print("-" * 68)
    for d in depths:
        vgg_signal = (layer_factor ** d) * 100.0
        # ResNet skip connections preserve the identity highway across blocks
        resnet_signal = (0.992 ** d) * 100.0
        print(f"{d:9d} Layers | {vgg_signal:20.2f}% | {resnet_signal:23.2f}%")

    print("-" * 68)
    print("Takeaway:")
    print("  - By 50-100 layers, a plain VGG-style chain loses >99% of its gradient signal.")
    print("  - ResNet's skip connections act like an express highway, keeping >90% signal!\n")


def simulate_transformer_self_attention():
    """
    Demonstrates how a Transformer uses Self-Attention to connect related words
    across a sentence regardless of distance (unlike a local 3x3 CNN filter).
    """
    print("=" * 76)
    print("SIMULATION 2: How Transformers Understand Context (Self-Attention)")
    print("=" * 76)

    words = ["The", "trophy", "did", "not", "fit", "the", "suitcase", "because", "it", "was", "big"]
    print("Sentence: 'The trophy did not fit the suitcase because it was big'")
    print("Question: What does the word 'it' refer to?\n")

    # Simple 4D semantic embeddings for each word:
    # [is_object, size_large, size_small, pronoun_query]
    embeddings = {
        "The":      np.array([0.1, 0.0, 0.0, 0.0]),
        "trophy":   np.array([1.0, 0.9, 0.1, 0.8]),
        "did":      np.array([0.0, 0.0, 0.0, 0.0]),
        "not":      np.array([0.0, 0.0, 0.0, 0.0]),
        "fit":      np.array([0.2, 0.4, 0.4, 0.1]),
        "the":      np.array([0.1, 0.0, 0.0, 0.0]),
        "suitcase": np.array([1.0, 0.3, 0.8, 0.3]),
        "because":  np.array([0.0, 0.0, 0.0, 0.0]),
        "it":       np.array([0.9, 0.8, 0.1, 0.9]),
        "was":      np.array([0.0, 0.0, 0.0, 0.0]),
        "big":      np.array([0.3, 1.0, 0.0, 0.7]),
    }

    query_it = embeddings["it"]
    scores = np.array([np.dot(query_it, embeddings[w]) * 2.2 for w in words])
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))

    print("Transformer Self-Attention Weights for the word 'it':")
    for w, weight in zip(words, attention_weights):
        bar = "#" * int(weight * 40)
        print(f"  {w:<10} | {weight * 100:5.1f}% | {bar}")

    print("\nTakeaway:")
    print("  - A CNN (VGG/ResNet) looks at neighbouring pixels through a small 3x3 window.")
    print("  - A Transformer compares 'it' with every word in the sentence at once,")
    print("    instantly linking 'it' -> 'trophy' and 'big'!\n")


def generate_titans_plot(output_path: str = "titans_comparison_simulation.png"):
    """
    Generates a clean 3-panel visual comparison of VGG, ResNet, and Transformers.
    """
    np.random.seed(42)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#1e293b",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": "#0f172a",
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.9,
    })

    fig = plt.figure(figsize=(15, 5.2), dpi=160)
    fig.suptitle(
        "Titans of Transfer Learning — VGG vs. ResNet vs. Transformers",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # -------------------------------------------------------------------------
    # PANEL 1: Vanishing Gradient vs. Network Depth (VGG vs. ResNet)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1)
    depths = np.arange(1, 101)
    vgg_grad = (0.91 ** depths) * 100.0
    resnet_grad = (0.993 ** depths) * 100.0

    ax1.plot(depths, resnet_grad, "-", color="#0d9488", linewidth=2.6,
             label="ResNet (With Skip Connections: y = F(x) + x)")
    ax1.plot(depths, vgg_grad, "--", color="#dc2626", linewidth=2.4,
             label="VGG-Style Plain Chain (No Skip Connections)")

    ax1.axvspan(16, 19, color="#d97706", alpha=0.18, label="VGG16 / VGG19 Depth Limit")
    ax1.annotate(
        "Vanishing Gradient:\nSignal drops near 0%\npast 40+ layers",
        xy=(45, vgg_grad[44]),
        xytext=(38, 22),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        color="#b91c1c",
        fontsize=8.5,
        fontweight="bold",
    )

    ax1.set_title("1. Signal Strength vs. Depth (VGG vs ResNet)")
    ax1.set_xlabel("Number of Network Layers")
    ax1.set_ylabel("Gradient Learning Signal (%)")
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 105)
    ax1.grid(True)
    ax1.legend(loc="upper right", facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.0)

    # -------------------------------------------------------------------------
    # PANEL 2: Fine-Tuning Accuracy Over 100 Epochs (Vision vs. Language)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2)
    epochs = np.arange(1, 101)

    acc_resnet = 70 + 25.5 * (1 - np.exp(-epochs / 14.0))
    acc_vgg = 66 + 24.0 * (1 - np.exp(-epochs / 20.0))
    acc_transformer = 74 + 23.5 * (1 - np.exp(-epochs / 11.0))

    ax2.plot(epochs, acc_transformer, "-", color="#1d4ed8", linewidth=2.5,
             label="Transformer (BERT/ViT — Global Self-Attention)")
    ax2.plot(epochs, acc_resnet, "-", color="#0d9488", linewidth=2.4,
             label="ResNet50 (Skip Connections — Fast & Deep)")
    ax2.plot(epochs, acc_vgg, "--", color="#d97706", linewidth=2.2,
             label="VGG16 (3x3 Convolutions — Heavier & Slower)")

    ax2.set_title("2. Transfer Learning Convergence (100 Epochs)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Target Validation Accuracy (%)")
    ax2.set_xlim(1, 100)
    ax2.set_ylim(55, 102)
    ax2.grid(True)
    ax2.legend(loc="lower right", facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.0)

    # -------------------------------------------------------------------------
    # PANEL 3: Local Receptive Field (CNN) vs Global Attention (Transformer)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    tokens = ["The", "trophy", "didn't", "fit", "because", "it", "was", "big"]
    n_tok = len(tokens)

    # Construct an illustrative attention matrix
    att_matrix = np.eye(n_tok) * 0.35
    # "it" (index 5) attends strongly to "trophy" (1) and "big" (7)
    att_matrix[5, 1] = 0.85
    att_matrix[1, 5] = 0.85
    att_matrix[5, 7] = 0.75
    att_matrix[7, 5] = 0.75
    att_matrix[1, 7] = 0.65
    att_matrix[7, 1] = 0.65
    att_matrix[3, 1] = 0.50

    im = ax3.imshow(att_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax3.set_xticks(np.arange(n_tok))
    ax3.set_yticks(np.arange(n_tok))
    ax3.set_xticklabels(tokens, rotation=35, ha="right", fontsize=8.5)
    ax3.set_yticklabels(tokens, fontsize=8.5)
    ax3.grid(False)
    ax3.set_title("3. Transformer Self-Attention Map (Global Context)")
    ax3.set_xlabel("Words Attended To (Keys)")
    ax3.set_ylabel("Query Word")

    # Highlight the "it" -> "trophy" connection
    ax3.text(1, 5, "0.85", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=8.5)
    ax3.text(7, 5, "0.75", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=8.5)

    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 3-panel Titans comparison plot to: {output_path}")


if __name__ == "__main__":
    simulate_vgg_vs_resnet_gradients()
    simulate_transformer_self_attention()
    generate_titans_plot()
