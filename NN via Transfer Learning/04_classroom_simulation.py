"""
===============================================================================
Classroom Visual Simulation: Neural Networks via Transfer Learning
===============================================================================
Generates a 4-panel visual dashboard (`transfer_learning_simulation.png`)
illustrating the core concepts of Transfer Learning:
  1. Validation Accuracy over Epochs (Scratch vs Feature Extraction
     vs Fine-Tuning)
  2. The Danger: Catastrophic Forgetting (Effect of Learning Rate & Warm-up
     Phase on pre-trained knowledge)
  3. What Exactly is Transferred? (Actual 2D Convolution Simulation showing
     Early Layer Edges -> Middle Layer Shapes/Textures -> Deep Features)
  4. The Decision Matrix & Sample Efficiency (Validation Accuracy vs Target
     Dataset Size)
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def conv2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution in pure NumPy to demonstrate CNN feature extraction."""
    kh, kw = kernel.shape
    h, w = image.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i : i + kh, j : j + kw] * kernel)
    return out


def generate_classroom_plots(output_path: str = "transfer_learning_simulation.png"):
    np.random.seed(42)
    # Clean, high-contrast academic presentation color scheme (off-white & navy/teal/crimson)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
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

    fig = plt.figure(figsize=(15, 10), dpi=160)
    fig.suptitle(
        "Neural Networks via Transfer Learning — Classroom Simulation Suite",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )

    # -------------------------------------------------------------------------
    # PANEL 1: Validation Accuracy over 100 Epochs
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = np.arange(1, 101)

    # Training from Scratch
    acc_scratch = 12 + 74 * (1 - np.exp(-epochs / 38.0))
    # Feature Extraction (Lab 1 - Frozen Base)
    acc_feat_ext = 68 + 21 * (1 - np.exp(-epochs / 10.0))
    # Fine-Tuning after Warm-up at Epoch 20 (Lab 2)
    acc_finetune = np.where(
        epochs <= 20,
        acc_feat_ext,
        acc_feat_ext + 8.5 * (1 - np.exp(-(epochs - 20) / 12.0)),
    )

    ax1.plot(epochs, acc_finetune, "-", color="#1d4ed8", linewidth=2.5,
             label="Transfer Learning + Fine-Tuning (Lab 2)")
    ax1.plot(epochs, acc_feat_ext, "-", color="#0d9488", linewidth=2.0,
             label="Feature Extraction Only (Lab 1: Frozen Base)")
    ax1.plot(epochs, acc_scratch, "--", color="#64748b", linewidth=2.2,
             label="Training from Scratch (Random Init)")

    ax1.axvline(x=20, color="#d97706", linestyle=":", linewidth=1.8)
    ax1.text(22, 32, "Epoch 20: Unfreeze Top Layers\n(After Warm-up Phase)", color="#b45309", fontsize=8.5, fontweight="bold")

    ax1.set_title("1. Validation Accuracy over 100 Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xlim(1, 100)
    ax1.grid(True)
    ax1.legend(loc="lower right", facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL 2: Catastrophic Forgetting Simulation (100 Epochs)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ft_epochs = np.arange(0, 101)

    # Micro LR (1e-5) + Gradual Unfreezing + Weight Decay
    good_ft = 86 + 10.5 * (1 - np.exp(-ft_epochs / 18.0))
    # Moderate LR (1e-3) without warm-up -> initial dip then partial recovery
    med_ft = np.concatenate([[86], 62 + 24 * (1 - np.exp(-ft_epochs[1:] / 24.0))])
    # Aggressive LR (1e-2) -> Catastrophic Forgetting! Overwrites pre-trained base
    bad_ft = np.concatenate([[86], 38 + 20 * (1 - np.exp(-ft_epochs[1:] / 42.0))])

    ax2.plot(ft_epochs, good_ft, "-", color="#0f766e", linewidth=2.5,
             label="Best Practice: Micro LR (1e-5) + Warm-up + L2")
    ax2.plot(ft_epochs, med_ft, "-", color="#d97706", linewidth=2.0,
             label="Risky: Medium LR (1e-3) Without Warm-up (Gradient Shock)")
    ax2.plot(ft_epochs, bad_ft, "-", color="#dc2626", linewidth=2.5,
             label="Catastrophic Forgetting: Large LR (1e-2) Overwrites Base")

    ax2.annotate(
        "Pre-trained weights destroyed!\nDegrades to scratch performance",
        xy=(8, bad_ft[8]),
        xytext=(24, 43),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        color="#b91c1c",
        fontsize=8.5,
        fontweight="bold",
    )

    ax2.set_title("2. The Danger: Catastrophic Forgetting (100 Epochs)")
    ax2.set_xlabel("Fine-Tuning Epochs (After Unfreezing Base)")
    ax2.set_ylabel("Target Validation Accuracy (%)")
    ax2.set_ylim(30, 102)
    ax2.set_xlim(0, 100)
    ax2.grid(True)
    ax2.legend(loc="lower right", facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL 3: Sample Efficiency & Decision Matrix
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3)
    dataset_sizes = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000])

    acc_by_size_feat = 78 + 12 * np.log10(dataset_sizes / 100) / np.log10(250)
    acc_by_size_ft = np.array([73, 81, 88, 92.5, 95.0, 96.5, 97.2, 97.8])
    acc_by_size_scratch = np.array([28, 36, 45, 58, 72, 81, 88, 93.0])

    ax3.semilogx(dataset_sizes, acc_by_size_feat, "-s", color="#0d9488", linewidth=2.2,
                 label="1. Feature Extraction (Frozen Base — Best for Small Data)")
    ax3.semilogx(dataset_sizes, acc_by_size_ft, "-o", color="#1d4ed8", linewidth=2.5,
                 label="2. Fine-Tuning (Unfreeze Top — Best for Larger Data)")
    ax3.semilogx(dataset_sizes, acc_by_size_scratch, "--^", color="#64748b", linewidth=2.0,
                 label="3. Training from Scratch (Overfits on Small Data)")

    ax3.axvspan(100, 600, color="#0d9488", alpha=0.10, label="Small Data Zone -> Feature Extraction")
    ax3.axvspan(600, 25000, color="#1d4ed8", alpha=0.06, label="Medium/Large Data Zone -> Fine-Tuning")

    ax3.set_title("3. Decision Matrix: Dataset Size vs Strategy")
    ax3.set_xlabel("Number of Labeled Target Images (Log Scale)")
    ax3.set_ylabel("Final Test Accuracy (%)")
    ax3.set_ylim(20, 102)
    ax3.grid(True)
    ax3.legend(loc="lower right", facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.0)

    # -------------------------------------------------------------------------
    # PANEL 4: What Exactly is Transferred? Hierarchical Features
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("4. What is Transferred? Hierarchical CNN Feature Maps", pad=12)

    # Create a synthetic 32x32 image (e.g. a geometric "vehicle/object" icon)
    H, W = 32, 32
    img = np.zeros((H, W))
    img[10:22, 6:26] = 0.85               # Main body rectangle
    img[6:12, 11:21] = 0.70               # Top cabin
    yy, xx = np.ogrid[:H, :W]
    wheel1 = (yy - 22) ** 2 + (xx - 11) ** 2 <= 3.2**2
    wheel2 = (yy - 22) ** 2 + (xx - 21) ** 2 <= 3.2**2
    img[wheel1 | wheel2] = 1.0            # Two circular wheels

    # Early Layer: Sobel Edge Detector (Generic low-level edges - HIGHLY transferable)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T
    early_edges = np.hypot(conv2d_simple(img, sobel_x), conv2d_simple(img, sobel_y))

    # Middle Layer: Corner / Texture & Ring pattern detector (ReLU(Conv(Early)))
    ring_kernel = np.array([[0, 1, 0], [1, -2, 1], [0, 1, 0]], dtype=float)
    middle_shapes = np.maximum(0, conv2d_simple(early_edges, ring_kernel))

    # Deep Layer: Task-Specific Detector (e.g., activates strongly on wheel/cabin motifs)
    deep_map = np.zeros_like(middle_shapes)
    deep_map[16:25, 6:24] = middle_shapes[16:25, 6:24] * 1.5
    deep_map = np.power(deep_map / (np.max(deep_map) + 1e-6), 1.5)

    # Plot the 4 mini feature maps inside Panel 4
    sub_maps = [
        (img, "Input Image\n(Raw Pixels)", "Blues"),
        (early_edges, "Early Layers\nEdges & Gradients\n(Keep Frozen)", "YlGnBu"),
        (middle_shapes, "Middle Layers\nShapes & Textures\n(Usually Frozen)", "PuBuGn"),
        (deep_map, "Deep Layers\nTask-Specific Parts\n(Fine-Tune)", "OrRd"),
    ]
    bbox = ax4.get_position()
    sub_w = bbox.width / 4.3
    for idx, (fmap, title, cmap) in enumerate(sub_maps):
        left = bbox.x0 + idx * (bbox.width / 4.0) + 0.005
        sub_ax = fig.add_axes([left, bbox.y0 + 0.04, sub_w * 0.88, bbox.height * 0.72])
        sub_ax.imshow(fmap, cmap=cmap)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for spine in sub_ax.spines.values():
            spine.set_edgecolor("#94a3b8")
            spine.set_linewidth(1.2)
        sub_ax.set_title(title, fontsize=8.5, color="#1e293b", pad=6)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved 4-panel classroom simulation plot to: {output_path}")


if __name__ == "__main__":
    generate_classroom_plots()
