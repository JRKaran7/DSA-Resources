"""
Practice 4: Modern Computer Vision Applications (Patches, Edge Quantization & Privacy)
======================================================================================
Applies modern vision operations directly on real photographs:
1. Vision Transformer (ViT) Patch Tokenization on 'butterfly.jpg'.
2. Edge Inference Optimization: Symmetric INT8 weight quantization.
3. Privacy Engineering: On-device blurring of sensitive regions (car/sign in 'building.jpg').
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

SAMPLE_DIR = Path(__file__).parent / "sample_images"


def vit_patch_tokenizer(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Divides a real RGB image of shape (H, W, C) into non-overlapping patches for Vision Transformers.
    """
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch_size."

    patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    patches = patches.swapaxes(1, 2).reshape(-1, patch_size * patch_size * c)
    return patches


def quantize_weights_to_int8(weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Demonstrates symmetric INT8 weight quantization [-127, 127] so models use
    4x less memory and run faster on edge devices (cameras, drones, sensors).
    """
    scale = float(weights.abs().max() / 127.0)
    quantized_int8 = torch.clamp(torch.round(weights / scale), -127, 127).to(torch.int8)
    return quantized_int8, scale


def anonymize_sensitive_regions(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Privacy-by-design helper: blurs sensitive regions (e.g., parked vehicles, signs, faces)
    on-device before storing or transmitting video frames.
    Each box is (x, y, width, height).
    """
    output = image.copy()
    for (x, y, w, h) in boxes:
        roi = output[y:y + h, x:x + w]
        output[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (31, 31), 30)
    return output


if __name__ == "__main__":
    # 1. Vision Transformer Patch Extraction on real 'butterfly.jpg' resized to 256x256
    butterfly_bgr = cv2.imread(str(SAMPLE_DIR / "butterfly.jpg"))
    butterfly_256 = cv2.resize(butterfly_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    patches = vit_patch_tokenizer(butterfly_256, patch_size=16)
    print(f"ViT Patch Tokenization on 'butterfly.jpg' (256x256): "
          f"{patches.shape[0]} patches of dimension {patches.shape[1]}")

    # 2. Edge INT8 Quantization Demo
    sample_layer = nn.Linear(256, 64)
    q_weights, scale = quantize_weights_to_int8(sample_layer.weight.data)
    print(f"Edge Quantization: FP32 ({sample_layer.weight.element_size() * sample_layer.weight.numel()} bytes) "
          f"-> INT8 ({q_weights.element_size() * q_weights.numel()} bytes), scale={scale:.5f}")

    # 3. Privacy Anonymization Demo on real 'building.jpg' (blurring parked car & street sign)
    building_bgr = cv2.imread(str(SAMPLE_DIR / "building.jpg"))
    sensitive_boxes = [
        (105, 495, 85, 35),  # Parked car at bottom-left of building.jpg
        (60, 475, 35, 35),   # Street sign on left of building.jpg
    ]
    anonymized_img = anonymize_sensitive_regions(building_bgr, sensitive_boxes)
    out_path = SAMPLE_DIR / "output_04_anonymized_building.jpg"
    cv2.imwrite(str(out_path), anonymized_img)
    print(f"Privacy Anonymization completed on {len(sensitive_boxes)} regions in 'building.jpg' -> {out_path.name}")
