"""
Complete Computer Vision Coding Practice (All Modules Combined on Real Photos)
==============================================================================
Runs all 4 progressive Computer Vision practice modules using real photographs
in sample_images/ (fruits.jpg, building.jpg, butterfly.jpg):
  Module 1: Pixels, 2D Grayscale Matrices, and 3D RGB Tensors
  Module 2: Image Enhancement, 2D Convolution Filters, Edge Detection & Contours
  Module 3: Baseline Convolutional Neural Network (CNN) Image Classifier
  Module 4: Modern Vision Applications (ViT Patches, Edge Quantization & Privacy Blur)
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_DIR = Path(__file__).parent / "sample_images"


# =====================================================================
# MODULE 1: PIXELS, MATRICES & RGB TENSORS
# =====================================================================

def load_real_image(path: Path) -> np.ndarray:
    """Loads a real photograph from disk and converts it from BGR to RGB."""
    bgr_image = cv2.imread(str(path))
    if bgr_image is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def run_module_1_pixels_and_matrices():
    print("\n" + "=" * 65)
    print("MODULE 1: Pixels, 2D Grayscale Matrices & 3D RGB Tensors (Real Photo)")
    print("=" * 65)

    img_path = SAMPLE_DIR / "fruits.jpg"
    rgb_tensor = load_real_image(img_path)
    print(f"Loaded Real Photo: {img_path.name} -> Shape (H, W, C): {rgb_tensor.shape}")
    print(f"Pixel [R, G, B] at center (240, 256): {rgb_tensor[240, 256]}")

    grayscale_matrix = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2GRAY)
    print(f"Grayscale Matrix Shape (H, W): {grayscale_matrix.shape}")
    print("5x5 Grayscale Matrix Slice around center (238:243, 254:259):")
    print(grayscale_matrix[238:243, 254:259])

    brightened_matrix = np.clip(grayscale_matrix.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    print(f"Original center intensity: {grayscale_matrix[240, 256]} -> Brightened: {brightened_matrix[240, 256]}")


# =====================================================================
# MODULE 2: CONVOLUTION FILTERS, EDGE DETECTION & CONTOUR DETECTION
# =====================================================================

def apply_custom_convolution(gray_image: np.ndarray) -> np.ndarray:
    """Applies a 3x3 Sobel vertical edge detection kernel via 2D convolution."""
    sobel_vertical_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    filtered = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=sobel_vertical_kernel)
    return cv2.convertScaleAbs(filtered)


def run_module_2_filters_and_tracking():
    print("\n" + "=" * 65)
    print("MODULE 2: Convolution Filters, Edge Detection & Object Detection")
    print("=" * 65)

    building_gray = cv2.imread(str(SAMPLE_DIR / "building.jpg"), cv2.IMREAD_GRAYSCALE)
    conv_edges = apply_custom_convolution(building_gray)
    denoised = cv2.medianBlur(building_gray, ksize=3)
    equalized = cv2.equalizeHist(denoised)
    edges = cv2.Canny(equalized, threshold1=80, threshold2=180)

    print(f"Convolved 'building.jpg' Output Shape: {conv_edges.shape}")
    print(f"Detected Edge Pixels (Canny): {np.count_nonzero(edges)}")

    butterfly_bgr = cv2.imread(str(SAMPLE_DIR / "butterfly.jpg"))
    butterfly_gray = cv2.cvtColor(butterfly_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(cv2.Canny(butterfly_gray, 100, 200), np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Top detected object bounding boxes in 'butterfly.jpg':")
    for idx, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:3], start=1):
        if cv2.contourArea(contour) > 1500:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"  Region {idx}: Bounding Box (x={x}, y={y}, w={w}, h={h})")


# =====================================================================
# MODULE 3: BASELINE CONVOLUTIONAL NEURAL NETWORK (CNN) CLASSIFIER
# =====================================================================

class BaselineCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc2(F.relu(self.fc1(x)))


def run_module_3_baseline_cnn():
    print("\n" + "=" * 65)
    print("MODULE 3: Baseline CNN Classifier on Real Photos")
    print("=" * 65)

    torch.manual_seed(42)
    filenames = ["fruits.jpg", "building.jpg", "butterfly.jpg"]
    class_names = ["Fruits", "Building", "Butterfly"]
    tensors = []
    for fname in filenames:
        bgr = cv2.imread(str(SAMPLE_DIR / fname))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        tensors.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)

    real_images = torch.stack(tensors)
    labels = torch.tensor([0, 1, 2])

    model = BaselineCNNClassifier(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        loss = criterion(model(real_images), labels)
        loss.backward()
        optimizer.step()

    print(f"Training Loss after 15 steps on real photos: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(real_images[:1]), dim=1).squeeze().tolist()
        for cls_name, prob in zip(class_names, probabilities):
            print(f"  Predicted probability for '{cls_name}' on fruits.jpg: {prob * 100:.1f}%")
    return model


# =====================================================================
# MODULE 4: MODERN VISION (ViT PATCHES, EDGE QUANTIZATION & PRIVACY)
# =====================================================================

def vit_patch_tokenizer(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Divides an image of shape (H, W, C) into non-overlapping patches."""
    h, w, c = image.shape
    patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    return patches.swapaxes(1, 2).reshape(-1, patch_size * patch_size * c)


def quantize_weights_to_int8(weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Demonstrates symmetric INT8 weight quantization [-127, 127] for edge deployment."""
    scale = float(weights.abs().max() / 127.0)
    quantized_int8 = torch.clamp(torch.round(weights / scale), -127, 127).to(torch.int8)
    return quantized_int8, scale


def anonymize_sensitive_regions(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Blurs detected sensitive bounding boxes on-device."""
    output = image.copy()
    for (x, y, w, h) in boxes:
        roi = output[y:y + h, x:x + w]
        output[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (31, 31), 30)
    return output


def run_module_4_modern_cv(model: nn.Module):
    print("\n" + "=" * 65)
    print("MODULE 4: ViT Patch Tokenization, Edge Quantization & Privacy")
    print("=" * 65)

    butterfly_bgr = cv2.imread(str(SAMPLE_DIR / "butterfly.jpg"))
    butterfly_256 = cv2.resize(butterfly_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    patches = vit_patch_tokenizer(butterfly_256, patch_size=16)
    print(f"ViT Patch Tokenization on 'butterfly.jpg' (256x256): "
          f"{patches.shape[0]} patches of size {patches.shape[1]}")

    fp32_weights = model.fc1.weight.data
    q_weights, scale = quantize_weights_to_int8(fp32_weights)
    print(f"Edge Quantization Complete: FP32 ({fp32_weights.element_size() * fp32_weights.numel()} B) "
          f"-> INT8 ({q_weights.element_size() * q_weights.numel()} B), scale={scale:.5f}")

    building_bgr = cv2.imread(str(SAMPLE_DIR / "building.jpg"))
    anonymized = anonymize_sensitive_regions(building_bgr, [(105, 495, 85, 35), (60, 475, 35, 35)])
    print(f"Privacy Anonymization completed on 'building.jpg' of shape {anonymized.shape}.")


if __name__ == "__main__":
    run_module_1_pixels_and_matrices()
    run_module_2_filters_and_tracking()
    trained_cnn = run_module_3_baseline_cnn()
    run_module_4_modern_cv(trained_cnn)
