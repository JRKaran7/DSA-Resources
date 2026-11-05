"""
Practice 3: Building a Baseline Convolutional Neural Network (CNN) Image Classifier
===================================================================================
Loads real photographs (fruits.jpg, building.jpg, butterfly.jpg), resizes them into
normalized 32x32 RGB tensors, and trains/evaluates a baseline CNN in PyTorch.
"""

from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_DIR = Path(__file__).parent / "sample_images"


class BaselineCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Block 1: Input (3 RGB channels, 32x32) -> 16 Feature Maps (16x16 after pooling)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16 Feature Maps (16x16) -> 32 Feature Maps (8x8 after pooling)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully Connected Classifier Head
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc2(F.relu(self.fc1(x)))


def load_real_images_as_batch() -> tuple[torch.Tensor, list[str]]:
    """Loads the 3 real sample photographs and converts them to a (3, 3, 32, 32) tensor."""
    filenames = ["fruits.jpg", "building.jpg", "butterfly.jpg"]
    tensors = []
    for fname in filenames:
        bgr = cv2.imread(str(SAMPLE_DIR / fname))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        # Convert (H, W, C) uint8 [0, 255] -> (C, H, W) float32 [0.0, 1.0]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(tensor)
    return torch.stack(tensors), ["Fruits", "Building", "Butterfly"]


def run_training_and_inference_demo():
    torch.manual_seed(42)
    real_images, class_names = load_real_images_as_batch()
    labels = torch.tensor([0, 1, 2])

    model = BaselineCNNClassifier(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train briefly on the 3 real photographs
    model.train()
    for _ in range(15):
        optimizer.zero_grad()
        logits = model(real_images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Loaded Real Photo Batch Shape (B, C, H, W): {tuple(real_images.shape)}")
    print(f"Training Loss after 15 steps: {loss.item():.4f}\n")

    # Inference on the real 'fruits.jpg' photograph
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(real_images[:1]), dim=1).squeeze().tolist()
        print("CNN Predictions for 'fruits.jpg':")
        for cls_name, prob in zip(class_names, probs):
            print(f"  {cls_name}: {prob * 100:.1f}%")


if __name__ == "__main__":
    run_training_and_inference_demo()
