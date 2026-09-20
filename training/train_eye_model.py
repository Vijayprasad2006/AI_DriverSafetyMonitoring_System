"""
DriveGuard AI - PyTorch Eye State Model Training Pipeline
Trains EyeStateCNN on eye patches (MRL Eye Dataset, CEW, or generated test splits).
Includes data augmentation, validation splits, and model weight persistence.
"""

import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.eye_classifier import EyeStateCNN

class EyePatchDataset(Dataset):
    """Loads eye patch images organized in class subfolders (Closed / Open)."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Supported folder structure: root_dir/Closed/*.png and root_dir/Open/*.png
        for label_idx, label_name in enumerate(["Closed", "Open"]):
            class_folder = self.root_dir / label_name
            if class_folder.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for file_path in class_folder.glob(ext):
                        self.samples.append((file_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_from_real_dms(output_dir: Path):
    """Extracts real driver eye patches from Simuletic DMS Dataset."""
    from training.train_with_real_dms import extract_eye_patches
    samples = extract_eye_patches()
    for idx, (arr, label) in enumerate(samples):
        label_name = "Closed" if label == 0 else "Open"
        folder = output_dir / label_name
        folder.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(arr)
        img.save(folder / f"dms_sample_{idx:04d}.png")

def train(dataset_dir: str, epochs: int = 5, batch_size: int = 32, lr: float = 0.001, device: str = "cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Eye Training] Training on device: {device}")

    data_path = Path(dataset_dir)
    if not data_path.exists() or len(list(data_path.glob("*/*"))) == 0:
        print(f"[Eye Training] Populating from real Simuletic DMS dataset...")
        extract_from_real_dms(data_path)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = EyePatchDataset(str(data_path), transform=transform)
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = EyeStateCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    out_path = config.WEIGHTS_DIR / "eye_model.pth"
    torch.save(model.state_dict(), str(out_path))
    print(f"[Eye Training] Model saved successfully to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Eye State CNN")
    parser.add_argument("--dataset_dir", type=str, default=str(config.BASE_DIR / "data" / "eye_dataset"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(args.dataset_dir, args.epochs, args.batch_size, args.lr, args.device)
