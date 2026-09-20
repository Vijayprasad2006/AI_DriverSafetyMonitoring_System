"""
DriveGuard AI - PyTorch Emotion Model Fine-Tuning Pipeline
Fine-tunes MobileNetV3-Small on FER-2013 or AffectNet facial crops across 7 classes:
Neutral, Happy, Sad, Surprised, Angry, Fearful, Disgusted.
"""

import os
import argparse
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
from models.emotion_classifier import EmotionCNN

class EmotionDataset(Dataset):
    """Loads facial emotion crop images organized in subfolders by emotion label."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = config.EMOTION_CLASSES

        for label_idx, class_name in enumerate(self.classes):
            folder = self.root_dir / class_name
            if folder.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for file_path in folder.glob(ext):
                        self.samples.append((file_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_real_faces_from_dms(output_dir: Path):
    """Extracts real driver face crops from the Simuletic DMS dataset."""
    import cv2
    dms_img_dir = Path(r"C:\Users\Vijay\Downloads\archive (1)\Simuletic_DMS_Dataset\images")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    neutral_folder = output_dir / "Neutral"
    neutral_folder.mkdir(parents=True, exist_ok=True)
    
    if dms_img_dir.exists():
        for idx, img_path in enumerate(sorted(list(dms_img_dir.glob("*.jpg")))[:30]):
            img = cv2.imread(str(img_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    crop = img[y:y+h, x:x+w]
                    crop = cv2.resize(crop, (224, 224))
                    cv2.imwrite(str(neutral_folder / f"real_driver_{idx:03d}.jpg"), crop)

def train_emotion(dataset_dir: str, epochs: int = 3, batch_size: int = 16, lr: float = 0.0003, device: str = "cpu"):
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Emotion Training] Training on device: {device}")

    data_path = Path(dataset_dir)
    if not data_path.exists() or len(list(data_path.glob("*/*"))) == 0:
        print(f"[Emotion Training] Extracting real driver faces from Simuletic DMS dataset...")
        extract_real_faces_from_dms(data_path)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EmotionDataset(str(data_path), transform=transform_train)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = EmotionCNN(num_classes=len(config.EMOTION_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = running_loss / total if total > 0 else 0.0

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

    out_file = config.WEIGHTS_DIR / "emotion_model.pth"
    torch.save(model.state_dict(), str(out_file))
    print(f"[Emotion Training] Fine-tuned model saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Emotion CNN")
    parser.add_argument("--dataset_dir", type=str, default=str(config.BASE_DIR / "data" / "fer_dataset"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_emotion(args.dataset_dir, args.epochs, args.batch_size, args.lr, args.device)
