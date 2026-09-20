"""
DriveGuard AI - Real DMS Dataset Training Pipeline
Extracts real driver eye patches from Simuletic DMS Dataset (archive (1)),
applies data augmentation, and trains EyeStateCNN to classify real human eye states.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.eye_classifier import EyeStateCNN
from src.face_detection import FaceDetector

DMS_ROOT = config.BASE_DIR / "dataset"
if not DMS_ROOT.exists():
    DMS_ROOT = Path(r"C:\Users\Vijay\Downloads\archive (1)\Simuletic_DMS_Dataset")
LABELS_DIR = DMS_ROOT / "labels"
IMAGES_DIR = DMS_ROOT / "images"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "simuletic_eyes"

class RealDMSEyeDataset(Dataset):
    """Loads extracted real eye patches."""

    def __init__(self, samples, transform_augment=False):
        self.samples = samples
        self.transform_augment = transform_augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_arr, label = self.samples[idx]
        
        # Augmentation on numpy
        if self.transform_augment:
            if np.random.rand() > 0.5:
                img_arr = np.fliplr(img_arr).copy()
            # Random brightness jitter
            factor = np.random.uniform(0.85, 1.15)
            img_arr = np.clip(img_arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Convert to float tensor (1, 32, 32)
        gray = img_arr.astype(np.float32)
        tensor = torch.from_numpy(gray).unsqueeze(0)
        # Normalize to [-1, 1]
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor, label

def extract_eye_patches():
    """Extracts eye crops from the DMS dataset."""
    print("=" * 60)
    print("EXTRACTING REAL EYE PATCHES FROM SIMULETIC DMS DATASET")
    print(f"Source: {DMS_ROOT}")
    print("=" * 60)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    extracted_samples = [] # (32x32 gray arr, label_idx) where 0=Closed, 1=Open

    json_files = list(LABELS_DIR.glob("*.json"))
    print(f"Found {len(json_files)} annotation files.")

    stats = {"Closed": 0, "Open": 0, "Skipped": 0}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data.get("samples", []):
            img_name = sample.get("image")
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                stats["Skipped"] += 1
                continue

            raw_eye_state = sample.get("attributes", {}).get("eye_state", "")
            if raw_eye_state in ["Closed", "Drowsy/Microsleep"]:
                label = 0 # Closed
                label_name = "Closed"
            elif raw_eye_state == "Open":
                label = 1 # Open
                label_name = "Open"
            else:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Detect face
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
            if len(faces) == 0:
                # Estimate eye region from upper face geometry if face detection missed
                eye_region = gray[int(h*0.25):int(h*0.45), int(w*0.3):int(w*0.7)]
                patch = cv2.resize(eye_region, (32, 32))
                extracted_samples.append((patch, label))
                stats[label_name] += 1
                continue

            # Driver face is primary (largest box)
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]

            # Detect eyes inside face ROI
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_patch = face_roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_patch = cv2.resize(eye_patch, (32, 32))
                    extracted_samples.append((eye_patch, label))
                    stats[label_name] += 1
            else:
                # Approximate upper half of face for eyes
                eye_roi = face_roi_gray[int(fh*0.2):int(fh*0.45), int(fw*0.15):int(fw*0.85)]
                if eye_roi.size > 0:
                    eye_patch = cv2.resize(eye_roi, (32, 32))
                    extracted_samples.append((eye_patch, label))
                    stats[label_name] += 1

    print(f"Extraction Complete: {len(extracted_samples)} real eye crops.")
    print(f"Class counts: Closed/Drowsy = {stats['Closed']}, Open = {stats['Open']}")
    return extracted_samples

def train_eye_model_on_real_data(epochs: int = 15, batch_size: int = 16, lr: float = 0.001):
    samples = extract_eye_patches()
    if len(samples) < 20:
        print("Insufficient samples extracted.")
        return

    # Train / Val split (80 / 20)
    split_idx = int(len(samples) * 0.8)
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    
    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]

    train_dataset = RealDMSEyeDataset(train_samples, transform_augment=True)
    val_dataset = RealDMSEyeDataset(val_samples, transform_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining EyeStateCNN on {device} for {epochs} epochs...")

    model = EyeStateCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    weights_path = config.MODELS_DIR / "weights" / "eye_model.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = correct / total if total > 0 else 0.0
        avg_train_loss = train_loss / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Trained model checkpoint successfully saved to: {weights_path}")

    # Generate updated evaluation report
    report = {
        "dataset_name": "Simuletic_DMS_Dataset (Real Driver Monitoring)",
        "total_extracted_patches": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_accuracy": round(best_val_acc, 4),
        "classes": ["Closed/Drowsy", "Open"],
        "checkpoint": str(weights_path),
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    report_file = config.DATA_DIR / "real_dms_training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to: {report_file}")

if __name__ == "__main__":
    train_eye_model_on_real_data(epochs=15, batch_size=16)
