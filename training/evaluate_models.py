"""
DriveGuard AI - Model Evaluation & Benchmarking Suite
Evaluates trained eye or emotion models against test sets.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
and persists results to database/evaluation_report.json.
"""

import os
import json
import time
import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.eye_classifier import EyeClassifier

def evaluate_eye_model(dataset_dir: str, weights_path: str = None) -> dict:
    """Evaluates the eye classifier on test images."""
    data_path = Path(dataset_dir)
    print(f"[Evaluation] Evaluating eye model on: {data_path}")

    classifier = EyeClassifier(weights_path=weights_path)
    
    y_true = []
    y_pred = []
    latencies = []

    classes = ["Closed", "Open"]
    
    for label_idx, class_name in enumerate(classes):
        class_folder = data_path / class_name
        if class_folder.exists():
            for img_file in class_folder.glob("*.png"):
                img = np.array(Image.open(img_file).convert("L"))
                # Add channel dimension
                img_3ch = np.stack([img]*3, axis=-1)

                t0 = time.perf_counter()
                res = classifier.predict(img_3ch)
                latencies.append((time.perf_counter() - t0) * 1000.0)

                pred_idx = 0 if res["label"] == "Closed" else 1
                y_true.append(label_idx)
                y_pred.append(pred_idx)

    if not y_true:
        print("[Evaluation] No evaluation images found in target path.")
        return {}

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "dataset_name": data_path.name,
        "total_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "class_labels": classes,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    out_file = config.DATA_DIR / "evaluation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Evaluation] Report saved to {out_file}")
    print(f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}% | Latency: {np.mean(latencies):.2f}ms")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DriveGuard AI Models")
    parser.add_argument("--dataset_dir", type=str, default=str(config.BASE_DIR / "data" / "eye_dataset"))
    parser.add_argument("--weights_path", type=str, default=None)
    args = parser.parse_args()

    evaluate_eye_model(args.dataset_dir, args.weights_path)
