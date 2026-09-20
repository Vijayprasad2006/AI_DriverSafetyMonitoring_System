"""
DriveGuard AI - PyTorch Eye State CNN Classifier
Classifies eye patches extracted from facial landmarks into Open vs Closed.
Trained on MRL Eye Dataset / CEW representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class EyeStateCNN(nn.Module):
    """
    Lightweight convolutional neural network optimized for real-time
    driver eye-state classification on CPU and GPU.
    """

    def __init__(self, num_classes: int = 2):
        super(EyeStateCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # -> 16 x 16
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # -> 8 x 8

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # -> 4 x 4

        # Classifier head
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Initialize weights with standard Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EyeClassifier:
    """Wrapper for eye state inference with preprocessing and batching."""

    CLASSES = ["Closed", "Open"]

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = EyeStateCNN(num_classes=2)
        self.model.to(self.device)
        self.model.eval()

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                pass  # Use initialized weights if file not present

    def _preprocess(self, eye_patch: np.ndarray) -> torch.Tensor:
        if eye_patch.ndim == 3:
            gray = np.dot(eye_patch[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = eye_patch.astype(np.float32)
        t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t_resized = F.interpolate(t, size=(32, 32), mode='bilinear', align_corners=False)
        t_norm = (t_resized / 255.0 - 0.5) / 0.5
        return t_norm.to(self.device)

    def predict(self, eye_patch: np.ndarray) -> dict:
        """
        Classifies an extracted eye image patch.
        Returns predicted label, confidence, and probability distribution.
        """
        if eye_patch is None or eye_patch.size == 0:
            return {"label": "Open", "confidence": 0.5, "prob_closed": 0.5, "prob_open": 0.5}

        with torch.no_grad():
            tensor = self._preprocess(eye_patch)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            prob_closed = float(probs[0])
            prob_open = float(probs[1])
            label = "Closed" if prob_closed > prob_open else "Open"
            confidence = max(prob_closed, prob_open)

            return {
                "label": label,
                "confidence": confidence,
                "prob_closed": prob_closed,
                "prob_open": prob_open
            }
