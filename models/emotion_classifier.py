"""
DriveGuard AI - PyTorch Facial Expression Recognition (FER) Model
Classifies face crops into 7 standard emotional categories:
Neutral, Happy, Sad, Surprised, Angry, Fearful, Disgusted.
Built on a high-efficiency MobileNetV3-Small backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Any, List
import config

class EmotionCNN(nn.Module):
    """MobileNetV3-Small transfer learning architecture for FER-2013 / AffectNet."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(EmotionCNN, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        
        # Feature extraction layers
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classification head for 7 emotions
        in_features = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EmotionClassifier:
    """Facial emotion classifier wrapper with preprocessing and probability distributions."""

    CLASSES = config.EMOTION_CLASSES

    def __init__(self, weights_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = EmotionCNN(num_classes=len(self.CLASSES), pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                pass  # Fallback to pretrained backbone

    def _preprocess(self, face_crop_rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(face_crop_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        t_resized = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        t_norm = t_resized / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((t_norm - mean) / std).to(self.device)

    def predict(self, face_crop_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Infers emotion from an RGB cropped face array.
        Returns:
            dominant_emotion: str
            confidence: float
            distribution: Dict[str, float]
        """
        if face_crop_rgb is None or face_crop_rgb.size == 0:
            return {
                "dominant_emotion": "Neutral",
                "confidence": 0.50,
                "distribution": {c: round(1.0 / len(self.CLASSES), 3) for c in self.CLASSES}
            }

        with torch.no_grad():
            tensor = self._preprocess(face_crop_rgb)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            top_idx = int(np.argmax(probs))
            dominant = self.CLASSES[top_idx]
            confidence = float(probs[top_idx])

            distribution = {self.CLASSES[i]: round(float(probs[i]), 3) for i in range(len(self.CLASSES))}

            return {
                "dominant_emotion": dominant,
                "confidence": round(confidence, 3),
                "distribution": distribution
            }
