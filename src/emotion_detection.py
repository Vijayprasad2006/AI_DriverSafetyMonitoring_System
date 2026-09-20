"""
DriveGuard AI - Facial Emotion Detection Wrapper
Provides temporal smoothing and history tracking for the deep learning FER model.
NOTE: Facial expressions represent visual classifications, not psychological diagnoses.
"""

import time
from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from models.model_loader import ModelManager

class EmotionDetector:
    """Manages facial expression inference and rolling emotion timelines."""

    def __init__(self, history_len: int = 60):
        self.manager = ModelManager()
        self.history_len = history_len
        self.emotion_history = deque(maxlen=history_len)
        self.last_inference_time = 0.0
        self.cached_result: Dict[str, Any] = {
            "dominant_emotion": "Neutral",
            "confidence": 0.5,
            "distribution": {e: 0.14 for e in ["Neutral", "Happy", "Sad", "Surprised", "Angry", "Fearful", "Disgusted"]}
        }

    def predict(self, face_rgb: np.ndarray, throttle_fps: float = 5.0) -> Dict[str, Any]:
        """
        Runs emotion inference throttled at target fps to conserve CPU/GPU resources.
        face_rgb: Cropped face image in RGB format.
        """
        now = time.time()
        # Throttle inference to prevent UI blocking
        if (now - self.last_inference_time) < (1.0 / throttle_fps):
            return self.cached_result

        classifier = self.manager.get_emotion_classifier()
        if classifier is None:
            return self.cached_result

        t0 = time.perf_counter()
        result = classifier.predict(face_rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.manager.update_latency("emotion_classifier_ms", elapsed_ms)

        self.cached_result = result
        self.last_inference_time = now
        self.emotion_history.append((now, result["dominant_emotion"], result["confidence"]))

        return result

    def get_history(self):
        """Returns recent emotion time-series."""
        return list(self.emotion_history)
