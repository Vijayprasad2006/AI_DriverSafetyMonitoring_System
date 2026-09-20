"""
DriveGuard AI - Centralized Model Loader & Cache Manager
Loads and caches AI models (Eye State CNN, Emotion CNN, MediaPipe)
Detects CPU/GPU acceleration, profiles inference latency, and manages memory.
"""

import time
import logging
import torch
from typing import Dict, Any, Optional
from models.eye_classifier import EyeClassifier
from models.emotion_classifier import EmotionClassifier

logger = logging.getLogger("DriveGuard.ModelLoader")

class ModelManager:
    """Singleton model manager ensuring neural networks are loaded only once."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[str] = None):
        if self._initialized:
            return

        # Determine optimal hardware device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Initializing ModelManager on device: {self.device}")
        
        # Models
        self.eye_model: Optional[EyeClassifier] = None
        self.emotion_model: Optional[EmotionClassifier] = None
        
        # Latency telemetry (in milliseconds)
        self.latencies: Dict[str, float] = {
            "face_mesh_ms": 0.0,
            "eye_classifier_ms": 0.0,
            "emotion_classifier_ms": 0.0,
            "head_pose_ms": 0.0,
            "total_inference_ms": 0.0
        }

        self.model_status: Dict[str, str] = {
            "eye_classifier": "NOT_LOADED",
            "emotion_classifier": "NOT_LOADED",
            "face_landmarker": "READY"
        }

        self._load_models()
        self._initialized = True

    def _load_models(self):
        """Loads models with timing and error handling."""
        try:
            t0 = time.perf_counter()
            self.eye_model = EyeClassifier(device=self.device)
            self.latencies["eye_classifier_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self.model_status["eye_classifier"] = "LOADED_ACTIVE"
        except Exception as e:
            self.model_status["eye_classifier"] = f"ERROR: {e}"
            logger.error(f"Failed to load EyeClassifier: {e}")

        try:
            t0 = time.perf_counter()
            self.emotion_model = EmotionClassifier(device=self.device)
            self.latencies["emotion_classifier_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self.model_status["emotion_classifier"] = "LOADED_ACTIVE"
        except Exception as e:
            self.model_status["emotion_classifier"] = f"ERROR: {e}"
            logger.error(f"Failed to load EmotionClassifier: {e}")

    def get_eye_classifier(self) -> Optional[EyeClassifier]:
        return self.eye_model

    def get_emotion_classifier(self) -> Optional[EmotionClassifier]:
        return self.emotion_model

    def update_latency(self, model_key: str, elapsed_ms: float):
        """Updates measured inference latency for UI monitoring."""
        if model_key in self.latencies:
            # Exponential moving average for smooth display
            prev = self.latencies[model_key]
            self.latencies[model_key] = round(0.3 * elapsed_ms + 0.7 * prev, 2)

    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Provides hardware status, CUDA info, and model execution profiles."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU",
            "model_status": self.model_status,
            "latencies_ms": self.latencies
        }
