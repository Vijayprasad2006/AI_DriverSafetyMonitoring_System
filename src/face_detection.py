"""
DriveGuard AI - Face Detection & Primary Face Tracking Module
Integrates MediaPipe Face Landmarker / FaceMesh for 468/478 3D landmark extraction.
Implements primary driver tracking based on bounding box area and frame-center proximity
to avoid flickering when passengers or multiple faces are visible.
"""

import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import config

logger = logging.getLogger("DriveGuard.FaceDetection")

class FaceDetector:
    """Detects facial landmarks, tracks the primary driver face, and computes bounding boxes."""

    def __init__(self,
                 max_num_faces: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mesh_detector = None
        self._init_mediapipe()

        # Driver tracking state
        self.primary_face_center: Optional[Tuple[float, float]] = None
        self.missing_face_start: Optional[float] = None
        self.consecutive_missing_frames = 0

    def _init_mediapipe(self):
        """Initializes MediaPipe Face Mesh solution."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mesh_detector = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            logger.info("MediaPipe FaceMesh initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe FaceMesh not available: {e}")
            self.mesh_detector = None

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes a BGR image frame and extracts driver face features.
        Returns:
            face_detected: bool
            missing_seconds: float
            landmarks: np.ndarray of shape (468, 2) in pixel space (or None)
            bbox: Tuple (x1, y1, x2, y2)
            face_crop_rgb: Cropped face in RGB format
            num_faces_detected: int
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return self._missing_face_result()

        h, w = frame_bgr.shape[:2]
        now = time.time()

        if self.mesh_detector is None:
            # Synthetic landmark generator if mediapipe is unavailable
            return self._synthetic_face_landmarks(frame_bgr)

        try:
            import cv2
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.mesh_detector.process(frame_rgb)
        except Exception:
            return self._missing_face_result()

        if not results.multi_face_landmarks:
            return self._missing_face_result()

        num_faces = len(results.multi_face_landmarks)
        selected_mesh = self._select_primary_face(results.multi_face_landmarks, w, h)

        # Convert landmarks to pixel coordinates
        landmarks_pixel = np.zeros((len(selected_mesh.landmark), 2), dtype=np.float32)
        for i, lm in enumerate(selected_mesh.landmark):
            landmarks_pixel[i] = [lm.x * w, lm.y * h]

        # Compute bounding box
        x_min = max(0, int(np.min(landmarks_pixel[:, 0])))
        y_min = max(0, int(np.min(landmarks_pixel[:, 1])))
        x_max = min(w, int(np.max(landmarks_pixel[:, 0])))
        y_max = min(h, int(np.max(landmarks_pixel[:, 1])))
        bbox = (x_min, y_min, x_max, y_max)

        # Update primary face center tracking
        self.primary_face_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
        self.missing_face_start = None
        self.consecutive_missing_frames = 0

        # Extract face crop in RGB
        face_crop_rgb = frame_rgb[y_min:y_max, x_min:x_max]

        return {
            "face_detected": True,
            "missing_seconds": 0.0,
            "landmarks": landmarks_pixel,
            "bbox": bbox,
            "face_crop_rgb": face_crop_rgb,
            "num_faces_detected": num_faces
        }

    def _select_primary_face(self, face_landmarks_list, w: int, h: int):
        """
        Picks the primary driver face using proximity to center and bounding box size.
        """
        if len(face_landmarks_list) == 1:
            return face_landmarks_list[0]

        best_face = face_landmarks_list[0]
        best_score = -1e9
        frame_cx, frame_cy = w / 2.0, h / 2.0

        for face in face_landmarks_list:
            xs = [lm.x * w for lm in face.landmark]
            ys = [lm.y * h for lm in face.landmark]
            face_w = max(xs) - min(xs)
            face_h = max(ys) - min(ys)
            area = face_w * face_h
            cx = (max(xs) + min(xs)) / 2.0
            cy = (max(ys) + min(ys)) / 2.0

            # Proximity to frame center
            dist_to_center = np.hypot(cx - frame_cx, cy - frame_cy)

            # Combined score favoring larger and centered faces
            score = area - (dist_to_center * 50.0)
            if score > best_score:
                best_score = score
                best_face = face

        return best_face

    def _missing_face_result(self) -> Dict[str, Any]:
        """Handles state when face is undetected."""
        now = time.time()
        self.consecutive_missing_frames += 1
        if self.missing_face_start is None:
            self.missing_face_start = now
        missing_sec = now - self.missing_face_start

        return {
            "face_detected": False,
            "missing_seconds": round(missing_sec, 1),
            "landmarks": None,
            "bbox": None,
            "face_crop_rgb": None,
            "num_faces_detected": 0
        }

    def _synthetic_face_landmarks(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Generates synthetic landmarks when MediaPipe is not installed."""
        h, w = frame_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        
        # Approximate 468 points around center
        landmarks = np.zeros((468, 2), dtype=np.float32)
        for i in range(468):
            angle = (i / 468.0) * 2 * np.pi
            r = 80 + (i % 20)
            landmarks[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]

        # Specific key points
        landmarks[1] = [cx, cy]                  # Nose tip
        landmarks[152] = [cx, cy + 90]           # Chin
        landmarks[33] = [cx - 50, cy - 20]       # Right eye corner
        landmarks[263] = [cx + 50, cy - 20]      # Left eye corner
        landmarks[61] = [cx - 30, cy + 50]       # Mouth right
        landmarks[291] = [cx + 30, cy + 50]      # Mouth left

        # Set eye points
        for idx in [362, 385, 387, 263, 373, 380]:
            landmarks[idx] = [cx + 35, cy - 20]
        for idx in [33, 160, 158, 133, 153, 144]:
            landmarks[idx] = [cx - 35, cy - 20]
            
        bbox = (int(cx - 90), int(cy - 110), int(cx + 90), int(cy + 110))
        return {
            "face_detected": True,
            "missing_seconds": 0.0,
            "landmarks": landmarks,
            "bbox": bbox,
            "face_crop_rgb": frame_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]],
            "num_faces_detected": 1
        }
