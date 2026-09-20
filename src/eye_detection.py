"""
DriveGuard AI - Eye Detection & Landmark Aspect Ratio (EAR) Module
Computes geometric Eye Aspect Ratio (EAR) from 3D/2D facial landmarks
and extracts normalized eye crops for deep learning eye-state classification.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# MediaPipe canonical 468-point mesh landmark indices for eyes
# Left Eye indices (6 points for EAR calculation)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right Eye indices (6 points for EAR calculation)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

class EyeDetector:
    """Calculates EAR and extracts eye regions for CNN verification."""

    @staticmethod
    def calculate_ear(eye_landmarks: np.ndarray) -> float:
        """
        Computes the Eye Aspect Ratio (EAR):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        where p1, p4 are eye corners and p2, p3, p5, p6 are vertical eyelid points.
        """
        if len(eye_landmarks) < 6:
            return 0.30

        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]

        # Vertical distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)

        # Horizontal distance
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal < 1e-6:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(ear)

    @classmethod
    def get_eye_features(cls, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts EAR for both eyes and crops eye patches.
        landmarks: shape (N, 2) or (N, 3) in pixel coordinates.
        """
        h, w = frame.shape[:2]

        left_pts = landmarks[LEFT_EYE_INDICES, :2]
        right_pts = landmarks[RIGHT_EYE_INDICES, :2]

        left_ear = cls.calculate_ear(left_pts)
        right_ear = cls.calculate_ear(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Extract eye patches for deep learning model
        left_crop = cls._extract_patch(frame, left_pts)
        right_crop = cls._extract_patch(frame, right_pts)

        return {
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "avg_ear": round(avg_ear, 3),
            "left_eye_pts": left_pts,
            "right_eye_pts": right_pts,
            "left_crop": left_crop,
            "right_crop": right_crop
        }

    @staticmethod
    def _extract_patch(frame: np.ndarray, pts: np.ndarray, padding: int = 10) -> Optional[np.ndarray]:
        """Crops eye bounding box with padding for CNN inference."""
        h, w = frame.shape[:2]
        min_x = max(0, int(np.min(pts[:, 0])) - padding)
        max_x = min(w, int(np.max(pts[:, 0])) + padding)
        min_y = max(0, int(np.min(pts[:, 1])) - padding)
        max_y = min(h, int(np.max(pts[:, 1])) + padding)

        if max_x <= min_x or max_y <= min_y:
            return None

        return frame[min_y:max_y, min_x:max_x]
