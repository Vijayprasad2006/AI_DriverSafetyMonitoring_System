"""
DriveGuard AI - Head Pose Estimation & Nodding Detection Module
Uses OpenCV solvePnP with a 3D canonical facial model and 2D facial landmarks
to estimate Pitch, Yaw, and Roll. Detects head drops and repetitive nodding.
"""

import time
import math
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional
import config

# Key MediaPipe facial landmark indices for 3D-2D correspondence
# 1: Nose tip, 152: Chin, 263: Left eye outer corner, 33: Right eye outer corner,
# 291: Left mouth corner, 61: Right mouth corner
POSE_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]

# 3D canonical facial model coordinates (in millimeters)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

class HeadPoseEstimator:
    """Estimates head orientation Euler angles and tracks nodding / distraction events."""

    def __init__(self,
                 pitch_down_thresh: float = config.HEAD_PITCH_DOWN_THRESHOLD,
                 yaw_thresh: float = config.HEAD_YAW_LEFT_THRESHOLD):
        
        self.pitch_down_thresh = pitch_down_thresh
        self.yaw_thresh = yaw_thresh

        # Temporal smoothing buffers
        self.pitch_history = deque(maxlen=30)
        self.yaw_history = deque(maxlen=30)
        self.roll_history = deque(maxlen=30)

        # Nodding detection: track pitch oscillation
        self.pitch_peaks = deque(maxlen=10)
        self.is_nodding = False
        self.consecutive_head_down_frames = 0

    def estimate_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Estimates Pitch, Yaw, and Roll from facial landmarks.
        landmarks: shape (N, 2) or (N, 3) pixel coordinates.
        frame_shape: (height, width).
        """
        h, w = frame_shape[:2]

        try:
            import cv2
            # 2D points from landmarks
            image_points = landmarks[POSE_LANDMARK_INDICES, :2].astype(np.float64)

            # Approximate camera intrinsic matrix
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return self._default_pose()

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles
            pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rmat)

        except Exception:
            # Fallback estimation if cv2 is not available yet
            pitch, yaw, roll = 0.0, 0.0, 0.0

        # Temporal smoothing
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.roll_history.append(roll)

        smooth_pitch = float(np.mean(self.pitch_history))
        smooth_yaw = float(np.mean(self.yaw_history))
        smooth_roll = float(np.mean(self.roll_history))

        # Check prolonged downward head slump
        if smooth_pitch < self.pitch_down_thresh:
            self.consecutive_head_down_frames += 1
        else:
            self.consecutive_head_down_frames = 0

        # Check nodding pattern (oscillation in pitch)
        self._detect_nodding(smooth_pitch)

        is_head_down = self.consecutive_head_down_frames >= 20
        is_distracted = abs(smooth_yaw) > self.yaw_thresh

        return {
            "pitch": round(smooth_pitch, 1),
            "yaw": round(smooth_yaw, 1),
            "roll": round(smooth_roll, 1),
            "is_head_down": is_head_down,
            "is_distracted": is_distracted,
            "is_nodding": self.is_nodding,
            "head_down_frames": self.consecutive_head_down_frames
        }

    def _detect_nodding(self, current_pitch: float):
        """Analyzes pitch history for cyclical nodding oscillations."""
        if len(self.pitch_history) >= 15:
            arr = list(self.pitch_history)
            # Find local extrema
            diffs = np.diff(arr)
            zero_crossings = np.where(np.diff(np.sign(diffs)))[0]
            if len(zero_crossings) >= config.NODDING_OSCILLATION_THRESHOLD:
                # Check amplitude of oscillation
                amp = np.max(arr) - np.min(arr)
                self.is_nodding = (amp > 10.0 and np.mean(arr) < 0.0)
            else:
                self.is_nodding = False
        else:
            self.is_nodding = False

    @staticmethod
    def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
        """Calculates Euler angles (in degrees) from 3x3 rotation matrix."""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = math.degrees(x)
        yaw = math.degrees(y)
        roll = math.degrees(z)
        return pitch, yaw, roll

    def _default_pose(self) -> Dict[str, Any]:
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "is_head_down": False, "is_distracted": False,
            "is_nodding": False, "head_down_frames": 0
        }
