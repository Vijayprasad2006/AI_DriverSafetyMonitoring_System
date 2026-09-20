"""
DriveGuard AI - Yawning & Mouth Aspect Ratio (MAR) Detection Module
Computes MAR from inner and outer lip landmarks, validates sustained duration
to differentiate yawns from speech, and maintains rolling yawn frequency.
"""

import time
import numpy as np
from collections import deque
from typing import Dict, Any
import config

# MediaPipe canonical 468 mesh lip landmark indices
# Outer lip: 61 (left corner), 291 (right corner), 0 (top center), 17 (bottom center)
# Inner lip: 78 (left inner), 308 (right inner), 13 (top inner), 14 (bottom inner), 82, 87, 312, 317
MOUTH_OUTER_INDICES = [61, 291, 0, 17]
MOUTH_INNER_INDICES = [78, 308, 13, 14, 82, 87, 312, 317]

class YawnDetector:
    """Detects yawning events using Mouth Aspect Ratio (MAR) with duration filters."""

    def __init__(self,
                 mar_threshold: float = config.MAR_YAWN_THRESHOLD,
                 min_frames: int = config.YAWN_FRAMES_THRESHOLD,
                 cooldown_sec: float = config.YAWN_COOLDOWN_SECONDS):
        
        self.mar_threshold = mar_threshold
        self.min_frames = min_frames
        self.cooldown_sec = cooldown_sec

        self.consecutive_yawn_frames = 0
        self.last_yawn_end_time = 0.0
        self.total_yawns = 0
        self.is_currently_yawning = False

        # Rolling history of confirmed yawn timestamps for rate analysis (last 3 minutes)
        self.recent_yawns = deque()

    @staticmethod
    def calculate_mar(mouth_landmarks: np.ndarray) -> float:
        """
        Computes the Mouth Aspect Ratio (MAR):
        Uses Euclidean distances between inner vertical lip landmarks and outer lip corners.
        """
        if len(mouth_landmarks) < 8:
            return 0.20

        # Points: p78 (left corner), p308 (right corner), p13 (upper lip), p14 (lower lip),
        # p82, p87 (left vertical pair), p312, p317 (right vertical pair)
        p_left = mouth_landmarks[0]
        p_right = mouth_landmarks[1]
        p_top = mouth_landmarks[2]
        p_bot = mouth_landmarks[3]
        p_vert1_top = mouth_landmarks[4]
        p_vert1_bot = mouth_landmarks[5]
        p_vert2_top = mouth_landmarks[6]
        p_vert2_bot = mouth_landmarks[7]

        # Vertical distances
        v1 = np.linalg.norm(p_top - p_bot)
        v2 = np.linalg.norm(p_vert1_top - p_vert1_bot)
        v3 = np.linalg.norm(p_vert2_top - p_vert2_bot)

        # Horizontal distance
        h = np.linalg.norm(p_left - p_right)

        if h < 1e-6:
            return 0.0

        mar = (v1 + v2 + v3) / (3.0 * h)
        return float(mar)

    def update(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Processes facial landmarks to detect yawns.
        landmarks: (N, 2) or (N, 3) pixel coordinates.
        """
        now = time.time()
        mouth_pts = landmarks[MOUTH_INNER_INDICES, :2]
        mar = self.calculate_mar(mouth_pts)

        just_completed_yawn = False

        if mar >= self.mar_threshold:
            self.consecutive_yawn_frames += 1
            # If open long enough, flag active yawn
            if self.consecutive_yawn_frames >= self.min_frames:
                self.is_currently_yawning = True
        else:
            if self.is_currently_yawning:
                # Mouth has now closed after a sustained open period
                if (now - self.last_yawn_end_time) >= self.cooldown_sec:
                    self.total_yawns += 1
                    self.recent_yawns.append(now)
                    self.last_yawn_end_time = now
                    just_completed_yawn = True
                self.is_currently_yawning = False

            self.consecutive_yawn_frames = 0

        # Evict yawns older than 3 minutes (180 seconds)
        cutoff = now - 180.0
        while self.recent_yawns and self.recent_yawns[0] < cutoff:
            self.recent_yawns.popleft()

        return {
            "mar": round(mar, 3),
            "is_yawning": self.is_currently_yawning,
            "yawn_frames": self.consecutive_yawn_frames,
            "just_completed_yawn": just_completed_yawn,
            "total_yawns": self.total_yawns,
            "recent_yawn_count": len(self.recent_yawns)
        }

    def reset(self):
        """Resets yawn counters."""
        self.consecutive_yawn_frames = 0
        self.total_yawns = 0
        self.is_currently_yawning = False
        self.recent_yawns.clear()
