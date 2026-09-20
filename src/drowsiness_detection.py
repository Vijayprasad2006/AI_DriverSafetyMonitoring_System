"""
DriveGuard AI - Temporal Drowsiness & PERCLOS Tracking Module
Analyzes eye closures across consecutive frames, calculates moving PERCLOS,
and detects microsleep events and prolonged eye closure.
"""

import time
from collections import deque
from typing import Dict, Any, Optional
import config

class DrowsinessDetector:
    """Tracks eye closure state over time, calculating PERCLOS and sustained drowsiness."""

    def __init__(self,
                 ear_threshold: float = config.EAR_DROWSY_THRESHOLD,
                 warning_frames: int = config.DROWSINESS_FRAMES_WARNING,
                 critical_frames: int = config.DROWSINESS_FRAMES_CRITICAL,
                 microsleep_frames: int = config.MICROSLEEP_FRAMES_THRESHOLD,
                 perclos_window_sec: float = config.PERCLOS_WINDOW_SECONDS):
        
        self.ear_threshold = ear_threshold
        self.warning_frames = warning_frames
        self.critical_frames = critical_frames
        self.microsleep_frames = microsleep_frames
        self.perclos_window_sec = perclos_window_sec

        self.consecutive_closed_frames = 0
        self.total_closed_events = 0
        self.microsleep_events = 0
        self.last_state = "OPEN"
        
        # Buffer of (timestamp, is_closed: bool) for PERCLOS
        self.closure_history = deque()

    def update(self, ear: float, dl_eye_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Updates drowsiness telemetry with current frame's EAR and optional deep learning label.
        Returns:
            Dict containing closed_frames, is_drowsy, is_critical, is_microsleep, perclos.
        """
        now = time.time()
        
        # Determine if eyes are closed in this frame:
        # Fuse geometric EAR with CNN prediction when available
        is_closed = (ear < self.ear_threshold)
        if dl_eye_state == "Closed" and ear < (self.ear_threshold + 0.04):
            is_closed = True

        if is_closed:
            self.consecutive_closed_frames += 1
            if self.last_state == "OPEN":
                self.total_closed_events += 1
            self.last_state = "CLOSED"
        else:
            self.consecutive_closed_frames = 0
            self.last_state = "OPEN"

        # Record into PERCLOS history
        self.closure_history.append((now, is_closed))

        # Evict samples older than perclos_window_sec
        cutoff = now - self.perclos_window_sec
        while self.closure_history and self.closure_history[0][0] < cutoff:
            self.closure_history.popleft()

        # Calculate PERCLOS (% of time eyes are closed in the window)
        if self.closure_history:
            closed_count = sum(1 for _, closed in self.closure_history if closed)
            perclos = closed_count / len(self.closure_history)
        else:
            perclos = 0.0

        # Evaluate severity
        is_microsleep = self.consecutive_closed_frames >= self.microsleep_frames
        if is_microsleep and (self.consecutive_closed_frames == self.microsleep_frames):
            self.microsleep_events += 1

        is_critical = (self.consecutive_closed_frames >= self.critical_frames) or (perclos >= config.PERCLOS_CRITICAL_THRESHOLD)
        is_warning = (self.consecutive_closed_frames >= self.warning_frames) or (perclos >= config.PERCLOS_WARNING_THRESHOLD)

        return {
            "is_closed": is_closed,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "perclos": round(perclos, 3),
            "is_warning": is_warning and not is_critical,
            "is_critical": is_critical,
            "is_microsleep": is_microsleep,
            "total_closed_events": self.total_closed_events,
            "microsleep_count": self.microsleep_events
        }

    def reset(self):
        """Resets counters for a new session."""
        self.consecutive_closed_frames = 0
        self.total_closed_events = 0
        self.microsleep_events = 0
        self.closure_history.clear()
