"""
DriveGuard AI - Resilient Camera & Video Stream Manager
Handles OpenCV webcam capture with FPS profiling, resolution configuration,
and synthetic video test pattern generation for head-less / test environments.
"""

import time
import logging
import numpy as np
from typing import Optional, Tuple
import config

logger = logging.getLogger("DriveGuard.Camera")

class CameraStream:
    """Manages webcam capture with graceful degradation and simulated driver frame generation."""

    def __init__(self,
                 camera_index: int = config.DEFAULT_CAMERA_INDEX,
                 width: int = config.DEFAULT_FRAME_WIDTH,
                 height: int = config.DEFAULT_FRAME_HEIGHT):
        
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        self.is_synthetic = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_frame_time = time.time()
        self._synthetic_phase = 0.0

    def start(self) -> bool:
        """Initializes the webcam device."""
        try:
            import cv2
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.is_opened = True
                self.is_synthetic = False
                logger.info(f"Connected to camera index {self.camera_index}")
                return True
        except Exception as e:
            logger.warning(f"Could not open hardware camera {self.camera_index}: {e}")

        # Fallback to high-fidelity synthetic driver simulation
        logger.info("Initializing synthetic driver video generator.")
        self.is_opened = True
        self.is_synthetic = True
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next video frame.
        Returns:
            (success: bool, frame: np.ndarray in BGR)
        """
        if not self.is_opened:
            return False, None

        t0 = time.time()
        if not self.is_synthetic and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._update_fps(t0)
                return True, frame
            else:
                logger.warning("Hardware frame read failed; switching to synthetic stream.")
                self.is_synthetic = True

        # Generate synthetic driver frame
        frame = self._generate_synthetic_frame()
        self._update_fps(t0)
        return True, frame

    def _update_fps(self, t0: float):
        """Calculates rolling frame rate."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed >= 1.0:
            self.fps = round(self.frame_count / elapsed, 1)
            self.frame_count = 0
            self.last_frame_time = now

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Plays back real driver monitoring camera frames from the Simuletic DMS dataset."""
        try:
            from pathlib import Path
            import config
            dms_img_dir = config.BASE_DIR / "dataset" / "images"
            if not dms_img_dir.exists():
                dms_img_dir = Path(r"C:\Users\Vijay\Downloads\archive (1)\Simuletic_DMS_Dataset\images")
            if dms_img_dir.exists():
                if not hasattr(self, "_real_frame_paths") or not self._real_frame_paths:
                    self._real_frame_paths = sorted(list(dms_img_dir.glob("*.jpg")))
                    self._real_frame_idx = 0

                if self._real_frame_paths:
                    img_path = self._real_frame_paths[self._real_frame_idx % len(self._real_frame_paths)]
                    self._real_frame_idx += 1
                    import cv2
                    frame = cv2.imread(str(img_path))
                    if frame is not None:
                        frame = cv2.resize(frame, (self.width, self.height))
                        return frame
        except Exception:
            pass

        # Clean high-tech camera standby card
        standby_frame = np.full((self.height, self.width, 3), (12, 16, 26), dtype=np.uint8)
        return standby_frame

    def release(self):
        """Safely releases the webcam hardware."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_opened = False
        logger.info("Camera stream released.")
