"""
Unit tests for Eye Detection and Temporal Drowsiness Tracker
Validates geometric EAR, frame counting, PERCLOS calculation, and microsleep flags.
"""

import numpy as np
import pytest
import config
from src.eye_detection import EyeDetector
from src.drowsiness_detection import DrowsinessDetector

def test_ear_calculation():
    # Synthetic eye landmarks: 6 points
    # p1=(0,0), p2=(5, 5), p3=(15, 5), p4=(20, 0), p5=(15, -5), p6=(5, -5)
    landmarks = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
        [15.0, 5.0],
        [20.0, 0.0],
        [15.0, -5.0],
        [5.0, -5.0]
    ], dtype=np.float32)

    ear = EyeDetector.calculate_ear(landmarks)
    assert ear > 0.0
    # Expected: (10 + 10) / (2 * 20) = 0.5
    assert abs(ear - 0.5) < 1e-3

def test_drowsiness_tracker_consecutive_closed():
    detector = DrowsinessDetector(warning_frames=10, critical_frames=25, microsleep_frames=50)

    # 50 frames of open eyes (baseline driving)
    for _ in range(50):
        res = detector.update(ear=0.30)
        assert res["is_closed"] is False
        assert res["consecutive_closed_frames"] == 0

    # 12 frames of closed eyes (ear < 0.22) -> Warning (> 10 frames, PERCLOS = 12/62 = 19.3% < 35%)
    for i in range(1, 13):
        res = detector.update(ear=0.15)
        assert res["is_closed"] is True
        assert res["consecutive_closed_frames"] == i

    assert res["is_warning"] is True
    assert res["is_critical"] is False

def test_drowsiness_microsleep_trigger():
    detector = DrowsinessDetector(warning_frames=10, critical_frames=20, microsleep_frames=30)
    for _ in range(35):
        res = detector.update(ear=0.12)

    assert res["is_critical"] is True
    assert res["is_microsleep"] is True
    assert res["microsleep_count"] >= 1
