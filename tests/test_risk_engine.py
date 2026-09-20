"""
Unit tests for Driver Risk Assessment Engine
Validates multi-signal fusion, score boundaries, and hysteresis state changes.
"""

import pytest
import config
from src.risk_engine import RiskEngine

def test_risk_engine_normal_state():
    engine = RiskEngine()
    result = engine.compute_risk(
        ear=0.32,
        closed_frames=0,
        perclos=0.04,
        is_yawning=False,
        recent_yawns=0,
        pitch=0.0,
        yaw=0.0,
        is_nodding=False,
        face_detected=True,
        missing_face_sec=0.0
    )
    assert result["risk_level"] == config.RISK_LEVEL_NORMAL
    assert result["raw_score"] < config.RISK_SCORE_WARNING_MIN
    assert "Normal driver alertness maintained." in result["reasons"]

def test_risk_engine_critical_eye_closure():
    engine = RiskEngine()
    # Emulate sustained eye closure for 60 frames (microsleep)
    for _ in range(config.RISK_ESCALATION_FRAMES + 2):
        result = engine.compute_risk(
            ear=0.12,
            closed_frames=65,
            perclos=0.45,
            is_yawning=False,
            recent_yawns=1,
            pitch=-5.0,
            yaw=0.0,
            is_nodding=False,
            face_detected=True,
            missing_face_sec=0.0
        )
    assert result["risk_level"] == config.RISK_LEVEL_CRITICAL
    assert result["smoothed_score"] >= config.RISK_SCORE_CRITICAL_MIN
    assert any("Microsleep" in r or "closure" in r for r in result["reasons"])

def test_risk_engine_missing_face():
    engine = RiskEngine()
    result = engine.compute_risk(
        ear=0.0,
        closed_frames=0,
        perclos=0.0,
        is_yawning=False,
        recent_yawns=0,
        pitch=0.0,
        yaw=0.0,
        is_nodding=False,
        face_detected=False,
        missing_face_sec=4.5
    )
    assert any("face not visible" in r for r in result["reasons"])
