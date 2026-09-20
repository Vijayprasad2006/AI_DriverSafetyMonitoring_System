"""
Unit tests for Emergency Notification Service
Validates Twilio configuration handling, cooldown throttling, and simulation dispatch.
"""

import time
import pytest
from src.emergency_service import EmergencyService

def test_emergency_simulation_mode():
    service = EmergencyService(enabled=True)
    # Empty credentials trigger simulation mode
    res = service.send_emergency_alert("Severe driver fatigue detected", 90.0, force=True)
    assert res["status"] == "SENT_SIMULATED"
    assert res["mode"] == "SIMULATION"
    assert "simulated_body" in res

def test_emergency_cooldown_suppression():
    service = EmergencyService(enabled=True)
    # First send succeeds
    res1 = service.send_emergency_alert("Initial alert", 80.0, force=True)
    assert "SENT" in res1["status"]

    # Immediate second send should be blocked by cooldown
    res2 = service.send_emergency_alert("Second alert", 82.0, force=False)
    assert res2["status"] == "COOLDOWN"
    assert "remaining" in res2["message"]

def test_emergency_disabled():
    service = EmergencyService(enabled=False)
    res = service.send_emergency_alert("Alert while disabled", 85.0, force=False)
    assert res["status"] == "DISABLED"
