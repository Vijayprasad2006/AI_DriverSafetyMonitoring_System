"""
Unit tests for SQLite Database Layer
Validates session tracking, event insertions, telemetry sampling, and privacy wipe.
"""

import tempfile
import os
import pytest
from database.database import Database

@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_driveguard.db")
    db = Database(db_path=db_path)
    yield db

def test_session_lifecycle(temp_db):
    session_id = "test_sess_001"
    temp_db.start_session(session_id, "Test Driver")

    sess = temp_db.get_session(session_id)
    assert sess is not None
    assert sess["driver_name"] == "Test Driver"
    assert sess["status"] == "ACTIVE"

    temp_db.end_session(session_id, duration_sec=120.5)
    sess_updated = temp_db.get_session(session_id)
    assert sess_updated["status"] == "COMPLETED"
    assert sess_updated["total_duration_sec"] == 120.5

def test_event_logging(temp_db):
    session_id = "test_sess_002"
    temp_db.start_session(session_id)

    event_id = temp_db.log_event(
        session_id=session_id,
        risk_level="CRITICAL",
        event_type="PROLONGED_CLOSURE",
        severity="CRITICAL",
        details="Eyes closed for 3.2 seconds",
        telemetry={"ear": 0.11, "perclos": 0.42},
        sms_sent=True,
        vehicle_stop_simulated=True
    )
    assert event_id > 0

    events = temp_db.get_events(session_id=session_id)
    assert len(events) == 1
    assert events[0]["severity"] == "CRITICAL"
    assert events[0]["sms_sent"] == 1
    assert events[0]["vehicle_stop_simulated"] == 1

def test_telemetry_logging_and_privacy_clear(temp_db):
    session_id = "test_sess_003"
    temp_db.start_session(session_id)

    temp_db.log_telemetry(
        session_id=session_id,
        ear=0.28, mar=0.15, perclos=0.08,
        pitch=-2.0, yaw=1.0, roll=0.0,
        risk_score=22.0, risk_level="NORMAL"
    )

    records = temp_db.get_telemetry(session_id)
    assert len(records) == 1
    assert records[0]["ear"] == 0.28

    # Privacy wipe
    temp_db.clear_all_data()
    assert len(temp_db.get_telemetry(session_id)) == 0
    assert len(temp_db.get_events(session_id)) == 0
    assert len(temp_db.get_all_sessions()) == 0
