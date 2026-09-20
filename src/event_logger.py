"""
DriveGuard AI - Event & Telemetry Logger
Handles non-blocking queuing and persistence of safety events and metrics.
"""

import time
import logging
from typing import Optional, Dict, Any
from database.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriveGuard.EventLogger")

class EventLogger:
    """Buffers and logs telemetry and discrete safety alerts to the SQLite database."""

    def __init__(self, db: Optional[Database] = None, session_id: Optional[str] = None):
        self.db = db or Database()
        self.session_id = session_id or f"session_{int(time.time())}"
        self.last_telemetry_time = 0.0
        self.telemetry_interval = 1.0  # Log telemetry once per second

    def set_session(self, session_id: str, driver_name: str = "Primary Driver"):
        self.session_id = session_id
        self.db.start_session(session_id, driver_name)

    def log_event(self, risk_level: str, event_type: str, severity: str,
                  details: str, telemetry: Optional[Dict[str, Any]] = None,
                  sms_sent: bool = False, vehicle_stop_simulated: bool = False) -> int:
        """Logs an alert or safety incident."""
        logger.info(f"[{severity}] Event '{event_type}': {details}")
        return self.db.log_event(
            session_id=self.session_id,
            risk_level=risk_level,
            event_type=event_type,
            severity=severity,
            details=details,
            telemetry=telemetry,
            sms_sent=sms_sent,
            vehicle_stop_simulated=vehicle_stop_simulated
        )

    def log_telemetry_sample(self, ear: float, mar: float, perclos: float,
                             pitch: float, yaw: float, roll: float,
                             risk_score: float, risk_level: str,
                             dominant_emotion: str = "Neutral",
                             face_detected: bool = True):
        """Throttled logging of telemetry samples (once per second)."""
        now = time.time()
        if now - self.last_telemetry_time >= self.telemetry_interval:
            self.db.log_telemetry(
                session_id=self.session_id,
                ear=ear,
                mar=mar,
                perclos=perclos,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                risk_score=risk_score,
                risk_level=risk_level,
                dominant_emotion=dominant_emotion,
                face_detected=face_detected
            )
            self.last_telemetry_time = now

    def end_current_session(self, duration_sec: float):
        self.db.end_session(self.session_id, duration_sec)
