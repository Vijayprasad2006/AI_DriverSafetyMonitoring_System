"""
DriveGuard AI - SQLite Database Interface
Handles session tracking, safety event logging, and telemetry persistence.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import config

class Database:
    """Manages SQLite connection and operations for driver safety telemetry and events."""

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates tables if they do not exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                driver_name TEXT DEFAULT 'Primary Driver',
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_duration_sec REAL DEFAULT 0.0,
                warning_count INTEGER DEFAULT 0,
                critical_count INTEGER DEFAULT 0,
                yawn_count INTEGER DEFAULT 0,
                microsleep_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'ACTIVE'
            )
            """)

            # Events table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_level TEXT,
                event_type TEXT,
                severity TEXT,
                details TEXT,
                telemetry_snapshot TEXT,
                sms_sent INTEGER DEFAULT 0,
                vehicle_stop_simulated INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
            """)

            # Telemetry logs table (sampled every 1s for analytics)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                ear REAL,
                mar REAL,
                perclos REAL,
                pitch REAL,
                yaw REAL,
                roll REAL,
                risk_score REAL,
                risk_level TEXT,
                dominant_emotion TEXT,
                face_detected INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
            """)

            conn.commit()

    def start_session(self, session_id: str, driver_name: str = "Primary Driver") -> str:
        """Starts a new driver monitoring session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, driver_name, start_time, status)
            VALUES (?, ?, datetime('now', 'localtime'), 'ACTIVE')
            """, (session_id, driver_name))
            conn.commit()
        return session_id

    def end_session(self, session_id: str, duration_sec: float = 0.0):
        """Ends an active session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE sessions
            SET end_time = datetime('now', 'localtime'),
                total_duration_sec = ?,
                status = 'COMPLETED'
            WHERE session_id = ?
            """, (duration_sec, session_id))
            conn.commit()

    def log_event(self, session_id: str, risk_level: str, event_type: str,
                  severity: str, details: str, telemetry: Optional[Dict[str, Any]] = None,
                  sms_sent: bool = False, vehicle_stop_simulated: bool = False) -> int:
        """Logs a critical safety or fatigue event."""
        telemetry_str = json.dumps(telemetry) if telemetry else "{}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO events (session_id, timestamp, risk_level, event_type, severity, details, telemetry_snapshot, sms_sent, vehicle_stop_simulated)
            VALUES (?, datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, risk_level, event_type, severity, details, telemetry_str, int(sms_sent), int(vehicle_stop_simulated)))

            # Update session aggregates
            if severity == "WARNING":
                cursor.execute("UPDATE sessions SET warning_count = warning_count + 1 WHERE session_id = ?", (session_id,))
            elif severity == "CRITICAL":
                cursor.execute("UPDATE sessions SET critical_count = critical_count + 1 WHERE session_id = ?", (session_id,))
            
            if "yawn" in event_type.lower():
                cursor.execute("UPDATE sessions SET yawn_count = yawn_count + 1 WHERE session_id = ?", (session_id,))
            elif "microsleep" in event_type.lower():
                cursor.execute("UPDATE sessions SET microsleep_count = microsleep_count + 1 WHERE session_id = ?", (session_id,))

            conn.commit()
            return cursor.lastrowid

    def log_telemetry(self, session_id: str, ear: float, mar: float, perclos: float,
                      pitch: float, yaw: float, roll: float, risk_score: float,
                      risk_level: str, dominant_emotion: str = "Neutral",
                      face_detected: bool = True):
        """Logs periodic telemetry for analytics charts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO telemetry (session_id, timestamp, ear, mar, perclos, pitch, yaw, roll, risk_score, risk_level, dominant_emotion, face_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, time.time(), ear, mar, perclos, pitch, yaw, roll, risk_score, risk_level, dominant_emotion, int(face_detected)))
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves all sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_events(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves safety events, optionally filtered by session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("SELECT * FROM events WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit))
            else:
                cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_session_events(self, session_id: str, limit: int = 100):
        """Retrieves safety events for a session as a pandas DataFrame."""
        import pandas as pd
        events = self.get_events(session_id=session_id, limit=limit)
        if events:
            return pd.DataFrame(events)
        return pd.DataFrame(columns=["timestamp", "risk_level", "event_type", "severity", "details"])

    def get_telemetry(self, session_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieves telemetry data for charts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM telemetry WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?", (session_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def clear_all_data(self):
        """Wipes all tables for driver privacy."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM telemetry")
            cursor.execute("DELETE FROM events")
            cursor.execute("DELETE FROM sessions")
            conn.commit()
