"""
DriveGuard AI - Real-time Telemetry Bridge Server
Lightweight HTTP micro-daemon on localhost:8502 that receives 30 FPS client-side
computer vision telemetry from the browser HUD and ingests it into Python's
SQLite database and shared state.
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger("DriveGuard.TelemetryBridge")

# Global in-memory telemetry state accessible across threads
SHARED_TELEMETRY = {
    "ear": 0.32,
    "mar": 0.18,
    "perclos": 0.05,
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "risk_score": 12.0,
    "risk_level": "NORMAL",
    "dominant_emotion": "Neutral",
    "emotion_confidence": 0.88,
    "face_detected": True,
    "reasons": ["Normal driver alertness maintained."],
    "recommendation": "Driver attentive. Safe to continue driving.",
    "is_yawning": False,
    "is_nodding": False,
    "is_closed": False,
    "fps": 30.0,
    "timestamp": 0.0
}

_server_instance: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_event_logger_ref = None

class TelemetryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence routine request logging to keep console clean
        pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(SHARED_TELEMETRY).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _event_logger_ref
        if self.path == "/api/telemetry":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                data = json.loads(post_body.decode("utf-8"))
                
                # Update in-memory telemetry
                for k, v in data.items():
                    if k in SHARED_TELEMETRY:
                        SHARED_TELEMETRY[k] = v
                
                # Forward to SQLite event logger if available
                if _event_logger_ref is not None:
                    _event_logger_ref.log_telemetry_sample(
                        ear=float(data.get("ear", 0.32)),
                        mar=float(data.get("mar", 0.18)),
                        perclos=float(data.get("perclos", 0.0)),
                        pitch=float(data.get("pitch", 0.0)),
                        yaw=float(data.get("yaw", 0.0)),
                        roll=float(data.get("roll", 0.0)),
                        risk_score=float(data.get("risk_score", 10.0)),
                        risk_level=str(data.get("risk_level", "NORMAL")),
                        dominant_emotion=str(data.get("dominant_emotion", "Neutral")),
                        face_detected=bool(data.get("face_detected", True))
                    )
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def start_telemetry_bridge(event_logger=None, port: int = 8502) -> bool:
    """Starts the background telemetry HTTP server if not already running."""
    global _server_instance, _server_thread, _event_logger_ref
    if event_logger is not None:
        _event_logger_ref = event_logger

    if _server_instance is not None:
        return True

    try:
        server_address = ("127.0.0.1", port)
        _server_instance = HTTPServer(server_address, TelemetryHandler)
        _server_thread = threading.Thread(target=_server_instance.serve_forever, daemon=True)
        _server_thread.start()
        logger.info(f"Telemetry bridge server active on http://127.0.0.1:{port}")
        return True
    except Exception as e:
        logger.warning(f"Could not bind telemetry bridge server on port {port}: {e}")
        return False
