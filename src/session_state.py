"""
DriveGuard AI - Streamlit Shared State Manager
Initializes and coordinates singletons (camera, models, risk engine, alerts)
across Streamlit multi-page application.
"""

import time
import streamlit as st
import config
from database.database import Database
from src.event_logger import EventLogger
from src.risk_engine import RiskEngine
from src.audio_alerts import AudioAlertSystem
from src.emergency_service import EmergencyService
from src.vehicle_stop import VehicleStopSimulation
from src.camera import CameraStream
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.drowsiness_detection import DrowsinessDetector
from src.yawn_detection import YawnDetector
from src.head_pose import HeadPoseEstimator
from src.emotion_detection import EmotionDetector
from models.model_loader import ModelManager

def init_session_state():
    """Ensures all components and state keys are initialized once in st.session_state."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.session_id = f"DG_{int(time.time())}"
        st.session_state.driver_name = "Primary Driver"
        st.session_state.is_monitoring = False
        st.session_state.session_start_time = time.time()
        
        # Core Singletons
        st.session_state.db = Database()
        st.session_state.event_logger = EventLogger(st.session_state.db, st.session_state.session_id)
        st.session_state.db.start_session(st.session_state.session_id, st.session_state.driver_name)
        
        st.session_state.risk_engine = RiskEngine()
        st.session_state.audio_alerts = AudioAlertSystem()
        st.session_state.emergency_service = EmergencyService()
        st.session_state.vehicle_stop = VehicleStopSimulation()
        
        # CV components
        st.session_state.camera = CameraStream()
        st.session_state.face_detector = FaceDetector()
        st.session_state.eye_detector = EyeDetector()
        st.session_state.drowsiness_detector = DrowsinessDetector()
        st.session_state.yawn_detector = YawnDetector()
        st.session_state.head_pose = HeadPoseEstimator()
        st.session_state.emotion_detector = EmotionDetector()
        st.session_state.model_manager = ModelManager()

        # Instantaneous telemetry store
        st.session_state.telemetry = {
            "ear": 0.32,
            "mar": 0.18,
            "perclos": 0.05,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "risk_score": 12.0,
            "risk_level": config.RISK_LEVEL_NORMAL,
            "dominant_emotion": "Neutral",
            "emotion_confidence": 0.88,
            "face_detected": True,
            "reasons": ["Normal driver alertness maintained."],
            "recommendation": "Driver attentive. Safe to continue driving.",
            "is_yawning": False,
            "is_nodding": False,
            "is_closed": False,
            "fps": 0.0
        }

    # Always ensure telemetry bridge daemon is running and sync live telemetry
    from src.telemetry_bridge import start_telemetry_bridge, SHARED_TELEMETRY
    start_telemetry_bridge(st.session_state.event_logger)
    if "telemetry" in st.session_state and SHARED_TELEMETRY.get("timestamp", 0) > 0:
        st.session_state.telemetry.update(SHARED_TELEMETRY)

def load_custom_css():
    """Injects custom CSS design system into current Streamlit page."""
    css_file = config.ASSETS_DIR / "styles.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
