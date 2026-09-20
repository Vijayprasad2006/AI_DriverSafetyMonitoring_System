"""
DriveGuard AI - Global Configuration & Parameter Hub
Defines all detection thresholds, risk scoring weights, audio settings,
SMS settings, database paths, and UI theme constants.
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
DATA_DIR = BASE_DIR / "database"
SOUNDS_DIR = ASSETS_DIR / "sounds"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = str(DATA_DIR / "driveguard.db")

# Webcam and Vision Defaults
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
TARGET_FPS = 30

# Drowsiness & Eye Aspect Ratio (EAR) Thresholds
EAR_DROWSY_THRESHOLD = 0.22      # Eye considered closed if EAR is below this
EAR_CLOSED_THRESHOLD = 0.18      # Definite eye closure
DROWSINESS_FRAMES_WARNING = 20   # ~0.65 - 0.75 seconds at 30 FPS
DROWSINESS_FRAMES_CRITICAL = 45  # ~1.5 seconds at 30 FPS
MICROSLEEP_FRAMES_THRESHOLD = 60 # ~2.0 seconds sustained closure

# MediaPipe 468 Mesh Eye Landmark Indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# PERCLOS (Percentage of Eye Closure over moving time window)
PERCLOS_WINDOW_SECONDS = 30
PERCLOS_WARNING_THRESHOLD = 0.20  # 20% closure in 30s window
PERCLOS_CRITICAL_THRESHOLD = 0.35 # 35% closure in 30s window

# Yawning & Mouth Aspect Ratio (MAR) Thresholds
MAR_YAWN_THRESHOLD = 0.60        # Mouth open wide
YAWN_FRAMES_THRESHOLD = 30       # Must persist for ~1.0s to confirm yawn (not speaking)
YAWN_COOLDOWN_SECONDS = 3.0      # Cooldown before counting another yawn
YAWN_WARNING_RATE = 2            # 2 yawns in 3 minutes -> Warning
YAWN_CRITICAL_RATE = 4           # 4+ yawns in 3 minutes -> Critical

# Head Pose Estimation Thresholds (Euler angles in degrees)
HEAD_PITCH_DOWN_THRESHOLD = -15.0 # Chin tilted downwards
HEAD_PITCH_UP_THRESHOLD = 20.0    # Tilted up
HEAD_YAW_LEFT_THRESHOLD = -25.0   # Looking left
HEAD_YAW_RIGHT_THRESHOLD = 25.0   # Looking right
HEAD_ROLL_TILT_THRESHOLD = 20.0   # Extreme side tilt
NODDING_OSCILLATION_THRESHOLD = 3 # 3 nod cycles within 10s window

# Face Tracking & Safety Failsafe
MAX_MISSING_FACE_SECONDS = 3.0   # If driver face is absent for >3s, trigger warning

# Risk Scoring Weights (Total scale 0 - 100)
WEIGHT_EAR_CLOSURE = 35          # Weight for active eye closure duration
WEIGHT_PERCLOS = 25              # Weight for moving PERCLOS score
WEIGHT_YAWN_FREQUENCY = 15       # Weight for recent yawning rate
WEIGHT_HEAD_POSE = 15            # Weight for downward gaze or nodding
WEIGHT_FACE_PRESENCE = 10        # Weight for face tracking status

# Risk Engine Categories
RISK_LEVEL_NORMAL = "NORMAL"
RISK_LEVEL_WARNING = "WARNING"
RISK_LEVEL_CRITICAL = "CRITICAL"

RISK_SCORE_WARNING_MIN = 40.0
RISK_SCORE_CRITICAL_MIN = 75.0

# Hysteresis & Temporal Smoothing
RISK_ESCALATION_FRAMES = 5       # Frames sustained before upgrading risk level
RISK_RECOVERY_SECONDS = 3.0      # Seconds of normal behavior before downgrading risk level
EMA_ALPHA = 0.25                 # Exponential Moving Average factor for telemetry

# Emotion Recognition Classes
EMOTION_CLASSES = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprised",
    "Angry",
    "Fearful",
    "Disgusted"
]

# Audio Alerts
AUDIO_COOLDOWN_WARNING = 5.0     # Minimum seconds between warning beeps
AUDIO_COOLDOWN_CRITICAL = 3.0    # Minimum seconds between critical sirens
DEFAULT_AUDIO_VOLUME = 0.8       # Range 0.0 to 1.0

# Twilio SMS Configuration (Override with environment variables or UI)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
EMERGENCY_CONTACT_PHONE = os.getenv("EMERGENCY_CONTACT_PHONE", "")
EMERGENCY_CONTACT_NAME = os.getenv("EMERGENCY_CONTACT_NAME", "Emergency Contact")
SMS_COOLDOWN_SECONDS = 120.0     # Prevent spamming emergency contacts (2 min cooldown)
ENABLE_SMS_ALERTS = os.getenv("ENABLE_SMS_ALERTS", "false").lower() in ("true", "1", "yes")

# UI Styling Constants
THEME_COLORS = {
    "background": "#0b0f19",
    "card_bg": "#131a29",
    "card_border": "#1f293d",
    "accent_cyan": "#00f0ff",
    "accent_blue": "#38bdf8",
    "status_normal": "#10b981",    # Emerald green
    "status_warning": "#f59e0b",   # Amber
    "status_critical": "#ef4444",  # Crimson red
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "grid_color": "#1e293b"
}
