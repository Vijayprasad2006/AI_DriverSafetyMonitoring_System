# DRIVEGUARD AI 🛡️
### Advanced AI-Powered Driver Safety, Drowsiness & Emotion Monitoring System

DriveGuard AI is an advanced, research-grade automotive safety platform that uses laptop/vehicle webcam video feeds to continuously monitor driver alertness, fatigue, distraction, and visible facial expressions in real-time.

The system combines geometric landmark estimation (MediaPipe Face Mesh), deep convolutional neural networks (PyTorch EyeStateCNN & MobileNetV3 FER), a weighted risk assessment engine with temporal hysteresis, non-blocking audio alarms, Twilio SMS notifications, a simulated vehicle-stop sequence, and a modern Streamlit fleet dashboard.

---

## 🌟 Key Features

1. **Face Mesh & Primary Driver Lock:**
   - Detects 468 3D facial landmarks in real time.
   - Primary face tracking locks onto the driver using bounding box area and frame-center proximity, preventing accidental switching when passengers are visible.
   - Failsafe alert when the driver's face is obscured or absent for >3 seconds.

2. **Dual-Tier Drowsiness & Fatigue Detection:**
   - **Geometric Eye Aspect Ratio (EAR):** Sub-millisecond calculation of eyelid closure.
   - **PyTorch EyeStateCNN:** Deep learning verification on normalized eye patches.
   - **Temporal PERCLOS:** Moving percentage of eye closure over 30-second rolling windows.
   - **Microsleep Detection:** Instant trigger when sustained closure exceeds 2.0 seconds.

3. **Yawn Detection & Duration Filter (MAR):**
   - Inner and outer lip Mouth Aspect Ratio (MAR).
   - Validates mouth opening duration ($\ge 1.0\text{s}$) to avoid false-positives from speaking.
   - Tracks 3-minute rolling yawn frequency.

4. **3D Head Pose & Nodding Estimation:**
   - Fits 3D canonical facial model points to 2D landmarks via `cv2.solvePnP`.
   - Derives Euler angles: Pitch (downward head slump), Yaw (distraction left/right), Roll (tilt).
   - Detects periodic nodding oscillations indicative of falling asleep.

5. **Facial Expression Recognition (FER):**
   - PyTorch MobileNetV3-Small classifier predicting 7 emotional classes: `Neutral`, `Happy`, `Sad`, `Surprised`, `Angry`, `Fearful`, `Disgusted`.
   - Contextual awareness metric (explicitly decoupled from safety-critical braking).

6. **Driver Risk Engine with Hysteresis:**
   - Synthesizes all fatigue indicators into an explainable score ($0 - 100$).
   - Asymmetrical hysteresis prevents alert flickering between `NORMAL`, `WARNING`, and `CRITICAL`.
   - Provides clear text explanations and actionable rest recommendations.

7. **Multi-Tier Emergency Services:**
   - **Dual Audio Alarms:** Threaded, non-blocking warning tones and critical sirens with alert cooldowns.
   - **Twilio SMS Alerts:** Cloud SMS alerts to emergency contacts with rate limiting and simulation fallback mode.
   - **Vehicle Stop Simulation:** Multi-stage pullover sequence (Hazards -> Lane Change -> Deceleration -> Safe Stop).

8. **Telemetry & Event Persistence (SQLite):**
   - Logs 1Hz telemetry time-series and safety events to local SQLite database (`driveguard.db`).
   - One-click privacy purge erases all driver history.

---

## 📁 Project Structure

```
DriveGuard-AI/
├── app.py                          # Overview Dashboard & Navigation
├── config.py                       # Thresholds, weights, audio/SMS settings
├── requirements.txt                # Dependencies
├── README.md                       # Comprehensive Guide
├── .env.example                    # Environment credentials template
├── .gitignore                      # Git rules
├── conftest.py                     # Pytest path configuration
│
├── assets/
│   ├── styles.css                  # Fleet dark design system
│   └── sounds/                     # Synthesized warning and critical WAV files
│
├── models/
│   ├── model_loader.py             # Singleton manager & latency profiler
│   ├── eye_classifier.py           # PyTorch EyeStateCNN (Open/Closed)
│   ├── emotion_classifier.py       # PyTorch MobileNetV3-Small FER
│   └── weights/                    # Saved weights (.pth)
│
├── src/
│   ├── camera.py                   # Resilient webcam & synthetic stream
│   ├── face_detection.py           # MediaPipe Mesh & primary face tracking
│   ├── eye_detection.py            # EAR calculation & eye patch crops
│   ├── drowsiness_detection.py     # Consecutive closure & PERCLOS
│   ├── yawn_detection.py           # MAR calculation & yawn duration
│   ├── head_pose.py                # solvePnP Euler angles & nodding
│   ├── emotion_detection.py        # FER inference wrapper
│   ├── risk_engine.py              # Weighted risk engine with hysteresis
│   ├── audio_alerts.py             # Non-blocking audio sirens
│   ├── emergency_service.py        # Twilio SMS dispatch & simulation
│   ├── vehicle_stop.py             # Safe pullover simulation
│   ├── event_logger.py             # Telemetry bridge to SQLite
│   └── session_state.py            # Streamlit multi-page state coordinator
│
├── pages/
│   ├── 1_Live_Monitoring.py        # Live webcam stream with HUD telemetry
│   ├── 2_Fatigue_Analytics.py      # Plotly timelines (EAR, MAR, PERCLOS)
│   ├── 3_Emotion_Analysis.py       # Emotion radar & distribution charts
│   ├── 4_Emergency_Center.py       # SMS controls & vehicle stop simulation
│   ├── 5_Model_Performance.py      # Latency benchmarks & evaluation report
│   └── 6_Settings.py               # Threshold calibration & privacy purge
│
├── training/
│   ├── train_eye_model.py          # PyTorch training pipeline for Eye CNN
│   ├── train_emotion_model.py      # Fine-tuning pipeline for Emotion CNN
│   └── evaluate_models.py          # Empirical evaluation & confusion matrix
│
├── database/
│   └── database.py                 # SQLite database schema & CRUD operations
│
├── tests/
│   ├── test_risk_engine.py         # Unit tests for scoring & hysteresis
│   ├── test_drowsiness.py          # Unit tests for EAR & PERCLOS
│   ├── test_emergency_service.py   # Unit tests for SMS & cooldowns
│   └── test_database.py            # Unit tests for SQLite persistence
│
└── docs/
    ├── MODEL_CARD.md               # Specifications, ethics & limitations
    ├── DATASET_GUIDE.md            # MRL Eye, CEW & FER-2013 setup
    └── SYSTEM_ARCHITECTURE.md      # Dataflow & state machine diagrams
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Standard laptop webcam

### 2. Clone and Setup Environment
```powershell
# Navigate to workspace
cd AI_Driver_Safety_Monitoring

# Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Credentials (Optional)
Copy `.env.example` to `.env` to configure Twilio credentials:
```powershell
copy .env.example .env
```
> **Note:** If Twilio credentials are not set, DriveGuard AI automatically operates in **High-Fidelity Simulation Mode** so all SMS alert workflows remain fully demonstrable!

### 4. Launch the Application
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite covering database persistence, drowsiness tracking, risk scoring, and emergency workflows:

```powershell
python -m pytest tests/ -v
```

Expected output:
```
tests/test_database.py::test_session_lifecycle PASSED
tests/test_database.py::test_event_logging PASSED
tests/test_database.py::test_telemetry_logging_and_privacy_clear PASSED
tests/test_drowsiness.py::test_ear_calculation PASSED
tests/test_drowsiness.py::test_drowsiness_tracker_consecutive_closed PASSED
tests/test_drowsiness.py::test_drowsiness_microsleep_trigger PASSED
tests/test_emergency_service.py::test_emergency_simulation_mode PASSED
tests/test_emergency_service.py::test_emergency_cooldown_suppression PASSED
tests/test_emergency_service.py::test_emergency_disabled PASSED
tests/test_risk_engine.py::test_risk_engine_normal_state PASSED
tests/test_risk_engine.py::test_risk_engine_critical_eye_closure PASSED
tests/test_risk_engine.py::test_risk_engine_missing_face PASSED

================ 12 passed in 0.21s =================
```

---

## 🏋️ Training & Fine-Tuning Pipelines

DriveGuard AI includes standalone PyTorch training workflows with data augmentation, validation splits, and checkpointing:

### Train the Eye State Model:
```powershell
python training/train_eye_model.py --epochs 5 --batch_size 32 --device cpu
```

### Fine-Tune the Facial Emotion Model:
```powershell
python training/train_emotion_model.py --epochs 3 --batch_size 16 --device cpu
```

### Evaluate Models:
```powershell
python training/evaluate_models.py --weights_path models/weights/eye_model.pth
```
This updates `database/evaluation_report.json` with genuine empirical Accuracy, Precision, Recall, F1-Score, and Confusion Matrix metrics displayed on the **Model Performance** page.

---

## 🛡️ Privacy & Safety Governance

- **Zero Cloud Streaming:** Video frames are analyzed in local host RAM and immediately discarded. No video is ever uploaded or retained.
- **Biometric Non-Identification:** MediaPipe detects geometric points; no face recognition, identity hashing, or driver indexing is performed.
- **Driver Consent & Purge:** Telemetry consists of numeric aggregates. Drivers can wipe all session history with one click from the Settings page.

---

## ⚠️ Research Disclaimer
DriveGuard AI is an advanced AI research prototype and is not a certified automotive safety device (ASIL ISO 26262). It must not be deployed to directly actuate real physical vehicle braking or steering systems.
