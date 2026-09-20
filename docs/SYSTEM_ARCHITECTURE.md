# DriveGuard AI — System Architecture & Signal Fusion

## 1. High-Level Dataflow

```mermaid
flowchart TD
    A[Webcam / Video Stream] --> B[Face Detector & Mesh Regressor]
    B --> C{Primary Driver Tracking}
    
    C -->|468 3D Landmarks| D1[Eye Aspect Ratio - EAR]
    C -->|Eye Crops| D2[EyeStateCNN PyTorch]
    C -->|Lip Landmarks| D3[Mouth Aspect Ratio - MAR]
    C -->|3D-2D Correspondences| D4[Head Pose solvePnP]
    C -->|Face Crop| D5[EmotionCNN MobileNetV3]
    
    D1 & D2 --> E1[Temporal Drowsiness & PERCLOS Engine]
    D3 --> E2[Yawn Duration & Frequency Tracker]
    D4 --> E3[Pitch Slump & Nodding Tracker]
    D5 --> E4[FER Probability Distribution]
    
    E1 & E2 & E3 --> F[Driver Risk Assessment Engine]
    
    F -->|Weighted Scoring + Hysteresis| G{Risk Level}
    G -->|NORMAL| H1[Green HUD Banner]
    G -->|WARNING| H2[Amber Audio Tone + Rest Advice]
    G -->|CRITICAL| H3[Red Critical Siren + SMS Dispatch + Safe Stop Sim]
    
    F & E1 & E2 & E3 & E4 --> I[(SQLite Telemetry & Event DB)]
    I --> J[Streamlit Multi-Page Fleet Dashboard]
```

---

## 2. Risk Engine State Machine & Hysteresis

To prevent alert flickering when indicators fluctuate around threshold boundaries, DriveGuard AI implements an asymmetrical state transition machine:

```mermaid
stateDiagram-v2
    [*] --> NORMAL
    
    NORMAL --> WARNING: Score >= 40 for 5 frames
    NORMAL --> CRITICAL: Acute Microsleep OR Score >= 75
    
    WARNING --> CRITICAL: Score >= 75 (Immediate)
    WARNING --> NORMAL: Sustained recovery for 3.0 seconds
    
    CRITICAL --> WARNING: Sustained recovery for 3.0 seconds
    CRITICAL --> NORMAL: Sustained recovery for 6.0 seconds
```

---

## 3. Database Schema (SQLite)

- **`sessions`:**
  - `session_id` (PK, TEXT)
  - `driver_name` (TEXT)
  - `start_time` (TIMESTAMP)
  - `end_time` (TIMESTAMP)
  - `total_duration_sec` (REAL)
  - `warning_count` (INT)
  - `critical_count` (INT)
  - `yawn_count` (INT)
  - `microsleep_count` (INT)
  - `status` (TEXT)

- **`events`:**
  - `id` (PK, INT AUTOINCREMENT)
  - `session_id` (FK, TEXT)
  - `timestamp` (TIMESTAMP)
  - `risk_level` (TEXT)
  - `event_type` (TEXT)
  - `severity` (TEXT)
  - `details` (TEXT)
  - `telemetry_snapshot` (JSON)
  - `sms_sent` (BOOLEAN)
  - `vehicle_stop_simulated` (BOOLEAN)

- **`telemetry`:**
  - `id` (PK, INT AUTOINCREMENT)
  - `session_id` (FK, TEXT)
  - `timestamp` (REAL)
  - `ear` (REAL)
  - `mar` (REAL)
  - `perclos` (REAL)
  - `pitch` (REAL)
  - `yaw` (REAL)
  - `roll` (REAL)
  - `risk_score` (REAL)
  - `risk_level` (TEXT)
  - `dominant_emotion` (TEXT)
  - `face_detected` (BOOLEAN)
