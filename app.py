"""
Driver Drowsiness and Safety Monitoring System
Clean, basic, straightforward interface.
"""

import time
import streamlit as st
import pandas as pd
import config
from src.session_state import init_session_state, load_custom_css

st.set_page_config(
    page_title="Driver Safety Monitoring",
    page_icon="🚗",
    layout="wide"
)

init_session_state()
load_custom_css()

# Sync latest live telemetry
from src.telemetry_bridge import SHARED_TELEMETRY
if SHARED_TELEMETRY.get("timestamp", 0) > 0:
    st.session_state.telemetry.update(SHARED_TELEMETRY)

tel = st.session_state.telemetry
risk_level = tel.get("risk_level", config.RISK_LEVEL_NORMAL)
risk_score = tel.get("risk_score", 10.0)
ear = tel.get("ear", 0.30)
mar = tel.get("mar", 0.18)
pitch = tel.get("pitch", 0.0)
yaw = tel.get("yaw", 0.0)
dominant_emotion = tel.get("dominant_emotion", "Neutral")
exact_driver_state = tel.get("exact_driver_state", "ATTENTIVE (NORMAL)")

st.title("🚗 Driver Drowsiness and Safety Monitoring System")
st.write("A computer vision and deep learning system to detect driver fatigue, eye closure (EAR), yawning (MAR), and head distraction in real time.")

st.divider()

# Status Banner
if risk_level == "CRITICAL":
    st.error(f"🚨 CRITICAL ALERT: {exact_driver_state} (Risk Score: {risk_score:.0f}/100)")
elif risk_level == "WARNING":
    st.warning(f"⚠️ WARNING: {exact_driver_state} (Risk Score: {risk_score:.0f}/100)")
else:
    st.success(f"✅ DRIVER STATUS: {exact_driver_state} (Risk Score: {risk_score:.0f}/100)")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Driver State", value=exact_driver_state)
with col2:
    st.metric(label="Eye Aspect Ratio (EAR)", value=f"{ear:.2f}", delta="Open" if ear >= config.EAR_DROWSY_THRESHOLD else "Closed")
with col3:
    st.metric(label="Mouth Aspect Ratio (MAR)", value=f"{mar:.2f}", delta="Normal" if mar < config.MAR_YAWN_THRESHOLD else "Yawning")
with col4:
    st.metric(label="Head Pitch", value=f"{pitch:.1f}°", delta="Looking Forward" if pitch > config.HEAD_PITCH_DOWN_THRESHOLD else "Slump")

st.divider()

# Navigation & Features Overview
st.subheader("System Modules (Select from Sidebar)")
m1, m2, m3 = st.columns(3)
with m1:
    st.info("📹 **1. Live Video Monitoring**\n\nOpen webcam or test video to view real-time face detection, eye landmarks, and drowsiness prediction.")
with m2:
    st.info("📈 **2. Fatigue Analytics**\n\nView historical time-series graphs of EAR, PERCLOS fatigue curves, and yawn counts.")
with m3:
    st.info("🎭 **3. Emotion Analysis**\n\nInspect 7-class facial emotion predictions and confidence charts.")

n1, n2, n3 = st.columns(3)
with n1:
    st.info("🚨 **4. Emergency Center**\n\nTest acoustic alerts (750 Hz / 1400 Hz), SMS notifications, and simulated vehicle safe stop.")
with n2:
    st.info("🎓 **5. Model Performance and Dataset**\n\nExplore 168 real driver images from the Simuletic DMS dataset and inspect test accuracy (91.8%).")
with n3:
    st.info("⚙️ **6. Settings and Thresholds**\n\nAdjust detection thresholds for EAR (0.22), MAR (0.60), and buzzer volume.")

st.divider()

# Recent Events Log
st.subheader("📋 Recent Safety Events")
events = st.session_state.db.get_events(limit=5)
if events:
    df_events = pd.DataFrame(events)[["timestamp", "risk_level", "event_type", "severity", "details"]]
    st.dataframe(df_events, use_container_width=True, hide_index=True)
else:
    st.caption("No events logged yet. Start live monitoring from the sidebar to record sessions.")
