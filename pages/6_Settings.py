"""
DriveGuard AI - Settings, Threshold Calibration & Privacy Controls
Enables dynamic threshold tuning, audio calibration, Twilio credentials,
and privacy data wiping.
"""

import streamlit as st
import config
from src.session_state import init_session_state, load_custom_css

st.set_page_config(page_title="System Configuration | DriveGuard DMS", page_icon="[DMS]", layout="wide")
init_session_state()
load_custom_css()

st.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #1f293d; margin-bottom: 20px;">
    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase;">
        OEM Calibration Protocol &bull; Edge Hardware Parameter Management
    </div>
    <h1 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        System Configuration &amp; Telemetry Calibration Console
    </h1>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
        Calibrate computer vision thresholds, alert trigger latencies, acoustic beacons, and data governance policies.
    </p>
</div>
""", unsafe_allow_html=True)

risk_engine = st.session_state.risk_engine
audio_alerts = st.session_state.audio_alerts
drowsiness_detector = st.session_state.drowsiness_detector
yawn_detector = st.session_state.yawn_detector
head_pose = st.session_state.head_pose
emergency_service = st.session_state.emergency_service
db = st.session_state.db

# Tabs for organization
tab_cv, tab_audio, tab_sms, tab_privacy = st.tabs([
    "[Computer Vision Calibration]",
    "[Acoustic Safety Beacons]",
    "[Telematics Gateway]",
    "[Data Governance & Privacy]"
])

# -------------------------------------------------------------
# TAB 1: VISION & SENSITIVITY THRESHOLDS
# -------------------------------------------------------------
with tab_cv:
    st.markdown("#### [CALIBRATION] Biometric Threshold Tuning")
    st.caption("Calibrate feature extraction bounds for individual cabin geometry, infrared illumination, or driver ergonomics.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Eye Closure (EAR) Parameters")
        new_ear = st.slider("EAR Drowsy Threshold", 0.15, 0.30, float(risk_engine.ear_threshold), 0.01,
                            help="Eye Aspect Ratio below this value flags the eye as closed.")
        new_warn_frames = st.slider("Warning Eye-Closure Frames", 10, 40, int(drowsiness_detector.warning_frames), 1,
                                    help="Consecutive frames of eye closure required for Warning.")
        new_crit_frames = st.slider("Critical Eye-Closure Frames", 30, 90, int(drowsiness_detector.critical_frames), 5,
                                    help="Consecutive frames of eye closure required for Critical alarm.")

        st.markdown("##### Cumulative PERCLOS (P80) Thresholds")
        new_perclos_warn = st.slider("PERCLOS Warning Threshold (%)", 10, 30, int(risk_engine.perclos_warning * 100), 1) / 100.0
        new_perclos_crit = st.slider("PERCLOS Critical Threshold (%)", 25, 60, int(risk_engine.perclos_critical * 100), 1) / 100.0

    with col2:
        st.markdown("##### Yawning (MAR) Parameters")
        new_mar = st.slider("MAR Yawn Threshold", 0.40, 0.80, float(yawn_detector.mar_threshold), 0.02,
                            help="Mouth Aspect Ratio above this value indicates an open mouth.")
        new_yawn_frames = st.slider("Yawn Duration Filter (Frames)", 15, 60, int(yawn_detector.min_frames), 5,
                                    help="Prevents speech from registering as yawns.")

        st.markdown("##### Cranial Pose Deviation Bounds")
        new_pitch = st.slider("Head Slump Downward Pitch (°)", -30.0, -5.0, float(head_pose.pitch_down_thresh), 1.0)
        new_yaw = st.slider("Distraction Yaw Angle (°)", 15.0, 45.0, float(head_pose.yaw_thresh), 1.0)

    if st.button("Persist Vision Thresholds", type="primary"):
        risk_engine.ear_threshold = new_ear
        drowsiness_detector.ear_threshold = new_ear
        drowsiness_detector.warning_frames = new_warn_frames
        drowsiness_detector.critical_frames = new_crit_frames
        risk_engine.perclos_warning = new_perclos_warn
        risk_engine.perclos_critical = new_perclos_crit
        yawn_detector.mar_threshold = new_mar
        yawn_detector.min_frames = new_yawn_frames
        head_pose.pitch_down_thresh = new_pitch
        head_pose.yaw_thresh = new_yaw
        st.success("Vision and sensitivity thresholds successfully applied.")

# -------------------------------------------------------------
# TAB 2: AUDIO ALERTS
# -------------------------------------------------------------
with tab_audio:
    st.markdown("#### [ACOUSTICS] Cabin Auditory Alert Calibration")
    vol = st.slider("Alarm Volume Multiplier", 0.0, 1.0, float(audio_alerts.volume), 0.05)
    enabled = st.toggle("Enable Acoustic Beacons", value=audio_alerts.enabled)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("Dispatch Warning Chime (750 Hz)", use_container_width=True):
            audio_alerts.volume = vol
            audio_alerts.enabled = enabled
            audio_alerts.play_warning(force=True)
            st.toast("Dispatched 750 Hz Advisory Tone.")
    with c_btn2:
        if st.button("Dispatch Emergency Siren (1400 Hz)", use_container_width=True):
            audio_alerts.volume = vol
            audio_alerts.enabled = enabled
            audio_alerts.play_critical(force=True)
            st.toast("Dispatched 1400 Hz Emergency Tone.")

    if st.button("Persist Audio Settings"):
        audio_alerts.volume = vol
        audio_alerts.enabled = enabled
        st.success("Audio beacon configuration saved.")

# -------------------------------------------------------------
# TAB 3: EMERGENCY SMS
# -------------------------------------------------------------
with tab_sms:
    st.markdown("#### [TELEMATICS] Cellular SMS Gateway Configuration")
    st.caption("Configure cloud SMS API credentials for automated telemetry dispatch to emergency responders or fleet telematics.")

    sid = st.text_input("Twilio Account SID", value=emergency_service.account_sid, type="password")
    token = st.text_input("Twilio Auth Token", value=emergency_service.auth_token, type="password")
    from_num = st.text_input("Twilio Sender Phone Number", value=emergency_service.from_number)
    contact_p = st.text_input("Emergency Recipient Number", value=emergency_service.contact_phone)
    contact_n = st.text_input("Fleet Dispatch / Recipient Name", value=emergency_service.contact_name)
    sms_on = st.checkbox("Enable Automated SMS Escalation", value=emergency_service.enabled)

    if st.button("Persist Telematics Credentials"):
        emergency_service.account_sid = sid
        emergency_service.auth_token = token
        emergency_service.from_number = from_num
        emergency_service.contact_phone = contact_p
        emergency_service.contact_name = contact_n
        emergency_service.enabled = sms_on
        st.success("Gateway credentials updated.")

# -------------------------------------------------------------
# TAB 4: PRIVACY & DATA GOVERNANCE
# -------------------------------------------------------------
with tab_privacy:
    st.markdown("#### [GOVERNANCE] Edge Computing & Privacy Architecture")
    st.markdown("""
    DriveGuard DMS enforces strict privacy-by-design standards:
    - **No Cloud Video Transmission:** Video buffers reside strictly in volatile local memory (RAM) and are discarded upon inference completion.
    - **Zero Biometric Identity Storage:** Face landmark extraction operates purely on mathematical coordinate geometry (Euler angles, aspect ratios) without biometric facial recognition or facial embedding databases.
    - **Local SQLite Audit Trail:** Only numerical telemetry time-series and timestamped ISO risk events are stored locally.
    """)

    st.markdown("##### Purge Telemetry & Event Audit Trail")
    st.caption("Permanently purges all historical session logs, time-series telemetry, and safety event records from SQLite.")

    if st.button("Purge Database Records", type="primary"):
        db.clear_all_data()
        st.session_state.drowsiness_detector.reset()
        st.session_state.yawn_detector.reset()
        st.success("All historical database records and session telemetry have been purged.")
