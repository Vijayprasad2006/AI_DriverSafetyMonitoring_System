"""
DriveGuard AI - Emergency Center & Safe Vehicle Stop Protocol
Manages emergency alert workflows, Twilio SMS dispatch, audio alarms,
and the simulated vehicle pullover sequence.
"""

import time
import streamlit as st
import config
from src.session_state import init_session_state, load_custom_css

st.set_page_config(page_title="Intervention Protocols | DriveGuard DMS", page_icon="[DMS]", layout="wide")
init_session_state()
load_custom_css()

st.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #1f293d; margin-bottom: 20px;">
    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase;">
        SAE J3016 Fallback &bull; ISO 26262 Emergency Escalation Gateway
    </div>
    <h1 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        Emergency Dispatch &amp; Intervention Protocol Console
    </h1>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
        Automated incident escalation, telematics SMS alerts, dual-frequency acoustic alarms, and autonomous safe-stop sequence.
    </p>
</div>
""", unsafe_allow_html=True)

# Mandatory Automotive Disclaimer
st.markdown("""
<div style="background-color: rgba(15, 23, 42, 0.8); border-left: 3px solid #ef4444; border: 1px solid #1e293b; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #f87171; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
        [FUNCTIONAL SAFETY MANDATE &bull; SIMULATION NOTICE]
    </div>
    <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        DriveGuard DMS safe-pullover routines operate in software hardware-in-the-loop (HIL) simulation mode. No direct CAN-bus actuator control or physical braking override is commanded without certified ISO 26262 ASIL-D gateway verification.
    </div>
</div>
""", unsafe_allow_html=True)

vehicle_stop = st.session_state.vehicle_stop
emergency_service = st.session_state.emergency_service
audio_alerts = st.session_state.audio_alerts

# Progress simulation if active
stop_status = vehicle_stop.update_simulation()

# -------------------------------------------------------------
# 1. VEHICLE STOP SIMULATION BANNER
# -------------------------------------------------------------
if stop_status["is_active"]:
    stage = stop_status["stage"]
    speed = stop_status["speed_kmh"]
    st.markdown(f"""
    <div class="vehicle-stop-banner" style="background: rgba(69, 10, 10, 0.5); border: 1px solid #ef4444; border-radius: 4px; padding: 14px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 800; color: #f87171; letter-spacing: 0.08em;">
                [AUTONOMOUS VEHICLE SAFE-STOP SEQUENCE ENGAGED &bull; SIMULATION PROTOCOL]
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                Phase: <strong>{stage}</strong> | Regulated Deceleration: <strong>{speed} km/h</strong> | Hazard Beacon: <strong>ACTIVE</strong>
            </div>
        </div>
        <div>
            <span class="badge-critical" style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 6px 14px; border: 1px solid #ef4444; border-radius: 2px;">PULLOVER ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. EMERGENCY STATUS CARDS
# -------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card {'critical' if stop_status['is_active'] else 'normal'}">
        <div class="metric-title">Safe Stop Sequence</div>
        <div style="font-size: 1.5rem; font-weight: 800; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{stop_status['stage']}</div>
        <div class="metric-subtext">Velocity: <strong>{stop_status['speed_kmh']} km/h</strong> (Elapsed: {stop_status['elapsed_seconds']}s)</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    sms_status = emergency_service.last_status
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Telematics SMS Dispatch</div>
        <div style="font-size: 1.4rem; font-weight: 700; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{sms_status['status']}</div>
        <div class="metric-subtext">Mode: <strong>{sms_status['mode']}</strong> | Transmissions: <strong>{emergency_service.sent_count}</strong></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Acoustic Safety Beacons</div>
        <div style="font-size: 1.4rem; font-weight: 700; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">ONLINE</div>
        <div class="metric-subtext">Calibrated Volume: <strong>{int(audio_alerts.volume*100)}%</strong> | Synthesizer: <strong>Armed</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. INTERACTIVE PROTOCOL CONTROLS
# -------------------------------------------------------------
st.markdown("#### [CONTROLS] Emergency Protocol Execution Gateway")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### Autonomous Safe-Stop Simulation")
    st.caption("Executes the ISO/TR 21959 graduated emergency deceleration protocol upon sustained driver unresponsiveness.")
    
    stop_col1, stop_col2 = st.columns(2)
    with stop_col1:
        if st.button("Trigger Safe-Stop Sequence", type="primary", use_container_width=True):
            vehicle_stop.trigger_stop_request("Manual Test Dispatch", 95.0)
            st.rerun()
    with stop_col2:
        if st.button("Manual Driver Override", use_container_width=True):
            vehicle_stop.cancel_stop_request()
            st.rerun()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Acoustic Beacon Diagnostic Check")
    audio_col1, audio_col2 = st.columns(2)
    with audio_col1:
        if st.button("Auditory Advisory Chime (750 Hz)", use_container_width=True):
            audio_alerts.play_warning(force=True)
            st.toast("Dispatched 750 Hz Advisory Tone.")
    with audio_col2:
        if st.button("Emergency Siren Pulse (1400 Hz)", use_container_width=True):
            audio_alerts.play_critical(force=True)
            st.toast("Dispatched 1400 Hz Emergency Tone.")

with col_right:
    st.markdown("##### Telematics Emergency Dispatch Registry")
    contact_name = st.text_input("Fleet Dispatch / Contact Name", value=emergency_service.contact_name)
    contact_phone = st.text_input("Emergency Telemetry Phone Number", value=emergency_service.contact_phone)
    sms_enabled = st.checkbox("Enable Automated Incident Escalation", value=emergency_service.enabled)

    if st.button("Save Gateway Configuration", use_container_width=True):
        emergency_service.contact_name = contact_name
        emergency_service.contact_phone = contact_phone
        emergency_service.enabled = sms_enabled
        st.success("Gateway configuration successfully persisted.")

    if st.button("Send Test Telematics SMS", use_container_width=True):
        res = emergency_service.send_emergency_alert("Manual Operator Gateway Test", 88.0, force=True)
        st.info(f"Gateway Response: {res.get('status')} &bull; {res.get('message')}")

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. EMERGENCY EVENT AUDIT LOG
# -------------------------------------------------------------
st.markdown("#### [AUDIT TRAIL] Critical Safety Incident Log")
events = st.session_state.db.get_events(limit=25)
critical_events = [e for e in events if e["severity"] == "CRITICAL" or e["vehicle_stop_simulated"] or e["sms_sent"]]

if critical_events:
    audit_data = []
    for ev in critical_events:
        audit_data.append({
            "Timestamp": ev["timestamp"],
            "Incident Class": ev["event_type"],
            "Severity": ev["severity"],
            "Details": ev["details"],
            "SMS Status": "Transmitted" if ev["sms_sent"] else "None",
            "Safe-Stop Status": "Engaged" if ev["vehicle_stop_simulated"] else "Inactive"
        })
    st.dataframe(audit_data, use_container_width=True)
else:
    st.caption("Incident register nominal. No critical escalations logged.")
