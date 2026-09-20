"""
DriveGuard AI - Facial Emotion Analysis Dashboard
Displays deep learning facial expression recognition (FER) predictions,
confidence distributions, and expression timelines with ethical safety disclosures.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
from src.session_state import init_session_state, load_custom_css

st.set_page_config(page_title="Affective Telemetry | DriveGuard DMS", page_icon="[DMS]", layout="wide")
init_session_state()
load_custom_css()

st.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #1f293d; margin-bottom: 20px;">
    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase;">
        ISO/TR 21959 Human Factors &bull; Ocular &amp; Facial Affective Telemetry
    </div>
    <h1 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        In-Cabin Driver Affective &amp; Expression Telemetry
    </h1>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
        Computer vision classification across standard facial micro-expression vectors per ISO/TR 21959 human factor guidelines.
    </p>
</div>
""", unsafe_allow_html=True)

# Ethical & Scientific Disclaimer Alert
st.markdown("""
<div style="background-color: rgba(15, 23, 42, 0.8); border-left: 3px solid #38bdf8; border: 1px solid #1e293b; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
        [REGULATORY &amp; SCIENTIFIC DISCLOSURE &bull; ISO/IEC 23894 AI RISK MANAGEMENT]
    </div>
    <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
        Facial expression classification measures superficial facial muscle movements and Action Units (AUs). In strict accordance with automotive functional safety (ISO 26262), affective indicators serve exclusively as secondary context and <strong>never unilaterally trigger emergency vehicle intervention protocols</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

tel = st.session_state.telemetry
emotion_detector = st.session_state.emotion_detector
model_manager = st.session_state.model_manager

# Current dominant emotion & confidence
dominant_emotion = tel.get("dominant_emotion", "Neutral")
confidence = tel.get("emotion_confidence", 0.82)

# -------------------------------------------------------------
# 1. TOP CARDS
# -------------------------------------------------------------
e1, e2, e3 = st.columns([1.2, 1, 1])

with e1:
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Observed Expression Vector</div>
        <div style="font-size: 2.0rem; font-weight: 800; color: #f8fafc; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">{dominant_emotion.upper()}</div>
        <div class="metric-subtext">Confidence Score: <strong>{confidence*100:.1f}%</strong></div>
    </div>
    """, unsafe_allow_html=True)

with e2:
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Neural Architecture</div>
        <div style="font-size: 1.2rem; font-weight: 700; margin-top: 10px; font-family: 'JetBrains Mono', monospace;">MobileNetV3-Small</div>
        <div class="metric-subtext">Transfer Learned on FER-2013 | 7 Vectors</div>
    </div>
    """, unsafe_allow_html=True)

with e3:
    diag = model_manager.get_system_diagnostics()
    latency = diag["latencies_ms"].get("emotion_classifier_ms", 12.0)
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Inference Latency</div>
        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 5px; font-family: 'JetBrains Mono', monospace;">{latency:.1f} ms</div>
        <div class="metric-subtext">Compute Unit: <strong>{diag['device'].upper()}</strong></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. PROBABILITY DISTRIBUTION ACROSS 7 CLASSES
# -------------------------------------------------------------
col_chart1, col_chart2 = st.columns([1.2, 1.0])

# Get distribution
dist = emotion_detector.cached_result.get("distribution", {})
if not dist:
    dist = {
        "Neutral": 0.65, "Happy": 0.12, "Surprised": 0.08,
        "Sad": 0.06, "Angry": 0.04, "Fearful": 0.03, "Disgusted": 0.02
    }

dist_df = pd.DataFrame({
    "Emotion": list(dist.keys()),
    "Probability": [v * 100 for v in dist.values()]
}).sort_values(by="Probability", ascending=False)

with col_chart1:
    st.markdown("#### [DISTRIBUTION] Affective Class Probabilities")
    fig_bar = px.bar(
        dist_df, x="Emotion", y="Probability",
        color="Probability",
        color_continuous_scale=["#1e293b", "#38bdf8"],
        labels={"Probability": "Confidence (%)"}
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(15, 23, 42, 0.6)', plot_bgcolor='rgba(11, 15, 25, 0.8)',
        font=dict(color='#94a3b8', family="JetBrains Mono"),
        yaxis=dict(range=[0, 100], gridcolor="#1e293b"),
        margin=dict(l=30, r=20, t=20, b=30), height=300,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    st.markdown("#### [RADAR] Expression Vector Topology")
    categories = list(dist.keys())
    values = list(dist.values())
    values.append(values[0])
    categories.append(categories[0])

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.12)',
        line=dict(color='#38bdf8', width=2)
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], gridcolor="#1e293b"),
            angularaxis=dict(gridcolor="#1e293b")
        ),
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color='#94a3b8', family="JetBrains Mono"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=300
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------------
# 3. TEMPORAL EMOTION TIMELINE
# -------------------------------------------------------------
st.markdown("#### [CHRONOLOGY] Rolling Facial Expression Transitions")
history = emotion_detector.get_history()
if history:
    t0 = history[0][0]
    hist_df = pd.DataFrame([
        {"Time (s)": round(t - t0, 1), "Emotion": emo, "Confidence": conf * 100}
        for t, emo, conf in history
    ])
    fig_hist = px.scatter(
        hist_df, x="Time (s)", y="Emotion",
        color="Confidence", size="Confidence",
        color_continuous_scale="Blues",
        title="Detected Expression Trajectory"
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(15, 23, 42, 0.6)', plot_bgcolor='rgba(11, 15, 25, 0.8)',
        font=dict(color='#94a3b8', family="JetBrains Mono"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(l=30, r=20, t=40, b=30), height=260
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.caption("Awaiting live session telemetry stream. Start camera monitoring to log expression transitions.")
