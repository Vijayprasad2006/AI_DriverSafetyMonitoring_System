"""
DriveGuard AI - Model Performance & Verification Suite
Inspects architecture parameters, hardware latency profiling,
and genuine evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
NOTE: Never displays fabricated metrics. If unevaluated, explicitly marked as 'Not Evaluated'.
"""

import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
from src.session_state import init_session_state, load_custom_css

st.set_page_config(page_title="Model Verification | DriveGuard DMS", page_icon="[DMS]", layout="wide")
init_session_state()
load_custom_css()

st.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #1f293d; margin-bottom: 20px;">
    <div style="font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase;">
        Euro NCAP 2026 Protocol &bull; Empirical Hardware Benchmarking
    </div>
    <h1 style="margin: 4px 0 0 0; font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        AI Model Performance &amp; Empirical Validation Suite
    </h1>
    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
        Empirical latency benchmarks, neural architecture specifications, and real DMS ground-truth confusion matrix audits.
    </p>
</div>
""", unsafe_allow_html=True)

model_manager = st.session_state.model_manager
diag = model_manager.get_system_diagnostics()

# -------------------------------------------------------------
# 1. HARDWARE & LATENCY CARDS
# -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Inference Hardware</div>
        <div style="font-size: 1.5rem; font-weight: 700; margin: 4px 0;">{diag['device'].upper()}</div>
        <div class="metric-subtext">{diag['device_name']}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    face_lat = diag['latencies_ms'].get('face_mesh_ms', 0.0)
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Face Landmarker</div>
        <div class="metric-value">{face_lat:.1f} <span style="font-size: 1rem;">ms</span></div>
        <div class="metric-subtext">MediaPipe 468-pt Mesh</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    eye_lat = diag['latencies_ms'].get('eye_classifier_ms', 0.0)
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Eye CNN Latency</div>
        <div class="metric-value">{eye_lat:.1f} <span style="font-size: 1rem;">ms</span></div>
        <div class="metric-subtext">PyTorch 32x32 ConvNet</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    emo_lat = diag['latencies_ms'].get('emotion_classifier_ms', 0.0)
    st.markdown(f"""
    <div class="metric-card normal">
        <div class="metric-title">Emotion CNN Latency</div>
        <div class="metric-value">{emo_lat:.1f} <span style="font-size: 1rem;">ms</span></div>
        <div class="metric-subtext">MobileNetV3-Small</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. RUN REAL LATENCY BENCHMARK
# -------------------------------------------------------------
st.markdown("### ⏱️ Live Latency & Throughput Benchmark")
st.write("Execute 50 consecutive inference iterations across all neural pipelines to compute empirical p50 and p95 latencies on this machine.")

if st.button("🚀 Run Empirical Benchmark (50 Iterations)", type="primary"):
    import torch
    import numpy as np

    eye_clf = model_manager.get_eye_classifier()
    emo_clf = model_manager.get_emotion_classifier()

    # Synthetic eye & face inputs
    dummy_eye = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    eye_times = []
    emo_times = []

    progress_bar = st.progress(0)
    for i in range(50):
        t0 = time.perf_counter()
        eye_clf.predict(dummy_eye)
        eye_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        emo_clf.predict(dummy_face)
        emo_times.append((time.perf_counter() - t1) * 1000.0)

        progress_bar.progress((i + 1) / 50)

    b1, b2 = st.columns(2)
    with b1:
        st.success(f"**Eye State CNN:** Mean: **{np.mean(eye_times):.2f} ms** | p95: **{np.percentile(eye_times, 95):.2f} ms** | Throughput: **{1000.0/np.mean(eye_times):.0f} FPS**")
    with b2:
        st.success(f"**Emotion CNN:** Mean: **{np.mean(emo_times):.2f} ms** | p95: **{np.percentile(emo_times, 95):.2f} ms** | Throughput: **{1000.0/np.mean(emo_times):.0f} FPS**")

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. GENUINE DATASET & EVALUATION METRICS
# -------------------------------------------------------------
st.markdown("### 📊 Offline Evaluation & Model Metrics")

eval_results_file = config.DATA_DIR / "evaluation_report.json"

if eval_results_file.exists():
    try:
        with open(eval_results_file, "r") as f:
            eval_data = json.load(f)
        
        st.markdown(f"#### Results for: `{eval_data.get('dataset_name', 'Benchmark Dataset')}`")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{eval_data.get('accuracy', 0.0)*100:.1f}%")
        m_col2.metric("Precision", f"{eval_data.get('precision', 0.0)*100:.1f}%")
        m_col3.metric("Recall", f"{eval_data.get('recall', 0.0)*100:.1f}%")
        m_col4.metric("F1-Score", f"{eval_data.get('f1_score', 0.0)*100:.1f}%")

        if "confusion_matrix" in eval_data:
            cm = eval_data["confusion_matrix"]
            labels = eval_data.get("class_labels", ["Negative", "Positive"])
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted Class", y="Ground Truth Class"))
            st.plotly_chart(fig_cm, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading evaluation report: {e}")
else:
    st.info("""
    📋 **Evaluation Status: NOT EVALUATED**
    
    Offline test dataset metrics (Accuracy, F1-Score, Confusion Matrix) have not been generated yet.
    In accordance with research integrity standards, DriveGuard AI **never fabricates model accuracy or evaluation numbers**.
    
    To run genuine evaluation on an eye-state or emotion dataset (e.g. MRL Eye Dataset, FER-2013):
    ```powershell
    python training/evaluate_models.py --dataset_dir <path_to_dataset>
    ```
    """)

# -------------------------------------------------------------
# 4. MODEL SPECIFICATIONS & REPRODUCIBILITY TABLE
# -------------------------------------------------------------
st.markdown("### 📖 Architecture Specifications")

specs_data = [
    {
        "Subsystem": "Face Mesh & Landmarks",
        "Architecture": "MediaPipe FaceLandmarker",
        "Input Resolution": "640 x 480",
        "Parameters / Backbone": "BlazeFace / MobileNet Mesh",
        "Primary Metric": "NME (Normalized Mean Error)",
        "Inference Type": "Real-Time 3D Mesh"
    },
    {
        "Subsystem": "Eye State Classifier",
        "Architecture": "Custom EyeStateCNN",
        "Input Resolution": "32 x 32 x 1 Grayscale",
        "Parameters / Backbone": "3-Block ConvNet + BatchNorm",
        "Primary Metric": "Balanced Accuracy / F1",
        "Inference Type": "PyTorch CPU/CUDA"
    },
    {
        "Subsystem": "Facial Expression (FER)",
        "Architecture": "EmotionCNN (MobileNetV3)",
        "Input Resolution": "224 x 224 x 3 RGB",
        "Parameters / Backbone": "MobileNetV3-Small (9.8MB)",
        "Primary Metric": "Macro F1 / Cross-Entropy",
        "Inference Type": "PyTorch CPU/CUDA"
    },
    {
        "Subsystem": "Head Pose Estimator",
        "Architecture": "3D-2D solvePnP (Levenberg-Marquardt)",
        "Input Resolution": "6 3D Canonical Points",
        "Parameters / Backbone": "Direct Geometry",
        "Primary Metric": "MAE (Degrees Pitch/Yaw/Roll)",
        "Inference Type": "Geometric Solver"
    }
]

st.dataframe(pd.DataFrame(specs_data), use_container_width=True)

# -------------------------------------------------------------
# 5. FACULTY PRESENTATION & REAL DMS DATASET EXPLORER
# -------------------------------------------------------------
st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="padding: 12px 16px; background: rgba(0, 240, 255, 0.08); border-left: 4px solid #00f0ff; border-radius: 6px; margin-bottom: 20px;">
    <h3 style="margin: 0; color: #00f0ff; font-family: 'Orbitron', monospace;">
        🎓 Faculty Review & Real DMS Dataset Explorer
    </h3>
    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
        Self-contained project dataset directory: <code>AI_Driver_Safety_Monitoring/dataset/</code> (168 annotated driver frames + ground-truth videos).
    </p>
</div>
""", unsafe_allow_html=True)

dataset_dir = config.BASE_DIR / "dataset"
if dataset_dir.exists():
    d_col1, d_col2 = st.columns([1.2, 1.0])
    
    with d_col1:
        st.markdown("##### 🖼️ Ground-Truth Frame Inspection")
        seq_options = ["driver_no_sleep", "driver_microsleep2", "driver_full_sleep", "driver_full_sleep2", "driver_full_sleep3", "driver_full_sleep4"]
        selected_seq = st.selectbox("Select Driving Session Sequence:", options=seq_options, index=0)
        
        # Load label JSON
        label_file = dataset_dir / "labels" / f"{selected_seq}.json"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            samples = label_data.get("samples", [])
            
            if samples:
                frame_idx = st.slider("Select Frame Sequence:", min_value=0, max_value=len(samples)-1, value=0,
                                      format="Frame %d")
                s = samples[frame_idx]
                img_name = s.get("image")
                img_path = dataset_dir / "images" / img_name
                attr = s.get("attributes", {})
                
                if img_path.exists():
                    from PIL import Image
                    st.image(Image.open(img_path), caption=f"Sequence: {selected_seq} | Frame: {img_name}", use_container_width=True)
                    
                    # Annotations breakdown
                    st.markdown(f"""
                    **Ground Truth Annotations:**
                    - **Eye State:** `<span style="color: {'#ef4444' if attr.get('eye_state') != 'Open' else '#10b981'}; font-weight: bold;">{attr.get('eye_state')}</span>`
                    - **PERCLOS:** `{(attr.get('perclos', 0.0)*100):.1f}%`
                    - **Head Pitch:** `{attr.get('head_pose', {}).get('p', 0.0):+.2f}°` | **Yaw:** `{attr.get('head_pose', {}).get('y', 0.0):+.2f}°`
                    - **Driver Zone:** `{attr.get('zone', 'Unknown')}`
                    """, unsafe_allow_html=True)
                    
    with d_col2:
        st.markdown("##### 🎥 Ground-Truth Video Playback")
        vid_file = dataset_dir / "visualizations" / f"{selected_seq}_annotated.mp4"
        if vid_file.exists():
            st.video(str(vid_file))
            st.caption(f"Annotated benchmark video playback: `{vid_file.name}`")
        else:
            st.info("No video visualization available for this sequence.")
else:
    st.info("Dataset folder not found. Ensure `dataset/` is present in the project directory.")

