import streamlit as st
import cv2
import tempfile
import time
import os
import sys
import numpy as np

# Import Engine
sys.path.append(os.getcwd())
try:
    from sentinel_engine import SentinelEngine
except ImportError:
    pass

st.set_page_config(page_title="SENTINEL OMEGA", page_icon="🦅", layout="wide")

# --- LOAD ENGINE SAFELY ---
@st.cache_resource
def load_engine():
    if not os.path.exists("yolov8m.pt"):
        from ultralytics import YOLO
        YOLO('yolov8m.pt')
    return SentinelEngine(model_size='m')

engine = None
try:
    engine = load_engine()
    status = "ONLINE"
    color = "#0f0"
except Exception as e:
    status = f"OFFLINE: {e}"
    color = "#f00"

# --- SIDEBAR ---
with st.sidebar:
    st.title("SENTINEL // OMEGA")
    st.markdown(f"STATUS: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    
    # [PHASE 2] BIOMETRICS
    st.markdown("### 👤 BIOMETRIC REGISTRY")
    with st.expander("Register New Personnel"):
        reg_name = st.text_input("Personnel Name", "Officer Doe")
        reg_photo = st.file_uploader("Upload ID Photo", type=['jpg', 'png'])
        if reg_photo and st.button("ENCODE TO DATABASE"):
            if engine.register_face(reg_photo.getbuffer(), reg_name):
                st.success(f"User '{reg_name}' Registered.")
            else:
                st.error("Registration Failed.")

    # [PHASE 1+2] VISUALS & INTEL
    st.markdown("### ⚙️ CONFIGURATION")
    tab_vis, tab_intel = st.tabs(["VISUALS", "INTEL"])
    
    with tab_vis:
        show_heatmap = st.toggle("Thermal Heatmap", False)
        show_trace = st.toggle("Movement Tracers", True)
        conf_thres = st.slider("Sensitivity", 0.1, 1.0, 0.35)
        
    with tab_intel:
        use_face = st.toggle("Biometric Face ID", False)
        use_anpr = st.toggle("Vehicle OCR (ANPR)", False)

    st.markdown("### 📊 MISSION LOG")
    live_stats_area = st.empty()
    report_area = st.empty()

# --- MAIN UI ---
st.title("SENTINEL AI: OMEGA")
st.markdown("##### AUTONOMOUS SURVEILLANCE & FORENSICS")

tab1, tab2 = st.tabs(["📷 FORENSIC IMAGE ANALYSIS", "🎥 LIVE STREAM INTELLIGENCE"])

# TAB 1: IMAGES
with tab1:
    uploaded_img = st.file_uploader("Upload Recon Image", type=['jpg', 'png'])
    if uploaded_img and st.button("ANALYZE TARGET"):
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Build Active Modules List
        active_mods = []
        if use_face: active_mods.append('face_id')
        if use_anpr: active_mods.append('anpr')
        
        with st.spinner("RUNNING MULTI-LAYER ANALYSIS..."):
            # Process
            res_img, stats, detections = engine.process_frame_detailed(
                img, conf_thres, active_mods, show_heatmap, False
            )
            
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Metric Cards
            cols = st.columns(4)
            for i, (k,v) in enumerate(stats.items()):
                cols[i%4].metric(k, v)
            
            # Report
            cv2.imwrite("temp_img.jpg", res_img)
            pdf = engine.generate_image_report(stats, "temp_img.jpg")
            with open(pdf, "rb") as f:
                st.download_button("DOWNLOAD FORENSIC REPORT", f, "Forensic_Report.pdf")

# TAB 2: VIDEO
with tab2:
    uploaded_video = st.file_uploader("Upload Surveillance Feed", type=['mp4', 'avi'])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        col_vid, col_stat = st.columns([3, 1])
        
        with col_stat:
            st.markdown("### LIVE INTEL")
            stat_box = st.empty()
        
        with col_vid:
            st.markdown("#### REAL-TIME FEED")
            st_frame = st.empty()
            
            # Session State
            unique_ids = set()
            cumulative_stats = {}
            
            if st.button("▶️ START OMEGA STREAM"):
                engine.reset_trackers()
                cap = cv2.VideoCapture(tfile.name)
                stop_btn = st.button("⏹️ STOP")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or stop_btn: break
                    
                    frame = cv2.resize(frame, (854, 480))
                    
                    # Intel Config
                    active_mods = []
                    if use_face: active_mods.append('face_id')
                    if use_anpr: active_mods.append('anpr')
                    
                    # RUN ENGINE
                    res_img, frame_stats, detections = engine.process_frame_detailed(
                        frame, conf_thres, active_mods, show_heatmap, show_trace
                    )
                    
                    # Update History
                    if detections.tracker_id is not None:
                        for tid in detections.tracker_id:
                            unique_ids.add(tid)
                            
                    # Update Cumulative Counts (Approximate based on frame stats)
                    for k, v in frame_stats.items():
                        cumulative_stats[k] = max(cumulative_stats.get(k, 0), v)

                    # Display
                    st_frame.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Live Metrics Side Panel
                    with stat_box.container():
                        st.metric("UNIQUE TARGETS", len(unique_ids))
                        for k, v in frame_stats.items():
                            st.metric(f"{k} (LIVE)", v)
                            
                cap.release()
                
                # Report
                pdf = engine.generate_video_report(cumulative_stats)
                with report_area:
                    st.success("MISSION COMPLETE")
                    with open(pdf, "rb") as f:
                        st.download_button("DOWNLOAD MISSION LOG", f, "Mission_Log.pdf")
