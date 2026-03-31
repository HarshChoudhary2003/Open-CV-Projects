"""
app.py  —  AI Smart Surveillance System
Streamlit Command-Center Dashboard
"""

import os
import sys
import time
import base64
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    CAMERA_INDEX, KNOWN_FACES_DIR, SNAPSHOTS_DIR,
    DASHBOARD_TITLE, MAX_LOG_ROWS
)
from database.db_manager import init_db, fetch_recent_events, fetch_event_stats
from core_engine import SurveillanceEngine

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS — Glassmorphism Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #050a14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid rgba(0,220,130,0.12);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Main content ── */
.block-container { padding: 1.5rem 2.5rem 2rem; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(13,20,35,0.95) 0%, rgba(18,28,50,0.9) 100%);
    border: 1px solid rgba(0,220,130,0.18);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(16px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00dc82, transparent);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,220,130,0.12);
}
.kpi-number {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #00dc82, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 4px;
}
.kpi-card.alert { border-color: rgba(255,59,59,0.3); }
.kpi-card.alert::before { background: linear-gradient(90deg, transparent, #ff3b3b, transparent); }
.kpi-card.alert .kpi-number {
    background: linear-gradient(135deg, #ff3b3b, #ff8c42);
    -webkit-background-clip: text;
}
.kpi-card.warn { border-color: rgba(255,165,0,0.3); }
.kpi-card.warn::before { background: linear-gradient(90deg, transparent, #ff9d00, transparent); }
.kpi-card.warn .kpi-number {
    background: linear-gradient(135deg, #ff9d00, #ffcc44);
    -webkit-background-clip: text;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,220,130,0.12);
}
.section-header h3 { margin: 0; font-size: 1rem; font-weight: 700; color: #cdd6f4; }

/* ── Status Badge ── */
.badge-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,220,130,0.12);
    border: 1px solid rgba(0,220,130,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem; font-weight: 600;
    color: #00dc82; text-transform: uppercase; letter-spacing: 0.08em;
}
.badge-live::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00dc82;
    animation: pulse 1.4s infinite;
}
.badge-offline {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,59,59,0.1);
    border: 1px solid rgba(255,59,59,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem; font-weight: 600;
    color: #ff3b3b; text-transform: uppercase; letter-spacing: 0.08em;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(1.4); }
}

/* ── Alert Banner ── */
.alert-banner {
    background: linear-gradient(135deg, rgba(255,59,59,0.1), rgba(255,140,66,0.08));
    border: 1px solid rgba(255,59,59,0.35);
    border-left: 4px solid #ff3b3b;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    animation: slideIn 0.3s ease;
}
.alert-banner.warning {
    background: linear-gradient(135deg, rgba(255,165,0,0.1), rgba(255,200,0,0.06));
    border-color: rgba(255,165,0,0.35);
    border-left-color: #ff9d00;
}
@keyframes slideIn {
    from { transform: translateX(-8px); opacity: 0; }
    to   { transform: translateX(0);    opacity: 1; }
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Video Feed ── */
.camera-frame img {
    border-radius: 14px;
    border: 1px solid rgba(0,220,130,0.2);
    width: 100%;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00dc82, #00b4d8) !important;
    color: #050a14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,220,130,0.35) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e3a2f; border-radius: 4px; }

/* ── Metric override ── */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
}

/* ── Snapshot grid ── */
.snap-img {
    border-radius: 10px;
    border: 1px solid rgba(255,59,59,0.25);
    width: 100%;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
init_db()

if "engine" not in st.session_state:
    st.session_state.engine   = None
if "cam_running" not in st.session_state:
    st.session_state.cam_running = False
if "face_recog" not in st.session_state:
    st.session_state.face_recog = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLOR = {
    "ALERT":   "🔴",
    "WARNING": "🟡",
    "INFO":    "🟢",
}

def severity_badge(sev: str) -> str:
    return SEVERITY_COLOR.get(sev, "⚪") + f" **{sev}**"


def img_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.8rem;'>🛡️</div>
        <div style='font-size:1.1rem; font-weight:800; color:#00dc82; letter-spacing:0.05em;'>
            AI Surveillance
        </div>
        <div style='font-size:0.7rem; color:#475569; letter-spacing:0.15em; text-transform:uppercase;'>
            Command Center
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Camera Settings ────────────────────────────────────────────────────
    st.markdown("**📷 Camera**")
    cam_src = st.selectbox("Source", [0, 1, 2, "Video File"], index=0,
                           label_visibility="collapsed")
    video_path = None
    if cam_src == "Video File":
        video_path = st.text_input("File path", placeholder="e.g. D:/clips/test.mp4")

    # ── Features ───────────────────────────────────────────────────────────
    st.markdown("**🧠 Features**")
    use_face = st.toggle("Face Recognition", value=False)
    st.caption("⚠️ Requires `face_recognition` library and known faces in `assets/known_faces/`")

    st.markdown("**⚡ Performance**")
    target_fps = st.slider("Target FPS", 5, 30, 20)

    st.divider()

    # ── Control ────────────────────────────────────────────────────────────
    if not st.session_state.cam_running:
        if st.button("▶  Start Surveillance", use_container_width=True):
            src = video_path if cam_src == "Video File" and video_path else cam_src
            engine = SurveillanceEngine(camera_src=src, use_face_recog=use_face)
            engine.start()
            time.sleep(0.8)
            if engine.error_msg:
                st.error(f"Engine error: {engine.error_msg}")
            else:
                st.session_state.engine      = engine
                st.session_state.cam_running = True
                st.session_state.face_recog  = use_face
                st.rerun()
    else:
        if st.button("⏹  Stop Surveillance", use_container_width=True):
            if st.session_state.engine:
                st.session_state.engine.stop()
            st.session_state.engine      = None
            st.session_state.cam_running = False
            st.rerun()

    st.divider()

    # ── Known Faces Management ─────────────────────────────────────────────
    with st.expander("👤 Known Faces"):
        st.caption(f"Folder: `assets/known_faces/`")
        uploaded = st.file_uploader("Add face image (filename = person name)",
                                    type=["jpg","jpeg","png"],
                                    accept_multiple_files=True,
                                    key="face_upload")
        if uploaded:
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            for f in uploaded:
                save_p = os.path.join(KNOWN_FACES_DIR, f.name)
                with open(save_p, "wb") as fp:
                    fp.write(f.getbuffer())
            st.success(f"Saved {len(uploaded)} face(s).")
            if st.session_state.engine and st.session_state.engine._face_eng:
                n = st.session_state.engine._face_eng.load_known_faces()
                st.info(f"Reloaded {n} known faces into engine.")

        # List existing
        faces = list(Path(KNOWN_FACES_DIR).glob("*.*"))
        if faces:
            for fp in faces:
                st.markdown(f"• {fp.stem}")
        else:
            st.caption("No known faces yet.")

    # ── Nav ────────────────────────────────────────────────────────────────
    st.divider()
    page = st.radio("Navigate", ["🎥 Live Feed", "📊 Analytics", "📋 Event Log",
                                  "🖼️ Snapshots", "⚙️ Settings"],
                    label_visibility="collapsed")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
engine: SurveillanceEngine = st.session_state.engine
is_live = st.session_state.cam_running and engine is not None

col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown(f"# 🛡️ {DASHBOARD_TITLE}")
with col_badge:
    if is_live:
        st.markdown('<span class="badge-live">● LIVE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-offline">● OFFLINE</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────────────────────────────────────
stats_db = fetch_event_stats()
live_stats = engine.get_stats() if is_live else {"fps": 0, "people": 0, "alerts": 0}

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-number'>{live_stats['people']}</div>
        <div class='kpi-label'>People Detected</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-number'>{live_stats['fps']}</div>
        <div class='kpi-label'>Live FPS</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class='kpi-card alert'>
        <div class='kpi-number'>{stats_db['alerts']}</div>
        <div class='kpi-label'>Total Alerts</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class='kpi-card warn'>
        <div class='kpi-number'>{stats_db['warnings']}</div>
        <div class='kpi-label'>Warnings</div>
    </div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-number'>{stats_db['today']}</div>
        <div class='kpi-label'>Events Today</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
if "🎥 Live Feed" in page:
# ══════════════════════════════════════════════════════════════════════════════
    col_cam, col_side = st.columns([3, 1.1])

    with col_cam:
        st.markdown("<div class='section-header'><h3>📡 Camera Feed</h3></div>",
                    unsafe_allow_html=True)
        frame_placeholder = st.empty()

        if not is_live:
            frame_placeholder.info(
                "🔴 Camera offline — click **▶ Start Surveillance** in the sidebar."
            )

    with col_side:
        st.markdown("<div class='section-header'><h3>🚨 Live Alerts</h3></div>",
                    unsafe_allow_html=True)
        alert_placeholder = st.empty()

        st.markdown("<div class='section-header'><h3>👥 Active Tracks</h3></div>",
                    unsafe_allow_html=True)
        track_placeholder = st.empty()

    # ── Live loop ─────────────────────────────────────────────────────────
    if is_live:
        while st.session_state.cam_running:
            frame = engine.get_frame()
            if frame is not None:
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

            # Alerts panel
            with alert_placeholder.container():
                alerts = engine.active_alerts
                if alerts:
                    for a in alerts[:5]:
                        css_cls = "alert-banner warning" if a["severity"] == "WARNING" else "alert-banner"
                        st.markdown(
                            f"<div class='{css_cls}'>{a['label']}<br>"
                            f"<small style='color:#94a3b8;'>{a['description'][:70]}</small></div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.success("✅ No active alerts")

            # Tracks panel
            with track_placeholder.container():
                tracks = engine.tracks
                if tracks:
                    rows = [{"ID": f"#{t.track_id}", "Name": t.label,
                             "Duration": f"{t.duration:.0f}s"} for t in tracks]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("No persons in frame.")

            time.sleep(0.04)   # ~25 FPS UI refresh


# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Analytics" in page:
# ══════════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'><h3>📊 Event Analytics</h3></div>",
                unsafe_allow_html=True)

    events = fetch_recent_events(limit=500)
    if not events:
        st.info("No events recorded yet. Start surveillance to generate data.")
    else:
        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Events by Type**")
            type_counts = df["event_type"].value_counts().reset_index()
            type_counts.columns = ["Event Type", "Count"]
            st.bar_chart(type_counts.set_index("Event Type"))

        with c2:
            st.markdown("**Events by Severity**")
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            st.bar_chart(sev_counts.set_index("Severity"))

        st.markdown("**Event Timeline (last 200)**")
        timeline_df = df[["timestamp", "event_type", "severity"]].head(200)
        timeline_df["hour"] = timeline_df["timestamp"].dt.floor("h")
        hourly = timeline_df.groupby("hour").size().reset_index(name="Events")
        st.line_chart(hourly.set_index("hour"))


# ══════════════════════════════════════════════════════════════════════════════
elif "📋 Event Log" in page:
# ══════════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'><h3>📋 Event Log</h3></div>",
                unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        sev_filter = st.selectbox("Filter Severity", ["ALL", "ALERT", "WARNING", "INFO"])
    with col_f2:
        limit = st.slider("Max rows", 50, 500, MAX_LOG_ROWS)

    events = fetch_recent_events(limit=limit, severity_filter=sev_filter)

    if not events:
        st.info("No events match the filter.")
    else:
        df = pd.DataFrame(events)
        df = df[["timestamp", "event_type", "severity", "description", "track_id"]]
        df.columns = ["Timestamp", "Event", "Severity", "Description", "Track ID"]
        df["Severity"] = df["Severity"].apply(
            lambda s: SEVERITY_COLOR.get(s, "⚪") + " " + s
        )
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={
                         "Timestamp":   st.column_config.TextColumn(width=160),
                         "Event":       st.column_config.TextColumn(width=140),
                         "Severity":    st.column_config.TextColumn(width=120),
                         "Description": st.column_config.TextColumn(width=400),
                         "Track ID":    st.column_config.NumberColumn(width=80),
                     })

        # CSV download
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Export CSV", csv, "surveillance_events.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
elif "🖼️ Snapshots" in page:
# ══════════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'><h3>🖼️ Alert Snapshots</h3></div>",
                unsafe_allow_html=True)

    snaps = sorted(Path(SNAPSHOTS_DIR).glob("*.jpg"), key=os.path.getmtime, reverse=True)

    if not snaps:
        st.info("No snapshots yet. Alerts will save snapshots automatically.")
    else:
        st.caption(f"Showing {min(len(snaps), 24)} most recent out of {len(snaps)} total.")
        cols = st.columns(4)
        for i, snap_path in enumerate(snaps[:24]):
            with cols[i % 4]:
                b64 = img_to_b64(str(snap_path))
                if b64:
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{b64}" class="snap-img">',
                        unsafe_allow_html=True
                    )
                    st.caption(snap_path.stem[:32])


# ══════════════════════════════════════════════════════════════════════════════
elif "⚙️ Settings" in page:
# ══════════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'><h3>⚙️ System Settings</h3></div>",
                unsafe_allow_html=True)

    from config import settings as cfg
    st.markdown("**Detection**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("YOLO Model", cfg.YOLO_MODEL)
        st.metric("Confidence Threshold", cfg.YOLO_CONF)
    with c2:
        st.metric("IoU Threshold", cfg.YOLO_IOU)
        st.metric("Max Track Age (frames)", cfg.MAX_TRACK_AGE)

    st.markdown("**Alerts**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Loiter Threshold (sec)", cfg.LOITER_SECONDS)
        st.metric("Alert Cooldown (sec)", cfg.ALERT_COOLDOWN_SEC)
    with c2:
        st.metric("Night Start Hour", cfg.NIGHT_START_HOUR)
        st.metric("Night End Hour", cfg.NIGHT_END_HOUR)

    st.markdown("**Paths**")
    st.code(f"Database: {cfg.DB_PATH}\nKnown Faces: {cfg.KNOWN_FACES_DIR}\nSnapshots: {cfg.SNAPSHOTS_DIR}")

    st.markdown("**⚠️ Danger Zone**")
    if st.button("🗑️  Clear All Events from Database", type="secondary"):
        import sqlite3
        conn = sqlite3.connect(cfg.DB_PATH)
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()
        st.success("All events cleared.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:0.75rem; padding:0.5rem 0;'>"
    "🛡️ AI Smart Surveillance System &nbsp;|&nbsp; "
    "Built with YOLOv8 · OpenCV · Streamlit &nbsp;|&nbsp; "
    f"{datetime.now().strftime('%Y')}"
    "</div>",
    unsafe_allow_html=True
)

# Auto-refresh when live (Streamlit experimental)
if is_live and "🎥 Live Feed" not in page:
    time.sleep(3)
    st.rerun()
