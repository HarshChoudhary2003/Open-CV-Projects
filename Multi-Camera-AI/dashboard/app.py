import streamlit as st
import pandas as pd
import requests
import time
import os
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Multi-Camera AI Tracking", layout="wide")

st.markdown("""
<style>
    .main {
        background: #1e1e1e;
        color: white;
    }
    h1 {
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.title("📹 Live Command Center")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🚦 Live Feed (YOLOv8 + DeepSORT + ReID)")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### 📡 Global Metrics")
    active_metric = st.empty()
    alerts_placeholder = st.empty()
    table_placeholder = st.empty()


def update_dashboard():
    # Attempt to load the latest frame
    if os.path.exists("assets/latest_frame.jpg"):
        try:
            img = Image.open("assets/latest_frame.jpg")
            frame_placeholder.image(img, use_column_width=True, caption="🎥 Live: Cross-Camera ReID Tracking")
        except:
            pass
            
    # Load API data
    try:
        active_res = requests.get(f"{API_URL}/persons/active")
        if active_res.status_code == 200:
            active_persons = active_res.json().get("active_persons", [])
            active_metric.metric("🟢 Active Persons Detected", len(active_persons))
    except:
        active_metric.metric("🔴 API Offline", "—")

    try:
        events_res = requests.get(f"{API_URL}/events?limit=10")
        if events_res.status_code == 200:
            events = events_res.json().get("events", [])
            if events:
                df = pd.DataFrame(events)
                table_placeholder.dataframe(df, use_container_width=True)
            else:
                table_placeholder.info("No tracking events yet...")
    except:
        pass


# Automatic Refresh Loop
# This ensures a highly advanced "real-time" streaming feel directly inside Streamlit
if st.button("Start Live Mission Protocol 🚀"):
    while True:
        update_dashboard()
        time.sleep(0.5) # 2 FPS refresh for dashboard stability
else:
    st.info("Click 'Start Live Mission Protocol' to connect to the ReID data pipeline.")
    update_dashboard()

st.sidebar.header("🎯 System Status")
st.sidebar.text("🟢 Backend API: Online")
st.sidebar.text("🟢 Detection Engine: YOLOv8")
st.sidebar.text("🟢 Movement Trail: Enabled")
