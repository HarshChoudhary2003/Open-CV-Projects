# 🛒 AI Retail Analytics System
### Customer Behavior Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)

> **The kind of system used by Amazon 🛒, Walmart 🏬, and Reliance Retail 🇮🇳**
> Recruiters love this: Business Impact + AI + Real-World CV = 🔥

</div>

---

## 🧠 What This System Does

This is a **production-grade retail intelligence platform** that uses computer vision to:

| Feature | Description |
|---------|-------------|
| 👥 **Person Detection** | YOLOv8 detects customers in real-time |
| 🔢 **Footfall Counting** | Counts unique visitors per session & hour |
| ⏱️ **Dwell Time Analysis** | Tracks how long each customer stays |
| 🔥 **Spatial Heatmap** | Visualizes most-visited store zones |
| 🗄️ **SQLite Logging** | Persists all analytics for historical analysis |
| 📊 **Streamlit Dashboard** | Interactive charts, insights & live feed |

---

## 🏗️ Full System Architecture

```
Camera Feed
   ↓
OpenCV Frame Capture
   ↓
YOLOv8n (Person Detection @ 40%+ confidence)
   ↓
SORT / Centroid Tracker (Multi-Object Tracking)
   ↓
Analytics Engine
  ├── FootfallCounter  (unique IDs via set())
  ├── DwellTimeTracker (per-ID time.time())
  └── HeatmapEngine   (Gaussian accumulation)
   ↓
SQLite Database (sessions / events / snapshots)
   ↓
Streamlit Dashboard (live charts + AI insights)
```

---

## 📁 Project Structure

```
AI Retail Analytics System/
│
├── app.py                  ← Streamlit dashboard
├── main.py                 ← Core CV pipeline
├── requirements.txt
│
├── detection/
│   └── yolo.py             ← YOLOv8 person detector
│
├── tracking/
│   └── tracker.py          ← SORT + centroid fallback
│
├── analytics/
│   ├── footfall.py         ← Unique visitor counting
│   ├── dwell_time.py       ← Residence time tracker
│   └── heatmap.py          ← Spatial density engine
│
├── database/
│   └── db.py               ← SQLite ORM (sessions, logs)
│
├── utils/
│   └── helpers.py          ← Frame utils, demo generator, insights
│
├── data/
│   └── retail.db           ← Auto-created SQLite DB
│
└── assets/
    └── demo.mp4            ← Auto-generated simulation
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: advanced SORT tracker
pip install sort-tracker
```

### 2. Run the Pipeline

```bash
# ── No camera? No problem ── generate a synthetic store demo:
python main.py --demo

# ── Live webcam:
python main.py --source 0

# ── Existing video file:
python main.py --source store.mp4

# ── With live heatmap overlay:
python main.py --source 0 --heatmap

# ── Headless (dashboard-only, no OpenCV window):
python main.py --source 0 --no-display
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎬 Pipeline Controls

| Key | Action |
|-----|--------|
| `ESC` or `Q` | Stop the pipeline gracefully |

---

## 📊 Dashboard Pages

| Page | Contents |
|------|----------|
| 📊 **Dashboard** | KPIs, hourly trends, session history, dwell distribution, zone traffic |
| 🎥 **Live Analysis** | Live frame metrics, active persons chart, unique visitor trend |
| 🔥 **Heatmap Studio** | Store heatmap overlay, hot-zone rankings |
| 📈 **Trends & Insights** | Weekly patterns, hourly heatgrid, AI-generated insights |
| ℹ️ **About** | Architecture, folder structure, quick-start guide |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | **YOLOv8n** (Ultralytics) |
| Tracking | **SORT** / Centroid fallback |
| Video | **OpenCV** 4.8+ |
| Analytics | **NumPy** + custom engines |
| Database | **SQLite 3** (WAL mode) |
| Dashboard | **Streamlit** + **Plotly** |
| Language | **Python 3.10+** |

---

## 🔬 Key Technical Highlights

- **Dual-backend tracker** — automatically uses SORT if installed, falls back to a robust centroid tracker (no dependency failure)
- **Gaussian heatmap** with temporal decay (`0.995` per frame) for smooth, representative maps
- **Session-scoped analytics** — each pipeline run is a separate DB session, enabling historical comparison
- **DB batching** — writes every 30 frames to avoid I/O bottlenecks at high FPS
- **Headless mode** — run without a display for server/cloud deployments

---

## 💼 Why This Project Stands Out

✅ End-to-end production pipeline (not just a script)  
✅ Real business metrics (footfall, dwell, heatmap) — not toy demos  
✅ Industry-standard tech stack (YOLOv8, SORT, SQLite)  
✅ Beautiful Streamlit dashboard with Plotly visualizations  
✅ Scalable architecture with clean module separation  
✅ Works without a real camera (synthetic demo generator built in)  

---

## 📈 Use Cases

- **Retail chains**: Track store sections with highest dwell → optimize shelf placement
- **Shopping malls**: Identify peak hours → optimize staff allocation
- **Supermarkets**: Heatmap-driven product placement strategy
- **Event venues**: Real-time footfall monitoring and crowd analytics

---

*Built for portfolio use. Designed to demonstrate Computer Vision + AI + Business Intelligence.*
