<div align="center">

# 🔷 VisionAI Platform

### **Production-Grade Real-Time AI Computer Vision System**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red?style=flat-square&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

**A futuristic, Iron Man-style AI surveillance and computer vision platform with real-time object detection, face recognition, emotion analysis, anomaly detection, and an autonomous AI agent.**

</div>

---

## ⚡ What Is VisionAI Platform?

VisionAI is a **startup-quality, production-ready AI vision system** that transforms any camera into a real-time intelligence node. It is not a demo — it is a full SaaS-ready platform built with:

- 🎯 **YOLOv8** for state-of-the-art object detection + ByteTrack for multi-object tracking
- 👤 **InsightFace / face_recognition** for face detection and recognition with a persistent registry
- 😊 **FER** for real-time emotion detection (angry, happy, fear, surprised, and more)
- 🖐 **MediaPipe Holistic** for body pose estimation and gesture classification
- 📝 **EasyOCR** for in-frame text extraction
- 🔥 **Anomaly Detector** combining optical flow, background subtraction, crowd density, and dwell-time analysis
- 🤖 **AI Agent** (Observe → Analyse → Decide → Act) with a rule engine and optional LLM reasoning (GPT-4o / Ollama)
- ⚡ **FastAPI + WebSockets** for real-time streaming
- 🎨 **Iron Man HUD Dashboard** with live video, threat meters, and analytics charts

---

## 📁 Folder Structure

```
VisionAI-Platform/
├── backend/
│   ├── app/
│   │   ├── api/               ← FastAPI routers
│   │   │   ├── auth.py        ← JWT login
│   │   │   ├── cameras.py     ← Camera start/stop
│   │   │   ├── websocket.py   ← MJPEG + JSON streams
│   │   │   ├── analytics.py   ← Stats API
│   │   │   ├── alerts.py      ← Alert management
│   │   │   ├── faces.py       ← Face registry
│   │   │   └── reports.py     ← CSV + PDF exports
│   │   ├── core/
│   │   │   ├── config.py      ← Pydantic settings
│   │   │   ├── database.py    ← SQLAlchemy async ORM
│   │   │   └── security.py    ← JWT auth
│   │   ├── services/
│   │   │   ├── detector.py    ← YOLOv8 + ByteTrack
│   │   │   ├── face_service.py ← Face recognition
│   │   │   ├── emotion_service.py ← Emotion detection
│   │   │   ├── pose_service.py ← MediaPipe Holistic
│   │   │   ├── ocr_service.py  ← EasyOCR
│   │   │   ├── anomaly_service.py ← Anomaly detection
│   │   │   ├── agent.py       ← AI Decision Agent
│   │   │   └── pipeline.py    ← Per-camera orchestrator
│   │   └── main.py            ← FastAPI app entry point
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html             ← Iron Man HUD dashboard
│   ├── style.css              ← Futuristic HUD styles
│   └── app.js                 ← Dashboard logic + WebSocket
├── docker/
│   └── nginx.conf             ← Reverse proxy
├── .github/
│   └── workflows/ci.yml       ← CI/CD pipeline
├── .env.example               ← Environment template
├── docker-compose.yml
└── start.py                   ← Quick-start launcher
```

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| pip | latest | `python -m pip install --upgrade pip` |
| Git | any | [git-scm.com](https://git-scm.com) |
| Webcam | required | Built-in or USB |

### 1. Clone & Navigate

```bash
git clone https://github.com/yourname/VisionAI-Platform.git
cd VisionAI-Platform
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch (do this first)

```bash
# CPU only (works everywhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.1) — much faster!
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install All Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** `dlib` (required by `face_recognition`) may need CMake. On Windows:
> ```bash
> pip install cmake
> pip install dlib
> pip install face-recognition
> ```
> Or skip it — the system auto-falls back to Haar cascade detection.

### 5. Configure Environment

```bash
# From the VisionAI-Platform root:
cp .env.example backend/.env

# Edit backend/.env and set your SECRET_KEY
```

### 6. Launch the Platform

```bash
# From VisionAI-Platform root:
python start.py
```

Or manually:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `frontend/index.html` in your browser.

---

## 🔐 Login Credentials

| Username | Password | Role |
|----------|----------|------|
| `admin` | `visionai2024` | Full admin |
| `operator` | `operator123` | View + camera ops |

> Change these in `backend/app/core/security.py` → `DEMO_USERS` (or wire to a database).

---

## 🧠 AI Agent Behaviour

The agent runs every frame and implements **Observe → Analyse → Decide → Act**:

```
Frame arrives
    │
    ├─ Detections? ─────► Count objects, extract classes
    ├─ Faces?      ─────► Compare against registry
    ├─ Emotions?   ─────► Detect anger/fear
    ├─ Pose?       ─────► Classify stance/gesture
    └─ Anomaly?    ─────► Flow + density + dwell score
                              │
                         Rule Engine
                              │
              ┌───────────────┼───────────────┐
           ALERT           SNAPSHOT           LOG
              │
         (optional LLM reasoning via GPT-4o / Ollama)
```

### Built-in Rules

| Rule | Trigger | Severity | Actions |
|------|---------|---------|---------|
| `suspicious_movement` | Anomaly score > 75% | HIGH | Alert + Snapshot |
| `unknown_face` | Unregistered face detected | MEDIUM | Alert + Log |
| `crowd_density` | > 10 people in frame | MEDIUM | Alert + Log |
| `negative_emotion` | Angry or fear detected | LOW | Log |
| `weapon_detected` | Knife/scissors in frame | CRITICAL | Alert + Snapshot + Log |

### Optional LLM Reasoning

In `.env`:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

Or use Ollama locally:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
```

---

## 📊 API Reference

Interactive docs at **http://localhost:8000/api/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/login` | JWT login |
| POST | `/api/v1/cameras/start` | Start camera pipeline |
| POST | `/api/v1/cameras/{id}/stop` | Stop camera |
| GET  | `/api/v1/cameras` | List active cameras |
| GET  | `/api/v1/analytics/summary` | Detection summary stats |
| GET  | `/api/v1/analytics/timeline` | Time-bucketed detections |
| GET  | `/api/v1/alerts` | Alert history |
| PATCH| `/api/v1/alerts/{id}/acknowledge` | Acknowledge alert |
| POST | `/api/v1/faces/register` | Register a face |
| GET  | `/api/v1/faces` | List registered faces |
| DELETE | `/api/v1/faces/{id}` | Remove face |
| GET  | `/api/v1/reports/csv/detections` | Export CSV |
| GET  | `/api/v1/reports/pdf/summary` | Export PDF |
| WS   | `/ws/stream/{camera_id}` | Binary MJPEG stream |
| WS   | `/ws/telemetry/{camera_id}` | JSON telemetry |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Visit **http://localhost** (Nginx serves the frontend).

---

## ⚙️ Config Reference (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLO weights (n/s/m/l/x) |
| `YOLO_CONFIDENCE` | `0.45` | Detection threshold |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `ANOMALY_SENSITIVITY` | `0.75` | Alert trigger threshold |
| `ALERT_COOLDOWN_SECONDS` | `10` | Min time between alerts |
| `LLM_PROVIDER` | `none` | `none` / `openai` / `ollama` |

---

## 🔥 Unique Feature: Behaviour Prediction

VisionAI includes a **dwell-time tracker** that correlates movement velocity with historical position data per track ID. When a person remains stationary beyond 30 seconds in a frame, a `LOITERING` anomaly is triggered.

Combined with LLM reasoning, the system can generate natural-language threat assessments like:

> *"A stationary individual (Track ID #7) has been detected for 45 seconds near entrance CAM-01. Emotion baseline is neutral. Recommend operator verification."*

---

## 🌐 Use Cases

| Industry | Application |
|----------|-------------|
| 🏢 Security | Intrusion detection, loitering alerts, face access control |
| 🛒 Retail | Customer emotion analysis, crowd density monitoring |
| 🏥 Healthcare | Patient fall detection, emotion wellbeing monitoring |
| 🚗 Transport | Vehicle counting, license plate OCR, anomaly detection |
| 🏭 Industrial | Worker safety posture, PPE detection, zone compliance |

---

## 📄 License

MIT License — free for personal and commercial use.

---

<div align="center">
<b>Built with ⚡ by VisionAI Labs</b><br/>
<i>Real intelligence. Real time.</i>
</div>
