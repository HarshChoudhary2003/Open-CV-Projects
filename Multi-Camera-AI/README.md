# AI Multi-Camera Tracking & Re-Identification System 🎥🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-blueviolet)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-yellow)
![Docker](https://img.shields.io/badge/Docker-Deploy-blue)

A production-ready AI system designed for smart cities, airports, and large-scale surveillance setups. This system ingests multi-camera feeds, detects individuals using YOLOv8, tracks them temporally using DeepSORT, and performs global identity matching (Person ReID) utilizing a PyTorch-based feature extraction engine.

## 🌟 Core Features
- **Multi-Camera Support:** Ingest multiple RTSP feeds, IP cameras, or local MP4 files concurrently.
- **YOLOv8 Detection Engine:** Real-time state-of-the-art bounding box predictions.
- **DeepSORT Tracking:** Handles occlusions and temporal consistency.
- **Person Re-Identification (ReID):** Cross-camera identity matching algorithm using PyTorch ResNet feature embeddings and cosine similarity.
- **Database Tracking:** Persistent event logging using SQLite (extendable to PostgreSQL/Redis).
- **FastAPI Backend:** Secure REST APIs to query active persons or historical tracking anomalies.
- **Streamlit Dashboard:** Glassmorphic, real-time control center for surveillance metrics.
- **Dockerized:** Instant cloud deployment.

## 🏗️ Architecture

```
Camera 1 ─┐
Camera 2 ─┼──> Frame Ingestion (OpenCV)
Camera N ─┘
            ↓
    YOLOv8 Class Detection 
            ↓
     DeepSORT Tracking
            ↓
  Person ReID (PyTorch ResNet50)
            ↓
  Cosine Similarity Matching
            ↓
   SQLite Event DB Manager
            ↓
       FastAPI Backend
            ↓
     Streamlit Dashboard
```

## 📁 System Modules

- `configs/cameras.yaml`: Define streaming URLs and location IDs.
- `detection/yolo.py`: YOLOv8 instantiation and processing layer.
- `tracking/deepsort.py`: Temporal smoothing and ID maintenance.
- `reid/reid_model.py`: Embeddings generation & global registry logic.
- `pipeline/processor.py`: The threaded multi-camera processing orchestrator.
- `database/db.py`: Database tables and interaction methods.
- `backend/main.py`: The REST API server.
- `dashboard/app.py`: Streamlit frontend for monitoring.

## 🚀 Installation & Usage

### Method 1: Docker (Recommended)
1. Build the system:
```bash
docker build -t multi-cam-ai .
```
2. Run the platform container:
```bash
docker run -p 8000:8000 -p 8501:8501 multi-cam-ai
```

### Method 2: Local Deployment
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Start the FastAPI Database Server:
```bash
uvicorn backend.main:app --reload --port 8000
```
3. Boot up the Intelligence Pipeline:
```bash
python pipeline/processor.py
```
4. Initialize the Streamlit Control Center:
```bash
streamlit run dashboard/app.py
```

## 🚨 API Endpoints
- `GET /events?limit=50`: Fetch recent tracking logs.
- `GET /persons/active`: Retrieve global IDs currently caught on camera.
- `POST /alerts`: Programmatic blacklist or anomaly alerts.

## 🧠 System Context for Interview Answers
> "I architected an end-to-end multi-camera AI system capable of detecting, tracking, and universally identifying individuals across disparate feeds. It utilizes a YOLOv8 detection engine piped into a DeepSORT temporal tracker, and leverages PyTorch for high-dimensional feature embedding (ReID) extraction, ensuring individuals maintain cohesive IDs globally. To scale operations, I abstracted the data layer into an SQLite DB consumed by a FastAPI server, actively visualized through a low-latency Streamlit front-end." 

---
*Built for Production. Scalable for the Future.*
