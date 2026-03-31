# 🛡️ AI Smart Surveillance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00DCFF?style=for-the-badge&logo=data:image/png;base64,)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Event%20Logs-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**Production-grade AI surveillance with real-time human detection, face recognition, and a glassmorphism command-center dashboard.**

</div>

---

## 🎯 What It Does

| Feature | Details |
|---|---|
| 🎥 **Real-time Camera Feed** | Thread-buffered OpenCV stream (no frame lag) |
| 🧠 **Human Detection** | YOLOv8n with IoU-based multi-object tracking |
| 👤 **Face Recognition** | Identify known vs unknown faces; enrol via CLI or UI |
| 🚨 **Alert Engine** | Rule-based: Unknown Face · Loitering · Night Intrusion · Crowd |
| 📊 **Live Dashboard** | Glassmorphism Streamlit UI with KPIs, analytics, event log |
| 🖼️ **Snapshots** | Auto-saved JPEG on every alert trigger |
| 🗄️ **Event Logging** | SQLite persistence; filterable; CSV export |
| ⏱️ **Loitering Detection** | Fires after configurable dwell time (default 15 s) |
| 🌙 **Night Intrusion** | Alerts when person detected during configured night hours |

---

## 🏗️ Architecture

```
Camera (OpenCV)
    │
    ▼
VideoStream (threaded buffer)
    │
    ▼
PersonDetector (YOLOv8 + IoU Tracker)
    │
    ├──▶ FaceEngine (face_recognition, per-track cache)
    │
    ▼
AlertEngine (rule evaluation, cooldown, snapshot)
    │
    ├──▶ SQLite (event log)
    └──▶ Snapshots (JPEG)
         │
         ▼
Streamlit Dashboard
    ├── 🎥 Live Feed + HUD
    ├── 📊 Analytics (charts)
    ├── 📋 Event Log (filterable)
    ├── 🖼️ Snapshots grid
    └── ⚙️ Settings
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | OpenCV 4.8+ |
| Detection | YOLOv8n (Ultralytics) |
| Tracking | Custom IoU-based tracker |
| Face Recognition | `face_recognition` (dlib) |
| UI | Streamlit + Custom CSS (glassmorphism) |
| Database | SQLite 3 |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
AI Smart Surveillance System/
│
├── app.py                    ← Streamlit dashboard (main entry point)
├── core_engine.py            ← Central pipeline (detection + tracking + alerts)
├── add_known_face.py         ← CLI face enrolment tool
├── requirements.txt
│
├── config/
│   └── settings.py           ← All tunable parameters (single source of truth)
│
├── detection/
│   └── yolo_detector.py      ← YOLOv8 wrapper + IoU multi-object tracker
│
├── recognition/
│   └── face_engine.py        ← Face recognition engine with per-track caching
│
├── utils/
│   ├── alert_engine.py       ← Rule-based alert evaluator
│   ├── frame_utils.py        ← HUD overlay helpers
│   └── video_stream.py       ← Threaded camera capture
│
├── database/
│   ├── db_manager.py         ← SQLite CRUD layer
│   └── surveillance.db       ← (auto-created on first run)
│
└── assets/
    ├── known_faces/          ← Drop face images here (filename = person name)
    └── snapshots/            ← Auto-saved alert screenshots
```

---

## ⚙️ Setup

### 1. Clone / navigate to the project
```bash
cd "AI Smart Surveillance System"
```

### 2. Install core dependencies
```bash
pip install opencv-python ultralytics streamlit numpy pandas Pillow
```

### 3. (Optional) Install face recognition
> Requires CMake and a C++ compiler. On Windows use Visual Studio Build Tools.
```bash
pip install cmake dlib face_recognition
```

---

## 🚀 Run

### Launch the dashboard
```bash
streamlit run app.py
```

### Headless (CLI) mode — no Streamlit
```bash
python core_engine.py
```

---

## 👤 Enrolling Known Faces

**Option A — From a photo file:**
```bash
python add_known_face.py --name "Harsh Choudhary" --image "D:/photos/harsh.jpg"
```

**Option B — Live webcam capture:**
```bash
python add_known_face.py --name "Harsh Choudhary" --capture
```

**Option C — Via the Dashboard:**
Open the sidebar → **Known Faces** expander → drag & drop an image.

> The filename (without extension) becomes the person's display name.  
> `harsh_choudhary.jpg` → **"Harsh Choudhary"**

---

## 🚨 Alert Rules

| Alert | Trigger | Severity |
|---|---|---|
| ⚠ Unknown Person | Face not matched to known library | ALERT |
| ⏱ Loitering | Same person in frame > 15 seconds | WARNING |
| 🌙 Night Intrusion | Any person detected between 21:00–06:00 | ALERT |
| 👥 Crowd Detected | ≥ 4 people simultaneously in scene | WARNING |

All thresholds are configurable in `config/settings.py`.

---

## 🛠️ Configuration (`config/settings.py`)

| Parameter | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | Swap to `yolov8s/m/l` for higher accuracy |
| `YOLO_CONF` | `0.45` | Detection confidence threshold |
| `FACE_TOLERANCE` | `0.55` | Lower = stricter face match |
| `LOITER_SECONDS` | `15` | Seconds before loitering alert |
| `ALERT_COOLDOWN_SEC` | `10` | Minimum gap between repeated same-type alerts |
| `NIGHT_START_HOUR` | `21` | Night mode start (9 PM) |
| `NIGHT_END_HOUR` | `6` | Night mode end (6 AM) |
| `SNAPSHOT_ON_ALERT` | `True` | Save JPEG on every alert |
| `CAMERA_INDEX` | `0` | Default webcam index |

---

## 📊 Dashboard Pages

| Page | Contents |
|---|---|
| 🎥 Live Feed | Real-time annotated camera stream, active alerts, track table |
| 📊 Analytics | Event-type bar chart, severity breakdown, hourly timeline |
| 📋 Event Log | Filterable, paginated log with CSV export |
| 🖼️ Snapshots | Grid of the 24 most-recent alert screenshots |
| ⚙️ Settings | Read current config; danger-zone clear events |

---

## 🎨 Design Decisions

- **Single-source config** — all tunable values live in `settings.py`; no magic numbers scattered throughout the codebase.
- **Per-track caching** — face recognition runs every 3rd frame and caches results per `track_id` to stay within real-time budget.
- **Alert cooldowns** — each `(track_id, alert_type)` pair has its own cooldown timer to prevent alert flooding.
- **Threaded capture** — `VideoStream` runs in a daemon thread, so the inference loop is never blocked by I/O.
- **Graceful degradation** — face recognition and `ultralytics` are imported lazily; dashboard loads even if they are absent.

---

## 🤝 Contributing

```bash
# Fork → feature branch → PR
git checkout -b feat/my-feature
git commit -m "feat: add zone-based intrusion detection"
git push origin feat/my-feature
```

---

<div align="center">
Built with ❤️ using YOLOv8 · OpenCV · Streamlit · SQLite
</div>
