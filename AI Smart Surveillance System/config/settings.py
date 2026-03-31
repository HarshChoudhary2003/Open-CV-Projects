"""
AI Smart Surveillance System — Global Configuration
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR   = os.path.join(BASE_DIR, "assets", "known_faces")
SNAPSHOTS_DIR     = os.path.join(BASE_DIR, "assets", "snapshots")
DB_PATH           = os.path.join(BASE_DIR, "database", "surveillance.db")

# ─── YOLO ─────────────────────────────────────────────────────────────────────
YOLO_MODEL        = "yolov8n.pt"      # nano — swap to yolov8s/m/l for higher accuracy
YOLO_CONF         = 0.45              # confidence threshold
YOLO_IOU          = 0.45              # NMS IoU threshold
PERSON_CLASS_ID   = 0                 # COCO class 0 = person

# ─── Face Recognition ─────────────────────────────────────────────────────────
FACE_TOLERANCE    = 0.55              # lower = stricter match
FACE_MODEL        = "hog"             # "hog" (fast/CPU) | "cnn" (accurate/GPU)
FACE_SCALE_FACTOR = 0.5               # downscale for speed (0‑1)

# ─── Tracking ─────────────────────────────────────────────────────────────────
LOITER_SECONDS    = 15                # seconds before "loitering" alert fires
MAX_TRACK_AGE     = 30                # frames to keep a lost track

# ─── Alert Engine ─────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SEC = 10               # minimum gap between repeated alerts (per type)
NIGHT_START_HOUR   = 21               # 9 PM — night mode start
NIGHT_END_HOUR     = 6                # 6 AM — night mode end

# ─── Video ────────────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0                # 0 = default webcam
FRAME_WIDTH        = 1280
FRAME_HEIGHT       = 720
TARGET_FPS         = 30

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_TITLE    = "AI Surveillance Command Center"
MAX_LOG_ROWS       = 200              # rows shown in live feed table
SNAPSHOT_ON_ALERT  = True             # save JPG snapshot on every alert
