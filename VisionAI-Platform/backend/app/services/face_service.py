"""
VisionAI Platform - Face Detection & Recognition Service
Uses OpenCV DNN + InsightFace / face_recognition for embeddings.
Falls back to Haar cascade if DNN weights not present.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# InsightFace (optional – deep embeddings)
try:
    from insightface.app import FaceAnalysis as InsightFaceAnalysis
    _HAS_INSIGHT = True
except ImportError:
    _HAS_INSIGHT = False

# face_recognition (fallback)
try:
    import face_recognition
    _HAS_FACE_REC = True
except ImportError:
    _HAS_FACE_REC = False


FACE_DB_PATH = Path("face_db.json")


class FaceRecord:
    def __init__(self, person_id: str, name: str, embedding: List[float],
                 access_level: str = "visitor"):
        self.person_id = person_id
        self.name = name
        self.embedding = np.array(embedding, dtype=np.float32)
        self.access_level = access_level
        self.last_seen: Optional[float] = None


class FaceResult:
    def __init__(self, bbox: Tuple, name: str, person_id: Optional[str],
                 confidence: float, emotion: Optional[str] = None):
        self.bbox = bbox           # (x1, y1, x2, y2)
        self.name = name
        self.person_id = person_id
        self.confidence = confidence
        self.emotion = emotion

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "name": self.name,
            "person_id": self.person_id,
            "confidence": round(self.confidence, 3),
            "emotion": self.emotion,
        }


class FaceService:
    """
    Multi-backend face service:
      1. InsightFace (best quality, GPU)
      2. face_recognition (CPU, good accuracy)
      3. OpenCV Haar Cascade (fallback)
    """

    _instance: Optional["FaceService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return

        self.db: Dict[str, FaceRecord] = {}
        self._load_db()
        self.backend = "haar"

        if _HAS_INSIGHT:
            try:
                self.app = InsightFaceAnalysis(name="buffalo_l",
                                               providers=["CUDAExecutionProvider",
                                                          "CPUExecutionProvider"])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.backend = "insightface"
                print("[FaceService] Backend: InsightFace")
            except Exception as e:
                print(f"[FaceService] InsightFace failed ({e}), trying face_recognition...")

        if self.backend == "haar" and _HAS_FACE_REC:
            self.backend = "face_recognition"
            print("[FaceService] Backend: face_recognition")

        if self.backend == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            print("[FaceService] Backend: Haar Cascade (fallback)")

        self._initialised = True

    # ── DB persistence ────────────────────────────────────────────

    def _load_db(self):
        if FACE_DB_PATH.exists():
            data = json.loads(FACE_DB_PATH.read_text())
            for pid, rec in data.items():
                self.db[pid] = FaceRecord(
                    person_id=rec["person_id"],
                    name=rec["name"],
                    embedding=rec["embedding"],
                    access_level=rec.get("access_level", "visitor"),
                )

    def _save_db(self):
        data = {
            pid: {
                "person_id": rec.person_id,
                "name": rec.name,
                "embedding": rec.embedding.tolist(),
                "access_level": rec.access_level,
            }
            for pid, rec in self.db.items()
        }
        FACE_DB_PATH.write_text(json.dumps(data, indent=2))

    # ── Registration ──────────────────────────────────────────────

    def register_face(self, frame: np.ndarray, name: str,
                      access_level: str = "visitor") -> Optional[str]:
        """Extract embedding from frame and store in DB."""
        emb = self._get_embedding(frame)
        if emb is None:
            return None
        pid = str(uuid.uuid4())
        self.db[pid] = FaceRecord(pid, name, emb.tolist(), access_level)
        self._save_db()
        return pid

    # ── Recognition ───────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[FaceResult]]:
        results: List[FaceResult] = []
        annotated = frame.copy()

        if self.backend == "insightface":
            faces = self.app.get(frame)
            for face in faces:
                bbox = tuple(map(int, face.bbox))
                name, pid, conf = self._match_embedding(face.embedding)
                r = FaceResult(bbox, name, pid, conf)
                results.append(r)
                self._draw_face(annotated, r)

        elif self.backend == "face_recognition":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for (top, right, bottom, left), enc in zip(locs, encs):
                name, pid, conf = self._match_face_rec(enc)
                r = FaceResult((left, top, right, bottom), name, pid, conf)
                results.append(r)
                self._draw_face(annotated, r)

        else:  # haar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in rects:
                r = FaceResult((x, y, x + w, y + h), "Unknown", None, 0.0)
                results.append(r)
                self._draw_face(annotated, r)

        return annotated, results

    # ── Matching ──────────────────────────────────────────────────

    def _match_embedding(self, emb: np.ndarray, threshold: float = 0.5):
        if not self.db:
            return "Unknown", None, 0.0
        best_name, best_pid, best_sim = "Unknown", None, 0.0
        for rec in self.db.values():
            sim = float(np.dot(emb, rec.embedding) /
                        (np.linalg.norm(emb) * np.linalg.norm(rec.embedding) + 1e-9))
            if sim > best_sim:
                best_sim, best_name, best_pid = sim, rec.name, rec.person_id
        if best_sim < threshold:
            return "Unknown", None, best_sim
        return best_name, best_pid, best_sim

    def _match_face_rec(self, encoding):
        if not self.db:
            return "Unknown", None, 0.0
        for rec in self.db.values():
            dist = float(np.linalg.norm(encoding - rec.embedding))
            if dist < 0.6:
                return rec.name, rec.person_id, 1 - dist
        return "Unknown", None, 0.0

    def _get_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.backend == "insightface":
            faces = self.app.get(frame)
            return faces[0].embedding if faces else None
        if self.backend == "face_recognition":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            return np.array(encs[0]) if encs else None
        return None

    # ── Drawing ───────────────────────────────────────────────────

    def _draw_face(self, frame: np.ndarray, r: FaceResult):
        x1, y1, x2, y2 = r.bbox
        color = (0, 255, 180) if r.name != "Unknown" else (0, 100, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{r.name} ({r.confidence:.0%})" if r.name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def get_face_service() -> FaceService:
    svc = FaceService()
    svc.initialise()
    return svc
