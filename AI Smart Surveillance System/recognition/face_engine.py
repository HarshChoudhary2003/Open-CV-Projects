"""
recognition/face_engine.py
Face recognition layer — loads known faces from disk,
compares them against person crops from the tracker.
"""

from __future__ import annotations
import os
import glob
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition as _fr
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    KNOWN_FACES_DIR, FACE_TOLERANCE,
    FACE_MODEL, FACE_SCALE_FACTOR
)


class FaceEngine:
    """
    Loads every image in KNOWN_FACES_DIR (filename = person name).
    Matches unknown faces against the library and returns identity labels.
    """

    def __init__(self) -> None:
        if not _FR_AVAILABLE:
            raise RuntimeError(
                "face_recognition is not installed.\n"
                "Run: pip install face_recognition dlib"
            )
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str]            = []
        self._cache: Dict[int, Tuple[str, float]] = {}  # track_id → (name, last_updated)
        self.load_known_faces()

    # ── Public ────────────────────────────────────────────────────────────────
    def load_known_faces(self) -> int:
        """(Re-)load all images from KNOWN_FACES_DIR. Returns count loaded."""
        self.known_encodings = []
        self.known_names     = []
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for path in glob.glob(os.path.join(KNOWN_FACES_DIR, ext)):
                name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
                img  = _fr.load_image_file(path)
                encs = _fr.face_encodings(img)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(name)

        print(f"[FaceEngine] Loaded {len(self.known_names)} known faces: {self.known_names}")
        return len(self.known_names)

    def identify(self, frame: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 track_id: int,
                 cache_ttl: float = 2.0) -> str:
        """
        Crop the person bbox from frame and try to match a face.
        Cache results per track_id for `cache_ttl` seconds.
        Returns display name or "Unknown".
        """
        # Use cache to avoid per-frame inference
        if track_id in self._cache:
            name, ts = self._cache[track_id]
            if time.time() - ts < cache_ttl:
                return name

        x1, y1, x2, y2 = bbox
        # Safety-clip to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "Unknown"

        # Downscale for speed
        small = cv2.resize(crop, (0, 0), fx=FACE_SCALE_FACTOR, fy=FACE_SCALE_FACTOR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = _fr.face_locations(rgb, model=FACE_MODEL)
        if not locations:
            name = "Unknown"
        else:
            encs = _fr.face_encodings(rgb, locations)
            name = "Unknown"
            if encs and self.known_encodings:
                distances = _fr.face_distance(self.known_encodings, encs[0])
                idx       = int(np.argmin(distances))
                if distances[idx] <= FACE_TOLERANCE:
                    name = self.known_names[idx]

        self._cache[track_id] = (name, time.time())
        return name

    def flush_cache(self, active_track_ids: List[int]) -> None:
        """Remove stale track entries."""
        stale = [tid for tid in self._cache if tid not in active_track_ids]
        for tid in stale:
            del self._cache[tid]
