"""
utils/alert_engine.py
Rule-based alert engine — evaluates tracks and fires structured alerts.
Supports cooldown per alert type per track, logging, and snapshot saving.
"""

from __future__ import annotations
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    ALERT_COOLDOWN_SEC, NIGHT_START_HOUR, NIGHT_END_HOUR,
    LOITER_SECONDS, SNAPSHOTS_DIR, SNAPSHOT_ON_ALERT
)
from database.db_manager import log_event


# ─── Alert Definition ─────────────────────────────────────────────────────────
ALERT_DEFS = {
    "UNKNOWN_FACE": {
        "severity": "ALERT",
        "label":    "⚠ Unknown Person",
        "color":    (0, 0, 220),
    },
    "LOITERING": {
        "severity": "WARNING",
        "label":    "⏱ Loitering Detected",
        "color":    (0, 140, 255),
    },
    "NIGHT_INTRUSION": {
        "severity": "ALERT",
        "label":    "🌙 Night Intrusion",
        "color":    (180, 0, 220),
    },
    "CROWD": {
        "severity": "WARNING",
        "label":    "👥 Crowd Detected",
        "color":    (200, 100, 0),
    },
}


class AlertEngine:
    """
    Evaluates active tracks each frame, fires alerts according to rules,
    persists them to SQLite, and optionally saves a snapshot JPG.
    """

    def __init__(self) -> None:
        # (track_id, alert_type) → last fired timestamp
        self._cooldowns: Dict[Tuple[int, str], float] = {}
        self._lock = threading.Lock()
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────
    def evaluate(self, tracks, frame: np.ndarray) -> List[Dict]:
        """
        `tracks` = list of Track objects from PersonDetector.
        Returns list of fired alert dicts for this frame.
        """
        fired: List[Dict] = []
        now   = time.time()
        hour  = datetime.now().hour
        is_night = (hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR)

        # ── Per-track rules ──────────────────────────────────────────────────
        for t in tracks:
            # 1. Unknown face
            if t.label == "Unknown":
                if self._can_fire(t.track_id, "UNKNOWN_FACE", now):
                    desc = f"Unknown person (Track #{t.track_id}) detected."
                    self._fire("UNKNOWN_FACE", desc, t.track_id, frame, t.bbox, fired)

            # 2. Loitering
            if t.duration >= LOITER_SECONDS:
                if self._can_fire(t.track_id, "LOITERING", now):
                    desc = (f"Person '{t.label}' (Track #{t.track_id}) "
                            f"loitering for {t.duration:.0f}s.")
                    self._fire("LOITERING", desc, t.track_id, frame, t.bbox, fired)

            # 3. Night intrusion
            if is_night:
                if self._can_fire(t.track_id, "NIGHT_INTRUSION", now):
                    desc = f"Person detected during night hours (Track #{t.track_id})."
                    self._fire("NIGHT_INTRUSION", desc, t.track_id, frame, t.bbox, fired)

        # ── Scene-level rule ─────────────────────────────────────────────────
        if len(tracks) >= 4:
            if self._can_fire(0, "CROWD", now):
                desc = f"Crowd detected: {len(tracks)} persons in scene."
                self._fire("CROWD", desc, 0, frame, None, fired)

        return fired

    # ── Internal ──────────────────────────────────────────────────────────────
    def _can_fire(self, track_id: int, alert_type: str, now: float) -> bool:
        key = (track_id, alert_type)
        with self._lock:
            last = self._cooldowns.get(key, 0.0)
            if now - last >= ALERT_COOLDOWN_SEC:
                self._cooldowns[key] = now
                return True
        return False

    def _fire(self, alert_type: str, description: str,
              track_id: int, frame: np.ndarray,
              bbox: Optional[Tuple], fired_list: List) -> None:
        defn     = ALERT_DEFS[alert_type]
        severity = defn["severity"]
        snapshot_path: Optional[str] = None

        if SNAPSHOT_ON_ALERT:
            snapshot_path = self._save_snapshot(frame, bbox, alert_type)

        log_event(
            event_type=alert_type,
            description=description,
            severity=severity,
            snapshot=snapshot_path,
            track_id=track_id if track_id > 0 else None
        )

        fired_list.append({
            "type":        alert_type,
            "severity":    severity,
            "label":       defn["label"],
            "color":       defn["color"],
            "description": description,
            "snapshot":    snapshot_path,
        })

    @staticmethod
    def _save_snapshot(frame: np.ndarray,
                       bbox: Optional[Tuple],
                       label: str) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        name = f"{label}_{ts}.jpg"
        path = os.path.join(SNAPSHOTS_DIR, name)
        save_frame = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 220), 3)
        cv2.imwrite(path, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path
