"""
detection/yolo_detector.py
YOLOv8-powered human (person) detector with lightweight ByteTrack-style ID assignment.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Lazy-import ultralytics so the rest of the app can load even without YOLO installed
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL, YOLO_CONF, YOLO_IOU,
    PERSON_CLASS_ID, MAX_TRACK_AGE
)


# ─── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class Track:
    track_id:   int
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    centroid:   Tuple[int, int]
    age:        int = 0                      # frames since last matched
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    label:      str = "Unknown"             # face-recognition label

    @property
    def duration(self) -> float:
        return time.time() - self.first_seen


@dataclass
class Detection:
    bbox:       Tuple[int, int, int, int]
    confidence: float
    centroid:   Tuple[int, int]


# ─── IOU helper ───────────────────────────────────────────────────────────────
def _iou(b1: Tuple, b2: Tuple) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


# ─── Detector + Tracker ───────────────────────────────────────────────────────
class PersonDetector:
    """
    Wraps YOLOv8 inference + a simple IoU-based multi-object tracker
    so each person gets a persistent integer ID across frames.
    """

    def __init__(self) -> None:
        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")
        self.model  = _YOLO(YOLO_MODEL)
        self.tracks: List[Track] = []
        self._next_id = 1

    # ── Public API ────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Track]]:
        """
        Run detection + tracking on one frame.
        Returns annotated frame and list of active Track objects.
        """
        detections = self._detect(frame)
        self._update_tracks(detections)
        annotated  = self._draw(frame.copy())
        return annotated, self.tracks

    # ── Internal ──────────────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=[PERSON_CLASS_ID],
            verbose=False
        )
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx   = (x1 + x2) // 2
                cy   = (y1 + y2) // 2
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf,
                                            centroid=(cx, cy)))
        return detections

    def _update_tracks(self, detections: List[Detection]) -> None:
        matched_track_ids: set = set()

        for det in detections:
            best_iou   = 0.0
            best_track: Optional[Track] = None

            for t in self.tracks:
                iou = _iou(t.bbox, det.bbox)
                if iou > best_iou:
                    best_iou   = iou
                    best_track = t

            if best_track and best_iou > 0.3:
                best_track.bbox     = det.bbox
                best_track.centroid = det.centroid
                best_track.age      = 0
                best_track.last_seen = time.time()
                matched_track_ids.add(best_track.track_id)
            else:
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    centroid=det.centroid
                )
                self._next_id += 1
                self.tracks.append(new_track)
                matched_track_ids.add(new_track.track_id)

        # Age unmatched tracks; prune old ones
        for t in self.tracks:
            if t.track_id not in matched_track_ids:
                t.age += 1
        self.tracks = [t for t in self.tracks if t.age <= MAX_TRACK_AGE]

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        for t in self.tracks:
            x1, y1, x2, y2 = t.bbox
            color = (0, 255, 120) if t.label != "Unknown" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Pill-style label
            label = f"#{t.track_id} {t.label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Duration badge
            dur_txt = f"{t.duration:.0f}s"
            cv2.putText(frame, dur_txt, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return frame
