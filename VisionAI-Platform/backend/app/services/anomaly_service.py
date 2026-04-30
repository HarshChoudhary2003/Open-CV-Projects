"""
VisionAI Platform - Anomaly Detection Service
Statistical motion analysis: optical flow + background subtraction + dwell time.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings


class AnomalyEvent:
    def __init__(self, anomaly_type: str, score: float, description: str,
                 frame: Optional[np.ndarray] = None):
        self.anomaly_type = anomaly_type
        self.score = score
        self.description = description
        self.snapshot = frame
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type,
            "score": round(self.score, 3),
            "description": self.description,
            "timestamp": self.timestamp,
        }


class AnomalyDetector:
    _instance: Optional["AnomalyDetector"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.prev_gray: Optional[np.ndarray] = None
        self._flow_window: deque = deque(maxlen=30)
        self._dwell: Dict[int, Dict] = {}
        self.sensitivity = settings.ANOMALY_SENSITIVITY
        self._last_alert_time: float = 0
        self.cooldown = settings.ALERT_COOLDOWN_SECONDS
        self._initialised = True

    def analyse(self, frame: np.ndarray, detections=None, camera_id: str = "cam0",
                ) -> Tuple[np.ndarray, float, List[AnomalyEvent]]:
        annotated = frame.copy()
        events: List[AnomalyEvent] = []
        detections = detections or []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow_score = self._flow_score(gray)
        bg_score = self._bg_score(frame)
        n_persons = sum(1 for d in detections if d.class_name == "person")
        density_score = min(n_persons / 15.0, 1.0)
        dwell_score = self._dwell_score(detections)

        composite = 0.3 * flow_score + 0.3 * bg_score + 0.2 * density_score + 0.2 * dwell_score

        now = time.time()
        if composite > self.sensitivity and (now - self._last_alert_time) > self.cooldown:
            if flow_score > 0.8:
                events.append(AnomalyEvent("RAPID_MOVEMENT", flow_score,
                                           "Erratic rapid movement detected.", frame.copy()))
            if density_score > 0.7:
                events.append(AnomalyEvent("CROWD_DENSITY", density_score,
                                           f"High crowd: {n_persons} persons.", frame.copy()))
            if dwell_score > 0.6:
                events.append(AnomalyEvent("LOITERING", dwell_score,
                                           "Person loitering detected.", frame.copy()))
            if events:
                self._last_alert_time = now

        self._draw_hud(annotated, composite)
        self.prev_gray = gray
        return annotated, composite, events

    def _flow_score(self, gray: np.ndarray) -> float:
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        self._flow_window.append(mean_mag)
        baseline = np.mean(self._flow_window) + 1e-6
        return min((mean_mag + float(np.var(mag)) * 0.5) / (baseline * 5 + 1e-6), 1.0)

    def _bg_score(self, frame: np.ndarray) -> float:
        mask = self.bg_subtractor.apply(frame)
        fg = np.sum(mask == 255)
        total = frame.shape[0] * frame.shape[1]
        return min(fg / (total * 0.4 + 1), 1.0)

    def _dwell_score(self, detections) -> float:
        now = time.time()
        max_dwell = 0.0
        for det in detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            tid = det.track_id
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            if tid not in self._dwell:
                self._dwell[tid] = {"first_seen": now, "pos": (cx, cy)}
            else:
                dist = ((cx - self._dwell[tid]["pos"][0]) ** 2 +
                        (cy - self._dwell[tid]["pos"][1]) ** 2) ** 0.5
                if dist < 30:
                    max_dwell = max(max_dwell, now - self._dwell[tid]["first_seen"])
                else:
                    self._dwell[tid] = {"first_seen": now, "pos": (cx, cy)}
        return min(max_dwell / 30.0, 1.0)

    def _draw_hud(self, frame: np.ndarray, composite: float):
        h, w = frame.shape[:2]
        color = (0, int(255 * (1 - composite)), int(255 * composite))
        bar_w = int(w * 0.2)
        cv2.rectangle(frame, (w - bar_w - 10, 10), (w - 10, 28), (40, 40, 40), -1)
        cv2.rectangle(frame, (w - bar_w - 10, 10),
                      (w - bar_w - 10 + int(bar_w * composite), 28), color, -1)
        cv2.putText(frame, f"THREAT {composite:.0%}", (w - bar_w - 10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if composite > 0.75:
            cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 220), 3)
            cv2.putText(frame, "ANOMALY DETECTED", (w // 2 - 130, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def get_anomaly_detector() -> AnomalyDetector:
    det = AnomalyDetector()
    det.initialise()
    return det
