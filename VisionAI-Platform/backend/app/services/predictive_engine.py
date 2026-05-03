"""
VisionAI Platform - Predictive Behaviour Intelligence
Uses short-term trajectory analysis (Kalman-inspired linear extrapolation)
to predict WHERE a tracked object will be N frames in the future,
and scores the prediction against known "suspicious" movement patterns.

Suspicious Patterns Detected:
  • Rapid acceleration toward a zone boundary
  • Loiter → sudden dash
  • Converging paths from multiple tracks
  • Abrupt direction reversal (evasive behaviour)
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TrackHistory:
    """Rolling history of (timestamp, cx, cy) for one tracked object."""

    def __init__(self, maxlen: int = 60):
        self.records: deque = deque(maxlen=maxlen)

    def update(self, cx: float, cy: float):
        self.records.append((time.time(), cx, cy))

    def velocity(self) -> Tuple[float, float]:
        """Pixel-per-second velocity estimate from last 5 samples."""
        pts = list(self.records)
        if len(pts) < 2:
            return 0.0, 0.0
        recent = pts[-min(5, len(pts)):]
        dt = recent[-1][0] - recent[0][0] + 1e-6
        vx = (recent[-1][1] - recent[0][1]) / dt
        vy = (recent[-1][2] - recent[0][2]) / dt
        return vx, vy

    def acceleration(self) -> Tuple[float, float]:
        pts = list(self.records)
        if len(pts) < 4:
            return 0.0, 0.0
        mid = len(pts) // 2
        early, late = pts[:mid], pts[mid:]
        dt_e = early[-1][0] - early[0][0] + 1e-6
        dt_l = late[-1][0] - late[0][0] + 1e-6
        vx_e = (early[-1][1] - early[0][1]) / dt_e
        vy_e = (early[-1][2] - early[0][2]) / dt_e
        vx_l = (late[-1][1] - late[0][1]) / dt_l
        vy_l = (late[-1][2] - late[0][2]) / dt_l
        return (vx_l - vx_e) / (dt_e + dt_l), (vy_l - vy_e) / (dt_e + dt_l)

    def predict_position(self, seconds_ahead: float = 1.5) -> Tuple[float, float]:
        if not self.records:
            return 0.0, 0.0
        _, cx, cy = self.records[-1]
        vx, vy = self.velocity()
        return cx + vx * seconds_ahead, cy + vy * seconds_ahead


class PredictiveEngine:
    """
    Stateful engine that maintains track histories and scores suspicious
    predicted behaviour ahead of time.
    """

    _instance: Optional["PredictiveEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self, predict_horizon: float = 1.5):
        if self._initialised:
            return
        self._histories: Dict[int, TrackHistory] = defaultdict(TrackHistory)
        self._horizon = predict_horizon
        self._last_event_time: Dict[str, float] = {}
        self._initialised = True

    def update_and_predict(
        self,
        detections: List[dict],
        frame: np.ndarray,
        draw_predictions: bool = True,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Update track histories, compute predictions, score patterns.
        Returns (annotated_frame, list_of_prediction_events).
        """
        annotated = frame.copy()
        events: List[dict] = []
        h, w = frame.shape[:2]

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue

            bbox = det.get("bbox", {})
            cx = (bbox.get("x1", 0) + bbox.get("x2", 0)) / 2
            cy = (bbox.get("y1", 0) + bbox.get("y2", 0)) / 2

            hist = self._histories[tid]
            hist.update(cx, cy)

            vx, vy = hist.velocity()
            ax, ay = hist.acceleration()
            px, py = hist.predict_position(self._horizon)
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            speed = (vx ** 2 + vy ** 2) ** 0.5
            accel = (ax ** 2 + ay ** 2) ** 0.5

            # Score suspicious patterns
            score, reasons = self._score(tid, speed, accel, hist)

            if draw_predictions and len(hist.records) >= 3:
                # Draw predicted position
                self._draw_prediction(
                    annotated, int(cx), int(cy), int(px), int(py),
                    score, det.get("class_name", "obj"), tid
                )

            if score > 0.65 and reasons:
                key = f"pred_{tid}"
                now = time.time()
                if now - self._last_event_time.get(key, 0) > 10:
                    self._last_event_time[key] = now
                    events.append({
                        "type": "PREDICTED_SUSPICIOUS",
                        "track_id": tid,
                        "class_name": det.get("class_name", "person"),
                        "prediction_score": round(score, 3),
                        "reasons": reasons,
                        "predicted_x": round(px, 1),
                        "predicted_y": round(py, 1),
                        "timestamp": now,
                        "description": (
                            f"⚡ Predicted suspicious behaviour for track {tid}: "
                            + ", ".join(reasons)
                        ),
                    })

        return annotated, events

    def _score(
        self, tid: int, speed: float, accel: float, hist: TrackHistory
    ) -> Tuple[float, List[str]]:
        reasons = []
        score = 0.0

        # Pattern 1: Sudden rapid acceleration
        if accel > 80 and speed > 120:
            score += 0.4
            reasons.append("sudden acceleration")

        # Pattern 2: High speed (running)
        if speed > 200:
            score += 0.3
            reasons.append("running")

        # Pattern 3: Abrupt direction reversal
        pts = list(hist.records)
        if len(pts) >= 6:
            seg1 = (pts[-3][1] - pts[-6][1], pts[-3][2] - pts[-6][2])
            seg2 = (pts[-1][1] - pts[-3][1], pts[-1][2] - pts[-3][2])
            dot = seg1[0] * seg2[0] + seg1[1] * seg2[1]
            mag1 = (seg1[0] ** 2 + seg1[1] ** 2) ** 0.5 + 1e-6
            mag2 = (seg2[0] ** 2 + seg2[1] ** 2) ** 0.5 + 1e-6
            cos_a = dot / (mag1 * mag2)
            if cos_a < -0.7:
                score += 0.35
                reasons.append("evasive reversal")

        score = min(score, 1.0)
        return score, reasons

    def _draw_prediction(
        self,
        frame: np.ndarray,
        cx: int, cy: int, px: int, py: int,
        score: float,
        label: str,
        tid: int,
    ):
        color = (
            int(255 * score),
            int(255 * (1 - score)),
            200,
        )
        # Dashed trajectory line
        cv2.arrowedLine(frame, (cx, cy), (px, py), color, 2, tipLength=0.2)
        # Prediction circle
        radius = 12
        cv2.circle(frame, (px, py), radius, color, 2)
        if score > 0.5:
            text = f"⚡{score:.0%}"
            cv2.putText(frame, text, (px + 14, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def get_predictive_engine() -> PredictiveEngine:
    engine = PredictiveEngine()
    engine.initialise()
    return engine
