"""
VisionAI Platform - Emotion Detection Service
Uses FER (Facial Expression Recognition) or a lightweight CNN.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try FER library first
try:
    from fer import FER
    _HAS_FER = True
except ImportError:
    _HAS_FER = False

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   215, 255),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 0),
    "surprise": (0,   165, 255),
}


class EmotionResult:
    def __init__(self, bbox: Tuple, emotion: str, scores: Dict[str, float]):
        self.bbox = bbox
        self.emotion = emotion
        self.scores = scores
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "emotion": self.emotion,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


class EmotionService:
    _instance: Optional["EmotionService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return

        if _HAS_FER:
            self.detector = FER(mtcnn=False)
            self.backend = "fer"
            print("[EmotionService] Backend: FER")
        else:
            # Haar-based fallback with a simple heuristic
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            self.backend = "haar_heuristic"
            print("[EmotionService] Backend: Haar+Heuristic (install 'fer' for real emotions)")

        self._initialised = True

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[EmotionResult]]:
        annotated = frame.copy()
        results: List[EmotionResult] = []

        if self.backend == "fer":
            detections = self.detector.detect_emotions(frame)
            for det in detections:
                bx, by, bw, bh = det["box"]
                scores = det["emotions"]
                dominant = max(scores, key=scores.get)
                r = EmotionResult((bx, by, bx + bw, by + bh), dominant, scores)
                results.append(r)
                self._draw(annotated, r)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in rects:
                # Simple brightness-based heuristic (demo only)
                roi = gray[y:y + h, x:x + w]
                brightness = float(np.mean(roi))
                dominant = "happy" if brightness > 130 else "neutral"
                scores = {e: 0.0 for e in EMOTIONS}
                scores[dominant] = 1.0
                r = EmotionResult((x, y, x + w, y + h), dominant, scores)
                results.append(r)
                self._draw(annotated, r)

        return annotated, results

    def _draw(self, frame: np.ndarray, r: EmotionResult):
        x1, y1, x2, y2 = r.bbox
        color = EMOTION_COLORS.get(r.emotion, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"😊 {r.emotion.upper()}"
        cv2.putText(frame, label, (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # Emotion bar chart overlay
        bar_x = x2 + 5
        bar_y = y1
        for i, (emo, score) in enumerate(r.scores.items()):
            bcolor = EMOTION_COLORS.get(emo, (200, 200, 200))
            bar_len = int(score * 60)
            cv2.rectangle(frame, (bar_x, bar_y + i * 12),
                          (bar_x + bar_len, bar_y + i * 12 + 9), bcolor, -1)
            cv2.putText(frame, emo[:3], (bar_x - 28, bar_y + i * 12 + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

    def dominant_emotion_stats(self, results: List[EmotionResult]) -> Dict[str, float]:
        totals: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        if not results:
            return totals
        for r in results:
            for k, v in r.scores.items():
                totals[k] = totals.get(k, 0) + v
        n = len(results)
        return {k: round(v / n, 3) for k, v in totals.items()}


def get_emotion_service() -> EmotionService:
    svc = EmotionService()
    svc.initialise()
    return svc
