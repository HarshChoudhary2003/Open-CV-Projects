"""
detection/yolo.py
-----------------
YOLOv8-based person detection module for Retail Analytics System.
Uses ultralytics YOLO to detect persons (class 0) in video frames.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Detects persons in a video frame using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_path: Path to YOLOv8 weights file. Auto-downloads if not found.
            confidence: Minimum detection confidence threshold (0–1).
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # COCO class 0 = person

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single frame.

        Returns:
            List of dicts with keys: x1, y1, x2, y2, confidence, class_id
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == self.person_class_id and conf >= self.confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class_id": cls,
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes for raw detections (before tracking)."""
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
        return frame

    @staticmethod
    def detections_to_array(detections: list[dict]) -> np.ndarray:
        """
        Convert detection dicts to numpy array format expected by SORT tracker.
        Shape: (N, 5) → [x1, y1, x2, y2, confidence]
        """
        if not detections:
            return np.empty((0, 5))
        return np.array([
            [d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]]
            for d in detections
        ], dtype=np.float32)
