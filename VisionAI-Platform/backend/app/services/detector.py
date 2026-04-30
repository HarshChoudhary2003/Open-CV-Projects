"""
VisionAI Platform - YOLOv8 Object Detection Service
Wraps Ultralytics YOLOv8 with ByteTrack/DeepSORT tracking integration.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import settings


# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Vibrant palette per-class
_PALETTE = np.random.default_rng(42).integers(100, 255, size=(len(COCO_CLASSES), 3)).tolist()


class Detection:
    """Single object detection result."""

    __slots__ = ("class_id", "class_name", "confidence", "bbox", "track_id", "timestamp")

    def __init__(
        self,
        class_id: int,
        confidence: float,
        bbox: Tuple[int, int, int, int],   # x1, y1, x2, y2
        track_id: Optional[int] = None,
    ):
        self.class_id = class_id
        self.class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"
        self.confidence = confidence
        self.bbox = bbox
        self.track_id = track_id
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "track_id": self.track_id,
            "timestamp": self.timestamp,
        }


class YOLODetector:
    """
    Production-grade YOLOv8 wrapper.
    - Lazy-loads model once per process
    - Supports ByteTrack (built-in Ultralytics) & DeepSORT
    - Draws annotated frames with HUD-style overlays
    """

    _instance: Optional["YOLODetector"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return
        weights = settings.YOLO_MODEL
        self.model = YOLO(weights)
        self.model.to(settings.DEVICE)
        self.conf = settings.YOLO_CONFIDENCE
        self.iou = settings.YOLO_IOU
        self._initialised = True
        print(f"[YOLODetector] Loaded {weights} on {settings.DEVICE}")

    # ──────────────────────────────────────────────────────────────
    def detect(
        self,
        frame: np.ndarray,
        track: bool = True,
        classes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Run inference and (optionally) tracking.
        Returns (annotated_frame, list_of_detections).
        """
        if track:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=classes,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=classes,
                verbose=False,
            )

        detections: List[Detection] = []
        annotated = frame.copy()

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else None

                det = Detection(cls_id, conf, (x1, y1, x2, y2), tid)
                detections.append(det)
                annotated = self._draw_detection(annotated, det)

        return annotated, detections

    # ──────────────────────────────────────────────────────────────
    def _draw_detection(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        x1, y1, x2, y2 = det.bbox
        color = tuple(_PALETTE[det.class_id % len(_PALETTE)])

        # Filled corner brackets instead of full rectangle
        thickness = 2
        length = max(15, (x2 - x1) // 5)
        pts = [
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        ]
        for corner, h, v in pts:
            cv2.line(frame, corner, h, color, thickness)
            cv2.line(frame, corner, v, color, thickness)

        # Semi-transparent label bg
        label = f"{'[' + str(det.track_id) + '] ' if det.track_id else ''}{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    # ──────────────────────────────────────────────────────────────
    def count_by_class(self, detections: List[Detection]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


def get_detector() -> YOLODetector:
    det = YOLODetector()
    det.initialise()
    return det
