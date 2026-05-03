"""
VisionAI Platform - Heatmap Analytics Service
Accumulates movement history per camera and renders
colour-coded density overlays (Jet colormap).
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class HeatmapService:
    """
    Per-camera heatmap generator.
    - Accumulates person centroid positions over a rolling window.
    - Renders an additive Gaussian-blurred density map in Jet palette.
    - Returns both the annotated frame and the raw heatmap array.
    """

    _instance: Optional["HeatmapService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self, decay_seconds: float = 120.0, blur_kernel: int = 51):
        if self._initialised:
            return
        self._maps: Dict[str, np.ndarray] = {}        # camera_id -> accumulation map
        self._decay = decay_seconds
        self._blur = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self._history: Dict[str, deque] = {}          # camera_id -> deque[(timestamp, cx, cy)]
        self._initialised = True

    # ─── Public API ───────────────────────────────────────────────────

    def update(
        self,
        camera_id: str,
        detections: List[dict],
        frame_shape: Tuple[int, int],
    ) -> None:
        """Feed new detections into the heat accumulator."""
        h, w = frame_shape[:2]

        if camera_id not in self._maps:
            self._maps[camera_id] = np.zeros((h, w), dtype=np.float32)
            self._history[camera_id] = deque()

        now = time.time()
        hist = self._history[camera_id]
        acc = self._maps[camera_id]

        # Resize accumulator if frame dimensions changed
        if acc.shape != (h, w):
            self._maps[camera_id] = np.zeros((h, w), dtype=np.float32)
            acc = self._maps[camera_id]

        # Decay old entries
        while hist and (now - hist[0][0]) > self._decay:
            _, cx, cy = hist.popleft()
            if 0 <= cy < h and 0 <= cx < w:
                acc[cy, cx] = max(0.0, acc[cy, cx] - 1.0)

        # Add new person centroids
        for det in detections:
            if det.get("class_name") != "person":
                continue
            bbox = det.get("bbox", {})
            cx = int((bbox.get("x1", 0) + bbox.get("x2", 0)) / 2)
            cy = int((bbox.get("y1", 0) + bbox.get("y2", 0)) / 2)
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            acc[cy, cx] += 1.0
            hist.append((now, cx, cy))

    def render_overlay(
        self,
        camera_id: str,
        frame: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Blend the Jet heatmap over the source frame.
        Returns the annotated frame.
        """
        if camera_id not in self._maps:
            return frame

        acc = self._maps[camera_id]
        h, w = frame.shape[:2]
        if acc.shape != (h, w):
            return frame

        blurred = cv2.GaussianBlur(acc, (self._blur, self._blur), 0)
        max_val = blurred.max()
        if max_val < 1e-6:
            return frame

        norm = (blurred / max_val * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = norm > 5  # only blend where there is actual heat
        result = frame.copy()
        result[mask] = cv2.addWeighted(
            frame[mask], 1 - alpha, colored[mask], alpha, 0
        )
        return result

    def get_raw(self, camera_id: str) -> Optional[np.ndarray]:
        """Return the raw accumulation array (float32)."""
        return self._maps.get(camera_id)

    def reset(self, camera_id: str) -> None:
        self._maps.pop(camera_id, None)
        self._history.pop(camera_id, None)


def get_heatmap_service() -> HeatmapService:
    svc = HeatmapService()
    svc.initialise()
    return svc
