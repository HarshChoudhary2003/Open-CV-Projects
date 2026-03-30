"""
analytics/heatmap.py
---------------------
Spatial heatmap: accumulates pixel-level foot-traffic density over time.
"""

import cv2
import numpy as np


class HeatmapEngine:
    """
    Builds & renders a spatial traffic heatmap.

    The heatmap accumulates Gaussian blobs at each detected centroid.
    An alpha-blended colored overlay is generated for visualization.
    """

    def __init__(self, frame_height: int, frame_width: int,
                 decay: float = 0.995, blur_radius: int = 31):
        """
        Args:
            frame_height: Height of the video frame in pixels.
            frame_width:  Width of the video frame in pixels.
            decay:        Multiplicative decay per frame (< 1 fades old heat).
            blur_radius:  Gaussian kernel radius for heatmap spreading.
        """
        self.h = frame_height
        self.w = frame_width
        self.decay = decay
        self.blur_radius = blur_radius | 1  # must be odd

        self._map = np.zeros((frame_height, frame_width), dtype=np.float32)

        # Colormap for visualization: COLORMAP_JET gives blue→green→red
        self._colormap = cv2.COLORMAP_JET

    def update(self, tracks: np.ndarray):
        """
        Add foot-traffic for current-frame tracks.

        Args:
            tracks: (N,5) [x1,y1,x2,y2,track_id]
        """
        self._map *= self.decay

        for track in tracks:
            x1, y1, x2, y2 = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            cx = np.clip((x1 + x2) // 2, 0, self.w - 1)
            cy = np.clip((y1 + y2) // 2, 0, self.h - 1)
            self._map[cy, cx] += 1.0

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """
        Return frame with heatmap blended on top.

        Args:
            frame: BGR frame of shape (H, W, 3)
            alpha: Heatmap opacity (0 = invisible, 1 = fully opaque)
        Returns:
            Blended BGR frame
        """
        # Smooth & normalise
        blurred = cv2.GaussianBlur(self._map, (self.blur_radius, self.blur_radius), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(norm, self._colormap)

        # Mask: only show heat where there's signal
        mask = (norm > 10).astype(np.float32)
        mask_3c = np.stack([mask] * 3, axis=-1)

        overlay = frame.copy().astype(np.float32)
        overlay = overlay * (1 - mask_3c * alpha) + colored.astype(np.float32) * mask_3c * alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def get_heatmap_image(self) -> np.ndarray:
        """
        Pure heatmap as a BGR image (for Streamlit display / saving).
        """
        blurred = cv2.GaussianBlur(self._map, (self.blur_radius, self.blur_radius), 0)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(norm, self._colormap)

    def hot_zones(self, n: int = 3) -> list[dict]:
        """
        Return the top-N hottest zones by centroid.

        Returns list of dicts: {x, y, intensity}
        """
        blurred = cv2.GaussianBlur(self._map, (self.blur_radius, self.blur_radius), 0)
        zones = []
        temp = blurred.copy()
        for _ in range(n):
            _, max_val, _, max_loc = cv2.minMaxLoc(temp)
            if max_val < 0.01:
                break
            zones.append({"x": max_loc[0], "y": max_loc[1], "intensity": float(max_val)})
            # Suppress neighbourhood so next peak is different
            x, y = max_loc
            r = 60
            cv2.circle(temp, (x, y), r, 0, -1)
        return zones

    def reset(self):
        self._map[:] = 0.0
