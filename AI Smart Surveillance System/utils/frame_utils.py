"""
utils/frame_utils.py
Frame overlay and HUD rendering helpers.
"""

from __future__ import annotations
from datetime import datetime
from typing import List

import cv2
import numpy as np


def draw_hud(frame: np.ndarray, fps: float, person_count: int,
             active_alerts: List[str]) -> np.ndarray:
    """
    Draw a translucent HUD overlay on the top-left corner:
      • FPS
      • Person count
      • Active alert banners
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Top bar ──────────────────────────────────────────────────────────────
    bar_h = 44
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "AI SURVEILLANCE", (12, 29),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 220, 130), 1, cv2.LINE_AA)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, ts, (w - tw - 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Info pills ───────────────────────────────────────────────────────────
    pill_y = bar_h + 14
    fps_txt = f"FPS: {fps:.1f}"
    ppl_txt = f"People: {person_count}"
    cv2.putText(frame, fps_txt, (12, pill_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, ppl_txt, (120, pill_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 160), 1, cv2.LINE_AA)

    # ── Alert banners ────────────────────────────────────────────────────────
    for i, alert_label in enumerate(active_alerts[:4]):
        ay = pill_y + 26 + i * 22
        cv2.putText(frame, alert_label, (12, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 80, 255), 1, cv2.LINE_AA)

    return frame


def draw_zone(frame: np.ndarray, zone_pts: np.ndarray,
              color=(0, 255, 200), alpha=0.18) -> np.ndarray:
    """Draw a semi-transparent restricted zone polygon."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone_pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [zone_pts], True, color, 2)
    return frame


def resize_for_display(frame: np.ndarray, max_w: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame
