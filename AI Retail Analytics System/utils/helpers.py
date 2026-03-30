"""
utils/helpers.py
-----------------
General utility functions shared across the pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional


# ── Frame helpers ─────────────────────────────────────────────────────────
def resize_frame(frame: np.ndarray, width: int = 960) -> np.ndarray:
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def overlay_text(frame: np.ndarray, lines: list[str],
                 origin: tuple[int, int] = (10, 30),
                 font_scale: float = 0.65,
                 color: tuple[int, int, int] = (240, 240, 240),
                 bg_color: tuple[int, int, int] = (10, 10, 30),
                 alpha: float = 0.65) -> np.ndarray:
    """
    Draw a semi-transparent text panel on the frame.

    Args:
        frame:      BGR frame to annotate.
        lines:      List of text strings (one per row).
        origin:     Top-left corner (x, y) of the panel.
        font_scale: Text size.
        color:      Text colour (BGR).
        bg_color:   Panel background colour (BGR).
        alpha:      Panel transparency (0 = transparent, 1 = opaque).
    Returns:
        Annotated frame (in-place modification).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(font_scale * 40)
    pad = 12

    max_w = max(
        cv2.getTextSize(line, font, font_scale, 2)[0][0] for line in lines
    )
    total_h = line_height * len(lines) + pad * 2
    x, y = origin

    # Clamp to frame bounds
    x2 = min(x + max_w + pad * 2, frame.shape[1])
    y2 = min(y + total_h, frame.shape[0])

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(lines):
        ty = y + pad + line_height * i + line_height // 2
        cv2.putText(frame, line, (x + pad, ty),
                    font, font_scale, color, 2, cv2.LINE_AA)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter in the top-right corner."""
    label = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - tw - 14
    cv2.putText(frame, label, (x, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)
    return frame


# ── Zone helpers ──────────────────────────────────────────────────────────
ZONE_LABELS = {
    "entrance": ((0.0, 0.0), (0.33, 1.0)),
    "center":   ((0.33, 0.0), (0.66, 1.0)),
    "back":     ((0.66, 0.0), (1.0, 1.0)),
}


def classify_zone(cx: float, cy: float, frame_w: float,
                  zones: dict | None = None) -> str:
    """
    Return zone name for a centroid (cx, cy in pixels).

    Args:
        cx, cy:   Centroid coordinates.
        frame_w:  Frame width in pixels (used for relative X).
        zones:    Dict of zone_name → ((x0_rel, y0_rel), (x1_rel, y1_rel)).
                  Defaults to ZONE_LABELS (entrance / center / back).
    """
    zones = zones or ZONE_LABELS
    rx = cx / frame_w
    for name, ((x0, y0), (x1, y1)) in zones.items():
        if x0 <= rx < x1:
            return name
    return "unknown"


# ── Video source helpers ──────────────────────────────────────────────────
def open_source(source: str | int) -> cv2.VideoCapture:
    """Open webcam index or video file path, returning a VideoCapture."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source!r}")
    return cap


def get_frame_dims(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Return (height, width) of a VideoCapture."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return h, w


# ── Synthetic demo video generator ───────────────────────────────────────
def generate_demo_video(
    output_path: str = "assets/demo.mp4",
    duration_sec: int = 30,
    fps: int = 25,
    width: int = 960,
    height: int = 540,
    n_people: int = 6,
) -> str:
    """
    Generate a synthetic retail store simulation video (no camera needed).

    Simulates n_people walking across a corridor with randomised paths.

    Returns:
        Path to generated video.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np.random.seed(42)
    # Each person: (x, y, vx, vy, color)
    people = []
    for _ in range(n_people):
        x = float(np.random.randint(50, width - 50))
        y = float(np.random.randint(100, height - 100))
        vx = float(np.random.choice([-1, 1]) * np.random.uniform(1.5, 4.0))
        vy = float(np.random.uniform(-1.0, 1.0))
        color = tuple(int(c) for c in np.random.randint(80, 230, 3).tolist())
        people.append([x, y, vx, vy, color])

    total_frames = duration_sec * fps
    for frame_idx in range(total_frames):
        # Store background: gradient floor
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height), (25, 20, 35), -1)

        # Aisle lines
        for ax in [width // 3, 2 * width // 3]:
            cv2.line(frame, (ax, 0), (ax, height), (50, 45, 65), 2)

        # Shelves
        for shelf_y in [int(height * 0.25), int(height * 0.75)]:
            cv2.rectangle(frame, (0, shelf_y - 12), (width, shelf_y + 12),
                          (60, 50, 80), -1)

        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"DEMO STORE  {ts}  Frame {frame_idx}/{total_frames}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)

        # Move & draw people
        for p in people:
            p[0] += p[2]
            p[1] += p[3]
            # Bounce off walls
            if p[0] < 30 or p[0] > width - 30:
                p[2] *= -1
            if p[1] < 60 or p[1] > height - 60:
                p[3] *= -1

            # Random direction changes
            if np.random.rand() < 0.02:
                p[3] = np.random.uniform(-2.0, 2.0)

            cx, cy = int(p[0]), int(p[1])
            color = p[4]
            # Body ellipse
            cv2.ellipse(frame, (cx, cy + 15), (14, 22), 0, 0, 360, color, -1)
            # Head
            cv2.circle(frame, (cx, cy - 14), 10, color, -1)

        out.write(frame)

    out.release()
    print(f"[Demo] Generated {total_frames} frames → {output_path}")
    return output_path


# ── Insight generator ─────────────────────────────────────────────────────
def generate_insights(
    footfall: int,
    avg_dwell: float,
    peak_hour: str,
    hot_zones: list[dict],
    frame_w: int,
) -> list[str]:
    """
    Produce human-readable analytics insights based on session metrics.
    """
    insights = []

    if footfall > 0:
        insights.append(f"👥 {footfall} unique visitors detected this session.")
    else:
        insights.append("📷 No visitors detected yet. Check camera source.")

    if avg_dwell >= 60:
        m, s = divmod(int(avg_dwell), 60)
        insights.append(f"⏱️ Avg dwell time: {m}m {s}s — customers are engaged!")
    elif avg_dwell > 0:
        insights.append(f"⏱️ Avg dwell time: {avg_dwell:.0f}s — quick browse session.")

    if peak_hour != "N/A":
        insights.append(f"🕒 Peak traffic hour: {peak_hour}")

    zone_labels = ["Entrance", "Center Aisle", "Back Section"]
    for i, zone in enumerate(hot_zones[:2]):
        rel_x = zone["x"] / max(frame_w, 1)
        if rel_x < 0.33:
            zone_name = zone_labels[0]
        elif rel_x < 0.66:
            zone_name = zone_labels[1]
        else:
            zone_name = zone_labels[2]
        insights.append(f"🔥 Hot zone #{i+1}: {zone_name} (intensity {zone['intensity']:.1f})")

    if not hot_zones:
        insights.append("🗺️ Heatmap empty — movement data accumulating...")

    return insights
