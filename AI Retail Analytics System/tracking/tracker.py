"""
tracking/tracker.py
--------------------
Multi-object tracking using SORT (Simple Online and Realtime Tracking).
Falls back to a lightweight centroid-based tracker if sort is unavailable.
"""

import numpy as np
import time

# ── Try to import SORT ──────────────────────────────────────────────────────
try:
    from sort import Sort as _Sort

    class _SORTWrapper:
        def __init__(self):
            self._tracker = _Sort(max_age=30, min_hits=3, iou_threshold=0.3)

        def update(self, dets: np.ndarray) -> np.ndarray:
            return self._tracker.update(dets)

    _BACKEND = "sort"

except ImportError:
    _BACKEND = "centroid"

# ── Lightweight centroid fallback ─────────────────────────────────────────
class _CentroidTracker:
    """Minimal centroid-based tracker used when SORT is unavailable."""

    def __init__(self, max_disappeared: int = 40):
        self.next_id = 1
        self.objects: dict[int, np.ndarray] = {}  # id → centroid [cx, cy]
        self.boxes: dict[int, np.ndarray] = {}    # id → [x1,y1,x2,y2]
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray, box: np.ndarray):
        self.objects[self.next_id] = centroid
        self.boxes[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.boxes[obj_id]
        del self.disappeared[obj_id]

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Args:
            dets: (N,5) [x1,y1,x2,y2,conf]
        Returns:
            (M,5) [x1,y1,x2,y2,track_id]
        """
        if dets is None or len(dets) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return np.empty((0, 5))

        input_centroids = np.array([
            [(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets
        ])

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, dets[i, :4])
        else:
            oids = list(self.objects.keys())
            oc = np.array(list(self.objects.values()))

            D = np.linalg.norm(oc[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 100:
                    continue
                obj_id = oids[row]
                self.objects[obj_id] = input_centroids[col]
                self.boxes[obj_id] = dets[col, :4]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(oids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                obj_id = oids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col], dets[col, :4])

        result = []
        for obj_id, box in self.boxes.items():
            result.append([box[0], box[1], box[2], box[3], float(obj_id)])
        return np.array(result) if result else np.empty((0, 5))


# ── Public tracker ─────────────────────────────────────────────────────────
class RetailTracker:
    """
    Unified tracker interface for the retail analytics pipeline.
    Automatically selects SORT or centroid tracker.
    """

    def __init__(self):
        if _BACKEND == "sort":
            self._tracker = _SORTWrapper()
        else:
            self._tracker = _CentroidTracker()

        print(f"[Tracker] Backend: {_BACKEND.upper()}")

        # Per-ID entry times for dwell-time computation
        self.entry_times: dict[int, float] = {}

    @property
    def backend(self) -> str:
        return _BACKEND

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Update tracker with current-frame detections.

        Args:
            dets: (N,5) array [x1,y1,x2,y2,conf]
        Returns:
            (M,5) array [x1,y1,x2,y2,track_id]
        """
        tracks = self._tracker.update(dets)

        now = time.time()
        for track in tracks:
            tid = int(track[4])
            if tid not in self.entry_times:
                self.entry_times[tid] = now

        return tracks

    def get_dwell_time(self, track_id: int) -> float:
        """Return seconds since first detection of this track_id."""
        if track_id in self.entry_times:
            return time.time() - self.entry_times[track_id]
        return 0.0

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray,
                    dwell_times: dict[int, float] | None = None) -> np.ndarray:
        """
        Draw tracking boxes and IDs on frame.

        Args:
            frame: BGR frame
            tracks: (M,5) [x1,y1,x2,y2,track_id]
            dwell_times: optional dict {track_id: seconds}
        """
        palette = [
            (0, 212, 255), (255, 100, 0), (0, 255, 150),
            (255, 0, 200), (255, 220, 0), (100, 0, 255),
        ]

        for track in tracks:
            x1, y1, x2, y2, tid = map(int, track[:5])
            color = palette[tid % len(palette)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, color, -1)

            label = f"ID:{tid}"
            if dwell_times and tid in dwell_times:
                label += f"  {dwell_times[tid]:.0f}s"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        return frame


# ── import cv2 here to avoid circular ────────────────────────────────────
import cv2  # noqa: E402
