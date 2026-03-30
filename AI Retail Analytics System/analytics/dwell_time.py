"""
analytics/dwell_time.py
------------------------
Tracks how long each customer stays in the monitored area.
"""

import time
import numpy as np


class DwellTimeTracker:
    """Computes dwell (residence) time for each tracked individual."""

    def __init__(self, alert_threshold_sec: float = 120.0):
        """
        Args:
            alert_threshold_sec: Flag customers dwelling longer than this value.
        """
        self.alert_threshold = alert_threshold_sec
        self._entry: dict[int, float] = {}    # id → entry timestamp
        self._last: dict[int, float] = {}     # id → last-seen timestamp
        self._total: dict[int, float] = {}    # id → accumulated dwell seconds

    def update(self, track_ids: list[int]):
        """Call every frame with currently visible track IDs."""
        now = time.time()
        for tid in track_ids:
            if tid not in self._entry:
                self._entry[tid] = now
                self._total[tid] = 0.0
            self._last[tid] = now

        # Accumulate time for IDs still visible
        for tid in list(self._last.keys()):
            if tid in track_ids:
                self._total[tid] = now - self._entry[tid]

    def get_dwell(self, track_id: int) -> float:
        """Seconds the given ID has been present."""
        return self._total.get(track_id, 0.0)

    def get_all_dwells(self) -> dict[int, float]:
        return dict(self._total)

    def average_dwell(self) -> float:
        """Mean dwell time across all seen IDs."""
        if not self._total:
            return 0.0
        return float(np.mean(list(self._total.values())))

    def long_dwell_ids(self) -> list[int]:
        """IDs that have exceeded the alert threshold."""
        return [tid for tid, t in self._total.items() if t >= self.alert_threshold]

    def max_dwell(self) -> tuple[int, float]:
        """(track_id, seconds) of the customer with longest dwell."""
        if not self._total:
            return (-1, 0.0)
        tid = max(self._total, key=self._total.get)
        return tid, self._total[tid]

    def format_time(self, seconds: float) -> str:
        """Convert seconds to mm:ss string."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
