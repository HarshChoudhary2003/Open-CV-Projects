"""
analytics/footfall.py
----------------------
Tracks unique visitors (footfall) entering the monitored zone.
"""

import time
from collections import defaultdict


class FootfallCounter:
    """Counts unique individuals detected by the tracking system."""

    def __init__(self):
        # id → first_seen_timestamp
        self._seen: dict[int, float] = {}
        # Hourly counts: hour_str → count
        self._hourly: dict[str, int] = defaultdict(int)

    def update(self, track_ids: list[int]):
        """Register new track IDs as visitors."""
        now = time.time()
        for tid in track_ids:
            if tid not in self._seen:
                self._seen[tid] = now
                hour_key = time.strftime("%H:00", time.localtime(now))
                self._hourly[hour_key] += 1

    @property
    def total(self) -> int:
        """Total unique visitors since session start."""
        return len(self._seen)

    @property
    def hourly_data(self) -> dict[str, int]:
        """Dict of {hour_str: visitor_count}."""
        return dict(self._hourly)

    def peak_hour(self) -> tuple[str, int]:
        """Returns (hour, count) of the busiest hour."""
        if not self._hourly:
            return ("N/A", 0)
        peak = max(self._hourly, key=self._hourly.get)
        return peak, self._hourly[peak]

    def reset(self):
        self._seen.clear()
        self._hourly.clear()
