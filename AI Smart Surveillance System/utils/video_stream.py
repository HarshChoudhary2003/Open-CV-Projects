"""
utils/video_stream.py
Thread-buffered OpenCV video capture — prevents frame lag.
"""

from __future__ import annotations
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class VideoStream:
    """
    Opens a camera (or video file) in a background thread so the main
    processing loop is never blocked waiting for the next frame.
    """

    def __init__(self, src: int | str = CAMERA_INDEX) -> None:
        self.src  = src
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray]     = None
        self._ok    = False
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> "VideoStream":
        self._cap = cv2.VideoCapture(self.src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.src!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimize internal buffer lag
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from source.")
        self._frame = frame
        self._ok    = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    # ── Frame Access ──────────────────────────────────────────────────────────
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    # ── Background Thread ─────────────────────────────────────────────────────
    def _update(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._ok = False
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._ok    = True
