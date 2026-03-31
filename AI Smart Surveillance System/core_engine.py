"""
core_engine.py
Central surveillance pipeline — ties together detection, recognition, and alerting.
Can be run standalone (headless) or driven by the Streamlit dashboard.
"""

from __future__ import annotations
import time
import threading
from typing import Optional, List, Dict

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import TARGET_FPS, CAMERA_INDEX
from database.db_manager import init_db, log_event
from detection.yolo_detector import PersonDetector
from utils.alert_engine import AlertEngine
from utils.frame_utils import draw_hud, resize_for_display
from utils.video_stream import VideoStream


class SurveillanceEngine:
    """
    Thread-safe engine that can be started/stopped independently.
    Exposes shared state (last_frame, tracks, fps, alerts) that the
    Streamlit dashboard polls every refresh cycle.
    """

    def __init__(self, camera_src: int | str = CAMERA_INDEX,
                 use_face_recog: bool = True) -> None:
        self.camera_src     = camera_src
        self.use_face_recog = use_face_recog

        # ── Shared state (read by dashboard) ──────────────────────────────────
        self._lock          = threading.Lock()
        self.last_frame:    Optional[np.ndarray] = None
        self.fps:           float = 0.0
        self.person_count:  int   = 0
        self.active_alerts: List[Dict] = []
        self.tracks:        List       = []
        self.running:       bool       = False
        self.error_msg:     Optional[str] = None

        # ── Components (lazy init in start()) ─────────────────────────────────
        self._stream:    Optional[VideoStream]    = None
        self._detector:  Optional[PersonDetector] = None
        self._alerter:   Optional[AlertEngine]    = None
        self._face_eng                            = None
        self._thread:    Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.running:
            return
        init_db()
        log_event("SYSTEM", "Surveillance engine started", severity="INFO")
        try:
            self._stream   = VideoStream(self.camera_src).start()
            self._detector = PersonDetector()
            self._alerter  = AlertEngine()
            if self.use_face_recog:
                from recognition.face_engine import FaceEngine
                self._face_eng = FaceEngine()
        except Exception as exc:
            self.error_msg = str(exc)
            return

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._stream:
            self._stream.stop()
        log_event("SYSTEM", "Surveillance engine stopped", severity="INFO")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        prev_t  = time.time()
        frame_i = 0

        while self.running:
            ok, frame = self._stream.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            frame_i += 1

            # YOLO detection + tracking
            annotated, tracks = self._detector.process_frame(frame)

            # Face recognition (every 3rd frame for performance)
            if self._face_eng and frame_i % 3 == 0:
                active_ids = [t.track_id for t in tracks]
                self._face_eng.flush_cache(active_ids)
                for t in tracks:
                    name     = self._face_eng.identify(frame, t.bbox, t.track_id)
                    t.label  = name

            # Alert evaluation
            alerts = self._alerter.evaluate(tracks, frame)

            # HUD overlay
            now   = time.time()
            fps   = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            alert_labels = [a["label"] for a in alerts]
            annotated    = draw_hud(annotated, fps, len(tracks), alert_labels)

            # Publish state
            with self._lock:
                self.last_frame   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                self.fps          = fps
                self.person_count = len(tracks)
                self.tracks       = tracks
                self.active_alerts = alerts

            # Throttle to TARGET_FPS
            elapsed = time.time() - now
            sleep   = max(0, (1.0 / TARGET_FPS) - elapsed)
            time.sleep(sleep)

    # ── State Accessors ───────────────────────────────────────────────────────
    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.last_frame

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "fps":          round(self.fps, 1),
                "people":       self.person_count,
                "alerts":       len(self.active_alerts),
                "running":      self.running,
            }


# ── CLI headless mode ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = SurveillanceEngine(use_face_recog=False)
    engine.start()
    if engine.error_msg:
        print("Error:", engine.error_msg)
        exit(1)

    print("Press ESC to quit...")
    while True:
        frame = engine.get_frame()
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("AI Surveillance", bgr)
        key = cv2.waitKey(1)
        if key == 27:
            break

    engine.stop()
    cv2.destroyAllWindows()
