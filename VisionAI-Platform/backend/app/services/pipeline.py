"""
VisionAI Platform - Pipeline Orchestrator v2.0
UPGRADED: Integrates Heatmap, Zone Detection, Predictive Engine, AI Copilot, TTS.
Coordinates all AI services into a single per-camera pipeline.
Runs in a background thread and streams results via asyncio Queue.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.services.agent import VisionAgent, AgentAction
from app.services.anomaly_service import get_anomaly_detector
from app.services.detector import get_detector
from app.services.emotion_service import get_emotion_service
from app.services.face_service import get_face_service
from app.services.ocr_service import get_ocr_service
from app.services.pose_service import get_pose_service
from app.services.heatmap_service import get_heatmap_service
from app.services.zone_service import get_zone_service
from app.services.predictive_engine import get_predictive_engine
from app.services.copilot import get_copilot
from app.services.tts_service import get_tts_service


@dataclass
class FrameResult:
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    fps: float = 0.0
    detections: List[dict] = field(default_factory=list)
    faces: List[dict] = field(default_factory=list)
    emotions: List[dict] = field(default_factory=list)
    pose: Optional[dict] = None
    ocr_texts: List[dict] = field(default_factory=list)
    anomaly_score: float = 0.0
    anomaly_events: List[dict] = field(default_factory=list)
    agent_actions: List[dict] = field(default_factory=list)
    zone_intrusions: List[dict] = field(default_factory=list)
    predicted_events: List[dict] = field(default_factory=list)
    llm_insight: Optional[str] = None
    scene_narrative: Optional[str] = None
    frame_jpeg: Optional[bytes] = None
    heatmap_jpeg: Optional[bytes] = None   # separate heatmap overlay JPEG

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("frame_jpeg", "heatmap_jpeg")}
        return d


class CameraPipeline:
    """
    Per-camera processing pipeline v2.
    Reads frames from VideoCapture, runs all enabled AI modules,
    pushes FrameResult objects into an asyncio Queue.
    """

    def __init__(self, camera_id: str, source: Any,
                 config: Optional[Dict] = None,
                 result_queue: Optional[asyncio.Queue] = None,
                 alert_callback=None):
        self.camera_id = camera_id
        self.source = source
        self.config = config or {}
        self.result_queue = result_queue or asyncio.Queue(maxsize=10)
        self.alert_callback = alert_callback

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Module toggles
        self.enable_detection  = self.config.get("enable_detection",  True)
        self.enable_face       = self.config.get("enable_face",       True)
        self.enable_emotion    = self.config.get("enable_emotion",     True)
        self.enable_pose       = self.config.get("enable_pose",       True)
        self.enable_ocr        = self.config.get("enable_ocr",        False)
        self.enable_anomaly    = self.config.get("enable_anomaly",    True)
        self.enable_agent      = self.config.get("enable_agent",      True)
        self.enable_heatmap    = self.config.get("enable_heatmap",    True)
        self.enable_zones      = self.config.get("enable_zones",      True)
        self.enable_predictive = self.config.get("enable_predictive", True)
        self.enable_tts        = self.config.get("enable_tts",        True)
        self.enable_copilot    = self.config.get("enable_copilot",    True)
        self.show_heatmap      = self.config.get("show_heatmap",      False)

        self.agent = VisionAgent(camera_id, alert_callback)

        # Lazy service references
        self._detector     = None
        self._face_svc     = None
        self._emotion_svc  = None
        self._pose_svc     = None
        self._ocr_svc      = None
        self._anomaly_det  = None
        self._heatmap_svc  = None
        self._zone_svc     = None
        self._predictor    = None
        self._tts          = None
        self._copilot      = None

        self._frame_count: int = 0
        self._tts_cooldown: Dict[str, float] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"pipeline-{self.camera_id}"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Internal thread ───────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  settings.STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialise services in worker thread
        if self.enable_detection:  self._detector    = get_detector()
        if self.enable_face:       self._face_svc    = get_face_service()
        if self.enable_emotion:    self._emotion_svc = get_emotion_service()
        if self.enable_pose:       self._pose_svc    = get_pose_service()
        if self.enable_ocr:        self._ocr_svc     = get_ocr_service()
        if self.enable_anomaly:    self._anomaly_det = get_anomaly_detector()
        if self.enable_heatmap:    self._heatmap_svc = get_heatmap_service()
        if self.enable_zones:      self._zone_svc    = get_zone_service()
        if self.enable_predictive: self._predictor   = get_predictive_engine()
        if self.enable_tts:        self._tts         = get_tts_service()
        if self.enable_copilot:    self._copilot     = get_copilot()

        prev_time = time.time()

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            self._frame_count += 1
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            result = FrameResult(camera_id=self.camera_id, timestamp=now, fps=fps)
            composite = frame.copy()
            detections_raw = []

            # ── 1. Object Detection + Tracking ───────────────────────
            if self._detector:
                composite, dets = self._detector.detect(composite, track=True)
                result.detections = [d.to_dict() for d in dets]
                detections_raw = dets

            # ── 2. Face Recognition ──────────────────────────────────
            if self._face_svc:
                composite, faces = self._face_svc.process_frame(composite)
                result.faces = [f.to_dict() for f in faces]

            # ── 3. Emotion Detection ─────────────────────────────────
            if self._emotion_svc:
                composite, emotions = self._emotion_svc.process_frame(composite)
                result.emotions = [e.to_dict() for e in emotions]

            # ── 4. Pose Estimation ───────────────────────────────────
            if self._pose_svc:
                composite, pose = self._pose_svc.process_frame(composite)
                result.pose = pose.to_dict() if pose else None

            # ── 5. OCR (throttled every 10 frames) ──────────────────
            if self._ocr_svc and self._frame_count % 10 == 0:
                _, ocr_results = self._ocr_svc.extract(frame)
                result.ocr_texts = [o.to_dict() for o in ocr_results]

            # ── 6. Anomaly Detection ─────────────────────────────────
            if self._anomaly_det:
                composite, a_score, a_events = self._anomaly_det.analyse(
                    composite, detections_raw, self.camera_id
                )
                result.anomaly_score = a_score
                result.anomaly_events = [e.to_dict() for e in a_events]

            # ── 7. Zone Intrusion Detection ──────────────────────────
            if self._zone_svc:
                composite, intrusions = self._zone_svc.process_frame(
                    composite, result.detections, self.camera_id
                )
                result.zone_intrusions = intrusions

            # ── 8. Predictive Intelligence ───────────────────────────
            if self._predictor and self._frame_count % 3 == 0:
                composite, pred_events = self._predictor.update_and_predict(
                    result.detections, composite, draw_predictions=True
                )
                result.predicted_events = pred_events

            # ── 9. Heatmap ───────────────────────────────────────────
            if self._heatmap_svc:
                self._heatmap_svc.update(self.camera_id, result.detections,
                                         frame.shape)
                if self.show_heatmap:
                    composite = self._heatmap_svc.render_overlay(
                        self.camera_id, composite, alpha=0.45
                    )
                # Always encode a separate heatmap frame
                hmap_frame = self._heatmap_svc.render_overlay(
                    self.camera_id, frame.copy(), alpha=0.65
                )
                _, h_jpeg = cv2.imencode(
                    ".jpg", hmap_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                result.heatmap_jpeg = h_jpeg.tobytes()

            # ── 10. HUD Overlays ─────────────────────────────────────
            composite = self._draw_hud(composite, result, fps)

            # ── 11. Agent Decision Loop ──────────────────────────────
            if self.enable_agent:
                person_count = sum(
                    1 for d in result.detections if d.get("class_name") == "person"
                )
                context = {
                    "anomaly_score": result.anomaly_score,
                    "detections":    result.detections,
                    "faces":         result.faces,
                    "person_count":  person_count,
                    "dominant_emotion": (
                        result.emotions[0]["emotion"] if result.emotions else None
                    ),
                    "zone_intrusions":  result.zone_intrusions,
                    "predicted_events": result.predicted_events,
                    "ocr_texts":        result.ocr_texts,
                }

                # Feed copilot memory
                if self._copilot:
                    self._copilot.ingest(context)
                    # Periodic scene narrative (every 30 frames)
                    if self._frame_count % 30 == 0:
                        result.scene_narrative = self._copilot.get_scene_narrative()

                agent_actions = asyncio.run_coroutine_threadsafe(
                    self.agent.process(context, composite), self._loop
                ).result(timeout=2)
                result.agent_actions = [a.to_dict() for a in agent_actions]

                # TTS alerts for critical actions
                if self._tts and agent_actions:
                    for action in agent_actions:
                        if action.priority >= 3:
                            rule = action.payload.get("rule", "alert")
                            severity = action.payload.get("severity", "HIGH")
                            msg = f"Alert! {severity} severity. {rule.replace('_', ' ')} detected."
                            self._tts.speak(msg, priority=(severity == "CRITICAL"))

            # ── 12. Encode JPEG ──────────────────────────────────────
            _, jpeg = cv2.imencode(
                ".jpg", composite,
                [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            result.frame_jpeg = jpeg.tobytes()

            # Push result to queue (drop frame if queue full)
            try:
                asyncio.run_coroutine_threadsafe(
                    self.result_queue.put(result), self._loop
                ).result(timeout=0.05)
            except Exception:
                pass

        cap.release()

    # ── HUD overlay ───────────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, result: FrameResult, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # Glassmorphism top-left info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 110), (5, 5, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Cyan accent border
        cv2.rectangle(frame, (0, 0), (300, 110), (0, 200, 160), 1)

        lines = [
            (f"VisionAI  |  CAM {self.camera_id}", (0, 255, 200)),
            (f"FPS: {fps:.1f}   {time.strftime('%H:%M:%S')}", (180, 180, 180)),
            (f"Objects: {len(result.detections)}  Faces: {len(result.faces)}", (180, 180, 180)),
            (f"Anomaly: {result.anomaly_score:.0%}",
             (0, int(255 * (1 - result.anomaly_score)), int(255 * result.anomaly_score))),
            (f"Zones: {len(result.zone_intrusions)} intrusion(s)",
             (255, 120, 0) if result.zone_intrusions else (100, 100, 100)),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

        # Predictive events banner
        if result.predicted_events:
            pred_text = f"⚡ PREDICTION: {result.predicted_events[0].get('reasons', [''])[0].upper()}"
            cv2.putText(frame, pred_text, (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 80, 255), 1, cv2.LINE_AA)

        # Zone intrusion banner
        if result.zone_intrusions:
            zn = result.zone_intrusions[0].get("zone_name", "ZONE")
            cv2.putText(frame, f"🚨 INTRUSION: {zn}",
                        (w // 2 - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # Animated scan line
        scan_y = int((time.time() * 80) % h)
        scan_overlay = frame.copy()
        cv2.line(scan_overlay, (0, scan_y), (w, scan_y), (0, 255, 120), 1)
        cv2.addWeighted(scan_overlay, 0.3, frame, 0.7, 0, frame)

        return frame


# ── Pipeline Registry ──────────────────────────────────────────────────────────

_pipelines: Dict[str, CameraPipeline] = {}


def get_pipeline(camera_id: str) -> Optional[CameraPipeline]:
    return _pipelines.get(camera_id)


def create_pipeline(camera_id: str, source: Any, config: Optional[Dict] = None,
                    alert_callback=None) -> CameraPipeline:
    if camera_id in _pipelines:
        _pipelines[camera_id].stop()
    p = CameraPipeline(camera_id, source, config, alert_callback=alert_callback)
    _pipelines[camera_id] = p
    return p


def list_pipelines() -> List[str]:
    return list(_pipelines.keys())


def stop_all():
    for p in _pipelines.values():
        p.stop()
    _pipelines.clear()
