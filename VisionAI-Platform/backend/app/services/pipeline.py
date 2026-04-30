"""
VisionAI Platform - Pipeline Orchestrator
Coordinates all AI services into a single per-camera pipeline.
Runs in a background thread and streams results via asyncio Queue.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

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
    llm_insight: Optional[str] = None
    frame_jpeg: Optional[bytes] = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "frame_jpeg"}
        return d


class CameraPipeline:
    """
    Per-camera processing pipeline.
    Reads frames from a VideoCapture, runs enabled modules,
    and pushes FrameResult objects into an asyncio Queue.
    """

    def __init__(self, camera_id: str, source: Any,
                 config: Optional[Dict] = None,
                 result_queue: Optional[asyncio.Queue] = None,
                 alert_callback=None):
        self.camera_id = camera_id
        self.source = source         # int (device index) or str (RTSP/file)
        self.config = config or {}
        self.result_queue = result_queue or asyncio.Queue(maxsize=10)
        self.alert_callback = alert_callback

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Module toggles (from config or defaults)
        self.enable_detection = self.config.get("enable_detection", True)
        self.enable_face = self.config.get("enable_face", True)
        self.enable_emotion = self.config.get("enable_emotion", True)
        self.enable_pose = self.config.get("enable_pose", True)
        self.enable_ocr = self.config.get("enable_ocr", False)
        self.enable_anomaly = self.config.get("enable_anomaly", True)
        self.enable_agent = self.config.get("enable_agent", True)

        self.agent = VisionAgent(camera_id, alert_callback)

        # Lazily initialise singletons
        self._detector = None
        self._face_svc = None
        self._emotion_svc = None
        self._pose_svc = None
        self._ocr_svc = None
        self._anomaly_det = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"pipeline-{self.camera_id}")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Internal thread ───────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Init services in this thread
        if self.enable_detection:
            self._detector = get_detector()
        if self.enable_face:
            self._face_svc = get_face_service()
        if self.enable_emotion:
            self._emotion_svc = get_emotion_service()
        if self.enable_pose:
            self._pose_svc = get_pose_service()
        if self.enable_ocr:
            self._ocr_svc = get_ocr_service()
        if self.enable_anomaly:
            self._anomaly_det = get_anomaly_detector()

        prev_time = time.time()

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video file
                    continue
                break

            # ── FPS tracking ──────────────────────────────────────
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            result = FrameResult(camera_id=self.camera_id, timestamp=now, fps=fps)
            composite = frame.copy()

            # 1. Object detection + tracking
            detections_raw = []
            if self._detector:
                composite, dets = self._detector.detect(composite, track=True)
                result.detections = [d.to_dict() for d in dets]
                detections_raw = dets

            # 2. Face recognition
            if self._face_svc:
                composite, faces = self._face_svc.process_frame(composite)
                result.faces = [f.to_dict() for f in faces]

            # 3. Emotion
            if self._emotion_svc:
                composite, emotions = self._emotion_svc.process_frame(composite)
                result.emotions = [e.to_dict() for e in emotions]

            # 4. Pose & gesture
            if self._pose_svc:
                composite, pose = self._pose_svc.process_frame(composite)
                result.pose = pose.to_dict() if pose else None

            # 5. OCR (expensive – throttle to every 10 frames)
            if self._ocr_svc and int(now * 3) % 10 == 0:
                _, ocr_results = self._ocr_svc.extract(frame)
                result.ocr_texts = [o.to_dict() for o in ocr_results]

            # 6. Anomaly detection
            if self._anomaly_det:
                composite, a_score, a_events = self._anomaly_det.analyse(
                    composite, detections_raw, self.camera_id
                )
                result.anomaly_score = a_score
                result.anomaly_events = [e.to_dict() for e in a_events]

            # 7. Draw HUD overlays
            composite = self._draw_hud(composite, result, fps)

            # 8. Agent decision
            if self.enable_agent:
                context = {
                    "anomaly_score": result.anomaly_score,
                    "detections": result.detections,
                    "faces": result.faces,
                    "person_count": sum(
                        1 for d in result.detections if d.get("class_name") == "person"
                    ),
                    "dominant_emotion": (
                        result.emotions[0]["emotion"] if result.emotions else None
                    ),
                }
                agent_actions = asyncio.run_coroutine_threadsafe(
                    self.agent.process(context, composite), self._loop
                ).result(timeout=2)
                result.agent_actions = [a.to_dict() for a in agent_actions]

            # 9. Encode JPEG
            _, jpeg = cv2.imencode(
                ".jpg", composite,
                [cv2.IMWRITE_JPEG_QUALITY, settings.JPEG_QUALITY]
            )
            result.frame_jpeg = jpeg.tobytes()

            # Push to queue (drop frame if queue full)
            try:
                asyncio.run_coroutine_threadsafe(
                    self.result_queue.put(result), self._loop
                ).result(timeout=0.05)
            except Exception:
                pass

        cap.release()

    # ── HUD overlay ───────────────────────────────────────────────

    def _draw_hud(self, frame: np.ndarray, result: FrameResult, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        # Top-left info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, 90), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        lines = [
            (f"VisionAI | CAM {self.camera_id}", (0, 255, 180)),
            (f"FPS: {fps:.1f}  |  {time.strftime('%H:%M:%S')}", (200, 200, 200)),
            (f"Objects: {len(result.detections)}  Faces: {len(result.faces)}", (200, 200, 200)),
            (f"Anomaly: {result.anomaly_score:.0%}", (0, int(255 * (1 - result.anomaly_score)),
                                                       int(255 * result.anomaly_score))),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Scanning line animation
        scan_y = int((time.time() * 100) % h)
        cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 100, 30), 1)
        return frame


# ── Pipeline Registry ─────────────────────────────────────────────────────────

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
