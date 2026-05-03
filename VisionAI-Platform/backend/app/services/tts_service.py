"""
VisionAI Platform - Text-to-Speech & Audio Alert Service
Provides async TTS alerts using pyttsx3 (offline) or gTTS (online).
Plays audio in a background thread to avoid blocking the pipeline.
"""

import asyncio
import io
import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger("visionai.tts")


class TTSService:
    """
    Audio alert service with dual backend:
      1. pyttsx3 – offline, zero-latency.
      2. gTTS    – higher quality, requires internet.
    Falls back gracefully if neither is available.
    """

    _instance: Optional["TTSService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return

        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._backend = "none"
        self._last_spoken: float = 0
        self._cooldown: float = 8.0   # don't repeat same alert within 8 s

        # Try pyttsx3 first (offline)
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 170)
            self._engine.setProperty("volume", 0.85)
            self._backend = "pyttsx3"
            logger.info("[TTS] Backend: pyttsx3 (offline)")
        except Exception:
            pass

        # Try playsound + gTTS as alternative
        if self._backend == "none":
            try:
                from gtts import gTTS   # noqa: F401
                import playsound       # noqa: F401
                self._backend = "gtts"
                logger.info("[TTS] Backend: gTTS (online)")
            except Exception:
                logger.warning("[TTS] No TTS backend available. Audio alerts disabled.")

        # Start worker thread
        self._worker = threading.Thread(target=self._worker_loop, daemon=True,
                                        name="tts-worker")
        self._worker.start()
        self._initialised = True

    # ── Public API ───────────────────────────────────────────────────

    def speak(self, text: str, priority: bool = False) -> None:
        """Enqueue a TTS message (drops if queue full and not priority)."""
        if self._backend == "none":
            return
        now = time.time()
        if now - self._last_spoken < self._cooldown and not priority:
            return
        try:
            self._queue.put_nowait(text)
            self._last_spoken = now
        except queue.Full:
            if priority:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(text)
                except Exception:
                    pass

    async def async_speak(self, text: str, priority: bool = False) -> None:
        """Async wrapper for speak."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.speak(text, priority))

    # ── Internal worker ──────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            try:
                text = self._queue.get(timeout=1.0)
                self._say(text)
            except queue.Empty:
                continue

    def _say(self, text: str):
        try:
            if self._backend == "pyttsx3":
                self._engine.say(text)
                self._engine.runAndWait()

            elif self._backend == "gtts":
                from gtts import gTTS
                import tempfile
                import os
                try:
                    import playsound
                    buf = io.BytesIO()
                    gTTS(text=text, lang="en").write_to_fp(buf)
                    buf.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        f.write(buf.read())
                        tmp_path = f.name
                    playsound.playsound(tmp_path, block=True)
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.error(f"[TTS] gTTS play error: {e}")
        except Exception as e:
            logger.error(f"[TTS] Say error: {e}")


def get_tts_service() -> TTSService:
    svc = TTSService()
    svc.initialise()
    return svc
