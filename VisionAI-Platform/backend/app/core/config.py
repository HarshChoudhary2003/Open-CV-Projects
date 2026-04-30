"""
VisionAI Platform - Core Configuration
Centralized, environment-driven configuration management.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    # ─── Application ──────────────────────────────────────────────
    APP_NAME: str = "VisionAI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ─── API / Security ───────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production-32-chars-min"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ─── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./visionai.db"

    # ─── Redis (optional caching / pub-sub) ───────────────────────
    REDIS_URL: Optional[str] = None

    # ─── Camera / Stream ──────────────────────────────────────────
    DEFAULT_CAMERA_INDEX: int = 0
    MAX_CAMERAS: int = 8
    STREAM_FPS: int = 30
    STREAM_WIDTH: int = 1280
    STREAM_HEIGHT: int = 720
    JPEG_QUALITY: int = 85

    # ─── AI Models ────────────────────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"          # nano for speed; swap to yolov8x for accuracy
    YOLO_CONFIDENCE: float = 0.45
    YOLO_IOU: float = 0.45
    FACE_CASCADE: str = "haarcascade_frontalface_default.xml"
    EMOTION_MODEL_PATH: str = "weights/emotion_model.pt"
    POSE_MODEL: str = "mediapipe"            # mediapipe | openpose
    OCR_ENGINE: str = "easyocr"             # easyocr | tesseract

    # ─── Tracking ─────────────────────────────────────────────────
    TRACKER_TYPE: str = "bytetrack"         # bytetrack | deepsort
    MAX_DISAPPEARED: int = 50
    MAX_DISTANCE: float = 0.3

    # ─── Anomaly / Alerts ─────────────────────────────────────────
    ANOMALY_SENSITIVITY: float = 0.75
    ALERT_COOLDOWN_SECONDS: int = 10
    SNAPSHOT_DIR: str = "snapshots"
    LOG_DIR: str = "logs"
    REPORT_DIR: str = "reports"

    # ─── GPU ──────────────────────────────────────────────────────
    USE_GPU: bool = True
    DEVICE: str = "auto"                    # auto | cpu | cuda | mps

    # ─── Email Alerts (optional) ──────────────────────────────────
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASS: Optional[str] = None
    ALERT_EMAIL_TO: Optional[str] = None

    # ─── LLM (optional AI reasoning) ─────────────────────────────
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: str = "none"              # none | openai | ollama
    LLM_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator("DEVICE", pre=True, always=True)
    def resolve_device(cls, v):
        if v == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
