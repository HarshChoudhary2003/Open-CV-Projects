"""
VisionAI Platform - Async Database Layer (SQLite via aiosqlite)
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


# ─── ORM Models ───────────────────────────────────────────────────────────────

class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    object_class = Column(String(128))
    confidence = Column(Float)
    track_id = Column(Integer, nullable=True)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    extra = Column(JSON, nullable=True)


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(128))
    severity = Column(String(32))          # LOW | MEDIUM | HIGH | CRITICAL
    description = Column(Text)
    snapshot_path = Column(String(512), nullable=True)
    acknowledged = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)


class FaceRecord(Base):
    __tablename__ = "face_records"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(128), unique=True, index=True)
    name = Column(String(256))
    embedding = Column(Text)               # JSON-serialised numpy array
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=True)
    access_level = Column(String(32), default="visitor")


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(64), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    people_count = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    dominant_emotion = Column(String(64), nullable=True)
    anomaly_score = Column(Float, default=0.0)
    fps = Column(Float, default=0.0)
    stats = Column(JSON, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user = Column(String(128), nullable=True)
    action = Column(String(256))
    detail = Column(Text, nullable=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
