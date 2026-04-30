"""VisionAI - Analytics API"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from app.core.security import get_current_user
from app.core.database import get_db, DetectionEvent, AnalyticsSnapshot

router = APIRouter()


@router.get("/analytics/summary")
async def analytics_summary(
    camera_id: Optional[str] = None,
    hours: int = Query(24, ge=1, le=720),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    since = datetime.utcnow() - timedelta(hours=hours)
    q = select(DetectionEvent).where(DetectionEvent.timestamp >= since)
    if camera_id:
        q = q.where(DetectionEvent.camera_id == camera_id)
    rows = (await db.execute(q)).scalars().all()

    counts: dict = {}
    for r in rows:
        counts[r.object_class] = counts.get(r.object_class, 0) + 1

    return {
        "total_detections": len(rows),
        "by_class": counts,
        "hours": hours,
        "camera_id": camera_id,
    }


@router.get("/analytics/timeline")
async def detection_timeline(
    camera_id: Optional[str] = None,
    hours: int = Query(6, ge=1, le=168),
    bucket_minutes: int = Query(15, ge=1, le=60),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Returns time-bucketed detection counts for charting."""
    since = datetime.utcnow() - timedelta(hours=hours)
    q = select(DetectionEvent).where(DetectionEvent.timestamp >= since)
    if camera_id:
        q = q.where(DetectionEvent.camera_id == camera_id)
    rows = (await db.execute(q)).scalars().all()

    buckets: dict = {}
    bucket_td = timedelta(minutes=bucket_minutes)
    for r in rows:
        bucket_key = (r.timestamp - since) // bucket_td
        ts = (since + bucket_key * bucket_td).isoformat()
        buckets[ts] = buckets.get(ts, 0) + 1

    timeline = [{"timestamp": k, "count": v} for k, v in sorted(buckets.items())]
    return {"timeline": timeline, "bucket_minutes": bucket_minutes}


@router.get("/analytics/snapshots")
async def get_snapshots(
    camera_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = select(AnalyticsSnapshot).order_by(AnalyticsSnapshot.timestamp.desc()).limit(limit)
    if camera_id:
        q = q.where(AnalyticsSnapshot.camera_id == camera_id)
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": r.id, "camera_id": r.camera_id, "timestamp": r.timestamp.isoformat(),
            "people_count": r.people_count, "anomaly_score": r.anomaly_score,
            "fps": r.fps, "dominant_emotion": r.dominant_emotion,
        }
        for r in rows
    ]
