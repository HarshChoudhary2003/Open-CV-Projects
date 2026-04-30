"""VisionAI - Alerts API"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select
from app.core.security import get_current_user
from app.core.database import get_db, AlertEvent

router = APIRouter()


@router.get("/alerts")
async def list_alerts(
    camera_id: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(50, ge=1, le=500),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    q = select(AlertEvent).order_by(AlertEvent.timestamp.desc()).limit(limit)
    if camera_id:
        q = q.where(AlertEvent.camera_id == camera_id)
    if severity:
        q = q.where(AlertEvent.severity == severity.upper())
    if acknowledged is not None:
        q = q.where(AlertEvent.acknowledged == acknowledged)
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": r.id, "camera_id": r.camera_id,
            "timestamp": r.timestamp.isoformat(),
            "alert_type": r.alert_type, "severity": r.severity,
            "description": r.description, "snapshot_path": r.snapshot_path,
            "acknowledged": r.acknowledged,
        }
        for r in rows
    ]


@router.patch("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    row = (await db.execute(
        select(AlertEvent).where(AlertEvent.id == alert_id)
    )).scalars().first()
    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")
    row.acknowledged = True
    await db.commit()
    return {"acknowledged": True, "id": alert_id}
