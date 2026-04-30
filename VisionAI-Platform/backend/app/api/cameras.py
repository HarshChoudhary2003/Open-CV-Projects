"""VisionAI - Camera Management API"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.core.security import get_current_user
from app.services.pipeline import create_pipeline, get_pipeline, list_pipelines, stop_all

router = APIRouter()


class CameraCreate(BaseModel):
    camera_id: str
    source: Any           # int or RTSP URL string
    config: Optional[Dict] = None


class CameraInfo(BaseModel):
    camera_id: str
    running: bool


_alert_log: List[dict] = []


async def _on_alert(action):
    """Global alert callback – broadcasts to WebSocket clients."""
    _alert_log.append(action.to_dict())
    if len(_alert_log) > 500:
        _alert_log.pop(0)


@router.post("/cameras/start", response_model=CameraInfo)
async def start_camera(
    req: CameraCreate,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
):
    loop = asyncio.get_event_loop()
    pipeline = create_pipeline(
        req.camera_id, req.source, req.config, alert_callback=_on_alert
    )
    pipeline.start(loop)
    return CameraInfo(camera_id=req.camera_id, running=True)


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str, current_user=Depends(get_current_user)):
    p = get_pipeline(camera_id)
    if not p:
        raise HTTPException(status_code=404, detail="Camera not found")
    p.stop()
    return {"status": "stopped", "camera_id": camera_id}


@router.get("/cameras", response_model=List[str])
async def list_cameras(current_user=Depends(get_current_user)):
    return list_pipelines()


@router.get("/cameras/{camera_id}/status")
async def camera_status(camera_id: str, current_user=Depends(get_current_user)):
    p = get_pipeline(camera_id)
    if not p:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {
        "camera_id": camera_id,
        "running": p._running,
        "source": str(p.source),
        "modules": {
            "detection": p.enable_detection,
            "face": p.enable_face,
            "emotion": p.enable_emotion,
            "pose": p.enable_pose,
            "ocr": p.enable_ocr,
            "anomaly": p.enable_anomaly,
            "agent": p.enable_agent,
        },
    }


@router.get("/alerts/recent")
async def recent_alerts(limit: int = 50, current_user=Depends(get_current_user)):
    return _alert_log[-limit:]
