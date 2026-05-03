"""
VisionAI Platform - Zone Management API
CRUD for restricted/safe zone polygons per camera.
"""

import uuid
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.security import verify_token
from app.services.zone_service import get_zone_service

router = APIRouter()


class ZoneCreate(BaseModel):
    name: str
    polygon: List[Tuple[int, int]]
    zone_type: str = "restricted"   # restricted | safe | monitor
    camera_id: str = "cam0"
    color: List[int] = [0, 0, 255]  # BGR


class ZoneOut(BaseModel):
    zone_id: str
    name: str
    polygon: List[List[int]]
    zone_type: str
    camera_id: str
    color: List[int]


@router.get("/zones", response_model=List[ZoneOut], summary="List all configured zones")
async def list_zones(camera_id: Optional[str] = None, _=Depends(verify_token)):
    svc = get_zone_service()
    zones = svc.list_zones(camera_id)
    return [z.to_dict() for z in zones]


@router.post("/zones", response_model=ZoneOut, status_code=status.HTTP_201_CREATED,
             summary="Create a new detection zone")
async def create_zone(body: ZoneCreate, _=Depends(verify_token)):
    svc = get_zone_service()
    zone_id = str(uuid.uuid4())[:8]
    zone = svc.add_zone(
        zone_id=zone_id,
        name=body.name,
        polygon=body.polygon,
        zone_type=body.zone_type,
        camera_id=body.camera_id,
        color=tuple(body.color[:3]),
    )
    return zone.to_dict()


@router.delete("/zones/{zone_id}", summary="Delete a zone")
async def delete_zone(zone_id: str, _=Depends(verify_token)):
    svc = get_zone_service()
    if not svc.remove_zone(zone_id):
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"deleted": zone_id}
