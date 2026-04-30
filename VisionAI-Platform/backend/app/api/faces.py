"""VisionAI - Face Registry API"""

import io
import numpy as np
import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from app.core.security import get_current_user
from app.services.face_service import get_face_service

router = APIRouter()


@router.post("/faces/register")
async def register_face(
    name: str = Form(...),
    access_level: str = Form("visitor"),
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    svc = get_face_service()
    pid = svc.register_face(frame, name, access_level)
    if pid is None:
        raise HTTPException(status_code=422, detail="No face detected in image")

    return {"person_id": pid, "name": name, "access_level": access_level}


@router.get("/faces")
async def list_faces(current_user=Depends(get_current_user)):
    svc = get_face_service()
    return [
        {
            "person_id": rec.person_id,
            "name": rec.name,
            "access_level": rec.access_level,
        }
        for rec in svc.db.values()
    ]


@router.delete("/faces/{person_id}")
async def delete_face(person_id: str, current_user=Depends(get_current_user)):
    svc = get_face_service()
    if person_id not in svc.db:
        raise HTTPException(status_code=404, detail="Person not found")
    del svc.db[person_id]
    svc._save_db()
    return {"deleted": person_id}
