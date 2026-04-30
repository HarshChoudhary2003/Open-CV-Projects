"""VisionAI - WebSocket Streaming API
Streams MJPEG frames and JSON telemetry to browser clients.
"""

import asyncio
import json
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.services.pipeline import get_pipeline

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        # camera_id -> set of websockets
        self.stream_clients: Dict[str, Set[WebSocket]] = {}
        self.telemetry_clients: Dict[str, Set[WebSocket]] = {}

    async def connect_stream(self, camera_id: str, ws: WebSocket):
        await ws.accept()
        self.stream_clients.setdefault(camera_id, set()).add(ws)

    async def connect_telemetry(self, camera_id: str, ws: WebSocket):
        await ws.accept()
        self.telemetry_clients.setdefault(camera_id, set()).add(ws)

    def disconnect(self, camera_id: str, ws: WebSocket):
        self.stream_clients.get(camera_id, set()).discard(ws)
        self.telemetry_clients.get(camera_id, set()).discard(ws)

    async def broadcast_frame(self, camera_id: str, jpeg_bytes: bytes):
        dead = set()
        for ws in list(self.stream_clients.get(camera_id, set())):
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.stream_clients.get(camera_id, set()).discard(ws)

    async def broadcast_telemetry(self, camera_id: str, data: dict):
        dead = set()
        for ws in list(self.telemetry_clients.get(camera_id, set())):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.telemetry_clients.get(camera_id, set()).discard(ws)


manager = ConnectionManager()


@router.websocket("/ws/stream/{camera_id}")
async def video_stream(websocket: WebSocket, camera_id: str):
    """Binary JPEG stream – clients render as img.src."""
    await manager.connect_stream(camera_id, websocket)
    pipeline = get_pipeline(camera_id)
    try:
        while True:
            if pipeline is None:
                pipeline = get_pipeline(camera_id)
                await asyncio.sleep(0.1)
                continue
            try:
                result = await asyncio.wait_for(
                    pipeline.result_queue.get(), timeout=1.0
                )
                if result.frame_jpeg:
                    await manager.broadcast_frame(camera_id, result.frame_jpeg)
                    # Also broadcast telemetry to JSON clients
                    await manager.broadcast_telemetry(camera_id, result.to_dict())
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        manager.disconnect(camera_id, websocket)


@router.websocket("/ws/telemetry/{camera_id}")
async def telemetry_stream(websocket: WebSocket, camera_id: str):
    """JSON telemetry stream (no video bytes)."""
    await manager.connect_telemetry(camera_id, websocket)
    pipeline = get_pipeline(camera_id)
    try:
        while True:
            if pipeline is None:
                pipeline = get_pipeline(camera_id)
                await asyncio.sleep(0.1)
                continue
            try:
                result = await asyncio.wait_for(
                    pipeline.result_queue.get(), timeout=1.0
                )
                await manager.broadcast_telemetry(camera_id, result.to_dict())
            except asyncio.TimeoutError:
                await websocket.send_text('{"ping":true}')
    except WebSocketDisconnect:
        manager.disconnect(camera_id, websocket)
