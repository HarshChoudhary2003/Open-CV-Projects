"""
VisionAI Platform - AI Copilot API
REST + WebSocket endpoints for the AI Copilot Q&A mode.
"""

import asyncio
import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.security import verify_token
from app.services.copilot import get_copilot

router = APIRouter()


class CopilotQuestion(BaseModel):
    question: str
    camera_id: str = "cam0"


class CopilotAnswer(BaseModel):
    question: str
    answer: str
    narrative: str


@router.post(
    "/copilot/ask",
    response_model=CopilotAnswer,
    summary="Ask the AI Copilot about the current scene",
)
async def ask_copilot(body: CopilotQuestion, _=Depends(verify_token)):
    copilot = get_copilot()
    answer = await copilot.ask(body.question)
    narrative = copilot.get_scene_narrative()
    return CopilotAnswer(
        question=body.question,
        answer=answer,
        narrative=narrative,
    )


@router.get(
    "/copilot/narrative",
    summary="Get the current scene narrative",
)
async def get_narrative(_=Depends(verify_token)):
    copilot = get_copilot()
    return {"narrative": copilot.get_scene_narrative()}


@router.get(
    "/copilot/history",
    summary="Get Q&A conversation history",
)
async def get_history(_=Depends(verify_token)):
    copilot = get_copilot()
    return {"history": copilot.get_qa_history()}


@router.websocket("/ws/copilot")
async def copilot_ws(websocket: WebSocket):
    """
    Interactive WebSocket for real-time AI Copilot chat.
    Send: {"question": "..."}
    Receive: {"answer": "...", "narrative": "..."}
    """
    await websocket.accept()
    copilot = get_copilot()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                question = payload.get("question", "")
                if not question:
                    continue
                answer = await copilot.ask(question)
                narrative = copilot.get_scene_narrative()
                await websocket.send_text(json.dumps({
                    "question": question,
                    "answer": answer,
                    "narrative": narrative,
                    "history": copilot.get_qa_history()[-5:],
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass
