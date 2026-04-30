"""
VisionAI Platform - AI Agent (Observe → Analyze → Decide → Act)
Rule-based + optional LLM reasoning layer.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings


class AgentAction:
    def __init__(self, action_type: str, payload: dict, priority: int = 1):
        self.action_type = action_type   # ALERT | LOG | SNAPSHOT | EMAIL | LLM_QUERY
        self.payload = payload
        self.priority = priority
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "payload": self.payload,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


class VisionAgent:
    """
    Autonomous agent loop:
    1. Observe  – receives per-frame analysis results
    2. Analyse  – applies rule engine + optional LLM
    3. Decide   – produces AgentActions
    4. Act      – dispatches actions (alert, log, snapshot, email)
    """

    def __init__(self, camera_id: str = "cam0",
                 action_callback: Optional[Callable] = None):
        self.camera_id = camera_id
        self.action_callback = action_callback
        self._rules = self._build_rules()
        self._last_action_time: Dict[str, float] = {}
        self.snapshot_dir = Path(settings.SNAPSHOT_DIR)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.log_dir = Path(settings.LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)

    # ── Rule engine ───────────────────────────────────────────────

    def _build_rules(self) -> List[dict]:
        """
        Each rule: condition (lambda) -> list of action types.
        Conditions receive the 'context' dict.
        """
        return [
            {
                "name": "suspicious_movement",
                "condition": lambda ctx: ctx.get("anomaly_score", 0) > 0.75,
                "actions": ["ALERT", "SNAPSHOT"],
                "severity": "HIGH",
                "cooldown": 15,
            },
            {
                "name": "unknown_face",
                "condition": lambda ctx: any(
                    f.get("name") == "Unknown" for f in ctx.get("faces", [])
                ),
                "actions": ["ALERT", "LOG"],
                "severity": "MEDIUM",
                "cooldown": 30,
            },
            {
                "name": "crowd_density",
                "condition": lambda ctx: ctx.get("person_count", 0) > 10,
                "actions": ["ALERT", "LOG"],
                "severity": "MEDIUM",
                "cooldown": 20,
            },
            {
                "name": "negative_emotion",
                "condition": lambda ctx: ctx.get("dominant_emotion") in ("angry", "fear"),
                "actions": ["LOG"],
                "severity": "LOW",
                "cooldown": 10,
            },
            {
                "name": "weapon_detected",
                "condition": lambda ctx: any(
                    d.get("class_name") in ("knife", "gun", "scissors")
                    for d in ctx.get("detections", [])
                ),
                "actions": ["ALERT", "SNAPSHOT", "LOG"],
                "severity": "CRITICAL",
                "cooldown": 5,
            },
        ]

    # ── Observe → Decide → Act ────────────────────────────────────

    async def process(self, context: dict, frame: Optional[np.ndarray] = None
                      ) -> List[AgentAction]:
        """
        Main agent tick. context keys:
          anomaly_score, detections, faces, emotions, person_count,
          dominant_emotion, ocr_texts, gesture, pose_label.
        """
        actions: List[AgentAction] = []
        now = time.time()

        for rule in self._rules:
            name = rule["name"]
            try:
                triggered = rule["condition"](context)
            except Exception:
                triggered = False

            if not triggered:
                continue

            cooldown = rule.get("cooldown", 10)
            if now - self._last_action_time.get(name, 0) < cooldown:
                continue

            self._last_action_time[name] = now

            for action_type in rule["actions"]:
                action = AgentAction(
                    action_type=action_type,
                    payload={
                        "rule": name,
                        "severity": rule["severity"],
                        "camera_id": self.camera_id,
                        "context_summary": self._summarise(context),
                    },
                    priority={"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(
                        rule["severity"], 1
                    ),
                )
                actions.append(action)

                await self._dispatch(action, frame)

        return actions

    async def _dispatch(self, action: AgentAction, frame: Optional[np.ndarray]):
        if action.action_type == "SNAPSHOT" and frame is not None:
            path = self._save_snapshot(frame, action.payload["rule"])
            action.payload["snapshot_path"] = str(path)

        if action.action_type == "LOG":
            self._write_log(action)

        if action.action_type == "ALERT" and self.action_callback:
            await self._run_callback(action)

    async def _run_callback(self, action: AgentAction):
        if asyncio.iscoroutinefunction(self.action_callback):
            await self.action_callback(action)
        else:
            self.action_callback(action)

    def _save_snapshot(self, frame: np.ndarray, tag: str) -> Path:
        fname = f"{self.camera_id}_{tag}_{int(time.time())}.jpg"
        path = self.snapshot_dir / fname
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path

    def _write_log(self, action: AgentAction):
        log_file = self.log_dir / f"agent_{self.camera_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(action.to_dict()) + "\n")

    def _summarise(self, ctx: dict) -> str:
        parts = []
        if ctx.get("person_count"):
            parts.append(f"{ctx['person_count']} persons")
        if ctx.get("dominant_emotion"):
            parts.append(f"emotion={ctx['dominant_emotion']}")
        if ctx.get("anomaly_score", 0) > 0.5:
            parts.append(f"anomaly={ctx['anomaly_score']:.0%}")
        return ", ".join(parts) if parts else "normal"

    # ── Optional LLM reasoning ────────────────────────────────────

    async def llm_reason(self, context: dict) -> Optional[str]:
        """
        Optionally call an LLM to generate natural-language insight.
        Requires OPENAI_API_KEY or Ollama running locally.
        """
        if settings.LLM_PROVIDER == "none":
            return None

        prompt = (
            f"You are a security AI. Analyse this scene:\n{json.dumps(context, indent=2)}\n"
            "Provide a concise threat assessment in 2 sentences."
        )

        if settings.LLM_PROVIDER == "openai":
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                resp = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"LLM error: {e}"

        if settings.LLM_PROVIDER == "ollama":
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": settings.LLM_MODEL, "prompt": prompt, "stream": False},
                        timeout=15,
                    )
                    return r.json().get("response", "")
            except Exception as e:
                return f"Ollama error: {e}"

        return None
