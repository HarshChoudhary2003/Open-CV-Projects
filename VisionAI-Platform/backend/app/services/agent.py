"""
VisionAI Platform v2.0 - AI Agent (Observe → Analyse → Decide → Act)
UPGRADED: Zone intrusion rules, predictive alerts, TTS integration.
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
        self.action_type = action_type   # ALERT | LOG | SNAPSHOT | EMAIL | LLM_QUERY | TTS
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
    4. Act      – dispatches actions (alert, log, snapshot, email, TTS)
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

    # ── Rule engine ───────────────────────────────────────────────────

    def _build_rules(self) -> List[dict]:
        """
        Each rule: condition (lambda ctx) → list of action types.
        Context keys: anomaly_score, detections, faces, emotions,
                      person_count, dominant_emotion, zone_intrusions,
                      predicted_events, ocr_texts.
        """
        return [
            # Tier 4 – CRITICAL ──────────────────────────────────────
            {
                "name": "weapon_detected",
                "condition": lambda ctx: any(
                    d.get("class_name") in ("knife", "gun", "scissors")
                    for d in ctx.get("detections", [])
                ),
                "actions": ["ALERT", "SNAPSHOT", "LOG", "TTS"],
                "severity": "CRITICAL",
                "cooldown": 5,
                "tts_msg": "Critical alert. Weapon detected on camera.",
            },
            {
                "name": "zone_intrusion",
                "condition": lambda ctx: bool(ctx.get("zone_intrusions")),
                "actions": ["ALERT", "SNAPSHOT", "LOG", "TTS"],
                "severity": "CRITICAL",
                "cooldown": 10,
                "tts_msg": "Alert! Intrusion detected in restricted zone.",
            },
            # Tier 3 – HIGH ──────────────────────────────────────────
            {
                "name": "suspicious_movement",
                "condition": lambda ctx: ctx.get("anomaly_score", 0) > 0.75,
                "actions": ["ALERT", "SNAPSHOT", "LOG"],
                "severity": "HIGH",
                "cooldown": 15,
            },
            {
                "name": "predicted_suspicious",
                "condition": lambda ctx: any(
                    e.get("prediction_score", 0) > 0.70
                    for e in ctx.get("predicted_events", [])
                ),
                "actions": ["ALERT", "LOG"],
                "severity": "HIGH",
                "cooldown": 12,
                "tts_msg": "Warning! Suspicious behaviour predicted by AI.",
            },
            # Tier 2 – MEDIUM ────────────────────────────────────────
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
            # Tier 1 – LOW ───────────────────────────────────────────
            {
                "name": "negative_emotion",
                "condition": lambda ctx: ctx.get("dominant_emotion") in ("angry", "fear"),
                "actions": ["LOG"],
                "severity": "LOW",
                "cooldown": 10,
            },
        ]

    # ── Observe → Decide → Act ────────────────────────────────────────

    async def process(self, context: dict, frame: Optional[np.ndarray] = None
                      ) -> List[AgentAction]:
        """
        Main agent tick. Returns list of actions taken this frame.
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
                        "tts_msg": rule.get("tts_msg", ""),
                    },
                    priority={
                        "CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1
                    }.get(rule["severity"], 1),
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

        if action.action_type == "TTS":
            msg = action.payload.get("tts_msg", "")
            if msg:
                try:
                    from app.services.tts_service import get_tts_service
                    get_tts_service().speak(msg, priority=(action.priority >= 4))
                except Exception:
                    pass

    async def _run_callback(self, action: AgentAction):
        if asyncio.iscoroutinefunction(self.action_callback):
            await self.action_callback(action)
        else:
            self.action_callback(action)

    def _save_snapshot(self, frame: np.ndarray, tag: str) -> Path:
        fname = f"{self.camera_id}_{tag}_{int(time.time())}.jpg"
        path = self.snapshot_dir / fname
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
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
        if ctx.get("zone_intrusions"):
            parts.append(f"zone_intrusions={len(ctx['zone_intrusions'])}")
        if ctx.get("predicted_events"):
            parts.append(f"predicted={len(ctx['predicted_events'])}")
        return ", ".join(parts) if parts else "normal"

    # ── Optional LLM reasoning ────────────────────────────────────────

    async def llm_reason(self, context: dict) -> Optional[str]:
        """
        Call LLM for a threat assessment narrative.
        Requires OPENAI_API_KEY or local Ollama.
        """
        if settings.LLM_PROVIDER == "none":
            return None

        prompt = (
            f"You are VisionAI Security Agent. Analyse this scene:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Provide a concise 2-sentence threat assessment."
        )

        if settings.LLM_PROVIDER == "openai":
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                resp = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3,
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
