"""
VisionAI Platform - AI Copilot Mode
"What is happening right now?" — answers using vision context + LLM reasoning.

Provides:
  • Scene narrative generation from structured frame data
  • Conversation memory (last N frames)
  • Copilot question-answering (sync + async)
  • Fallback rule-based narrative when no LLM configured
"""

import json
import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("visionai.copilot")


class FrameMemory:
    """Rolling buffer of summarised frame contexts for LLM prompts."""

    def __init__(self, maxlen: int = 10):
        self._buffer: Deque[dict] = deque(maxlen=maxlen)

    def push(self, ctx: dict):
        self._buffer.append({"t": time.strftime("%H:%M:%S"), **ctx})

    def to_prompt_text(self) -> str:
        lines = []
        for i, c in enumerate(self._buffer):
            parts = [f"[{c.get('t', '?')}]"]
            if c.get("person_count"):
                parts.append(f"{c['person_count']} persons")
            if c.get("detections"):
                classes = ", ".join({d["class_name"] for d in c["detections"][:5]})
                parts.append(f"objects: {classes}")
            if c.get("faces"):
                names = ", ".join({f["name"] for f in c["faces"][:3]})
                parts.append(f"faces: {names}")
            if c.get("dominant_emotion"):
                parts.append(f"emotion: {c['dominant_emotion']}")
            if c.get("anomaly_score", 0) > 0.4:
                parts.append(f"anomaly={c['anomaly_score']:.0%}")
            if c.get("zone_intrusions"):
                parts.append(f"⚠ zone intrusions: {len(c['zone_intrusions'])}")
            if c.get("predicted_events"):
                parts.append(f"⚡ predicted events: {len(c['predicted_events'])}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @property
    def latest(self) -> Optional[dict]:
        return self._buffer[-1] if self._buffer else None


class AICopilot:
    """
    Multimodal AI Copilot that fuses vision context with language reasoning.

    Usage:
        copilot = get_copilot()
        copilot.ingest(frame_context)
        narrative = await copilot.ask("What is happening right now?")
    """

    _instance: Optional["AICopilot"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self, llm_provider: str = "none", llm_model: str = "gpt-4o-mini",
                   openai_api_key: Optional[str] = None, ollama_url: str = "http://localhost:11434"):
        if self._initialised:
            return
        self.memory = FrameMemory(maxlen=12)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self.ollama_url = ollama_url
        self._qa_history: List[Dict] = []
        self._initialised = True

    def ingest(self, context: dict):
        """Feed latest frame context into the rolling memory."""
        self.memory.push(context)

    async def ask(self, question: str) -> str:
        """
        Answer a natural-language question about what is happening.
        Falls back to a rule-based narrative if no LLM is configured.
        """
        history_text = self.memory.to_prompt_text()
        latest = self.memory.latest or {}

        system_prompt = (
            "You are VisionAI Copilot, an intelligent security and scene analysis assistant. "
            "You receive real-time camera data and answer questions concisely and accurately. "
            "Always be factual. If uncertain, say so. Keep answers under 4 sentences."
        )

        user_prompt = (
            f"Recent camera observations (newest last):\n{history_text}\n\n"
            f"User question: {question}\n\n"
            "Answer based on the observations above."
        )

        answer = None

        if self.llm_provider == "openai" and self.openai_api_key:
            answer = await self._call_openai(system_prompt, user_prompt)
        elif self.llm_provider == "ollama":
            answer = await self._call_ollama(system_prompt, user_prompt)

        if not answer:
            answer = self._rule_based_answer(question, latest)

        record = {
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
        }
        self._qa_history.append(record)
        return answer

    def get_scene_narrative(self) -> str:
        """Generate a natural language description of the current scene."""
        latest = self.memory.latest
        if not latest:
            return "No scene data available yet."
        return self._rule_based_narrative(latest)

    # ── LLM backends ─────────────────────────────────────────────────

    async def _call_openai(self, system: str, user: str) -> Optional[str]:
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)
            resp = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=200,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[Copilot] OpenAI error: {e}")
            return None

    async def _call_ollama(self, system: str, user: str) -> Optional[str]:
        try:
            import httpx
            prompt = f"[SYSTEM]\n{system}\n[USER]\n{user}"
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.llm_model, "prompt": prompt, "stream": False},
                )
                return r.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"[Copilot] Ollama error: {e}")
            return None

    # ── Rule-based fallback ───────────────────────────────────────────

    def _rule_based_answer(self, question: str, ctx: dict) -> str:
        narrative = self._rule_based_narrative(ctx)
        q = question.lower()
        if "threat" in q or "danger" in q or "suspicious" in q:
            score = ctx.get("anomaly_score", 0)
            level = "HIGH" if score > 0.75 else "MEDIUM" if score > 0.4 else "LOW"
            return (
                f"Current threat level is {level} (anomaly score {score:.0%}). "
                f"{narrative}"
            )
        if "who" in q or "person" in q or "people" in q:
            faces = ctx.get("faces", [])
            names = [f["name"] for f in faces if f["name"] != "Unknown"]
            unknowns = sum(1 for f in faces if f["name"] == "Unknown")
            parts = []
            if names:
                parts.append(f"Known persons: {', '.join(names)}")
            if unknowns:
                parts.append(f"{unknowns} unknown individual(s)")
            return ". ".join(parts) if parts else "No faces currently identified."
        return narrative

    def _rule_based_narrative(self, ctx: dict) -> str:
        parts = []
        n = ctx.get("person_count", 0)
        if n == 0:
            parts.append("The scene is currently empty.")
        elif n == 1:
            parts.append("One person is present in the scene.")
        else:
            parts.append(f"{n} persons are present in the scene.")

        faces = ctx.get("faces", [])
        known = [f["name"] for f in faces if f["name"] != "Unknown"]
        if known:
            parts.append(f"Identified: {', '.join(known[:3])}.")

        emotion = ctx.get("dominant_emotion")
        if emotion:
            parts.append(f"Dominant emotion: {emotion}.")

        score = ctx.get("anomaly_score", 0)
        if score > 0.75:
            parts.append("⚠️ HIGH anomaly activity detected!")
        elif score > 0.4:
            parts.append("Moderate movement anomaly observed.")

        intrusions = ctx.get("zone_intrusions", [])
        if intrusions:
            names = list({e["zone_name"] for e in intrusions})
            parts.append(f"🚨 Zone intrusion in: {', '.join(names)}.")

        predicted = ctx.get("predicted_events", [])
        if predicted:
            reasons = list({r for e in predicted for r in e.get("reasons", [])})
            parts.append(f"⚡ Predicted behaviour: {', '.join(reasons[:2])}.")

        ocr = ctx.get("ocr_texts", [])
        if ocr:
            texts = [o["text"] for o in ocr[:2] if o.get("text")]
            if texts:
                parts.append(f"OCR text seen: \"{', '.join(texts)}\".")

        return " ".join(parts) or "Scene analysis in progress."

    def get_qa_history(self) -> List[Dict]:
        return self._qa_history[-20:]


def get_copilot() -> AICopilot:
    from app.core.config import settings
    copilot = AICopilot()
    if not copilot._initialised:
        copilot.initialise(
            llm_provider=settings.LLM_PROVIDER,
            llm_model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    return copilot
