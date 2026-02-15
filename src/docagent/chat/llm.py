from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    # Base64-encoded image(s), for vision-capable Ollama models.
    images: list[str] | None = None


class OllamaChatClient:
    def __init__(self, *, base_url: str, model: str, timeout_s: float = 120.0, options: dict[str, Any] | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.options = dict(options or {})

    def chat(self, messages: list[ChatMessage]) -> str:
        url = f"{self.base_url}/api/chat"
        msg_dicts: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.images:
                d["images"] = list(m.images)
            msg_dicts.append(d)
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": msg_dicts,
        }
        if self.options:
            payload["options"] = self.options

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(url, json=payload)
        except Exception as e:
            raise LLMError(
                f"Failed to connect to Ollama at {self.base_url}. Is it running? ({e})"
            ) from e

        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text}")

        data = r.json()
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise LLMError(f"Unexpected Ollama response: {data}")
        return content
