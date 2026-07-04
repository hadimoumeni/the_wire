"""Ollama client — the free, local model layer.

Talks to a local Ollama server (default http://localhost:11434). Uses Ollama's
structured-output support: pass a JSON schema as `format` and the model is
constrained to emit conforming JSON. No API key, no cost.
"""
from __future__ import annotations

import json
import os
import requests

DEFAULT_MODEL = os.environ.get("WIRE_MODEL", "qwen2.5:7b-instruct")
DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                 temperature: float = 0.0, timeout: int = 180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def available(self) -> bool:
        """True if the Ollama server is up and our model is pulled."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            tags = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return False
        base = self.model.split(":")[0]
        return any(t == self.model or t.startswith(base) for t in tags)

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """Single structured-output turn. Returns the parsed JSON object."""
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise OllamaError(str(e)) from e
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OllamaError(f"model returned non-JSON: {content[:200]}") from e


def get_client() -> OllamaClient | None:
    """Return a ready Ollama client, or None if unavailable (→ heuristic mode)."""
    if os.environ.get("WIRE_LLM", "").lower() == "heuristic":
        return None
    client = OllamaClient()
    return client if client.available() else None
