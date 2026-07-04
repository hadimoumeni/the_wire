"""Model layer — pluggable, all free.

Provider selection (in `get_client`):
  * WIRE_LLM=heuristic            → None (deterministic keyword baseline)
  * LLM_API_KEY set               → OpenAI-compatible hosted endpoint
                                     (default: Groq free tier, Llama-3.3-70B)
  * else, local Ollama reachable  → OllamaClient (Qwen2.5-7B)
  * else                          → None (heuristic)

Both clients expose the same `chat_json(system, user, schema)` returning parsed
JSON, so the agents don't care which one they got.
"""
from __future__ import annotations

import json
import os
import requests

DEFAULT_MODEL = os.environ.get("WIRE_MODEL", "qwen2.5:7b-instruct")
DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Hosted (OpenAI-compatible) defaults — Groq's free tier.
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_HOSTED_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")


class LLMError(RuntimeError):
    pass


OllamaError = LLMError  # back-compat alias (agents catch OllamaError)


def _schema_hint(schema: dict) -> str:
    return ("\n\nRespond with a single JSON object that matches this schema "
            "(JSON only, no prose):\n" + json.dumps(schema))


class OllamaClient:
    provider = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL,
                 temperature: float = 0.0, timeout: int = 180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            tags = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return False
        base = self.model.split(":")[0]
        return any(t == self.model or t.startswith(base) for t in tags)

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model, "stream": False, "format": schema,
            "options": {"temperature": self.temperature},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise LLMError(str(e)) from e
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON: {content[:200]}") from e


class OpenAICompatClient:
    """Any OpenAI-compatible chat endpoint (Groq, OpenRouter, HF router, …)."""
    provider = "openai"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_HOSTED_MODEL, temperature: float = 0.0,
                 timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + _schema_hint(schema)},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise LLMError(str(e)) from e
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON: {content[:200]}") from e


def get_client():
    """Return a ready client, or None for heuristic mode."""
    if os.environ.get("WIRE_LLM", "").lower() == "heuristic":
        return None

    key = os.environ.get("LLM_API_KEY")
    if key or os.environ.get("LLM_PROVIDER", "").lower() == "openai":
        return OpenAICompatClient(api_key=key) if key else None

    client = OllamaClient()
    return client if client.available() else None
