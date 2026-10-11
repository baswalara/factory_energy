"""LLM access behind one switch (LLM_PROVIDER, LLM_MODEL) plus MOCK_LLM=true.

The LLM has exactly two jobs: label the question, and word the brief. It never
calculates and never calls tools; code decides which tool runs.
"""
from __future__ import annotations

import os

import httpx

DEFAULT_MODEL = {"ollama": "llama3.1:8b", "openai": "gpt-6-luna",
                 "anthropic": "claude-haiku-4-5-20251001", "gemini": "gemini-3.8-flash"}
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


class LLM:
    def __init__(self, settings: dict):
        self.mock = settings["mock"]
        self.provider = settings["provider"]
        self.model = settings["model"] or DEFAULT_MODEL.get(self.provider, "")
        self.timeout = float(os.getenv("LLM_TIMEOUT_S", "40"))

    @property
    def name(self) -> str:
        return "mock" if self.mock else f"{self.provider}:{self.model}"

    def status(self) -> str:
        if self.mock:
            return "mock (MOCK_LLM=true): rule-based parsing and template brief"
        if self.provider not in DEFAULT_MODEL:
            return f"error: unknown LLM_PROVIDER '{self.provider}'"
        if self.provider == "ollama":
            try:
                httpx.get(self._ollama() + "/api/tags", timeout=3).raise_for_status()
                return f"ok ({self.name})"
            except Exception as e:
                return f"unreachable ({self.name}): {type(e).__name__}"
        return f"configured ({self.name})" if os.getenv(KEY_ENV[self.provider]) else f"missing {KEY_ENV[self.provider]}"

    def _ollama(self) -> str:
        return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def complete(self, system: str, user: str, want_json: bool = False) -> tuple[str, int]:
        """Returns (text, tokens used). Raises on any failure; callers fall back to rules or the template."""
        if self.mock:
            raise RuntimeError("mock mode")
        p, t = self.provider, self.timeout
        if p == "ollama":
            body = {"model": self.model, "stream": False, "options": {"temperature": 0},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if want_json:
                body["format"] = "json"
            r = httpx.post(self._ollama() + "/api/chat", json=body, timeout=t)
            r.raise_for_status()
            d = r.json()
            return d["message"]["content"], int(d.get("prompt_eval_count", 0)) + int(d.get("eval_count", 0))
        if p == "openai":
            body = {"model": self.model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if want_json:
                body["response_format"] = {"type": "json_object"}
            r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, timeout=t,
                           headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
            r.raise_for_status()
            d = r.json()
            return d["choices"][0]["message"]["content"], int(d.get("usage", {}).get("total_tokens", 0))
        if p == "anthropic":
            body = {"model": self.model, "max_tokens": 700, "system": system,
                    "messages": [{"role": "user", "content": user}]}
            r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=t,
                           headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
            r.raise_for_status()
            d = r.json()
            u = d.get("usage", {})
            return d["content"][0]["text"], int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
        if p == "gemini":
            body = {"system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}]}
            if want_json:
                body["generationConfig"] = {"responseMimeType": "application/json"}
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           json=body, timeout=t, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            r.raise_for_status()
            d = r.json()
            return (d["candidates"][0]["content"]["parts"][0]["text"],
                    int(d.get("usageMetadata", {}).get("totalTokenCount", 0)))
        raise RuntimeError(f"unknown LLM_PROVIDER '{p}'")
