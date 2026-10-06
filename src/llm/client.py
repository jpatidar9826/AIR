"""Pluggable LLM client.

Configure it from the MAAS guidelines in the AI Lab VM (Projects -> MAAS) via
.env. With LLM_PROVIDER=offline every feature still works using deterministic
fallbacks, so the demo never breaks if LLM access is unavailable.

Every call is logged (purpose, model, sizes - no text) to
artifacts/reports/llm_usage_log.jsonl as evidence of compliant usage.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import config
from src.text_utils import mask_pii


class LLMClient:
    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self.last_error: str | None = None
        self._client = None

    @property
    def available(self) -> bool:
        return self.provider in ("openai_compatible", "azure") and bool(config.LLM_API_KEY and self.model)

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import AzureOpenAI, OpenAI  # pip install openai
        if self.provider == "azure":
            self._client = AzureOpenAI(api_key=config.LLM_API_KEY, azure_endpoint=config.LLM_BASE_URL,
                                       api_version=config.LLM_API_VERSION, timeout=config.LLM_TIMEOUT)
        else:
            self._client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL or None,
                                  timeout=config.LLM_TIMEOUT)
        return self._client

    def _log(self, purpose: str, ok: bool, prompt_chars: int, out_chars: int, seconds: float):
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "purpose": purpose, "provider": self.provider,
                 "model": self.model, "ok": ok, "prompt_chars": prompt_chars, "output_chars": out_chars,
                 "seconds": round(seconds, 2), "pii_masked": config.MASK_PII_BEFORE_LLM}
        with open(config.LLM_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def chat(self, system: str, user: str, purpose: str = "general",
             temperature: float | None = None, max_tokens: int = 900) -> str | None:
        """Return the model's reply, or None if the LLM is unavailable/fails."""
        if not self.available:
            self.last_error = "LLM not configured (offline mode)"
            return None
        if config.MASK_PII_BEFORE_LLM:
            user = mask_pii(user)
        start = time.time()
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            self._log(purpose, True, len(system) + len(user), len(text), time.time() - start)
            self.last_error = None
            return text.strip()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._log(purpose, False, len(system) + len(user), 0, time.time() - start)
            return None


def parse_json(text: str | None):
    """Extract the first JSON object/array from an LLM reply."""
    if not text:
        return None
    cleaned = text.replace("```json", "").replace("```", "").strip()
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start, end = cleaned.find(open_c), cleaned.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
