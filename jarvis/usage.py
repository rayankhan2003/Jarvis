"""Count free-tier usage per day, so `jarvis usage` shows how much is left.

Everything is counted on the Mac from the pipeline's own metrics; nothing is
sent anywhere.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import Frame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from jarvis.config import HOME_DIR

USAGE_FILE = HOME_DIR / "usage.json"

# Approximate free-tier allowances (October 2026). Providers change these often.
FREE_LIMITS = {
    "mistral": {"tokens": 1_000_000_000 // 30, "note": "~1B tokens/month, ~1 request/second"},
    "groq": {"requests": 1000, "tokens": 200_000, "note": "per model, per day"},
    "gemini": {"requests": 500, "note": "Flash-Lite, per day"},
    "ollama": {"note": "runs on your Mac, unlimited"},
    "speech-to-text": {"requests": 2000, "note": "Groq Whisper, per day"},
}

PROVIDERS = ("mistral", "groq", "gemini", "ollama")


def provider_of(processor_name: str) -> str:
    name = processor_name.lower()
    if "google" in name or "gemini" in name:
        return "gemini"
    for provider in PROVIDERS:
        if provider in name:
            return provider
    return "other"


class Usage:
    def __init__(self, path: Path = USAGE_FILE, today: date | None = None):
        self._path = path
        self._today = today
        try:
            self._data: dict = json.loads(path.read_text())
        except (OSError, ValueError):
            self._data = {}

    def _day(self) -> dict:
        key = (self._today or date.today()).isoformat()
        return self._data.setdefault(key, {})

    def _bump(self, bucket: str, field: str, amount: int = 1):
        entry = self._day().setdefault(bucket, {})
        entry[field] = entry.get(field, 0) + amount
        self._save()

    def record_llm(self, provider: str, tokens: int):
        self._bump(provider, "requests")
        self._bump(provider, "tokens", tokens)

    def record_stt(self):
        self._bump("speech-to-text", "requests")

    def record_instant(self):
        self._bump("instant commands", "count")

    def record_ignored(self):
        self._bump("ignored noise", "count")

    def day(self, when: date) -> dict:
        return self._data.get(when.isoformat(), {})

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Keep two weeks of history.
            cutoff = ((self._today or date.today()) - timedelta(days=14)).isoformat()
            self._data = {k: v for k, v in self._data.items() if k >= cutoff}
            self._path.write_text(json.dumps(self._data, indent=1, sort_keys=True))
        except OSError as e:
            logger.warning(f"Couldn't save usage: {e}")


class UsageMeter(FrameProcessor):
    """Records the token usage each LLM reports. Put it at the end of the pipeline."""

    def __init__(self, usage: Usage):
        super().__init__()
        self._usage = usage

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, LLMUsageMetricsData):
                    self._usage.record_llm(provider_of(data.processor), data.value.total_tokens or 0)
        await self.push_frame(frame, direction)


def report(usage: Usage, today: date | None = None) -> str:
    today = today or date.today()
    day = usage.day(today)
    lines = [f"Jarvis usage for {today:%A %d %B}", ""]
    for name, limit in FREE_LIMITS.items():
        used = day.get(name, {})
        parts = []
        if "requests" in used or "requests" in limit:
            req = used.get("requests", 0)
            parts.append(f"{req} requests" + (f" of ~{limit['requests']:,}" if "requests" in limit else ""))
        if used.get("tokens") or "tokens" in limit:
            tok = used.get("tokens", 0)
            parts.append(f"{tok:,} tokens" + (f" of ~{limit['tokens']:,}" if "tokens" in limit else ""))
        lines.append(f"  {name:<15} {', '.join(parts) or '-':<45} ({limit['note']})")
    saved = day.get("instant commands", {}).get("count", 0)
    ignored = day.get("ignored noise", {}).get("count", 0)
    lines += ["", f"  {saved} instant commands and {ignored} bits of noise used no AI request."]
    week = [usage.day(today - timedelta(days=d)) for d in range(7)]
    week_requests = sum(v.get("requests", 0) for d in week for k, v in d.items() if k in PROVIDERS)
    lines.append(f"  Last 7 days: {week_requests} AI requests in total.")
    return "\n".join(lines)
