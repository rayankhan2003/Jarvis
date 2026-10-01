"""Failover between free providers.

Pipecat's built-in failover only switches when a provider is permanently
broken (bad key, unknown model). On free tiers the common failure is a
temporary one: 429 rate limits and exhausted daily quotas. This strategy
switches on those too, benches the provider for a cooldown, and goes back to
the preferred provider once the cooldown is over.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyFailover
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.errors import (
    ErrorCategory,
    classify_http_exception,
    classify_http_status_code,
)

# How long to bench a provider after each kind of failure, in seconds.
COOLDOWNS = {
    ErrorCategory.RATE_LIMIT: 60,
    ErrorCategory.QUOTA: 3600,
    ErrorCategory.CONNECTIVITY: 20,
    ErrorCategory.SERVER: 30,
}


def categorize(error: ErrorFrame) -> ErrorCategory:
    """Work out why a provider failed, including SDKs Pipecat can't classify."""
    if error.category and error.category != ErrorCategory.UNKNOWN:
        return error.category
    exc = error.exception
    if exc is not None:
        category = classify_http_exception(exc)
        if category != ErrorCategory.UNKNOWN:
            return category
        code = getattr(exc, "code", None)  # google-genai errors carry the HTTP status as `.code`
        if isinstance(code, int) and not isinstance(code, bool):
            category = classify_http_status_code(code)
            if category != ErrorCategory.UNKNOWN:
                return category
    text = f"{error.error} {exc or ''}".lower()
    if "quota" in text or "resource_exhausted" in text or "per day" in text:
        return ErrorCategory.QUOTA
    if "429" in text or "rate limit" in text or "rate_limit" in text or "too many requests" in text:
        return ErrorCategory.RATE_LIMIT
    if "timeout" in text or "timed out" in text or "connect" in text:
        return ErrorCategory.CONNECTIVITY
    return ErrorCategory.UNKNOWN


class FreeTierFailover(ServiceSwitcherStrategyFailover):
    """Failover that also handles rate limits and quotas, with cooldowns.

    Event handlers available (in addition to ``on_service_switched``):

    - on_failover: Called with (strategy, failed_service, new_service, category)
      when an error caused a switch, so the app can retry the request.
    """

    def __init__(self, services: list[FrameProcessor], clock: Callable[[], float] = time.monotonic):
        super().__init__(services)
        self._clock = clock
        self._benched_until: dict[int, float] = {}
        self._register_event_handler("on_failover")

    def benched(self, service: FrameProcessor) -> bool:
        return self._benched_until.get(id(service), 0.0) > self._clock()

    def available(self, service: FrameProcessor) -> bool:
        return service.is_usable and not self.benched(service)

    async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
        failed = error.processor or self._active_service
        if failed is not self._active_service:
            return None  # a stale error from a service we already left

        category = categorize(error)
        if failed.is_usable and category not in COOLDOWNS:
            return None  # e.g. a malformed reply; the provider itself is fine

        if failed.is_usable:
            self._benched_until[id(failed)] = self._clock() + COOLDOWNS[category]
        logger.warning(f"{failed.name} failed ({category.value}): {error.error}")

        start = self._services.index(failed)
        for offset in range(1, len(self._services)):
            candidate = self._services[(start + offset) % len(self._services)]
            if self.available(candidate):
                switched = await self._set_active_if_available(candidate)
                if switched:
                    logger.info(f"Switched to {candidate.name}")
                    await self._call_event_handler("on_failover", failed, candidate, category)
                return switched

        logger.error("Every provider is unavailable right now")
        return None

    async def restore_preferred(self) -> FrameProcessor | None:
        """Switch back to the first provider in the list that is available again."""
        for service in self._services:
            if service is self._active_service:
                return None  # already on the most preferred available provider
            if self.available(service):
                logger.info(f"{service.name} is available again; switching back")
                return await self._set_active_if_available(service)
        return None


def keep_only_messages_for(context: LLMContext, llm_id: str) -> int:
    """Drop provider-specific messages another provider left in the conversation.

    Gemini stores "thought signatures" in the context. They mean nothing to
    Groq or Ollama, which skip them and log an error on every request after
    a failover. Returns how many messages were dropped.
    """
    messages = context.get_messages()  # the context's own list, not a copy
    kept = [m for m in messages if not isinstance(m, LLMSpecificMessage) or m.llm == llm_id]
    dropped = len(messages) - len(kept)
    if dropped:
        context.set_messages(kept)
    return dropped
