import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import ErrorFrame
from pipecat.utils.errors import ErrorCategory

from jarvis.failover import FreeTierFailover, categorize


class FakeService:
    def __init__(self, name):
        self.name = name
        self.is_usable = True
        self.queued = []

    async def queue_frame(self, frame):
        self.queued.append(frame)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def setup():
    clock = Clock()
    services = [FakeService("gemini"), FakeService("groq"), FakeService("ollama")]
    strategy = FreeTierFailover(services, clock=clock)
    events = []

    @strategy.event_handler("on_failover")
    async def on_failover(strategy, failed, new, category):
        events.append((failed.name, new.name, category))

    return strategy, services, clock, events


def error(service, message, exception=None):
    return ErrorFrame(error=message, processor=service, exception=exception)


async def test_rate_limit_fails_over_and_comes_back(setup):
    strategy, (gemini, groq, _), clock, events = setup

    switched = await strategy.handle_error(error(gemini, "429 Too Many Requests"))
    assert switched is groq
    assert strategy.active_service is groq
    await asyncio.sleep(0.01)  # event handlers run as tasks
    assert events == [("gemini", "groq", ErrorCategory.RATE_LIMIT)]

    # Still benched: stays on groq.
    clock.now += 30
    assert await strategy.restore_preferred() is None
    assert strategy.active_service is groq

    # Cooldown over: back to the preferred provider.
    clock.now += 31
    assert await strategy.restore_preferred() is gemini
    assert strategy.active_service is gemini


async def test_quota_benches_for_an_hour(setup):
    strategy, (gemini, groq, _), clock, _ = setup
    await strategy.handle_error(error(gemini, "RESOURCE_EXHAUSTED: quota exceeded for requests per day"))
    clock.now += 600
    assert await strategy.restore_preferred() is None
    clock.now += 3001
    assert await strategy.restore_preferred() is gemini


async def test_skips_benched_providers(setup):
    strategy, (gemini, groq, ollama), _, _ = setup
    await strategy.handle_error(error(gemini, "429"))
    switched = await strategy.handle_error(error(groq, "429"))
    assert switched is ollama


async def test_ignores_errors_that_are_not_the_providers_fault(setup):
    strategy, (gemini, _, _), _, events = setup
    assert await strategy.handle_error(error(gemini, "could not parse tool call arguments")) is None
    assert strategy.active_service is gemini
    await asyncio.sleep(0.01)
    assert events == []


async def test_ignores_stale_errors_from_inactive_providers(setup):
    strategy, (gemini, groq, _), _, _ = setup
    assert await strategy.handle_error(error(groq, "429")) is None
    assert strategy.active_service is gemini


async def test_gives_up_when_everything_is_down(setup):
    strategy, services, _, _ = setup
    for s in services[1:]:
        s.is_usable = False
    assert await strategy.handle_error(error(services[0], "429")) is None


def test_categorize_reads_google_style_status_code():
    exc = Exception("boom")
    exc.code = 429
    assert categorize(ErrorFrame(error="boom", exception=exc)) == ErrorCategory.RATE_LIMIT


def test_categorize_reads_httpx_style_response():
    exc = Exception("boom")
    exc.response = SimpleNamespace(status_code=503)
    assert categorize(ErrorFrame(error="boom", exception=exc)) == ErrorCategory.SERVER
