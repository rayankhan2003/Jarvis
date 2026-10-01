"""jarvis doctor's Mistral check, against a fake Mistral API."""

import aiohttp
import pytest
from aiohttp import web

from jarvis import doctor
from jarvis.config import Config


async def fake_mistral(responses):
    """responses: model -> list of statuses to return in turn (last one repeats)."""

    async def models(request):
        return web.json_response({"data": [{"id": m} for m in responses]})

    async def chat(request):
        model = (await request.json())["model"]
        queue = responses[model]
        status = queue.pop(0) if len(queue) > 1 else queue[0]
        if status != 200:
            return web.json_response({"message": "nope"}, status=status)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"choices":[{"delta":{"content":"ready"}}]}\n\n')
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(doctor.asyncio, "sleep", instant)


async def run_check(monkeypatch, responses, **config):
    runner, url = await fake_mistral(responses)
    monkeypatch.setattr(doctor, "MISTRAL_API", url)
    try:
        async with aiohttp.ClientSession() as session:
            return await doctor.check_mistral(session, Config(mistral_api_key="k", **config))
    finally:
        await runner.cleanup()


async def test_both_models_work(monkeypatch):
    check, ttft = await run_check(monkeypatch, {"ministral-8b-latest": [200], "ministral-14b-latest": [200]})
    assert check.status == doctor.OK and ttft is not None
    assert "ministral-8b-latest" in check.detail and "ministral-14b-latest" in check.detail


async def test_retries_a_rate_limit(monkeypatch):
    check, _ = await run_check(monkeypatch, {"ministral-8b-latest": [429, 200], "ministral-14b-latest": [200]})
    assert check.status == doctor.OK


async def test_listed_but_unusable_model_fails_with_a_fix(monkeypatch):
    check, _ = await run_check(
        monkeypatch,
        {"mistral-small-latest": [200], "mistral-large-latest": [403]},
        mistral_model="mistral-small-latest",
        mistral_complex_model="mistral-large-latest",
    )
    assert check.status == doctor.FAIL
    assert "mistral-large-latest doesn't work" in check.detail
    assert "MISTRAL_COMPLEX_MODEL" in check.fix and "ministral-14b-latest" in check.fix


async def test_persistent_rate_limit_is_a_warning(monkeypatch):
    check, _ = await run_check(monkeypatch, {"ministral-8b-latest": [429], "ministral-14b-latest": [200]})
    assert check.status == doctor.WARN and "rate limited" in check.detail
