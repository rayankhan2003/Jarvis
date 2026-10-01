"""Model routing, conversation trimming and usage counting."""

from datetime import date

import pytest
from pipecat.frames.frames import LLMUpdateSettingsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.mistral.llm import MistralLLMService

from jarvis import conversation
from jarvis.router import ModelRouter, is_complex
from jarvis.usage import Usage, provider_of, report


@pytest.mark.parametrize(
    "text",
    [
        "Open Spotify, set the volume to 30 and play something chill",
        "Check the weather and then remind me to take an umbrella",
        "Plan my study schedule for the week",
        "Compare the iPhone and the Pixel cameras",
        "Write a short message to my manager saying I'll be late",
    ],
)
def test_complex_requests(text):
    assert is_complex(text)


@pytest.mark.parametrize(
    "text",
    ["What's the weather in Peshawar?", "Remind me to call Ali at six", "Who won the match yesterday?",
     "Tell me a joke"],
)
def test_simple_requests(text):
    assert not is_complex(text)


def test_router_only_switches_when_needed():
    service = object()
    router = ModelRouter(service, MistralLLMService.Settings, "small", "large")
    assert router.frames_for("What's the weather?") == []

    frames = router.frames_for("Plan my week and then remind me on Monday")
    assert len(frames) == 1 and isinstance(frames[0], LLMUpdateSettingsFrame)
    assert frames[0].delta.model == "large" and frames[0].service is service
    assert router.frames_for("Compare these two laptops") == []  # already large

    assert router.frames_for("What time is it?")[0].delta.model == "small"


def make_context(turns):
    context = LLMContext()
    for i in range(turns):
        context.add_message({"role": "user", "content": f"q{i}"})
        context.add_message({"role": "assistant", "tool_calls": [{"id": f"t{i}"}]})
        context.add_message({"role": "tool", "tool_call_id": f"t{i}", "content": "{}"})
        context.add_message({"role": "assistant", "content": f"a{i}"})
    return context


def test_trim_keeps_the_last_turns_whole():
    context = make_context(10)
    assert conversation.trim(context, max_user_turns=3) == 28
    messages = context.get_messages()
    assert messages[0] == {"role": "user", "content": "q7"}
    assert len(messages) == 12  # tool calls stay with their results


def test_trim_leaves_short_conversations_alone():
    context = make_context(2)
    assert conversation.trim(context, max_user_turns=6) == 0
    assert len(context.get_messages()) == 8


def test_clear():
    context = make_context(2)
    assert conversation.clear(context) == 8
    assert context.get_messages() == []


def test_usage_is_counted_per_day_and_reported(tmp_path):
    path = tmp_path / "usage.json"
    usage = Usage(path, today=date(2026, 10, 1))
    usage.record_llm("mistral", 1200)
    usage.record_llm("mistral", 800)
    usage.record_stt()
    usage.record_instant()
    usage.record_ignored()

    again = Usage(path, today=date(2026, 10, 1))
    day = again.day(date(2026, 10, 1))
    assert day["mistral"] == {"requests": 2, "tokens": 2000}
    assert day["speech-to-text"] == {"requests": 1}

    text = report(again, today=date(2026, 10, 1))
    assert "mistral" in text and "2,000 tokens" in text
    assert "1 instant commands and 1 bits of noise" in text
    assert "Last 7 days: 2 AI requests" in text


def test_usage_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json")
    Usage(path).record_stt()
    assert "speech-to-text" in path.read_text()


@pytest.mark.parametrize(
    "name, provider",
    [("MistralLLMService#0", "mistral"), ("GoogleLLMService#0", "gemini"), ("GroqLLMService#1", "groq"),
     ("OLLamaLLMService#0", "ollama")],
)
def test_provider_names(name, provider):
    assert provider_of(name) == provider


async def test_usage_meter_records_llm_metrics(tmp_path):
    from pipecat.frames.frames import MetricsFrame
    from pipecat.tests.utils import run_test

    from jarvis.usage import UsageMeter

    usage = Usage(tmp_path / "u.json", today=date(2026, 10, 1))
    data = LLMUsageMetricsData(
        processor="MistralLLMService#0",
        value=LLMTokenUsage(prompt_tokens=900, completion_tokens=40, total_tokens=940),
    )
    await run_test(UsageMeter(usage), frames_to_send=[MetricsFrame(data=[data])],
                   expected_down_frames=[MetricsFrame])
    assert usage.day(date(2026, 10, 1))["mistral"] == {"requests": 1, "tokens": 940}
