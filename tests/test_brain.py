"""Big brain routing, memory and the user's profile."""

import pytest
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.tests.utils import run_test

from jarvis.config import Config
from jarvis.failover import FreeTierFailover
from jarvis.memory import Memory
from jarvis.persona import system_prompt
from jarvis.router import BrainRouter


def brains():
    mistral = MistralLLMService(api_key="m", settings=MistralLLMService.Settings(model="ministral-8b-latest"))
    groq = GroqLLMService(api_key="g", settings=GroqLLMService.Settings(model="openai/gpt-oss-120b"))
    return mistral, groq, LLMSwitcher(llms=[mistral, groq], strategy_type=FreeTierFailover)


async def test_complex_requests_go_to_the_big_brain_for_one_turn():
    mistral, groq, switcher = brains()
    router = BrainRouter(switcher.strategy, groq)

    assert router.frames_for("What time is it?") == []
    frames = router.frames_for("Explain how black holes work")
    await run_test(switcher, frames_to_send=frames, expected_down_frames=None)
    assert switcher.strategy.active_service is groq

    # Next turn starts on the everyday brain again.
    assert await switcher.strategy.restore_preferred() is mistral
    assert switcher.strategy.active_service is mistral


async def test_big_brain_is_skipped_while_it_is_benched():
    mistral, groq, switcher = brains()
    switcher.strategy._benched_until[id(groq)] = float("inf")
    assert BrainRouter(switcher.strategy, groq).frames_for("Plan my week") == []


async def test_memory_changes_reach_every_brain(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    config = Config(user_name="Rayan")
    llms = [
        MistralLLMService(api_key="m", settings=MistralLLMService.Settings(system_instruction=system_prompt(config))),
        GroqLLMService(api_key="g", settings=GroqLLMService.Settings(system_instruction=system_prompt(config))),
        GoogleLLMService(api_key="x", settings=GoogleLLMService.Settings(system_instruction=system_prompt(config))),
    ]

    async def refresh():
        prompt = system_prompt(config, memories=memory.prompt_block())
        for llm in llms:
            await llm._update_settings(type(llm).Settings(system_instruction=prompt))

    memory.on_change = refresh
    await memory.remember("my sister's name is Ayesha")
    for llm in llms:
        assert "- My sister's name is Ayesha" in llm._settings.system_instruction


async def test_remember_replaces_and_forgets(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    assert (await memory.remember("My sister's name is Ayesha.")).say == "I'll remember that."
    await memory.remember("I like biryani")
    await memory.remember("My sister's name is Sara")  # newer fact about the same thing wins
    assert memory.facts == ["I like biryani", "My sister's name is Sara"]

    again = Memory(tmp_path / "memory.json")  # survives a restart
    assert again.facts == memory.facts

    assert (await memory.forget("sister")).say == "Forgotten."
    assert memory.facts == ["I like biryani"]
    assert (await memory.forget("cars")).say == "I don't have anything about that."
    assert (await memory.forget("everything")).say == "Forgotten, all 1 things."
    assert (await memory.recall()).say == "Nothing yet. Tell me to remember something."


async def test_recall_speaks_the_latest(tmp_path):
    memory = Memory(tmp_path / "memory.json")
    for fact in ["I like biryani", "My sister is Ayesha", "I study at Edwardes College", "I work at Strings Technologies",
                 "My car is a Corolla", "My birthday is in March"]:
        await memory.remember(fact)
    said = (await memory.recall()).say
    assert said.startswith("I remember:") and "And 1 more." in said


def test_profile_goes_into_the_instructions():
    prompt = system_prompt(Config(city="Peshawar", music_taste=("Talha Anjum", "Kaavish"), browser="Brave"))
    assert "Lives in Peshawar" in prompt and "Talha Anjum, Kaavish" in prompt and "Brave" in prompt


@pytest.mark.parametrize(
    "spoken, name",
    [
        ("Remember that my sister's name is Ayesha.", "remember"),
        ("Hey Jarvis, remember my favourite food is biryani", "remember"),
        ("Forget my sister's name", "forget"),
        ("What do you know about me?", "recall"),
        ("What's the weather?", "weather"),
        ("How's the weather in Lahore today?", "weather"),
    ],
)
def test_memory_and_weather_are_instant(spoken, name):
    from jarvis.fastpath import match_command

    assert match_command(spoken).name == name


@pytest.mark.parametrize("spoken", ["Remember to call mum at six", "Forget it", "Is it going to rain tomorrow?"])
def test_reminders_and_forecasts_go_to_the_brain(spoken):
    from jarvis.fastpath import match_command

    assert match_command(spoken) is None


async def test_remember_keeps_the_users_wording(monkeypatch, tmp_path):
    from jarvis import fastpath

    memory = Memory(tmp_path / "memory.json")
    monkeypatch.setattr(fastpath, "MEMORY", memory)
    await fastpath.match_command("Jarvis, remember that my sister's name is Ayesha!").run()
    assert memory.facts == ["My sister's name is Ayesha"]
