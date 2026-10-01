"""Run Jarvis's processors inside real Pipecat pipelines."""

import numpy as np
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from jarvis import fastpath
from jarvis.fastpath import FastPath
from jarvis.tools.system import Result
from jarvis.wake import CHUNK, WakeWordGate


def transcript(text):
    return TranscriptionFrame(text=text, user_id="me", timestamp="now", finalized=True)


async def test_instant_command_is_answered_without_the_brain(monkeypatch):
    async def fake_media(action):
        return Result(True, "Paused.")

    monkeypatch.setattr(fastpath.system, "media", fake_media)
    context = LLMContext()
    turns = []

    async def on_user_turn():
        turns.append(True)

    down, _ = await run_test(
        FastPath(context, on_user_turn=on_user_turn),
        frames_to_send=[transcript("Pause the music.")],
        expected_down_frames=[TTSSpeakFrame],
    )
    assert down[0].text == "Paused."
    assert context.get_messages()[-1] == {"role": "user", "content": "Pause the music."}
    assert turns == [True]


async def test_other_requests_reach_the_brain():
    await run_test(
        FastPath(LLMContext()),
        frames_to_send=[transcript("What's the weather like in Lahore?")],
        expected_down_frames=[TranscriptionFrame],
    )


async def test_failed_app_launch_falls_back_to_the_brain(monkeypatch):
    async def fake_open_app(name):
        return Result(False, f"I couldn't find an app called {name}.")

    monkeypatch.setattr(fastpath.system, "open_app", fake_open_app)
    await run_test(
        FastPath(LLMContext()),
        frames_to_send=[transcript("Open my portfolio")],
        expected_down_frames=[TranscriptionFrame],
    )


class FakeModel:
    def predict(self, chunk):
        return {"hey_jarvis": 1.0 if chunk.max() > 8000 else 0.0}

    def reset(self):
        pass


def audio(level):
    return InputAudioRawFrame(audio=np.full(CHUNK, level, dtype=np.int16).tobytes(),
                              sample_rate=16000, num_channels=1)


async def test_gate_only_lets_audio_through_after_wake_word():
    down, _ = await run_test(
        WakeWordGate(FakeModel()),
        frames_to_send=[audio(2000), audio(2000), audio(9000), audio(2000)],
        expected_down_frames=[InputAudioRawFrame, InputAudioRawFrame],
    )
