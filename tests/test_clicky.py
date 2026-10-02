"""Push-to-talk, the cursor buddy, and seeing/pointing at the screen."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

from jarvis import screen
from jarvis.buddy import FLIGHT_SECS, POINT_HOLD_SECS, Buddy, arc_point, top_left_to_cocoa
from jarvis.config import Config
from jarvis.ptt import FLAGS_CHANGED, KEY_DOWN, KEY_UP, KeySpec, key_spec, next_state
from jarvis.screen import Element, best_match, fit_size, image_to_points, match_score, parse_point
from jarvis.tools.system import Result
from jarvis.wake import CHUNK, RELEASE_TAIL_SECS, WakeWordGate

# --- push-to-talk ---------------------------------------------------------------

RIGHT_OPTION = key_spec("right_option")


def test_key_names():
    assert key_spec("Right Option") == KeySpec(61, True)
    assert key_spec("f13") == KeySpec(105, False)
    with pytest.raises(ValueError):
        key_spec("caps lock")


def test_modifier_toggles_on_flags_changed():
    assert next_state(False, FLAGS_CHANGED, 61, RIGHT_OPTION) is True
    assert next_state(True, FLAGS_CHANGED, 61, RIGHT_OPTION) is False
    assert next_state(False, FLAGS_CHANGED, 58, RIGHT_OPTION) is None  # left option: not ours


def test_function_key_uses_down_and_up():
    f13 = key_spec("f13")
    assert next_state(False, KEY_DOWN, 105, f13) is True
    assert next_state(True, KEY_UP, 105, f13) is False
    assert next_state(False, KEY_DOWN, 0, f13) is None


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def audio(level):
    return InputAudioRawFrame(audio=np.full(CHUNK, level, dtype=np.int16).tobytes(), sample_rate=16000,
                              num_channels=1)


def test_holding_the_key_lets_audio_through_without_a_wake_word():
    clock = Clock()
    gate = WakeWordGate(None, clock=clock)  # push-to-talk only: no wake word model
    assert gate._handle_audio(audio(3000)) is False
    gate.hold()
    assert gate.awake and gate._handle_audio(audio(3000)) is True
    gate.release()
    assert gate._in_release_tail()
    clock.now = RELEASE_TAIL_SECS + 0.1
    assert not gate._in_release_tail()
    assert gate._handle_audio(audio(3000)) is False  # listens again only on the next press


async def test_release_sends_silence_so_the_turn_ends():
    clock = Clock()
    gate = WakeWordGate(None, clock=clock)
    gate.hold()
    gate.release()
    down, _ = await run_test(gate, frames_to_send=[audio(3000)], expected_down_frames=[InputAudioRawFrame])
    assert not any(np.frombuffer(down[0].audio, dtype=np.int16))


def test_holding_works_alongside_the_wake_word():
    class Model:
        def predict(self, chunk):
            return {"hey_jarvis": 0.0}

        def reset(self):
            pass

    gate = WakeWordGate(Model(), clock=Clock())
    gate.hold()
    assert gate._handle_audio(audio(3000)) is True


# --- the cursor buddy -----------------------------------------------------------

def test_coordinates_flip_to_cocoa():
    assert top_left_to_cocoa(100, 50, 900) == (100, 850)


def test_arc_starts_and_ends_in_place_and_swoops_up():
    start, end = (0.0, 0.0), (400.0, 0.0)
    assert arc_point(start, end, 0) == start
    assert arc_point(start, end, 1) == pytest.approx(end)
    assert arc_point(start, end, 0.5)[1] > 40  # curves above the straight line


def test_buddy_follows_the_mouse_and_shows_its_mode():
    buddy = Buddy()
    first = buddy.frame((100, 100), 0)
    assert (first.x, first.y) == (118, 78)  # just below-right of the cursor
    for _ in range(30):
        frame = buddy.frame((300, 300), 0)
    assert frame.x == pytest.approx(318, abs=0.5)
    buddy.set_mode("listening")
    assert buddy.frame((300, 300), 0).color == (0.20, 0.85, 0.45, 0.95)


def test_buddy_flies_points_and_returns():
    buddy = Buddy()
    buddy.frame((100, 100), 0)
    buddy.point((600, 400), "Export", now=1.0)
    mid = buddy.frame((100, 100), 1.0 + FLIGHT_SECS / 2)
    assert mid.label == "" and 118 < mid.x < 600
    arrived = buddy.frame((100, 100), 1.0 + FLIGHT_SECS + 0.1)
    assert (arrived.x, arrived.y) == pytest.approx((600, 400))
    assert arrived.label == "Export" and arrived.ring_radius > 0
    later = buddy.frame((100, 100), 1.0 + FLIGHT_SECS + POINT_HOLD_SECS + 0.1)
    assert not buddy.pointing and later.label == ""
    assert math.hypot(later.x - 600, later.y - 400) > 1  # heading back to the cursor


# --- reading the model's reply ----------------------------------------------------

@pytest.mark.parametrize(
    "reply, text, label, hint",
    [
        ("Click Export at the top right. [POINT:Export@1180,40]", "Click Export at the top right.", "Export",
         (1180.0, 40.0)),
        ("That's the color inspector. [POINT:1100,42:color inspector]", "That's the color inspector.",
         "color inspector", (1100.0, 42.0)),
        ("It's a terminal. [POINT:400,300:terminal:screen2]", "It's a terminal.", "terminal", (400.0, 300.0)),
        ("HTML is the skeleton of a page. [POINT:none]", "HTML is the skeleton of a page.", None, None),
        ("Just text, no tag.", "Just text, no tag.", None, None),
        ("Use the File menu. [POINT:File]", "Use the File menu.", "File", None),
    ],
)
def test_parse_point(reply, text, label, hint):
    assert parse_point(reply) == (text, label, hint)


# --- finding the element ------------------------------------------------------------

def test_match_score_prefers_exact_and_whole_words():
    assert match_score("Export", "Export") == 1.0
    assert match_score("export", "Export…") == 1.0
    assert match_score("Export", "Export as PDF") > match_score("Export", "Exported files from last year")
    assert match_score("Export", "Import") < 0.62
    assert match_score("", "anything") == 0.0


def test_best_match_uses_the_hint_to_break_ties():
    elements = [Element("Save", 10, 10, 40, 20), Element("Save", 900, 10, 40, 20), Element("Cancel", 500, 10, 60, 20)]
    assert best_match("Save", elements, hint=(910, 20)).x == 900
    assert best_match("Save", elements, hint=(5, 5)).x == 10
    assert best_match("Delete everything", elements) is None


def test_image_sizes_and_coordinates():
    assert fit_size(2880, 1800) == (1280, 800)  # 16:10 Retina screenshot
    assert fit_size(1000, 700) == (1000, 700)  # never enlarged
    assert image_to_points(640, 400, (1280, 800), (1440, 900)) == (720, 450)


def make_shot(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"png")
    return screen.Shot(path, b"jpeg", (1280, 800), (2880, 1800), (1440, 900))


async def test_locate_prefers_accessibility_then_text_then_the_estimate(monkeypatch, tmp_path):
    shot = make_shot(tmp_path)
    monkeypatch.setattr(screen, "accessibility_elements", lambda: [Element("Export", 1000, 20, 80, 24)])
    monkeypatch.setattr(screen, "text_elements", lambda s: [Element("Export", 1, 1, 10, 10)])
    assert await screen.locate("Export", (1180, 40), shot) == (1040, 32, "accessibility")

    monkeypatch.setattr(screen, "accessibility_elements", lambda: [])
    assert await screen.locate("Export", (1180, 40), shot) == (6, 6, "text")

    monkeypatch.setattr(screen, "text_elements", lambda s: [])
    x, y, source = await screen.locate("Export", (1180, 40), shot)
    assert source == "estimate" and (x, y) == (1327.5, 45.0)
    assert await screen.locate("Export", None, shot) is None


# --- looking at the screen -------------------------------------------------------------

def vision_config(**kw):
    return Config(gemini_api_key="g", mistral_api_key="m", **kw)


def test_vision_providers_follow_the_order():
    assert [n for n, _ in screen.vision_providers(vision_config())] == ["mistral", "gemini"]
    assert [n for n, _ in screen.vision_providers(vision_config(vision_order=("gemini", "mistral")))] == [
        "gemini", "mistral"]
    local = Config(ollama_vision_model="qwen2.5vl:7b")
    assert [n for n, _ in screen.vision_providers(local)] == ["ollama"]
    assert screen.vision_providers(Config()) == []


async def test_vision_falls_back_to_the_next_provider(monkeypatch):
    calls = []

    async def broken(session, prompt, image):
        calls.append("mistral")
        raise RuntimeError("400 this model doesn't take images")

    async def working(session, prompt, image):
        calls.append("gemini")
        assert "1280x800" in prompt and "Where is Export?" in prompt
        return "Top right. [POINT:Export@1180,40]"

    monkeypatch.setattr(screen, "vision_providers", lambda config: [("mistral", broken), ("gemini", working)])
    reply = await screen.ask_vision(Config(), "Where is Export?", b"jpeg", (1280, 800))
    assert reply == "Top right. [POINT:Export@1180,40]" and calls == ["mistral", "gemini"]


async def test_look_answers_points_and_cleans_up(monkeypatch, tmp_path):
    shot = make_shot(tmp_path)
    pointed = []

    async def fake_capture():
        return shot

    async def fake_vision(config, question, jpeg, size):
        return "It's in the top right. [POINT:Export@1180,40]"

    async def fake_locate(label, hint, s):
        return (1040.0, 32.0, "accessibility")

    monkeypatch.setattr(screen, "capture", fake_capture)
    monkeypatch.setattr(screen, "ask_vision", fake_vision)
    monkeypatch.setattr(screen, "locate", fake_locate)
    result = await screen.look(vision_config(), "Where is export?", lambda x, y, label: pointed.append((x, y, label)))
    assert result.ok and result.say == "It's in the top right."
    assert pointed == [(1040.0, 32.0, "Export")]
    assert result.data["point"]["found_by"] == "accessibility"
    assert not Path(shot.path).exists()  # screenshots aren't left lying around


async def test_look_explains_missing_setup(monkeypatch):
    assert "vision model" in (await screen.look(Config(), "what's this?")).say

    async def no_shot():
        return None

    monkeypatch.setattr(screen, "capture", no_shot)
    assert "Screen Recording" in (await screen.look(vision_config(), "what's this?")).say


# --- instant commands and the overlay client ----------------------------------------------

@pytest.mark.parametrize("spoken", ["What's on my screen?", "Hey Jarvis, what is this error?", "Explain this",
                                    "Where is the export button?", "How do I add a filter in this app?",
                                    "Show me where to change the font", "Point to the save button"])
def test_screen_questions_are_instant(spoken):
    from jarvis.fastpath import match_command

    command = match_command(spoken)
    assert command.name == "screen" and command.ack == "Let me look."


@pytest.mark.parametrize("spoken", ["Where is Lahore?", "How do I make biryani?"])
def test_ordinary_questions_are_not_screen_questions(spoken):
    from jarvis.fastpath import match_command

    assert match_command(spoken) is None


async def test_screen_question_says_let_me_look_first(monkeypatch):
    from jarvis import fastpath
    from jarvis.fastpath import FastPath

    async def fake_look(config, question, point_at=None):
        assert question == "where is the export button?"
        return Result(True, "Top right.")

    monkeypatch.setattr(screen, "look", fake_look)
    down, _ = await run_test(
        FastPath(LLMContext()),
        frames_to_send=[TranscriptionFrame(text="Jarvis, where is the export button?", user_id="me",
                                           timestamp="now", finalized=True)],
        expected_down_frames=[TTSSpeakFrame, TTSSpeakFrame],
    )
    assert [f.text for f in down] == ["Let me look.", "Top right."]
    assert fastpath  # imported for monkeypatch targets


class FakeStdin:
    def __init__(self):
        self.lines = []

    def write(self, text):
        self.lines.append(json.loads(text))

    def flush(self):
        pass


class FakeProc:
    def __init__(self):
        self.stdin = FakeStdin()

    def poll(self):
        return None


def test_overlay_client_sends_only_changes():
    from jarvis.overlay import Overlay

    overlay = Overlay()
    overlay.status("idle")  # not running: ignored, no error
    overlay._proc = FakeProc()
    overlay.status("listening")
    overlay.status("listening")
    overlay.point(1040.04, 32, "Export")
    assert overlay._proc.stdin.lines == [
        {"cmd": "status", "mode": "listening"},
        {"cmd": "point", "x": 1040.0, "y": 32, "label": "Export"},
    ]


async def test_buddy_turns_blue_while_jarvis_speaks():
    from jarvis.overlay import Overlay, OverlayStatus

    overlay = Overlay()
    overlay._proc = FakeProc()
    await run_test(OverlayStatus(overlay), frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()],
                   expected_down_frames=None)
    assert [line["mode"] for line in overlay._proc.stdin.lines] == ["speaking", "idle"]


def test_doctor_test_image_has_text():
    import io

    from PIL import Image

    from jarvis.doctor import test_image

    img = Image.open(io.BytesIO(test_image()))
    assert img.size == (480, 160) and img.getextrema()[0][0] < 50  # some dark pixels: the text


def test_small_typos_still_match():
    assert match_score("Exprot", "Export") >= 0.62
