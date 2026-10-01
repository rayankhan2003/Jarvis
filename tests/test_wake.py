import numpy as np
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, InputAudioRawFrame

from jarvis.wake import CHUNK, WakeWordGate

WAKE = 9000  # fake model: chunks at this amplitude "contain the wake word"
SPEECH = 3000
QUIET = 0


class FakeModel:
    def predict(self, chunk):
        return {"hey_jarvis": 1.0 if chunk.max() >= WAKE else 0.0}

    def reset(self):
        pass


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def audio(level, samples=CHUNK):
    return InputAudioRawFrame(audio=np.full(samples, level, dtype=np.int16).tobytes(),
                              sample_rate=16000, num_channels=1)


def make_gate(**kwargs):
    clock = Clock()
    woke = []
    gate = WakeWordGate(FakeModel(), awake_secs=10, on_wake=lambda: woke.append(True), clock=clock, **kwargs)
    return gate, clock, woke


def test_drops_audio_until_wake_word():
    gate, _, woke = make_gate()
    assert gate._handle_audio(audio(SPEECH)) is False
    assert not gate.awake
    assert gate._handle_audio(audio(WAKE)) is True
    assert gate.awake and woke == [True]
    assert gate._handle_audio(audio(SPEECH)) is True


def test_detects_across_small_frames():
    gate, _, _ = make_gate()
    # 20 ms frames: the model only runs once 80 ms have been collected.
    for _ in range(3):
        assert gate._handle_audio(audio(WAKE, samples=320)) is False
    assert gate._handle_audio(audio(WAKE, samples=320)) is True


def test_goes_back_to_sleep_after_quiet():
    gate, clock, _ = make_gate()
    gate._handle_audio(audio(WAKE))
    clock.now = 5
    assert gate._handle_audio(audio(SPEECH)) is True  # talking extends the window
    clock.now = 14
    assert gate._handle_audio(audio(QUIET)) is True
    clock.now = 15.1
    assert gate._handle_audio(audio(QUIET)) is False
    assert not gate.awake


async def test_stays_awake_while_jarvis_talks_and_ignores_its_own_name():
    gate, clock, woke = make_gate(mute_while_speaking=False)

    await gate.process_frame(BotStartedSpeakingFrame(), _upstream())
    assert gate._handle_audio(audio(WAKE)) is False  # "Say 'Hey Jarvis'..." from the speakers
    assert woke == []

    await gate.process_frame(BotStoppedSpeakingFrame(), _upstream())
    gate._handle_audio(audio(WAKE))
    await gate.process_frame(BotStartedSpeakingFrame(), _upstream())
    clock.now = 60  # a long answer
    assert gate._handle_audio(audio(QUIET)) is True
    await gate.process_frame(BotStoppedSpeakingFrame(), _upstream())
    clock.now = 65
    assert gate._handle_audio(audio(QUIET)) is True


async def test_mutes_the_mic_while_jarvis_speaks_and_for_the_echo_tail():
    gate, clock, _ = make_gate()
    gate._handle_audio(audio(WAKE))
    assert gate._handle_audio(audio(SPEECH)) is True

    await gate.process_frame(BotStartedSpeakingFrame(), _upstream())
    clock.now = 30  # a long answer: Jarvis's own voice must not reach speech-to-text
    assert gate._handle_audio(audio(SPEECH)) is False
    assert gate.awake

    await gate.process_frame(BotStoppedSpeakingFrame(), _upstream())
    clock.now = 30.2  # echo still in the room
    assert gate._handle_audio(audio(SPEECH)) is False
    clock.now = 30.5  # your turn
    assert gate._handle_audio(audio(SPEECH)) is True


def test_sleep_on_dismiss():
    gate, _, _ = make_gate()
    gate._handle_audio(audio(WAKE))
    gate.sleep()
    assert gate._handle_audio(audio(SPEECH)) is False


def _upstream():
    from pipecat.processors.frame_processor import FrameDirection

    return FrameDirection.UPSTREAM
