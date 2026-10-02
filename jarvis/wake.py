"""Wake word gate: nothing leaves the Mac until you say "Hey Jarvis".

The gate sits right after the microphone. While asleep it runs openWakeWord
locally on every 80 ms of audio and drops the audio, so background talk never
reaches speech-to-text (and never spends free-tier quota). Once the wake word
is heard it lets audio through until the conversation goes quiet.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SAMPLE_RATE = 16000
CHUNK = 1280  # openWakeWord expects 80 ms frames at 16 kHz
SPEECH_RMS = 400  # int16 RMS above which we treat the mic as "someone is talking"
ECHO_TAIL_SECS = 0.4  # keep ignoring the mic briefly after Jarvis stops, for room echo
RELEASE_TAIL_SECS = 0.8  # after push-to-talk is released, send silence so the turn ends promptly


def load_wake_model(name: str = "hey_jarvis"):
    from openwakeword.model import Model
    from openwakeword.utils import download_models

    download_models([name])  # no-op once cached
    return Model(wakeword_models=[name], inference_framework="onnx")


class WakeWordGate(FrameProcessor):
    """Drops microphone audio until the wake word is detected.

    Push-to-talk works through ``hold`` and ``release``: while the key is held
    the microphone goes straight through; after release a moment of silence
    is sent, so voice activity detection ends the turn immediately.

    Args:
        model: An openWakeWord ``Model`` (see ``load_wake_model``), or None for
            push-to-talk only.
        threshold: Detection score (0-1) that counts as the wake word.
        awake_secs: Seconds of quiet (no one talking) before going back to sleep.
        on_wake: Called when the wake word is detected.
        on_sleep: Called when the gate goes back to sleep.
        mute_while_speaking: Drop microphone audio while Jarvis is talking, so
            laptop speakers don't make Jarvis interrupt itself. Turn off with
            headphones to be able to talk over Jarvis.
        clock: Injected for tests.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        awake_secs: float = 12.0,
        on_wake: Callable[[], None] | None = None,
        on_sleep: Callable[[], None] | None = None,
        mute_while_speaking: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__()
        self._model = model
        self._threshold = threshold
        self._awake_secs = awake_secs
        self._on_wake = on_wake
        self._on_sleep = on_sleep
        self._mute_while_speaking = mute_while_speaking
        self._clock = clock
        self._bot_stopped_at = float("-inf")
        self._buffer = np.zeros(0, dtype=np.int16)
        self._awake_until = 0.0
        self._bot_speaking = False
        self._held = False
        self._release_tail_until = float("-inf")

    @property
    def awake(self) -> bool:
        return self._awake_until > 0

    def wake(self):
        if not self.awake:
            logger.info("Wake word detected")
            if self._on_wake:
                self._on_wake()
        self._awake_until = self._clock() + self._awake_secs
        self._buffer = np.zeros(0, dtype=np.int16)
        if self._model is not None:
            self._model.reset()

    def hold(self):
        """Push-to-talk key pressed (key repeat calls this again; that's fine)."""
        if self._held:
            return
        self._held = True
        self._release_tail_until = float("-inf")
        self.wake()

    def release(self):
        """Push-to-talk key released."""
        if not self._held:
            return
        self._held = False
        self._release_tail_until = self._clock() + RELEASE_TAIL_SECS
        if self._model is None:
            # Push-to-talk only: listen again only when the key is pressed again.
            self._awake_until = self._release_tail_until

    def sleep(self):
        if self.awake:
            logger.info("Going back to sleep")
            if self._on_sleep:
                self._on_sleep()
        self._awake_until = 0.0

    def _extend(self):
        if self.awake:
            self._awake_until = self._clock() + self._awake_secs

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = self._clock()
            self._extend()

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._in_release_tail():
                frame.audio = bytes(len(frame.audio))  # true silence ends the turn
                await self.push_frame(frame, direction)
                return
            if self._handle_audio(frame):
                await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _handle_audio(self, frame: InputAudioRawFrame) -> bool:
        """Returns True if the audio should be passed on."""
        samples = np.frombuffer(frame.audio, dtype=np.int16)

        if self._held:
            self._extend()
            return True

        if self.awake:
            if self._model is None:
                # Push-to-talk only: the key, not the room, decides when Jarvis listens.
                self.sleep()
                return False
            if self._mute_while_speaking and self._hearing_jarvis():
                self._extend()
                return False
            if self._bot_speaking or _rms(samples) > SPEECH_RMS:
                self._extend()
            elif self._clock() > self._awake_until:
                self.sleep()
                return False
            return True

        if self._bot_speaking:
            # Jarvis saying its own name through the speakers must not wake it.
            self._buffer = np.zeros(0, dtype=np.int16)
            return False

        if self._model is None:  # push-to-talk only
            return False

        if frame.sample_rate != SAMPLE_RATE:
            logger.warning(f"Wake word needs {SAMPLE_RATE} Hz audio, got {frame.sample_rate} Hz")
            return False

        self._buffer = np.concatenate([self._buffer, samples])
        while len(self._buffer) >= CHUNK:
            chunk, self._buffer = self._buffer[:CHUNK], self._buffer[CHUNK:]
            scores = self._model.predict(chunk)
            if max(scores.values(), default=0.0) >= self._threshold:
                self.wake()
                return True
        return False

    def _in_release_tail(self) -> bool:
        return not self._held and self._clock() < self._release_tail_until

    def _hearing_jarvis(self) -> bool:
        return self._bot_speaking or self._clock() - self._bot_stopped_at < ECHO_TAIL_SECS


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
