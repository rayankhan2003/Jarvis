"""Jarvis's voices: the best of Kokoro's, and `jarvis voices` to hear them.

Grades are from Kokoro's own voice list (hexgrad/Kokoro-82M VOICES.md);
higher grades had more and cleaner training audio and sound more natural.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pipecat.transcriptions.language import Language

SAMPLE = "Good evening. I'm Jarvis. Shall I play some music, or check the weather for you?"


@dataclass(frozen=True)
class Voice:
    id: str
    grade: str
    description: str


VOICES = [
    Voice("af_heart", "A", "female, American, warm and natural"),
    Voice("af_bella", "A-", "female, American, bright and clear"),
    Voice("af_nicole", "B-", "female, American, soft and close"),
    Voice("bf_emma", "B-", "female, British, the closest to FRIDAY"),
    Voice("am_michael", "C+", "male, American, calm"),
    Voice("am_fenrir", "C+", "male, American, deeper"),
    Voice("bm_fable", "C", "male, British"),
    Voice("bm_george", "C", "male, British, the old default"),
]


def language_for(voice: str) -> Language:
    """Kokoro voice ids start with the accent: a = American, b = British."""
    return Language.EN_GB if voice.startswith("b") else Language.EN_US


def kokoro_lang(voice: str) -> str:
    return "en-gb" if voice.startswith("b") else "en-us"


def audition(names: list[str] | None = None, current: str = "") -> int:
    """Say a sample sentence in each voice, so you can pick one."""
    import numpy as np
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    from jarvis.doctor import to_wav

    model, voices_file = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices_file)
    kokoro = Kokoro(str(model), str(voices_file))

    chosen = [v for v in VOICES if not names or v.id in names]
    chosen += [Voice(n, "?", "") for n in (names or []) if n not in {v.id for v in VOICES}]
    print("\nJarvis voices (set one with JARVIS_VOICE=... in .env)\n")
    with tempfile.TemporaryDirectory() as tmp:
        for voice in chosen:
            mark = "  <- current" if voice.id == current else ""
            print(f"  {voice.id:<11} {voice.grade:<3} {voice.description}{mark}")
            try:
                samples, rate = kokoro.create(SAMPLE, voice=voice.id, speed=1.0, lang=kokoro_lang(voice.id))
            except Exception as e:
                print(f"     couldn't make this voice: {e}")
                continue
            path = Path(tmp) / f"{voice.id}.wav"
            path.write_bytes(to_wav(np.asarray(samples), rate))
            if sys.platform == "darwin":
                subprocess.run(["afplay", str(path)], check=False)
    print("\nPick one, put it in .env as JARVIS_VOICE=<id>, and restart jarvis.\n")
    return 0
