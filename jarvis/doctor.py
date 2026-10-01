"""`jarvis doctor`: checks this Mac, the keys and the latency before you rely on Jarvis.

Every check measures the real thing (a real request, a real microphone
recording, a real synthesis) and prints what to do when something fails.
"""

from __future__ import annotations

import asyncio
import io
import platform
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass

import aiohttp
import numpy as np

from jarvis.config import Config

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
GROQ_API = "https://api.groq.com/openai/v1"
TEST_SENTENCE = "Good evening. All systems are online."
VAD_STOP_SECS = 0.4  # roughly how long Silero waits to decide you've stopped talking

OK, WARN, FAIL = "ok", "warn", "fail"
ICONS = {OK: "\033[32m✔\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✘\033[0m"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _print(check: Check):
    print(f" {ICONS[check.status]} {check.name:<22} {check.detail}")
    if check.fix and check.status != OK:
        print(f"   {'':<22} → {check.fix}")


# ---------------------------------------------------------------- machine

def check_machine() -> list[Check]:
    checks = []
    is_mac = sys.platform == "darwin"
    arch = platform.machine()
    checks.append(Check(
        "macOS / Apple Silicon",
        OK if is_mac and arch == "arm64" else WARN,
        f"{platform.system()} {platform.release()} on {arch}",
        "Jarvis's Mac control and voice are built for Apple Silicon Macs.",
    ))
    py = sys.version_info
    checks.append(Check(
        "Python", OK if py >= (3, 11) else FAIL, platform.python_version(),
        "Install Python 3.11+ (brew install python@3.12).",
    ))
    if is_mac:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
        free = _mac_free_memory()
        gb = total / 1e9
        detail = f"{gb:.0f} GB total, ~{free / 1e9:.1f} GB free right now"
        status = OK if free > 1.5e9 else WARN
        checks.append(Check("Memory", status, detail,
                            "Close heavy apps (Chrome tabs, Docker). Jarvis itself needs ~1-1.5 GB."))
        checks.append(Check(
            "Offline brain", OK if gb >= 16 else WARN,
            "fits comfortably" if gb >= 16 else "8 GB: a 4B model will run, but slowly and with swapping",
            "Keep OLLAMA_MODEL as a backup only; the free cloud brains are faster on 8 GB.",
        ))
    return checks


def _mac_free_memory() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 16384
    pages = 0
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics") and "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            if line.startswith(key):
                pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page


def check_packages() -> list[Check]:
    checks = []
    for module, why, fix in [
        ("pipecat", "voice pipeline", "pip install -e ."),
        ("pyaudio", "microphone and speakers", "brew install portaudio && pip install pyaudio"),
        ("openwakeword", "wake word", "pip install -e ."),
        ("kokoro_onnx", "voice", "pip install -e ."),
    ]:
        try:
            __import__(module)
            checks.append(Check(module, OK, why))
        except Exception as e:
            checks.append(Check(module, FAIL, f"{why}: {e}", fix))
    for tool in ("osascript", "afplay"):
        found = shutil.which(tool)
        checks.append(Check(tool, OK if found else WARN, found or "missing", "Only available on macOS."))
    return checks


# ---------------------------------------------------------------- audio

def check_microphone(seconds: float = 3.0) -> Check:
    try:
        import pyaudio
    except Exception as e:
        return Check("Microphone", FAIL, str(e), "brew install portaudio && pip install pyaudio")
    pa = pyaudio.PyAudio()
    try:
        info = pa.get_default_input_device_info()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1600)
        print(f"   Recording {seconds:.0f}s from '{info['name']}'. Say something...")
        frames = [stream.read(1600, exception_on_overflow=False) for _ in range(int(seconds * 10))]
        stream.close()
    except Exception as e:
        return Check("Microphone", FAIL, str(e),
                     "System Settings → Privacy & Security → Microphone → allow your terminal app.")
    finally:
        pa.terminate()
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < 50:
        return Check("Microphone", FAIL, "recorded pure silence",
                     "Grant microphone access to your terminal (Privacy & Security → Microphone).")
    if peak < 1500:
        return Check("Microphone", WARN, f"very quiet (peak {peak:.0f})",
                     "Move closer or raise input volume in Sound settings.")
    return Check("Microphone", OK, f"'{info['name']}' (peak {peak:.0f})")


def check_wake_word() -> Check:
    try:
        from jarvis.wake import load_wake_model

        start = time.perf_counter()
        model = load_wake_model()
        load = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(25):
            model.predict(np.zeros(1280, dtype=np.int16))
        per_chunk = (time.perf_counter() - start) / 25 * 1000
    except Exception as e:
        return Check("Wake word", FAIL, str(e), "pip install -e . (needs internet once to download the model)")
    return Check("Wake word", OK, f"'hey jarvis' loaded in {load:.1f}s, {per_chunk:.1f} ms per 80 ms of audio")


def synthesize(config: Config) -> tuple[np.ndarray, int, float, float]:
    """Returns (samples, sample_rate, load secs, synthesis secs)."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model, voices = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    start = time.perf_counter()
    _ensure_model_files(model, voices)
    kokoro = Kokoro(str(model), str(voices))
    load = time.perf_counter() - start
    start = time.perf_counter()
    samples, rate = kokoro.create(TEST_SENTENCE, voice=config.voice, speed=1.0, lang="en-gb")
    return samples, rate, load, time.perf_counter() - start


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- cloud

async def check_gemini(session: aiohttp.ClientSession, config: Config) -> tuple[Check, float | None]:
    if not config.gemini_api_key:
        return Check("Gemini brain", WARN, "no GEMINI_API_KEY",
                     "Free key (no card): https://aistudio.google.com/apikey"), None
    headers = {"x-goog-api-key": config.gemini_api_key}
    async with session.get(f"{GEMINI_API}/models?pageSize=200", headers=headers) as resp:
        if resp.status != 200:
            return Check("Gemini brain", FAIL, f"key rejected ({resp.status})",
                         "Check GEMINI_API_KEY in .env"), None
        names = [m["name"].removeprefix("models/") for m in (await resp.json()).get("models", [])]
    if config.gemini_model not in names:
        lite = [n for n in names if "flash-lite" in n and "preview" not in n]
        return Check("Gemini brain", FAIL, f"model '{config.gemini_model}' not available to this key",
                     f"Set GEMINI_MODEL to one of: {', '.join(sorted(lite)[-3:]) or ', '.join(names[:5])}"), None
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ready"}]}]}
    url = f"{GEMINI_API}/models/{config.gemini_model}:streamGenerateContent?alt=sse"
    start = time.perf_counter()
    async with session.post(url, headers=headers, json=body) as resp:
        if resp.status == 429:
            return Check("Gemini brain", WARN, "rate limited / daily quota used up",
                         "Jarvis will fail over to Groq until it resets."), None
        if resp.status != 200:
            return Check("Gemini brain", FAIL, f"HTTP {resp.status}: {(await resp.text())[:120]}"), None
        async for _ in resp.content:
            ttft = time.perf_counter() - start
            break
        else:
            return Check("Gemini brain", FAIL, "empty response"), None
    return Check("Gemini brain", OK, f"{config.gemini_model}, first token in {ttft * 1000:.0f} ms"), ttft


async def check_groq_llm(session: aiohttp.ClientSession, config: Config) -> tuple[Check, float | None]:
    if not config.groq_api_key:
        return Check("Groq brain", WARN, "no GROQ_API_KEY",
                     "Free key (no card): https://console.groq.com/keys"), None
    headers = {"Authorization": f"Bearer {config.groq_api_key}"}
    body = {"model": config.groq_llm_model, "stream": True, "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}
    start = time.perf_counter()
    async with session.post(f"{GROQ_API}/chat/completions", headers=headers, json=body) as resp:
        if resp.status == 429:
            return Check("Groq brain", WARN, "rate limited / daily quota used up"), None
        if resp.status != 200:
            return Check("Groq brain", FAIL, f"HTTP {resp.status}: {(await resp.text())[:120]}",
                         "Check GROQ_API_KEY and GROQ_LLM_MODEL in .env"), None
        async for _ in resp.content:
            ttft = time.perf_counter() - start
            break
        else:
            return Check("Groq brain", FAIL, "empty response"), None
    return Check("Groq brain", OK, f"{config.groq_llm_model}, first token in {ttft * 1000:.0f} ms"), ttft


async def check_groq_stt(session: aiohttp.ClientSession, config: Config, wav: bytes | None) -> tuple[Check, float | None]:
    if config.local_stt:
        return Check("Speech-to-text", OK, "MLX Whisper on this Mac (JARVIS_LOCAL_STT=1)"), None
    if not config.groq_api_key:
        return Check("Speech-to-text", FAIL, "no GROQ_API_KEY",
                     "Free key: https://console.groq.com/keys (or JARVIS_LOCAL_STT=1)"), None
    if wav is None:
        return Check("Speech-to-text", WARN, "skipped (no test audio, voice check failed)"), None
    form = aiohttp.FormData()
    form.add_field("model", config.groq_stt_model)
    form.add_field("language", "en")
    form.add_field("file", wav, filename="test.wav", content_type="audio/wav")
    start = time.perf_counter()
    async with session.post(f"{GROQ_API}/audio/transcriptions",
                            headers={"Authorization": f"Bearer {config.groq_api_key}"}, data=form) as resp:
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return Check("Speech-to-text", FAIL, f"HTTP {resp.status}: {(await resp.text())[:120]}"), None
        text = (await resp.json()).get("text", "").strip()
    heard = "all systems" in text.lower()
    return Check("Speech-to-text", OK if heard else WARN,
                 f"Groq Whisper heard \"{text}\" in {elapsed * 1000:.0f} ms"), elapsed


async def check_ollama(session: aiohttp.ClientSession, config: Config) -> Check:
    if not config.ollama_model:
        return Check("Offline brain", OK, "disabled (set OLLAMA_MODEL to enable)")
    base = config.ollama_url.removesuffix("/v1")
    try:
        async with session.get(f"{base}/api/tags") as resp:
            models = [m["name"] for m in (await resp.json()).get("models", [])]
    except Exception:
        return Check("Offline brain", WARN, "Ollama isn't running", "Open the Ollama app, or unset OLLAMA_MODEL.")
    if config.ollama_model not in models:
        return Check("Offline brain", WARN, f"{config.ollama_model} not downloaded",
                     f"ollama pull {config.ollama_model}")
    return Check("Offline brain", OK, f"{config.ollama_model} ready")


# ---------------------------------------------------------------- main

async def run_doctor(config: Config, skip_mic: bool = False) -> int:
    print("\nJARVIS doctor\n")
    checks: list[Check] = []

    def add(check: Check):
        checks.append(check)
        _print(check)

    print("Machine")
    for c in check_machine() + check_packages():
        add(c)

    print("\nAudio")
    if not skip_mic:
        add(check_microphone())
    add(check_wake_word())
    wav, tts_secs = None, None
    try:
        samples, rate, load, tts_secs = synthesize(config)
        wav = to_wav(samples, rate)
        audio_secs = len(samples) / rate
        add(Check("Voice (Kokoro)", OK,
                  f"'{config.voice}' loaded in {load:.1f}s; {audio_secs:.1f}s of speech made in "
                  f"{tts_secs * 1000:.0f} ms"))
        tts_first = tts_secs * min(1.0, 1.5 / max(audio_secs, 0.1))  # first sentence ≈ first chunk
    except Exception as e:
        add(Check("Voice (Kokoro)", FAIL, str(e), "pip install -e . (downloads ~300 MB once)"))
        tts_first = None

    print("\nCloud (free tiers)")
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        gemini, gemini_ttft = await check_gemini(session, config)
        add(gemini)
        groq, groq_ttft = await check_groq_llm(session, config)
        add(groq)
        stt, stt_secs = await check_groq_stt(session, config, wav)
        add(stt)
        add(await check_ollama(session, config))
    if not config.brains():
        add(Check("Brain", FAIL, "no provider configured",
                  "Set GEMINI_API_KEY and/or GROQ_API_KEY in .env (both free)."))

    brain_ttft = gemini_ttft or groq_ttft
    print("\nEstimated response time (you stop talking → Jarvis starts talking)")
    if stt_secs and brain_ttft and tts_first:
        total = VAD_STOP_SECS + stt_secs + brain_ttft + tts_first
        print(f"   end of speech {VAD_STOP_SECS:.2f}s + speech-to-text {stt_secs:.2f}s + "
              f"brain {brain_ttft:.2f}s + voice {tts_first:.2f}s ≈ \033[1m{total:.1f}s\033[0m")
        print("   Instant commands (open app, volume, music, time) skip the brain: "
              f"≈ {VAD_STOP_SECS + stt_secs + tts_first:.1f}s")
    else:
        print("   Not enough checks passed to estimate yet.")

    failed = [c for c in checks if c.status == FAIL]
    print(f"\n{'All good. Run: jarvis' if not failed else f'{len(failed)} problem(s) to fix above.'}\n")
    return 1 if failed else 0


def main(skip_mic: bool = False) -> int:
    return asyncio.run(run_doctor(Config.load(), skip_mic=skip_mic))
