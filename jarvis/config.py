"""Runtime configuration, read from environment variables (and a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

HOME_DIR = Path(os.path.expanduser("~/.jarvis"))


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


@dataclass
class Config:
    # Brain: tried in `brain_order`; providers without a key are skipped.
    brain_order: tuple[str, ...] = ("mistral", "groq", "gemini", "ollama")
    mistral_api_key: str = ""
    # Both work on the free Experiment plan; mistral-large-latest does not.
    mistral_model: str = "ministral-8b-latest"  # everyday requests
    mistral_complex_model: str = "ministral-14b-latest"  # multi-step plans and long requests
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    groq_api_key: str = ""
    # The big brain: hard questions and multi-step plans go here (free on Groq).
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_reasoning: str = "low"  # how long it thinks first: low, medium, high
    smart_brain: bool = True  # send complex requests to Groq even when Mistral comes first
    ollama_model: str = ""  # e.g. "qwen3.5:4b"; empty disables the offline brain
    ollama_url: str = "http://localhost:11434/v1"

    # Ears
    groq_stt_model: str = "whisper-large-v3-turbo"
    local_stt: bool = False  # MLX Whisper on Apple Silicon instead of Groq

    # Voice
    voice: str = "bm_george"  # Kokoro British male voices: bm_george, bm_lewis, bm_daniel, bm_fable

    # Wake word
    always_listen: bool = False
    interruptions: bool = False  # talk over Jarvis; needs headphones
    wake_threshold: float = 0.5
    awake_secs: float = 12.0

    # Persona and what Jarvis knows about you
    user_name: str = ""
    honorific: str = "sir"
    city: str = ""  # default for the weather, e.g. "Peshawar"
    music_taste: tuple[str, ...] = ()  # what "play some music" picks from
    browser: str = ""  # default browser for searches and music, e.g. "Brave"

    @classmethod
    def load(cls) -> Config:
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(HOME_DIR / ".env")
        return cls(
            brain_order=tuple(
                n.strip() for n in _env("JARVIS_BRAIN_ORDER", ",".join(cls.brain_order)).split(",") if n.strip()
            ),
            mistral_api_key=_env("MISTRAL_API_KEY"),
            mistral_model=_env("MISTRAL_MODEL", cls.mistral_model),
            mistral_complex_model=_env("MISTRAL_COMPLEX_MODEL", cls.mistral_complex_model),
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_model=_env("GEMINI_MODEL", cls.gemini_model),
            groq_api_key=_env("GROQ_API_KEY"),
            groq_llm_model=_env("GROQ_LLM_MODEL", cls.groq_llm_model),
            groq_reasoning=_env("GROQ_REASONING", cls.groq_reasoning),
            smart_brain=_env_bool("JARVIS_SMART_BRAIN", True),
            ollama_model=_env("OLLAMA_MODEL"),
            ollama_url=_env("OLLAMA_URL", cls.ollama_url),
            groq_stt_model=_env("GROQ_STT_MODEL", cls.groq_stt_model),
            local_stt=_env_bool("JARVIS_LOCAL_STT", False),
            voice=_env("JARVIS_VOICE", cls.voice),
            always_listen=_env_bool("JARVIS_ALWAYS_LISTEN", False),
            interruptions=_env_bool("JARVIS_INTERRUPTIONS", False),
            wake_threshold=_env_float("JARVIS_WAKE_THRESHOLD", cls.wake_threshold),
            awake_secs=_env_float("JARVIS_AWAKE_SECS", cls.awake_secs),
            user_name=_env("JARVIS_USER_NAME"),
            honorific=_env("JARVIS_HONORIFIC", cls.honorific),
            city=_env("JARVIS_CITY"),
            music_taste=tuple(t.strip() for t in _env("JARVIS_MUSIC_TASTE").split(",") if t.strip()),
            browser=_env("JARVIS_BROWSER"),
        )

    def brains(self) -> list[str]:
        """Names of the configured LLM providers, in failover order."""
        configured = {
            "mistral": bool(self.mistral_api_key),
            "groq": bool(self.groq_api_key),
            "gemini": bool(self.gemini_api_key),
            "ollama": bool(self.ollama_model),
        }
        return [name for name in dict.fromkeys(self.brain_order) if configured.get(name)]
