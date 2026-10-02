"""What Jarvis remembers about you, kept in a small file on the Mac.

"Remember that my sister's name is Ayesha" saves a line here; the lines go
into Jarvis's instructions, so every brain knows them. Saving, listing and
forgetting cost no API request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path

from loguru import logger

from jarvis.config import HOME_DIR
from jarvis.tools.system import Result

MEMORY_FILE = HOME_DIR / "memory.json"
MAX_FACTS = 50  # keeps the instructions, and so every request, small
STOPWORDS = {"the", "a", "an", "my", "is", "that", "about", "to", "of", "i", "me", "and", "it", "what"}


def _words(text: str) -> set[str]:
    # "sister's" -> "sister": possessives shouldn't stop "forget my sister" matching.
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS}


class Memory:
    def __init__(self, path: Path = MEMORY_FILE):
        self._path = path
        self.on_change: Callable[[], Awaitable[None]] | None = None
        try:
            self._facts: list[dict] = json.loads(path.read_text())
        except (OSError, ValueError):
            self._facts = []

    @property
    def facts(self) -> list[str]:
        return [f["fact"] for f in self._facts]

    async def remember(self, fact: str) -> Result:
        fact = fact.strip().rstrip(".").strip()
        if not fact:
            return Result(False, "What should I remember?")
        fact = fact[0].upper() + fact[1:]
        # A newer fact about the same thing replaces the old one.
        new = _words(fact)
        self._facts = [f for f in self._facts if len(new & _words(f["fact"])) < max(2, len(new) * 0.6)]
        self._facts.append({"fact": fact, "saved": date.today().isoformat()})
        self._facts = self._facts[-MAX_FACTS:]
        await self._changed()
        return Result(True, "I'll remember that.", {"fact": fact})

    async def forget(self, about: str) -> Result:
        about = about.strip().lower()
        if about in {"everything", "it all", "all of it", "everything about me"}:
            count = len(self._facts)
            self._facts = []
            await self._changed()
            return Result(True, f"Forgotten, all {count} things." if count else "There was nothing to forget.")
        target = _words(about)
        keep = [f for f in self._facts if not (target and target <= _words(f["fact"]))]
        dropped = len(self._facts) - len(keep)
        if not dropped:
            return Result(True, "I don't have anything about that.")
        self._facts = keep
        await self._changed()
        return Result(True, "Forgotten." if dropped == 1 else f"Forgotten {dropped} things.")

    async def recall(self) -> Result:
        if not self._facts:
            return Result(True, "Nothing yet. Tell me to remember something.")
        facts = self.facts
        spoken = "; ".join(facts[-5:])
        more = f" And {len(facts) - 5} more." if len(facts) > 5 else ""
        return Result(True, f"I remember: {spoken}.{more}", {"facts": facts})

    def prompt_block(self) -> str:
        if not self._facts:
            return ""
        return "What you remember about the user:\n" + "\n".join(f"- {f}" for f in self.facts)

    async def _changed(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._facts, indent=1))
        except OSError as e:
            logger.warning(f"Couldn't save memory: {e}")
        if self.on_change:
            await self.on_change()


MEMORY = Memory()
