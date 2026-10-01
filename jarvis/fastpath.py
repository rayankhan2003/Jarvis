"""Instant commands: simple requests skip the LLM entirely.

"Open Spotify", "volume 40", "pause", "what time is it" are matched with plain
patterns and executed straight away, so they cost no LLM request and answer in
the time it takes to run the command. Anything that doesn't match, or a
command that fails, falls through to the brain.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from jarvis.tools import info, system
from jarvis.tools.system import Result

WAKE_PREFIX = re.compile(r"^(?:(?:hey|ok|okay|hi)\s+)?(?:jarvis|vis)\b[\s,]*")
POLITE = re.compile(r"\b(?:please|for me|thanks|thank you|now)\b")

NUMBER_WORDS = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "max": 100,
    "maximum": 100, "full": 100, "half": 50,
}
DISMISS = {"thats all", "thats it", "goodbye", "bye", "go to sleep", "sleep", "never mind",
           "nevermind", "stop listening", "dismissed"}


@dataclass
class Command:
    name: str
    run: Callable[[], Awaitable[Result]]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s%']", " ", text).replace("'", "")
    text = WAKE_PREFIX.sub("", text.strip())
    text = POLITE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _volume_level(word: str) -> int | None:
    word = word.removesuffix("%").removesuffix(" percent").strip()
    if word.isdigit():
        return int(word)
    return NUMBER_WORDS.get(word)


def match_command(text: str) -> Command | None:
    t = normalize(text)
    if not t:
        return None

    if t in DISMISS:
        return Command("dismiss", _ok("Very good."))

    if m := re.fullmatch(r"(?:open|launch|start)(?: up)? (?:the )?([\w .]{1,30}?)(?: app)?", t):
        target = m.group(1).strip()
        # "open the readme in my project" is a job for the brain, not `open -a`.
        if len(target.split()) <= 3 and not re.search(r"\b(?:file|folder|in|on|and|with|website|page)\b", target):
            return Command("open_app", lambda: system.open_app(target))

    if m := re.fullmatch(r"(?:close|quit|exit|kill) (?:the )?([\w .]{1,30}?)(?: app)?", t):
        target = m.group(1).strip()
        if len(target.split()) <= 3 and not re.search(r"\b(?:tab|window|file)\b", target):
            return Command("quit_app", lambda: system.quit_app(target))

    if m := re.fullmatch(r"(?:set (?:the )?)?volume (?:to |at )?(\w+(?:%| percent)?)", t):
        level = _volume_level(m.group(1))
        if level is not None:
            return Command("set_volume", lambda: system.set_volume(level))
    if re.fullmatch(r"(?:turn (?:the )?)?(?:volume up|turn it up|louder)", t):
        return Command("volume_up", lambda: system.change_volume(+15))
    if re.fullmatch(r"(?:turn (?:the )?)?(?:volume down|turn it down|quieter)", t):
        return Command("volume_down", lambda: system.change_volume(-15))
    if t in {"mute", "mute it", "mute the sound", "silence"}:
        return Command("mute", lambda: system.set_muted(True))
    if t in {"unmute", "unmute it", "sound on"}:
        return Command("unmute", lambda: system.set_muted(False))

    if t in {"pause", "pause music", "pause the music", "stop the music", "stop music"}:
        return Command("pause", lambda: system.media("pause"))
    if t in {"play", "resume", "play music", "resume music", "resume the music", "unpause"}:
        return Command("play", lambda: system.media("play"))
    if t in {"next", "skip", "next song", "next track", "skip this", "skip song", "skip this song"}:
        return Command("next", lambda: system.media("next"))
    if t in {"previous", "previous song", "previous track", "go back", "last song"}:
        return Command("previous", lambda: system.media("previous"))
    if t in {"whats playing", "what song is this", "whats this song", "who is this"}:
        return Command("now_playing", system.now_playing)

    if t in {"what time is it", "whats the time", "time", "the time", "tell me the time"}:
        return Command("time", _wrap(info.get_time))
    if t in {"battery", "whats my battery", "battery level", "how much battery", "battery status",
             "how much battery do i have", "how much battery is left"}:
        return Command("battery", system.battery)
    if t in {"lock", "lock screen", "lock the screen", "lock my mac", "lock the mac"}:
        return Command("lock", system.lock_screen)

    return None


def _ok(say: str) -> Callable[[], Awaitable[Result]]:
    async def run() -> Result:
        return Result(True, say)

    return run


def _wrap(fn: Callable[[], Result]) -> Callable[[], Awaitable[Result]]:
    async def run() -> Result:
        return fn()

    return run


class FastPath(FrameProcessor):
    """Sits between speech-to-text and the brain and handles instant commands.

    Args:
        context: The shared conversation, so the brain knows what was done.
        on_dismiss: Called when the user dismisses Jarvis ("that's all").
        on_user_turn: Called for every final transcript, handled here or not.
    """

    def __init__(
        self,
        context: LLMContext,
        on_dismiss: Callable[[], Awaitable[None]] | None = None,
        on_user_turn: Callable[[], Awaitable[None]] | None = None,
    ):
        super().__init__()
        self._context = context
        self._on_dismiss = on_dismiss
        self._on_user_turn = on_user_turn

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not (isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM):
            await self.push_frame(frame, direction)
            return

        if self._on_user_turn:
            await self._on_user_turn()

        command = match_command(frame.text)
        if command is None:
            await self.push_frame(frame, direction)
            return

        result = await command.run()
        logger.info(f"Instant command {command.name!r}: {result.say}")
        if not result.ok and command.name in {"open_app", "quit_app"}:
            # Maybe the brain can work out what was meant ("open my portfolio").
            await self.push_frame(frame, direction)
            return

        self._context.add_message({"role": "user", "content": frame.text})
        await self.push_frame(TTSSpeakFrame(result.say))
        if command.name == "dismiss" and self._on_dismiss:
            await self._on_dismiss()
