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

from jarvis import conversation
from jarvis.timers import TIMERS, parse_clock, parse_duration
from jarvis.tools import info, system
from jarvis.tools.system import Result
from jarvis.usage import Usage

WAKE_PREFIX = re.compile(r"^(?:(?:hey|ok|okay|hi)\s+)?(?:jarvis|vis)\b[\s,]*")
POLITE = re.compile(r"\b(?:please|for me|thanks|thank you|now)\b")

NUMBER_WORDS = {
    "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "max": 100,
    "maximum": 100, "full": 100, "half": 50,
}
BROWSER = r"(?:brave|chrome|google chrome|safari|firefox|arc|edge|opera)(?: browser)?"
SEARCH_VERB = r"(?:search|google|look up|find)(?: for)?"
NOT_A_WEB_SEARCH = re.compile(r"\b(?:file|files|folder|my mac|email|emails|message|messages)\b")
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

    if command := _match_timer(t) or _match_system(t):
        return command

    if command := _match_search(t):
        return command

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


def _match_timer(t: str) -> Command | None:
    if re.fullmatch(r"(?:cancel|stop|delete|clear|turn off)(?: the| my| all)? (?:timers?|alarms?)", t):
        return Command("cancel_timers", TIMERS.cancel)
    if re.fullmatch(r"(?:how (?:much time|long)(?: is)? left.*|whats left on the timer|timer status|check (?:the |my )?timer)", t):
        return Command("timer_status", TIMERS.status)
    for pattern in (
        r"(?:set |start )?(?:a |an )?timer (?:for |of )?(?P<d>.+)",
        r"(?:set |start )?(?:a |an )?(?P<d>.+?) timer",
    ):
        if (m := re.fullmatch(pattern, t)) and (seconds := parse_duration(m.group("d"))):
            return Command("timer", lambda seconds=seconds: TIMERS.start_timer(seconds))
    if m := re.fullmatch(r"(?:set (?:an |the |my )?alarm|wake me up|wake me)(?: for| at)? (?P<t>.+)", t):
        from datetime import datetime

        now = datetime.now()
        if when := parse_clock(m.group("t"), now):
            return Command("alarm", lambda: TIMERS.set_alarm(when, now))
    return None


def _match_system(t: str) -> Command | None:
    if re.fullmatch(r"(?:turn on |switch to |enable |go )?dark mode(?: on)?", t):
        return Command("dark_mode", lambda: system.set_dark_mode(True))
    if re.fullmatch(r"(?:turn off dark mode|dark mode off|disable dark mode|(?:turn on |switch to |go )?light mode)", t):
        return Command("light_mode", lambda: system.set_dark_mode(False))
    if re.fullmatch(r"(?:brighter|brightness up|(?:increase|raise|turn up)(?: the)? brightness|make (?:it|the screen) brighter)", t):
        return Command("brighter", lambda: system.change_brightness("up"))
    if re.fullmatch(r"(?:dimmer|dim(?: the)?(?: screen)?|brightness down|(?:decrease|lower|turn down)(?: the)? brightness|make (?:it|the screen) darker)", t):
        return Command("dimmer", lambda: system.change_brightness("down"))
    if re.fullmatch(r"(?:take a |take |grab a )?screenshot|screen ?shot", t):
        return Command("screenshot", system.screenshot)
    if re.fullmatch(r"(?:put (?:the |my )?mac to sleep|sleep (?:the |my )?mac|(?:mac|computer) sleep)", t):
        return Command("sleep_mac", system.sleep_mac)
    return None


# Speech-to-text often turns background noise into these.
NOISE = {"", "you", "uh", "um", "hmm", "mm", "mhm", "ah", "oh", "huh", "okay so", "so",
         "thanks for watching", "thank you for watching", "subscribe", "bye bye"}


def is_noise(text: str) -> bool:
    plain = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
    t = normalize(text)
    return plain in NOISE or t in NOISE or len(t) < 2


def _match_search(t: str) -> Command | None:
    """Browser searches: "open brave and search talha anjum", "search X on youtube", "google X"."""
    patterns = [
        # open brave and search X / brave search X
        (rf"(?:open |launch |go to )?(?:the )?(?P<browser>{BROWSER})(?: and)? {SEARCH_VERB} (?P<q>.+)", "google"),
        # open youtube and search X / youtube play X
        (rf"(?:open |go to )?youtube(?: and)? (?:{SEARCH_VERB}|play) (?P<q>.+?)(?: in (?:the )?(?P<browser>{BROWSER}))?", "youtube"),
        # search X on youtube (in brave) / play X on youtube
        (rf"(?:{SEARCH_VERB}|play) (?P<q>.+?) on youtube(?: in (?:the )?(?P<browser>{BROWSER}))?", "youtube"),
        # search X on/in brave
        (rf"{SEARCH_VERB} (?P<q>.+?) (?:on|in|using) (?:the )?(?P<browser>{BROWSER})", "google"),
        # search X / google X
        (rf"{SEARCH_VERB} (?P<q>.+)", "google"),
    ]
    for pattern, site in patterns:
        if m := re.fullmatch(pattern, t):
            query = m.group("q").strip()
            browser = (m.groupdict().get("browser") or "").removesuffix(" browser").strip()
            if not query or NOT_A_WEB_SEARCH.search(query) or re.fullmatch(BROWSER, query):
                return None
            return _search_command(query, browser, site)
    return None


def _search_command(query: str, browser: str, site: str) -> Command:
    return Command("search", lambda: system.search_in_browser(query, browser, site))


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
        before_brain: Returns frames to send ahead of a request the brain will
            handle, e.g. a model switch for a complex request.
        usage: Counts speech-to-text requests and the requests saved.
    """

    def __init__(
        self,
        context: LLMContext,
        on_dismiss: Callable[[], Awaitable[None]] | None = None,
        on_user_turn: Callable[[], Awaitable[None]] | None = None,
        before_brain: Callable[[str], list[Frame]] | None = None,
        usage: Usage | None = None,
    ):
        super().__init__()
        self._context = context
        self._on_dismiss = on_dismiss
        self._on_user_turn = on_user_turn
        self._before_brain = before_brain
        self._usage = usage

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not (isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM):
            await self.push_frame(frame, direction)
            return

        if self._usage:
            self._usage.record_stt()

        if is_noise(frame.text):
            logger.debug(f"Ignored noise: {frame.text!r}")
            if self._usage:
                self._usage.record_ignored()
            return

        if self._on_user_turn:
            await self._on_user_turn()

        command = match_command(frame.text)
        if command is None:
            await self._to_brain(frame)
            return

        result = await command.run()
        logger.info(f"Instant command {command.name!r}: {result.say}")
        if not result.ok and command.name in {"open_app", "quit_app"}:
            # Maybe the brain can work out what was meant ("open my portfolio").
            await self._to_brain(frame)
            return

        if self._usage:
            self._usage.record_instant()
        self._context.add_message({"role": "user", "content": frame.text})
        await self.push_frame(TTSSpeakFrame(result.say))
        if command.name == "dismiss" and self._on_dismiss:
            await self._on_dismiss()

    async def _to_brain(self, frame: TranscriptionFrame):
        conversation.trim(self._context)
        for extra in self._before_brain(frame.text) if self._before_brain else []:
            await self.push_frame(extra)
        await self.push_frame(frame)
