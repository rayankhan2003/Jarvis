"""Timers and alarms that run on the Mac and cost no API requests.

They live inside the running Jarvis process: when one is due Jarvis says so,
shows a notification and plays a sound. They don't survive quitting Jarvis;
for anything that must, ask for a reminder instead (it goes to Reminders).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger

from jarvis.tools.system import Result, applescript_string, osascript, run

ALERT_SOUND = "/System/Library/Sounds/Glass.aiff"

WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty five": 25, "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60,
    "ninety": 90,
}
UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def parse_duration(text: str) -> int | None:
    """Seconds in phrases like "10 minutes", "1 hour 30 minutes", "half an hour", "five mins"."""
    t = text.lower().strip()
    if re.fullmatch(r"(?:a |an )?half (?:an |a )?hour", t) or t == "half hour":
        return 1800
    if re.fullmatch(r"(?:a |an )?quarter (?:of an |an )?hour", t):
        return 900
    numbers = "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
    pattern = rf"(\d+(?:\.\d+)?|{numbers})( and a half)? ?(second|sec|minute|min|hour|hr)s?"
    total = 0.0
    matched = ""
    for m in re.finditer(pattern, t):
        number = float(m.group(1)) if m.group(1)[0].isdigit() else WORD_NUMBERS[m.group(1)]
        if m.group(2):
            number += 0.5
        total += number * UNIT_SECONDS[m.group(3)]
        matched += m.group(0)
    # Everything except the matched parts and "and" must be gone, or it wasn't a duration.
    leftover = re.sub(pattern, "", t).replace("and", "").strip()
    if total <= 0 or leftover:
        return None
    return int(total)


def parse_clock(text: str, now: datetime) -> datetime | None:
    """Next occurrence of a time like "7", "7 30", "7 30 am", "19 00", "730 pm"."""
    t = text.lower().strip().replace(".", " ").replace(":", " ")
    t = re.sub(r"\b(?:in the morning|morning)\b", "am", t)
    t = re.sub(r"\b(?:in the evening|at night|tonight|evening)\b", "pm", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = re.fullmatch(r"(\d{1,2})(?: ?(\d{2}))? ?(a ?m|p ?m)?(?: o ?clock)?", t)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    meridiem = (m.group(3) or "").replace(" ", "")
    if hour > 23 or minute > 59 or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    candidates = [hour] if meridiem or hour > 12 or hour == 0 else [hour, (hour + 12) % 24]
    options = []
    for h in candidates:
        when = now.replace(hour=h, minute=minute, second=0, microsecond=0)
        if when <= now:
            when += timedelta(days=1)
        options.append(when)
    return min(options)


def spoken_duration(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs and not hours:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


@dataclass
class _Entry:
    label: str
    due: float  # time.monotonic()
    task: asyncio.Task = field(repr=False)


class Timers:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._entries: list[_Entry] = []
        self.on_fire: Callable[[str], Awaitable[None]] | None = None

    def pending(self) -> list[_Entry]:
        self._entries = [e for e in self._entries if not e.task.done()]
        return sorted(self._entries, key=lambda e: e.due)

    def _schedule(self, seconds: float, label: str, spoken: str) -> _Entry:
        task = asyncio.create_task(self._fire_after(seconds, spoken))
        entry = _Entry(label, self._clock() + seconds, task)
        self._entries.append(entry)
        return entry

    async def _fire_after(self, seconds: float, spoken: str):
        await asyncio.sleep(seconds)
        logger.info(f"Timer done: {spoken}")
        await alert(spoken)
        if self.on_fire:
            await self.on_fire(spoken)

    async def start_timer(self, seconds: int) -> Result:
        length = spoken_duration(seconds)
        self._schedule(seconds, f"{length} timer", f"Your {length} timer is done.")
        return Result(True, f"Timer set for {length}.", {"seconds": seconds})

    async def set_alarm(self, when: datetime, now: datetime | None = None) -> Result:
        now = now or datetime.now()
        seconds = (when - now).total_seconds()
        at = when.strftime("%-I:%M %p").lower()
        self._schedule(seconds, f"{at} alarm", f"It's {at}. This is your alarm.")
        day = "" if when.date() == now.date() else " tomorrow"
        return Result(True, f"Alarm set for {at}{day}.", {"at": when.isoformat()})

    async def cancel(self) -> Result:
        pending = self.pending()
        for entry in pending:
            entry.task.cancel()
        self._entries = []
        if not pending:
            return Result(True, "There's nothing to cancel.")
        return Result(True, "Cancelled." if len(pending) == 1 else f"Cancelled all {len(pending)}.")

    async def status(self) -> Result:
        pending = self.pending()
        if not pending:
            return Result(True, "No timers running.")
        nxt = pending[0]
        left = spoken_duration(max(1, int(nxt.due - self._clock())))
        more = f" and {len(pending) - 1} more" if len(pending) > 1 else ""
        return Result(True, f"{left} left on the {nxt.label}{more}.")


async def alert(message: str):
    """Notification plus a sound, so a timer is noticed even when Jarvis is muted."""
    await osascript(f"display notification {applescript_string(message)} with title \"Jarvis\"")
    await run("afplay", ALERT_SOUND)


TIMERS = Timers()
