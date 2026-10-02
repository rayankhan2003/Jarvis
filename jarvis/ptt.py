"""Push-to-talk: hold a key, talk, let go. Like Clicky, and more reliable than a wake word.

A listen-only Quartz event tap watches one key in a background thread and
hands presses to the asyncio loop. It never translates keys into characters,
which is what crashes some key-listener libraries off the main thread on
recent macOS. macOS asks once for Input Monitoring permission for your
terminal app.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

FLAGS_CHANGED, KEY_DOWN, KEY_UP = "flags", "down", "up"


@dataclass(frozen=True)
class KeySpec:
    keycode: int  # macOS virtual key code
    modifier: bool  # modifiers arrive as "flags changed" events, not key down/up


# Right-hand modifiers are rarely used for anything else, so holding one is safe.
KEYS = {
    "right_option": KeySpec(61, True),
    "right_alt": KeySpec(61, True),
    "right_command": KeySpec(54, True),
    "right_cmd": KeySpec(54, True),
    "right_control": KeySpec(62, True),
    "right_ctrl": KeySpec(62, True),
    "right_shift": KeySpec(60, True),
    "f13": KeySpec(105, False),
    "f14": KeySpec(107, False),
    "f15": KeySpec(113, False),
}


def key_spec(name: str) -> KeySpec:
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    if name not in KEYS:
        raise ValueError(f"Unknown push-to-talk key {name!r}. Use one of: {', '.join(sorted(KEYS))}")
    return KEYS[name]


def next_state(down: bool, kind: str, keycode: int, spec: KeySpec) -> bool | None:
    """New held/not-held state after an event, or None if the event isn't about our key."""
    if keycode != spec.keycode:
        return None
    if spec.modifier:
        # Each press and each release of a modifier is one "flags changed" event;
        # toggling copes with the other side's modifier being held too.
        return (not down) if kind == FLAGS_CHANGED else None
    if kind == KEY_DOWN:
        return True
    if kind == KEY_UP:
        return False
    return None


class PushToTalk:
    """Calls ``on_press`` / ``on_release`` on the event loop when the key goes down / up."""

    def __init__(self, key: str, on_press: Callable[[], None], on_release: Callable[[], None]):
        self._spec = key_spec(key)
        self._key_name = key.replace("_", " ")
        self._on_press = on_press
        self._on_release = on_release
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_loop = None
        self._down = False

    def _handle(self, kind: str, keycode: int):
        new = next_state(self._down, kind, keycode, self._spec)
        if new is None or new == self._down:
            return
        self._down = new
        self._loop.call_soon_threadsafe(self._on_press if new else self._on_release)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> bool:
        try:
            import Quartz  # noqa: F401
        except ImportError:
            logger.warning("Push-to-talk needs macOS (PyObjC Quartz); use the wake word instead")
            return False
        self._loop = loop or asyncio.get_running_loop()
        ready = threading.Event()
        ok: list[bool] = []
        threading.Thread(target=self._run, args=(ready, ok), daemon=True, name="push-to-talk").start()
        ready.wait(timeout=3)
        if ok:
            logger.info(f"Push-to-talk: hold {self._key_name} and speak")
        return bool(ok)

    def _run(self, ready: threading.Event, ok: list):
        import Quartz
        from CoreFoundation import (
            CFMachPortCreateRunLoopSource,
            CFRunLoopAddSource,
            CFRunLoopGetCurrent,
            CFRunLoopRun,
            kCFRunLoopCommonModes,
        )

        kinds = {Quartz.kCGEventFlagsChanged: FLAGS_CHANGED, Quartz.kCGEventKeyDown: KEY_DOWN,
                 Quartz.kCGEventKeyUp: KEY_UP}
        tap_holder = []

        def callback(proxy, event_type, event, refcon):
            if event_type in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(tap_holder[0], True)  # macOS switched us off; switch back on
            elif event_type in kinds:
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                self._handle(kinds[event_type], keycode)
            return event

        mask = 0
        for event_type in kinds:
            mask |= Quartz.CGEventMaskBit(event_type)
        tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                      Quartz.kCGEventTapOptionListenOnly, mask, callback, None)
        if tap is None:
            logger.warning("Push-to-talk needs Input Monitoring: System Settings > Privacy & Security > "
                           "Input Monitoring > allow your terminal app, then restart Jarvis")
            ready.set()
            return
        tap_holder.append(tap)
        source = CFMachPortCreateRunLoopSource(None, tap, 0)
        self._run_loop = CFRunLoopGetCurrent()
        CFRunLoopAddSource(self._run_loop, source, kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        ok.append(True)
        ready.set()
        CFRunLoopRun()

    def stop(self):
        if self._run_loop is not None:
            from CoreFoundation import CFRunLoopStop

            CFRunLoopStop(self._run_loop)
            self._run_loop = None
