"""The cursor buddy's behaviour, with no macOS code so it can be tested anywhere.

The buddy is a small glowing dot that follows the mouse and shows what Jarvis
is doing (listening, thinking, speaking). When Jarvis points at something it
flies there on a curved path, pulses with a label, then comes back. The
overlay app (``overlay_app.py``) only draws what ``Buddy.frame`` returns.

Coordinates are macOS "Cocoa" screen points: origin at the bottom-left of the
main display, y going up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FOLLOW_OFFSET = (18.0, -22.0)  # sit just below-right of the cursor tip
FOLLOW_EASE = 0.35  # fraction of the remaining distance covered per frame
FLIGHT_SECS = 0.65
POINT_HOLD_SECS = 4.0
ARC_LIFT = 0.25  # how high the flight curves, as a fraction of its length

COLORS = {  # RGBA
    "idle": (0.55, 0.62, 0.78, 0.55),
    "listening": (0.20, 0.85, 0.45, 0.95),
    "thinking": (1.00, 0.72, 0.20, 0.95),
    "speaking": (0.30, 0.60, 1.00, 0.95),
    "pointing": (0.30, 0.60, 1.00, 1.00),
}
RADIUS = {"idle": 5.0, "listening": 8.0, "thinking": 7.0, "speaking": 8.0, "pointing": 9.0}


def top_left_to_cocoa(x: float, y: float, main_screen_height: float) -> tuple[float, float]:
    """Accessibility and screenshots measure from the top-left; Cocoa from the bottom-left."""
    return x, main_screen_height - y


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def arc_point(start: tuple[float, float], end: tuple[float, float], t: float) -> tuple[float, float]:
    """Point at progress ``t`` along a curved flight from ``start`` to ``end``."""
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    # Control point above the midpoint, so the buddy swoops rather than slides.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + length * ARC_LIFT
    t = _ease_in_out(t)
    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
    return x, y


@dataclass
class Frame:
    x: float
    y: float
    radius: float
    color: tuple[float, float, float, float]
    ring_radius: float = 0.0  # pulsing ring around a target; 0 = none
    ring_alpha: float = 0.0
    label: str = ""


class Buddy:
    def __init__(self):
        self.mode = "idle"
        self.pos: tuple[float, float] | None = None
        self._flight: tuple[tuple[float, float], tuple[float, float], float] | None = None  # start, end, t0
        self._label = ""
        self._hold_until = 0.0

    def set_mode(self, mode: str):
        if mode in COLORS and mode != "pointing":
            self.mode = mode

    def point(self, target: tuple[float, float], label: str, now: float):
        start = self.pos or target
        self._flight = (start, target, now)
        self._label = label
        self._hold_until = now + FLIGHT_SECS + POINT_HOLD_SECS

    @property
    def pointing(self) -> bool:
        return self._flight is not None

    def frame(self, mouse: tuple[float, float], now: float) -> Frame:
        if self._flight and now >= self._hold_until:
            self._flight = None  # done pointing: drift back to the cursor

        if self._flight:
            start, end, t0 = self._flight
            progress = (now - t0) / FLIGHT_SECS
            self.pos = arc_point(start, end, progress)
            arrived = progress >= 1
            pulse = (math.sin((now - t0) * 6) + 1) / 2
            return Frame(
                *self.pos,
                radius=RADIUS["pointing"],
                color=COLORS["pointing"],
                ring_radius=(18 + 10 * pulse) if arrived else 0.0,
                ring_alpha=(0.6 - 0.4 * pulse) if arrived else 0.0,
                label=self._label if arrived else "",
            )

        home = (mouse[0] + FOLLOW_OFFSET[0], mouse[1] + FOLLOW_OFFSET[1])
        if self.pos is None:
            self.pos = home
        else:
            x, y = self.pos
            self.pos = (x + (home[0] - x) * FOLLOW_EASE, y + (home[1] - y) * FOLLOW_EASE)

        radius = RADIUS[self.mode]
        if self.mode in {"listening", "speaking"}:
            radius += 2 * math.sin(now * 8)  # gentle breathing while talking
        elif self.mode == "thinking":
            radius += 1.5 * math.sin(now * 14)
        return Frame(*self.pos, radius=radius, color=COLORS[self.mode])
