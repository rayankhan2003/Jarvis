"""Jarvis's side of the cursor buddy: starts the overlay process and sends it commands.

Every method is safe to call even when the overlay isn't running (not on a
Mac, PyObjC missing, or the buddy turned off): it just does nothing.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def overlay_supported() -> bool:
    return sys.platform == "darwin" and importlib.util.find_spec("AppKit") is not None


class Overlay:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._mode = ""

    def start(self) -> bool:
        if not overlay_supported():
            logger.info("Cursor buddy needs macOS with PyObjC; running without it")
            return False
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "jarvis.overlay_app"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError as e:
            logger.warning(f"Couldn't start the cursor buddy: {e}")
            return False
        return True

    def _send(self, command: dict):
        if not self._proc or self._proc.poll() is not None:
            return
        try:
            self._proc.stdin.write(json.dumps(command) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            self._proc = None

    def status(self, mode: str):
        if mode != self._mode:  # only send changes
            self._mode = mode
            self._send({"cmd": "status", "mode": mode})

    def point(self, x: float, y: float, label: str = ""):
        """Fly to (x, y) in top-left screen points and show ``label``."""
        self._send({"cmd": "point", "x": round(x, 1), "y": round(y, 1), "label": label})

    def stop(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None


OVERLAY = Overlay()


class OverlayStatus(FrameProcessor):
    """Turns the buddy blue while Jarvis speaks. Put it right after the audio output."""

    def __init__(self, overlay: Overlay):
        super().__init__()
        self._overlay = overlay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._overlay.status("speaking")
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._overlay.status("idle")
        await self.push_frame(frame, direction)
