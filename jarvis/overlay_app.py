"""The cursor buddy's window: a transparent, click-through overlay on every screen.

Runs as its own small process (``python -m jarvis.overlay_app``) so drawing at
60 fps never stalls Jarvis's audio. Jarvis sends it one JSON command per line
on stdin:

    {"cmd": "status", "mode": "listening"}      idle | listening | thinking | speaking
    {"cmd": "point", "x": 640, "y": 40, "label": "Export"}   top-left screen points

All behaviour lives in ``buddy.py``; this file only draws what it returns.
macOS only (PyObjC).
"""

from __future__ import annotations

import json
import sys
import threading
import time

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSAttributedString,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSCompositingOperationClear,
    NSEvent,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSRectFillUsingOperation,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect, NSObject, NSTimer
from PyObjCTools import AppHelper

from jarvis.buddy import Buddy, top_left_to_cocoa

FPS = 60


def _color(rgba, alpha_scale: float = 1.0):
    r, g, b, a = rgba
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a * alpha_scale)


def _circle(x: float, y: float, radius: float):
    return NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - radius, y - radius, radius * 2, radius * 2))


class BuddyView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(BuddyView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.buddy_frame = None
        self.screen_origin = (frame.origin.x, frame.origin.y)
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        NSRectFillUsingOperation(rect, NSCompositingOperationClear)
        f = self.buddy_frame
        if f is None:
            return
        # Global Cocoa coordinates -> this screen's view coordinates.
        x, y = f.x - self.screen_origin[0], f.y - self.screen_origin[1]
        size = self.bounds().size
        if not (-60 <= x <= size.width + 60 and -60 <= y <= size.height + 60):
            return

        _color(f.color, 0.25).set()
        _circle(x, y, f.radius * 2.2).fill()  # soft glow
        _color(f.color).set()
        _circle(x, y, f.radius).fill()

        if f.ring_radius:
            _color(f.color, f.ring_alpha / max(f.color[3], 0.01)).set()
            ring = _circle(x, y, f.ring_radius)
            ring.setLineWidth_(3)
            ring.stroke()

        if f.label:
            text = NSAttributedString.alloc().initWithString_attributes_(
                f.label,
                {NSFontAttributeName: NSFont.boldSystemFontOfSize_(13),
                 NSForegroundColorAttributeName: NSColor.whiteColor()},
            )
            w, h = text.size().width, text.size().height
            lx, ly = x + f.ring_radius + 8, y - h / 2
            NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.82).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(lx - 8, ly - 4, w + 16, h + 8), 9, 9).fill()
            text.drawAtPoint_((lx, ly))


class Controller(NSObject):
    def setup(self):
        self.buddy = Buddy()
        self.views = []
        self.windows = []
        behaviour = (NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                     | NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorIgnoresCycle)
        for screen in NSScreen.screens():
            frame = screen.frame()
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
            window.setOpaque_(False)
            window.setBackgroundColor_(NSColor.clearColor())
            window.setIgnoresMouseEvents_(True)  # clicks go straight through
            window.setHasShadow_(False)
            window.setLevel_(NSScreenSaverWindowLevel)
            window.setCollectionBehavior_(behaviour)
            view = BuddyView.alloc().initWithFrame_(frame)
            window.setContentView_(view)
            window.orderFrontRegardless()  # shown without taking focus
            self.windows.append(window)
            self.views.append(view)
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1 / FPS, self, "tick:", None, True)
        return self

    def tick_(self, _timer):
        mouse = NSEvent.mouseLocation()
        frame = self.buddy.frame((mouse.x, mouse.y), time.monotonic())
        for view in self.views:
            view.buddy_frame = frame
            view.setNeedsDisplay_(True)

    @objc.python_method  # plain Python method: not exposed to Objective-C
    def handle(self, command: dict):
        if command.get("cmd") == "status":
            self.buddy.set_mode(command.get("mode", "idle"))
        elif command.get("cmd") == "point":
            main_height = NSScreen.screens()[0].frame().size.height
            target = top_left_to_cocoa(float(command["x"]), float(command["y"]), main_height)
            self.buddy.point(target, str(command.get("label", ""))[:40], time.monotonic())


def _read_commands(controller: Controller):
    for line in sys.stdin:
        try:
            command = json.loads(line)
        except ValueError:
            continue
        AppHelper.callAfter(controller.handle, command)
    # Jarvis exited: close the overlay too.
    AppHelper.callAfter(NSApplication.sharedApplication().terminate_, None)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
    controller = Controller.alloc().init().setup()
    threading.Thread(target=_read_commands, args=(controller,), daemon=True).start()
    AppHelper.runEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
