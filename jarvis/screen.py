"""Seeing the screen and pointing at things, like Clicky, on free providers.

1. Take a screenshot (built-in ``screencapture``), shrunk to the screen's own
   shape so the model sees undistorted proportions.
2. A vision model answers the question and, if it helps, names the element to
   click: ``[POINT:Export@640,40]`` (label, then its rough position).
3. Find the element exactly, cheapest first:
     a. the Mac's accessibility tree (exact, ~0.1 s, no AI)
     b. Apple's built-in text recognition on the screenshot (no AI)
     c. the model's rough position, as a last resort
4. The cursor buddy flies there.

Pure logic (parsing, matching, coordinate maths) is separate from the macOS
calls, so it can be tested anywhere.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import io
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from loguru import logger

from jarvis.tools.system import Result, osascript, run

CONFIG = None  # set by the bot at startup; the instant command uses it

MAX_IMAGE_WIDTH = 1280  # 16:10 MacBook screens -> 1280x800, which vision models handle well
MATCH_THRESHOLD = 0.62
AX_MAX_ELEMENTS = 2500
AX_MAX_SECONDS = 0.8

PROMPT = """You are looking at a screenshot of the user's Mac ({width}x{height} pixels). \
Answer their question in one or two short sentences that will be spoken aloud: no markdown, no lists, \
no URLs. If showing them where to click would help, end with [POINT:<the element's visible text or \
name>@<x>,<y>] where x,y is the element's centre in screenshot pixels. Otherwise end with [POINT:none].

Question: {question}"""

POINT_TAG = re.compile(r"\[POINT:(?P<body>[^\]]*)\]\s*$", re.I)


# --- Pure logic ---------------------------------------------------------------

@dataclass
class Element:
    label: str
    x: float  # top-left screen points
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def parse_point(reply: str) -> tuple[str, str | None, tuple[float, float] | None]:
    """Split a vision reply into (spoken text, label to point at, rough position in image pixels).

    Accepts our format ``[POINT:Export@640,40]`` and Clicky's ``[POINT:640,40:Export]``.
    """
    match = POINT_TAG.search(reply.strip())
    if not match:
        return reply.strip(), None, None
    text = reply.strip()[: match.start()].strip()
    body = match.group("body").strip()
    if not body or body.lower() == "none":
        return text, None, None
    if m := re.fullmatch(r"(?P<label>.+?)@\s*(?P<x>\d+(?:\.\d+)?)\s*,\s*(?P<y>\d+(?:\.\d+)?)", body):
        return text, m.group("label").strip(), (float(m.group("x")), float(m.group("y")))
    if m := re.fullmatch(r"(?P<x>\d+(?:\.\d+)?)\s*,\s*(?P<y>\d+(?:\.\d+)?)\s*:\s*(?P<label>[^:]+?)(?::screen\d+)?", body):
        return text, m.group("label").strip(), (float(m.group("x")), float(m.group("y")))
    return text, body, None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def match_score(query: str, candidate: str) -> float:
    """How well an on-screen label matches what the model asked for (0 to 1)."""
    q, c = _norm(query), _norm(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if len(c) <= 60 and (q in c.split() or f" {q} " in f" {c} "):
        return 0.9 - min(0.2, (len(c) - len(q)) / 200)  # whole-word containment, prefer short labels
    if len(q) >= 4 and c.startswith(q):
        return 0.8
    # Loose similarity counts for less: "Import" must not pass for "Export",
    # but a small slip ("Exprot") still should.
    return difflib.SequenceMatcher(None, q, c).ratio() * 0.85


def best_match(query: str, elements: list[Element], hint: tuple[float, float] | None = None) -> Element | None:
    """The element that best matches ``query``; ties go to the one nearest the model's hint."""
    scored = [(match_score(query, e.label), e) for e in elements if e.w > 0 and e.h > 0]
    scored = [(s, e) for s, e in scored if s >= MATCH_THRESHOLD]
    if not scored:
        return None
    top = max(s for s, _ in scored)
    contenders = [e for s, e in scored if s >= top - 0.05]
    if hint and len(contenders) > 1:
        return min(contenders, key=lambda e: (e.center[0] - hint[0]) ** 2 + (e.center[1] - hint[1]) ** 2)
    return contenders[0]


def fit_size(width: int, height: int, max_width: int = MAX_IMAGE_WIDTH) -> tuple[int, int]:
    """Shrink to ``max_width`` keeping the screen's aspect ratio (never enlarge)."""
    if width <= max_width:
        return width, height
    return max_width, round(height * max_width / width)


def image_to_points(x: float, y: float, image_size: tuple[int, int], screen_points: tuple[float, float]):
    """Screenshot pixels (as the model saw them) -> top-left screen points."""
    return x * screen_points[0] / image_size[0], y * screen_points[1] / image_size[1]


# --- macOS --------------------------------------------------------------------

@dataclass
class Shot:
    path: Path  # full-resolution PNG (for text recognition)
    jpeg: bytes  # shrunk copy for the vision model
    image_size: tuple[int, int]  # size of the shrunk copy
    full_size: tuple[int, int]  # size of the PNG in pixels
    screen_points: tuple[float, float]  # main screen size in points


async def screen_points() -> tuple[float, float] | None:
    code, out, _ = await osascript('tell application "Finder" to get bounds of window of desktop')
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", out)] if code == 0 else []
    return (nums[2], nums[3]) if len(nums) == 4 else None


async def capture() -> Shot | None:
    from PIL import Image

    path = Path(tempfile.gettempdir()) / f"jarvis-screen-{int(time.time() * 1000)}.png"
    code, _, err = await run("screencapture", "-x", "-m", "-t", "png", str(path))  # main display, silent
    if code or not path.exists():
        logger.warning(f"Screenshot failed: {err}")
        return None
    points = await screen_points()
    with Image.open(path) as img:
        full = img.size
        small_size = fit_size(*full)
        small = img.convert("RGB").resize(small_size, Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=85)
    if points is None:  # Retina screens are 2x; a reasonable guess
        points = (full[0] / 2, full[1] / 2)
    return Shot(path, buf.getvalue(), small_size, full, points)


def accessibility_elements() -> list[Element]:
    """Buttons, menus, fields... of the frontmost app, from the accessibility tree (exact positions)."""
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXIsProcessTrusted,
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
            AXValueGetValue,
            kAXValueTypeCGPoint,
            kAXValueTypeCGSize,
        )
    except ImportError:
        return []
    if not AXIsProcessTrusted():
        return []
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return []

    def attr(element, name):
        err, value = AXUIElementCopyAttributeValue(element, name, None)
        return value if err == 0 else None

    elements: list[Element] = []
    queue = [AXUIElementCreateApplication(app.processIdentifier())]
    deadline = time.monotonic() + AX_MAX_SECONDS
    seen = 0
    while queue and seen < AX_MAX_ELEMENTS and time.monotonic() < deadline:
        element = queue.pop(0)
        seen += 1
        label = next((str(v) for n in ("AXTitle", "AXDescription", "AXValue", "AXHelp")
                      if isinstance(v := attr(element, n), str) and v.strip()), "")
        position, size = attr(element, "AXPosition"), attr(element, "AXSize")
        if label and position is not None and size is not None:
            ok_p, point = AXValueGetValue(position, kAXValueTypeCGPoint, None)
            ok_s, dims = AXValueGetValue(size, kAXValueTypeCGSize, None)
            if ok_p and ok_s:  # index rather than .x/.width: works for structs and plain tuples
                elements.append(Element(label[:120], point[0], point[1], dims[0], dims[1]))
        queue.extend(attr(element, "AXChildren") or [])
    return elements


def text_elements(shot: Shot) -> list[Element]:
    """Words on screen with their positions, from Apple's built-in text recognition."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return []
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(shot.path)), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    sw, sh = shot.screen_points
    elements = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        (bx, by), (bw, bh) = observation.boundingBox()  # 0-1 of the image, origin bottom-left
        elements.append(Element(str(candidates[0].string()), bx * sw, (1 - by - bh) * sh, bw * sw, bh * sh))
    return elements


async def locate(label: str, hint_px: tuple[float, float] | None, shot: Shot) -> tuple[float, float, str] | None:
    """Where ``label`` is on screen, in top-left points, and how it was found."""
    hint = image_to_points(*hint_px, shot.image_size, shot.screen_points) if hint_px else None
    for source, finder in (("accessibility", accessibility_elements), ("text", lambda: text_elements(shot))):
        try:
            elements = await asyncio.to_thread(finder)
        except Exception as e:
            logger.debug(f"{source} lookup failed: {e}")
            continue
        if found := best_match(label, elements, hint):
            return (*found.center, source)
    if hint:
        return (*hint, "estimate")
    return None


# --- Vision models ------------------------------------------------------------

async def _ask_mistral(session, key: str, model: str, prompt: str, jpeg_b64: str) -> str:
    body = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{jpeg_b64}"},
    ]}]}
    async with session.post("https://api.mistral.ai/v1/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {key}"}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Mistral {resp.status}: {(await resp.text())[:160]}")
        return (await resp.json())["choices"][0]["message"]["content"]


async def _ask_gemini(session, key: str, model: str, prompt: str, jpeg_b64: str) -> str:
    body = {"contents": [{"role": "user", "parts": [
        {"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64}},
    ]}], "generationConfig": {"maxOutputTokens": 300}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    async with session.post(url, json=body, headers={"x-goog-api-key": key}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Gemini {resp.status}: {(await resp.text())[:160]}")
        data = await resp.json()
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])


async def _ask_ollama(session, url: str, model: str, prompt: str, jpeg_b64: str) -> str:
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt, "images": [jpeg_b64]}]}
    async with session.post(f"{url.removesuffix('/v1')}/api/chat", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Ollama {resp.status}: {(await resp.text())[:160]}")
        return (await resp.json())["message"]["content"]


def vision_providers(config) -> list[tuple[str, callable]]:
    """Configured vision models, in the order to try them."""
    available = {
        "mistral": (config.mistral_api_key and config.mistral_vision_model,
                    lambda s, p, i: _ask_mistral(s, config.mistral_api_key, config.mistral_vision_model, p, i)),
        "gemini": (config.gemini_api_key,
                   lambda s, p, i: _ask_gemini(s, config.gemini_api_key, config.gemini_model, p, i)),
        "ollama": (config.ollama_vision_model,
                   lambda s, p, i: _ask_ollama(s, config.ollama_url, config.ollama_vision_model, p, i)),
    }
    return [(name, available[name][1]) for name in config.vision_order if name in available and available[name][0]]


async def ask_vision(config, question: str, jpeg: bytes, image_size: tuple[int, int]) -> str:
    prompt = PROMPT.format(width=image_size[0], height=image_size[1], question=question)
    image = base64.b64encode(jpeg).decode()
    errors = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=25)) as session:
        for name, ask in vision_providers(config):
            try:
                reply = await ask(session, prompt, image)
                logger.info(f"Screen answered by {name}")
                return reply
            except Exception as e:
                logger.warning(f"Vision via {name} failed: {e}")
                errors.append(name)
    raise RuntimeError(f"no vision provider worked ({', '.join(errors) or 'none configured'})")


# --- Putting it together -----------------------------------------------------

async def look(config, question: str, point_at=None) -> Result:
    """Answer a question about the screen; point at the answer if that helps.

    ``point_at(x, y, label)`` is called with top-left screen points.
    """
    if not vision_providers(config):
        return Result(False, "I need a vision model for that. Add a Gemini or Mistral key, or a local one.")
    shot = await capture()
    if shot is None:
        return Result(False, "I couldn't see the screen. Allow Screen Recording for your terminal in System Settings.")
    try:
        try:
            reply = await ask_vision(config, question, shot.jpeg, shot.image_size)
        except Exception as e:
            return Result(False, f"I couldn't read the screen just now: {e}.")
        text, label, hint = parse_point(reply)
        data: dict = {"answer": text}
        if label:
            found = await locate(label, hint, shot)
            if found:
                x, y, source = found
                data.update(point={"x": round(x), "y": round(y), "label": label, "found_by": source})
                logger.info(f"Pointing at {label!r} ({source})")
                if point_at:
                    point_at(x, y, label)
        return Result(True, text or "Here.", data)
    finally:
        shot.path.unlink(missing_ok=True)
