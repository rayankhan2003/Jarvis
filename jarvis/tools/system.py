"""Mac control: apps, volume, media, battery, screen.

Every function returns a ``Result`` and never raises, so both the instant
command path and the LLM can report failures honestly.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

APP_DIRS = [Path("/Applications"), Path("/System/Applications"), Path("~/Applications").expanduser()]

# What people say -> what macOS calls the app.
APP_ALIASES = {
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "settings": "System Settings",
    "system preferences": "System Settings",
    "music": "Music",
    "apple music": "Music",
    "files": "Finder",
    "finder": "Finder",
    "mail": "Mail",
    "messages": "Messages",
    "notes": "Notes",
    "calendar": "Calendar",
    "reminders": "Reminders",
    "terminal": "Terminal",
    "whatsapp": "WhatsApp",
    "brave": "Brave Browser",
    "brave browser": "Brave Browser",
    "firefox": "Firefox",
    "arc": "Arc",
    "edge": "Microsoft Edge",
    "opera": "Opera",
    "teams": "Microsoft Teams",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
}


@dataclass
class Result:
    ok: bool
    say: str  # what Jarvis can say about it
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "message": self.say, **self.data}


async def run(*cmd: str, timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a command without a shell. Returns (exit code, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not found"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        return 124, "", f"{cmd[0]} timed out after {timeout:.0f}s"
    return proc.returncode or 0, out.decode().strip(), err.decode().strip()


async def osascript(script: str, timeout: float = 10.0) -> tuple[int, str, str]:
    return await run("osascript", "-e", script, timeout=timeout)


def applescript_string(value: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def is_macos() -> bool:
    return sys.platform == "darwin"


def installed_apps() -> list[str]:
    names = []
    for folder in APP_DIRS:
        if folder.is_dir():
            names.extend(p.stem for p in folder.glob("*.app"))
            names.extend(p.stem for p in folder.glob("*/*.app"))  # e.g. /Applications/Utilities
    return sorted(set(names))


def resolve_app(spoken: str, apps: list[str] | None = None) -> str | None:
    """Map a spoken app name to an installed app, tolerating transcription slips."""
    spoken = spoken.strip().lower().removesuffix(" app").strip()
    if spoken in APP_ALIASES:
        return APP_ALIASES[spoken]
    apps = installed_apps() if apps is None else apps
    by_lower = {a.lower(): a for a in apps}
    if spoken in by_lower:
        return by_lower[spoken]
    squashed = {a.lower().replace(" ", ""): a for a in apps}
    if spoken.replace(" ", "") in squashed:
        return squashed[spoken.replace(" ", "")]
    match = difflib.get_close_matches(spoken, list(by_lower), n=1, cutoff=0.75)
    return by_lower[match[0]] if match else None


async def open_app(name: str) -> Result:
    app = resolve_app(name)
    if not app:
        return Result(False, f"I couldn't find an app called {name}.")
    code, _, err = await run("open", "-a", app)
    if code:
        return Result(False, f"I couldn't open {app}: {err or 'unknown error'}.")
    return Result(True, f"Opening {app}.", {"app": app})


async def quit_app(name: str) -> Result:
    app = resolve_app(name)
    if not app:
        # Never tell an unknown name to quit: AppleScript can "succeed" without
        # closing anything, and Jarvis would claim it did.
        return Result(False, f"I couldn't find an app called {name}.")
    if not await _running(app):
        return Result(True, f"{app} isn't running.", {"app": app, "was_running": False})
    code, _, err = await osascript(f"tell application {applescript_string(app)} to quit")
    if code:
        return Result(False, f"I couldn't close {app}: {err or 'unknown error'}.")
    return Result(True, f"Closed {app}.", {"app": app, "was_running": True})


async def get_volume() -> int | None:
    code, out, _ = await osascript("output volume of (get volume settings)")
    return int(out) if code == 0 and out.isdigit() else None


async def set_volume(level: int) -> Result:
    level = max(0, min(100, int(level)))
    code, _, err = await osascript(f"set volume output volume {level} without output muted")
    if code:
        return Result(False, f"I couldn't change the volume: {err}.")
    return Result(True, f"Volume at {level}.", {"volume": level})


async def change_volume(delta: int) -> Result:
    current = await get_volume()
    if current is None:
        return Result(False, "I couldn't read the current volume.")
    return await set_volume(current + delta)


async def set_muted(muted: bool) -> Result:
    code, _, err = await osascript(f"set volume output muted {'true' if muted else 'false'}")
    if code:
        return Result(False, f"I couldn't change the mute setting: {err}.")
    return Result(True, "Muted." if muted else "Sound is back on.")


MEDIA_COMMANDS = {
    "play": "play",
    "pause": "pause",
    "toggle": "playpause",
    "next": "next track",
    "previous": "previous track",
}


async def _running(app: str) -> bool:
    code, out, _ = await osascript(f"application {applescript_string(app)} is running")
    return code == 0 and out == "true"


async def media(action: str) -> Result:
    """Control Spotify if it is running, otherwise Apple Music."""
    command = MEDIA_COMMANDS.get(action)
    if not command:
        return Result(False, f"I don't know the media action {action}.")
    player = "Spotify" if await _running("Spotify") else "Music"
    code, _, err = await osascript(f"tell application {applescript_string(player)} to {command}")
    if code:
        return Result(False, f"{player} didn't respond: {err}.")
    said = {"play": "Playing.", "pause": "Paused.", "toggle": "Done.", "next": "Next track.",
            "previous": "Previous track."}[action]
    return Result(True, said, {"player": player})


async def now_playing() -> Result:
    for player in ("Spotify", "Music"):
        if await _running(player):
            script = (
                f"tell application {applescript_string(player)} to "
                "(name of current track) & \" by \" & (artist of current track)"
            )
            code, out, _ = await osascript(script)
            if code == 0 and out:
                return Result(True, f"This is {out}.", {"player": player, "track": out})
    return Result(False, "Nothing is playing.")


def parse_battery(pmset_output: str) -> tuple[int, bool] | None:
    match = re.search(r"(\d+)%;\s*([a-zA-Z ]+);", pmset_output)
    if not match:
        return None
    state = match.group(2).strip().lower()
    return int(match.group(1)), state in {"charging", "charged", "finishing charge"}


async def battery() -> Result:
    code, out, err = await run("pmset", "-g", "batt")
    parsed = parse_battery(out) if code == 0 else None
    if not parsed:
        return Result(False, "I couldn't read the battery.")
    percent, charging = parsed
    return Result(True, f"Battery at {percent} percent{', charging' if charging else ''}.",
                  {"percent": percent, "charging": charging})


async def lock_screen() -> Result:
    code, _, err = await run("pmset", "displaysleepnow")
    if code:
        return Result(False, f"I couldn't lock the screen: {err}.")
    return Result(True, "Locking the screen.")


# Set from the user's profile at startup (JARVIS_BROWSER, JARVIS_MUSIC_TASTE).
DEFAULT_BROWSER = ""
MUSIC_TASTE: tuple[str, ...] = ()


async def open_url(url: str, browser: str = "") -> Result:
    """Open a page in ``browser`` (any spoken name, e.g. "brave"), or the user's usual browser."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
    browser = browser or DEFAULT_BROWSER
    cmd = ["open", url]
    app = ""
    if browser:
        app = resolve_app(browser) or ""
        if not app:
            return Result(False, f"I couldn't find a browser called {browser}.")
        cmd = ["open", "-a", app, url]
    code, _, err = await run(*cmd)
    if code:
        return Result(False, f"I couldn't open that page: {err}.")
    return Result(True, f"Opening it in {app or 'your browser'}.", {"url": url, "browser": app})


# Spoken site names -> addresses, so "open YouTube" opens the site rather than an app.
SITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "linkedin": "https://www.linkedin.com",
    "whatsapp web": "https://web.whatsapp.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org",
    "amazon": "https://www.amazon.com",
    "daraz": "https://www.daraz.pk",
}


async def open_site(name: str, browser: str = "") -> Result:
    url = SITES[name]
    result = await open_url(url, browser)
    if not result.ok:
        return result
    pretty = {"youtube": "YouTube", "github": "GitHub", "chatgpt": "ChatGPT", "linkedin": "LinkedIn",
              "whatsapp web": "WhatsApp Web"}.get(name, name.title())
    where = f" in {result.data['browser']}" if result.data["browser"] else ""
    return Result(True, f"Opening {pretty}{where}.", result.data)


SEARCH_SITES = {
    "google": "https://www.google.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
}


def search_url(query: str, site: str = "google") -> str:
    return SEARCH_SITES.get(site, SEARCH_SITES["google"]).format(quote_plus(query.strip()))


async def search_in_browser(query: str, browser: str = "", site: str = "google") -> Result:
    """Show search results for ``query`` on Google or YouTube in a browser."""
    site = site if site in SEARCH_SITES else "google"
    result = await open_url(search_url(query, site), browser)
    if not result.ok:
        return result
    where = "YouTube" if site == "youtube" else result.data["browser"] or "your browser"
    return Result(True, f"Searching {query} on {where}.", result.data)


async def set_dark_mode(on: bool | None = None) -> Result:
    """Turn dark mode on, off, or toggle it (``None``)."""
    value = "not dark mode" if on is None else ("true" if on else "false")
    script = f"tell application \"System Events\" to tell appearance preferences to set dark mode to {value}"
    code, _, err = await osascript(script)
    if code:
        return Result(False, f"I couldn't change the appearance: {err}.")
    if on is None:
        return Result(True, "Switched.")
    return Result(True, "Dark mode on." if on else "Light mode on.")


BRIGHTNESS_KEYS = {"up": 144, "down": 145}


async def change_brightness(direction: str, steps: int = 3) -> Result:
    key = BRIGHTNESS_KEYS[direction]
    script = f"tell application \"System Events\"\nrepeat {steps} times\nkey code {key}\nend repeat\nend tell"
    code, _, err = await osascript(script)
    if code:
        return Result(False, f"I couldn't change the brightness: {err}.")
    return Result(True, "Brighter." if direction == "up" else "Dimmer.")


async def screenshot() -> Result:
    folder = Path("~/Desktop").expanduser()
    path = folder / f"Jarvis screenshot {datetime.now():%Y-%m-%d at %H.%M.%S}.png"
    code, _, err = await run("screencapture", "-x", str(path))
    if code:
        return Result(False, f"I couldn't take a screenshot: {err}.")
    return Result(True, "Screenshot saved to your desktop.", {"path": str(path)})


async def sleep_mac() -> Result:
    code, _, err = await run("pmset", "sleepnow")
    if code:
        return Result(False, f"I couldn't put the Mac to sleep: {err}.")
    return Result(True, "Good night.")


# --- Playing on YouTube -----------------------------------------------------

YOUTUBE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
# Most specific first: the first search result, then any video on the page.
VIDEO_IDS = [
    re.compile(r'"videoRenderer":\{"videoId":"([A-Za-z0-9_-]{11})"'),
    re.compile(r'"videoId":"([A-Za-z0-9_-]{11})"'),
    re.compile(r"/watch\?v=([A-Za-z0-9_-]{11})"),
]

# What Jarvis puts on when you just say "play some music".
MOOD_QUERIES = {
    "chill": "chill music mix",
    "relaxing": "relaxing music mix",
    "upbeat": "upbeat feel good songs mix",
    "happy": "feel good songs mix",
    "sad": "sad songs mix",
    "romantic": "romantic songs mix",
    "focus": "lofi music for studying",
    "study": "lofi music for studying",
    "work": "focus music for work",
    "workout": "workout music mix",
    "gym": "workout music mix",
    "party": "party songs mix",
    "sleep": "calm sleep music",
}


def pick_music(mood: str = "", hour: int | None = None, taste: tuple[str, ...] | None = None) -> str:
    """A search for when the user doesn't name a song: by mood, else their taste, else time of day."""
    mood = mood.strip().lower()
    for word in sorted(MOOD_QUERIES, key=len, reverse=True):  # "workout" before "work"
        if word in mood:
            return MOOD_QUERIES[word]
    taste = MUSIC_TASTE if taste is None else taste
    if taste:
        import random

        return f"{random.choice(taste)} songs mix"
    hour = datetime.now().hour if hour is None else hour
    if 5 <= hour < 12:
        return "morning chill songs mix"
    if 12 <= hour < 18:
        return "popular songs mix"
    if 18 <= hour < 23:
        return "evening chill songs mix"
    return "late night lofi mix"


async def find_youtube_video(query: str) -> str | None:
    """ID of the first video in YouTube's results, read from the results page (no API key)."""
    import aiohttp

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            async with session.get(search_url(query, "youtube"), headers=YOUTUBE_HEADERS) as resp:
                page = await resp.text()
    except Exception:
        return None
    for pattern in VIDEO_IDS:
        if match := pattern.search(page):
            return match.group(1)
    return None


async def play_youtube(query: str = "", browser: str = "", mood: str = "") -> Result:
    """Start playing on YouTube; a "mix" keeps similar songs coming after the first."""
    chosen = query.strip() or pick_music(mood)
    video = await find_youtube_video(chosen)
    if not video:
        # Can't find a video to start: show the results instead of failing silently.
        result = await search_in_browser(chosen, browser, "youtube")
        if result.ok:
            result.say = f"I couldn't start it directly, so here are YouTube results for {chosen}."
        return result
    url = f"https://www.youtube.com/watch?v={video}&list=RD{video}&start_radio=1"
    result = await open_url(url, browser)
    if not result.ok:
        return result
    what = query.strip() or "some music"
    return Result(True, f"Playing {what} on YouTube.", {**result.data, "query": chosen, "video": video})
