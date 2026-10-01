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
    app = resolve_app(name) or name
    code, _, err = await osascript(f"tell application {applescript_string(app)} to quit")
    if code:
        return Result(False, f"I couldn't close {app}: {err or 'unknown error'}.")
    return Result(True, f"Closed {app}.", {"app": app})


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


async def open_url(url: str, browser: str = "") -> Result:
    """Open a page in ``browser`` (any spoken name, e.g. "brave"), or the default browser."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
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
