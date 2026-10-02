"""Looking things up: time, weather, the web, and reminders."""

from __future__ import annotations

import asyncio
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import quote

import aiohttp

from jarvis.tools.system import Result, applescript_string, osascript


def get_time(now: datetime | None = None) -> Result:
    now = now or datetime.now().astimezone()
    spoken = now.strftime("%-I:%M %p").lower().replace(":00", " o'clock")
    return Result(True, f"It's {spoken}.", {"iso": now.isoformat(), "weekday": now.strftime("%A")})


# Set from the user's profile at startup (JARVIS_CITY).
HOME_CITY = ""


async def weather(location: str = "") -> Result:
    """Current weather from wttr.in (free, no key). Empty location = home city, else a guess from IP."""
    location = location or HOME_CITY
    url = f"https://wttr.in/{quote(location)}?format=j1"
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": "jarvis"}) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
    except Exception as e:  # network errors, bad JSON
        return Result(False, f"I couldn't reach the weather service: {e}.")
    return parse_weather(data, location)


def parse_weather(data: dict, location: str = "") -> Result:
    try:
        now = data["current_condition"][0]
        today = data["weather"][0]
        area = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", location)
    except (KeyError, IndexError):
        return Result(False, "The weather service sent something I couldn't read.")
    desc = now["weatherDesc"][0]["value"].lower()
    temp, feels = now["temp_C"], now["FeelsLikeC"]
    high, low = today["maxtempC"], today["mintempC"]
    return Result(
        True,
        f"{temp} degrees and {desc} in {area}, feels like {feels}. High of {high}, low of {low}.",
        {"area": area, "temp_c": temp, "feels_like_c": feels, "high_c": high, "low_c": low,
         "description": desc},
    )


async def web_search(query: str, max_results: int = 5) -> Result:
    """DuckDuckGo search (free, no key)."""

    def _search() -> list[dict]:
        from ddgs import DDGS

        return list(DDGS().text(query, max_results=max_results))

    try:
        hits = await asyncio.wait_for(asyncio.to_thread(_search), timeout=10)
    except Exception as e:
        return Result(False, f"The search failed: {e}.")
    if not hits:
        return Result(False, f"I found nothing for {query}.")
    results = [{"title": h.get("title"), "snippet": h.get("body"), "url": h.get("href")} for h in hits]
    # Snippets are two lines; read the top pages so answers are right and current.
    pages = await asyncio.gather(*(read_page(r["url"]) for r in results[:PAGES_TO_READ]))
    for result, text in zip(results[:PAGES_TO_READ], pages, strict=True):
        if text:
            result["content"] = text
    return Result(True, f"Found {len(results)} results.", {"results": results})


PAGES_TO_READ = 3
PAGE_CHARS = 1500  # per page: enough for the facts, small enough for free-tier tokens


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"}

    def __init__(self):
        super().__init__()
        self._skipping = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skipping += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data):
        if not self._skipping and data.strip():
            self.parts.append(data.strip())


def page_text(html: str, limit: int = PAGE_CHARS) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    # Keep sentence-like chunks; menus and buttons are short.
    text = " ".join(p for p in parser.parts if len(p) > 30)
    return text[:limit]


async def read_page(url: str | None, timeout: float = 5.0) -> str:
    if not url:
        return ""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh) jarvis"}) as resp:
                if resp.status != 200 or "html" not in resp.headers.get("Content-Type", ""):
                    return ""
                return page_text(await resp.text(errors="ignore"))
    except Exception:
        return ""


def _applescript_date(var: str, when: datetime) -> str:
    """Build an AppleScript date without locale-dependent string parsing."""
    return "\n".join([
        f"set {var} to current date",
        f"set day of {var} to 1",
        f"set year of {var} to {when.year}",
        f"set month of {var} to {when.month}",
        f"set day of {var} to {when.day}",
        f"set time of {var} to {when.hour * 3600 + when.minute * 60}",
    ])


async def create_reminder(title: str, due: str = "") -> Result:
    """Add a reminder to the default Reminders list. ``due`` is ISO 8601, optional."""
    props = f"name:{applescript_string(title)}"
    lines = []
    when = None
    if due:
        try:
            when = datetime.fromisoformat(due)
        except ValueError:
            return Result(False, f"I couldn't understand the time {due}.")
        lines.append(_applescript_date("dueDate", when))
        props += ", due date:dueDate, remind me date:dueDate"
    lines += [
        'tell application "Reminders"',
        f"make new reminder with properties {{{props}}}",
        "end tell",
    ]
    code, _, err = await osascript("\n".join(lines), timeout=15)
    if code:
        return Result(False, f"Reminders said no: {err}.")
    at = f" for {when.strftime('%-I:%M %p on %A')}" if when else ""
    return Result(True, f"Reminder set{at}.", {"title": title, "due": due})
