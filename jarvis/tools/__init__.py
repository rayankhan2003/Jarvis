"""Tools the brain can call. Pipecat builds each tool's schema from its signature and docstring."""

from __future__ import annotations

from pipecat.services.llm_service import FunctionCallParams

from jarvis.tools import info, system


async def open_app(params: FunctionCallParams, name: str):
    """Open (launch or bring to front) a Mac application.

    Args:
        name (str): The app's name as the user said it, e.g. "Spotify", "VS Code", "Safari".
    """
    await params.result_callback((await system.open_app(name)).as_dict())


async def quit_app(params: FunctionCallParams, name: str):
    """Quit a running Mac application. The app will ask about unsaved work itself.

    Args:
        name (str): The app's name, e.g. "Spotify".
    """
    await params.result_callback((await system.quit_app(name)).as_dict())


async def set_volume(params: FunctionCallParams, level: int):
    """Set the Mac's output volume.

    Args:
        level (int): Volume from 0 to 100.
    """
    await params.result_callback((await system.set_volume(level)).as_dict())


async def set_muted(params: FunctionCallParams, muted: bool):
    """Mute or unmute the Mac's sound.

    Args:
        muted (bool): True to mute, False to unmute.
    """
    await params.result_callback((await system.set_muted(muted)).as_dict())


async def media_control(params: FunctionCallParams, action: str):
    """Control music playback in Spotify (if running) or Apple Music.

    Args:
        action (str): One of "play", "pause", "toggle", "next", "previous".
    """
    await params.result_callback((await system.media(action)).as_dict())


async def now_playing(params: FunctionCallParams):
    """Get the song currently playing in Spotify or Apple Music."""
    await params.result_callback((await system.now_playing()).as_dict())


async def battery_status(params: FunctionCallParams):
    """Get the battery percentage and whether the Mac is charging."""
    await params.result_callback((await system.battery()).as_dict())


async def lock_screen(params: FunctionCallParams):
    """Lock the Mac's screen (puts the display to sleep)."""
    await params.result_callback((await system.lock_screen()).as_dict())


async def open_url(params: FunctionCallParams, url: str):
    """Open a web page in the default browser.

    Args:
        url (str): The address, e.g. "github.com" or "https://news.ycombinator.com".
    """
    await params.result_callback((await system.open_url(url)).as_dict())


async def get_time(params: FunctionCallParams):
    """Get the current local date and time."""
    await params.result_callback(info.get_time().as_dict())


async def get_weather(params: FunctionCallParams, location: str = ""):
    """Get the current weather and today's high and low.

    Args:
        location (str): City name. Leave empty for the user's current location.
    """
    await params.result_callback((await info.weather(location)).as_dict())


async def web_search(params: FunctionCallParams, query: str):
    """Search the web and get the top results with short snippets. Use for news, facts you are unsure of, or anything recent.

    Args:
        query (str): What to search for.
    """
    await params.result_callback((await info.web_search(query)).as_dict())


async def create_reminder(params: FunctionCallParams, title: str, due: str = ""):
    """Add a reminder to the Mac's Reminders app.

    Args:
        title (str): What to be reminded about.
        due (str): Optional local date and time in ISO 8601, e.g. "2026-10-01T18:00". Work it out from the current time when the user says "in 20 minutes" or "at six".
    """
    await params.result_callback((await info.create_reminder(title, due)).as_dict())


ALL_TOOLS = [
    open_app,
    quit_app,
    set_volume,
    set_muted,
    media_control,
    now_playing,
    battery_status,
    lock_screen,
    open_url,
    get_time,
    get_weather,
    web_search,
    create_reminder,
]
