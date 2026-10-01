from datetime import datetime

import pytest

from jarvis.tools import info, system
from jarvis.tools.system import applescript_string, parse_battery, resolve_app

APPS = ["Safari", "Spotify", "Visual Studio Code", "Google Chrome", "System Settings", "Notion"]


@pytest.mark.parametrize(
    "spoken, app",
    [
        ("spotify", "Spotify"),
        ("Spotify app", "Spotify"),
        ("vs code", "Visual Studio Code"),
        ("chrome", "Google Chrome"),
        ("settings", "System Settings"),
        ("notiion", "Notion"),  # transcription slip
        ("photoshop", None),
    ],
)
def test_resolve_app(spoken, app):
    assert resolve_app(spoken, APPS) == app


def test_applescript_string_escapes_quotes():
    assert applescript_string('say "hi" \\ bye') == '"say \\"hi\\" \\\\ bye"'


@pytest.mark.parametrize(
    "output, expected",
    [
        ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t85%; discharging; 4:10 remaining", (85, False)),
        ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t42%; charging; 1:02 remaining", (42, True)),
        ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining", (100, True)),
        ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; AC attached; not charging", (80, False)),
        ("garbage", None),
    ],
)
def test_parse_battery(output, expected):
    assert parse_battery(output) == expected


async def test_set_volume_clamps_and_reports(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "", ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    result = await system.set_volume(140)
    assert result.ok and result.data["volume"] == 100
    assert scripts == ["set volume output volume 100 without output muted"]


async def test_open_app_reports_failure_honestly(monkeypatch):
    async def fake_run(*cmd, timeout=10.0):
        return 1, "", "Unable to find application named 'Spotify'"

    monkeypatch.setattr(system, "run", fake_run)
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    result = await system.open_app("spotify")
    assert not result.ok and "couldn't open" in result.say


async def test_media_prefers_spotify_when_running(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, ("true" if "is running" in script else ""), ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    result = await system.media("next")
    assert result.ok and result.data["player"] == "Spotify"
    assert scripts[-1] == 'tell application "Spotify" to next track'


def test_time_is_spoken_naturally():
    assert info.get_time(datetime(2026, 10, 1, 18, 0)).say == "It's 6 o'clock pm."
    assert info.get_time(datetime(2026, 10, 1, 9, 5)).say == "It's 9:05 am."


def test_parse_weather():
    data = {
        "current_condition": [{"temp_C": "31", "FeelsLikeC": "33", "weatherDesc": [{"value": "Sunny"}]}],
        "weather": [{"maxtempC": "35", "mintempC": "22"}],
        "nearest_area": [{"areaName": [{"value": "Peshawar"}]}],
    }
    result = info.parse_weather(data)
    assert result.ok
    assert result.say == "31 degrees and sunny in Peshawar, feels like 33. High of 35, low of 22."


def test_parse_weather_handles_garbage():
    assert not info.parse_weather({}).ok


async def test_reminder_builds_locale_independent_date(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "", ""

    monkeypatch.setattr(info, "osascript", fake_osascript)
    result = await info.create_reminder('Call "Ali"', "2026-10-01T18:30")
    assert result.ok
    script = scripts[0]
    assert "set month of dueDate to 10" in script
    assert "set time of dueDate to 66600" in script
    assert 'name:"Call \\"Ali\\""' in script


async def test_reminder_rejects_bad_dates():
    assert not (await info.create_reminder("x", "six-ish")).ok


def test_search_urls_are_encoded():
    assert system.search_url("talha anjum") == "https://www.google.com/search?q=talha+anjum"
    assert system.search_url("c++ & rust", "youtube") == "https://www.youtube.com/results?search_query=c%2B%2B+%26+rust"


async def test_search_opens_in_the_named_browser(monkeypatch):
    calls = []

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "run", fake_run)
    result = await system.search_in_browser("talha anjum", "brave")
    assert result.ok and result.say == "Searching talha anjum on Brave Browser."
    assert calls == [("open", "-a", "Brave Browser", "https://www.google.com/search?q=talha+anjum")]


async def test_search_uses_default_browser_when_none_named(monkeypatch):
    calls = []

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "run", fake_run)
    await system.search_in_browser("kaavish", site="youtube")
    assert calls == [("open", "https://www.youtube.com/results?search_query=kaavish")]


async def test_unknown_browser_is_reported(monkeypatch):
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    result = await system.open_url("github.com", "netscape")
    assert not result.ok and "couldn't find a browser" in result.say
