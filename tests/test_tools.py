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


@pytest.mark.parametrize(
    "on, expected",
    [(True, "set dark mode to true"), (False, "set dark mode to false"), (None, "set dark mode to not dark mode")],
)
async def test_dark_mode_scripts(monkeypatch, on, expected):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "", ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    assert (await system.set_dark_mode(on)).ok
    assert scripts[0].endswith(expected)


async def test_brightness_uses_the_brightness_keys(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "", ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    await system.change_brightness("down")
    assert "key code 145" in scripts[0] and "repeat 3 times" in scripts[0]


async def test_screenshot_goes_to_the_desktop_silently(monkeypatch):
    calls = []

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "run", fake_run)
    result = await system.screenshot()
    assert result.ok and calls[0][:2] == ("screencapture", "-x")
    assert "/Desktop/Jarvis screenshot " in calls[0][2]


async def test_quit_never_claims_to_close_an_unknown_app(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "", ""  # AppleScript can "succeed" without closing anything

    monkeypatch.setattr(system, "osascript", fake_osascript)
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    result = await system.quit_app("jarvis")
    assert not result.ok and "couldn't find" in result.say
    assert scripts == []


async def test_quit_reports_an_app_that_isnt_running(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "false", ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    result = await system.quit_app("spotify")
    assert result.say == "Spotify isn't running."
    assert all("to quit" not in s for s in scripts)


async def test_quit_closes_a_running_app(monkeypatch):
    scripts = []

    async def fake_osascript(script, timeout=10.0):
        scripts.append(script)
        return 0, "true" if "is running" in script else "", ""

    monkeypatch.setattr(system, "osascript", fake_osascript)
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    result = await system.quit_app("spotify")
    assert result.ok and result.say == "Closed Spotify."
    assert scripts[-1] == 'tell application "Spotify" to quit'


async def test_open_site_in_a_browser(monkeypatch):
    calls = []

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "run", fake_run)
    result = await system.open_site("youtube", "brave")
    assert result.say == "Opening YouTube in Brave Browser."
    assert calls == [("open", "-a", "Brave Browser", "https://www.youtube.com")]


@pytest.mark.parametrize(
    "mood, hour, expected",
    [("workout", 10, "workout music mix"), ("studying", 10, "lofi music for studying"),
     ("chill", 10, "chill music mix"), ("", 9, "morning chill songs mix"), ("", 23, "late night lofi mix")],
)
def test_pick_music(mood, hour, expected):
    assert system.pick_music(mood, hour) == expected


RESULTS_PAGE = 'var ytInitialData = {"contents":{"x":[{"videoRenderer":{"videoId":"dQw4w9WgXcQ","title":{}}}]}};'


async def test_play_opens_a_youtube_mix(monkeypatch):
    calls = []

    async def fake_find(query):
        calls.append(("find", query))
        return "dQw4w9WgXcQ"

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "find_youtube_video", fake_find)
    monkeypatch.setattr(system, "run", fake_run)
    result = await system.play_youtube("talha anjum", "brave")
    assert result.ok and result.say == "Playing talha anjum on YouTube."
    assert calls == [("find", "talha anjum"),
                     ("open", "-a", "Brave Browser",
                      "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1")]


async def test_play_without_a_song_picks_one(monkeypatch):
    found = []

    async def fake_find(query):
        found.append(query)
        return "dQw4w9WgXcQ"

    async def fake_run(*cmd, timeout=10.0):
        return 0, "", ""

    monkeypatch.setattr(system, "find_youtube_video", fake_find)
    monkeypatch.setattr(system, "run", fake_run)
    result = await system.play_youtube(mood="workout")
    assert found == ["workout music mix"] and result.say == "Playing some music on YouTube."


async def test_play_falls_back_to_results_if_no_video_found(monkeypatch):
    calls = []

    async def fake_find(query):
        return None

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "find_youtube_video", fake_find)
    monkeypatch.setattr(system, "run", fake_run)
    result = await system.play_youtube("kaavish")
    assert result.ok and "here are YouTube results" in result.say
    assert calls == [("open", "https://www.youtube.com/results?search_query=kaavish")]


@pytest.mark.parametrize(
    "page, video",
    [(RESULTS_PAGE, "dQw4w9WgXcQ"), ('{"videoId":"abcdefghijk"}', "abcdefghijk"),
     ('<a href="/watch?v=ZYXWVUTSRQP">', "ZYXWVUTSRQP"), ("<html>consent</html>", None)],
)
async def test_find_youtube_video_reads_the_results_page(monkeypatch, page, video):
    import aiohttp

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return page

    class FakeSession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, headers=None):
            assert "search_query=talha+anjum" in url
            return FakeResponse()

    monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
    assert await system.find_youtube_video("talha anjum") == video


def test_music_taste_is_used_when_no_mood_given():
    assert system.pick_music("", 10, taste=("Kaavish",)) == "Kaavish songs mix"
    assert system.pick_music("workout", 10, taste=("Kaavish",)) == "workout music mix"


async def test_default_browser_is_used_when_none_named(monkeypatch):
    calls = []

    async def fake_run(*cmd, timeout=10.0):
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(system, "run", fake_run)
    monkeypatch.setattr(system, "DEFAULT_BROWSER", "brave")
    monkeypatch.setattr(system, "installed_apps", lambda: APPS)
    await system.open_url("github.com")
    await system.open_url("github.com", "safari")
    assert calls[0][:3] == ("open", "-a", "Brave Browser")
    assert calls[1][:3] == ("open", "-a", "Safari")


async def test_weather_uses_home_city(monkeypatch):
    seen = []

    class FakeResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        async def json(self, content_type=None):
            return {}

    class FakeSession:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, headers=None):
            seen.append(url)
            return FakeResp()

    monkeypatch.setattr(info.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(info, "HOME_CITY", "Peshawar")
    await info.weather()
    await info.weather("Lahore")
    assert seen == ["https://wttr.in/Peshawar?format=j1", "https://wttr.in/Lahore?format=j1"]


def test_page_text_keeps_the_article_and_drops_the_clutter():
    html = """<html><head><script>var x = 'a very long script that should never be read aloud';</script></head>
    <body><nav>Home | About | Contact us today for the best deals around</nav>
    <article><p>Pakistan beat Australia by five wickets in the final match played in Lahore on Sunday.</p>
    <p>Short</p></article><footer>Copyright 2026 some site with a long footer line here</footer></body></html>"""
    text = info.page_text(html)
    assert text == "Pakistan beat Australia by five wickets in the final match played in Lahore on Sunday."


async def test_web_search_reads_the_top_pages(monkeypatch):
    import ddgs

    class FakeDDGS:
        def text(self, query, max_results=5):
            return [{"title": f"t{i}", "body": f"b{i}", "href": f"https://e.com/{i}"} for i in range(5)]

    async def fake_read(url, timeout=5.0):
        return "" if url.endswith("/1") else f"content of {url}"

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    monkeypatch.setattr(info, "read_page", fake_read)
    result = await info.web_search("pakistan cricket")
    results = result.data["results"]
    assert results[0]["content"] == "content of https://e.com/0"
    assert "content" not in results[1]  # page couldn't be read: snippet only
    assert results[2]["content"] == "content of https://e.com/2"
    assert "content" not in results[3]  # only the top three are read
