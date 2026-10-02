import pytest

from jarvis import fastpath
from jarvis.fastpath import match_command, normalize
from jarvis.tools.system import Result


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("Hey Jarvis, open Spotify.", "open spotify"),
        ("Jarvis open Spotify please", "open spotify"),
        ("...vis, what's the time?", "whats the time"),
        ("Set the volume to 40%.", "set the volume to 40%"),
    ],
)
def test_normalize(spoken, expected):
    assert normalize(spoken) == expected


@pytest.mark.parametrize(
    "spoken, name",
    [
        ("Open Spotify.", "open_app"),
        ("Hey Jarvis, launch VS Code", "open_app"),
        ("Quit Safari", "quit_app"),
        ("Volume 40", "set_volume"),
        ("Set the volume to fifty percent.", "set_volume"),
        ("Turn it up", "volume_up"),
        ("Mute", "mute"),
        ("Pause the music.", "pause"),
        ("Skip this song", "next"),
        ("What time is it?", "time"),
        ("How much battery do I have?", "battery"),
        ("Lock the screen.", "lock"),
        ("That's all, thanks.", "dismiss"),
    ],
)
def test_instant_commands(spoken, name):
    command = match_command(spoken)
    assert command is not None, spoken
    assert command.name == name


@pytest.mark.parametrize(
    "spoken",
    [
        "Open the readme file in my portfolio project",
        "Is it going to rain tomorrow in Peshawar?",
        "Remind me to call mum at six",
        "Open github and check my pull requests",
        "Set the volume to something reasonable",
        "",
    ],
)
def test_everything_else_goes_to_the_brain(spoken):
    assert match_command(spoken) is None


async def test_volume_level_is_passed_through(monkeypatch):
    seen = {}

    async def fake_set_volume(level):
        seen["level"] = level
        return Result(True, f"Volume at {level}.")

    monkeypatch.setattr(fastpath.system, "set_volume", fake_set_volume)
    result = await match_command("volume to 35 percent").run()
    assert result.ok and seen["level"] == 35


@pytest.mark.parametrize(
    "spoken, query, browser, site",
    [
        ("Hey Jarvis, open Brave browser and search Talha Anjum.", "talha anjum", "brave", "google"),
        ("Open Brave and search for Talha Anjum", "talha anjum", "brave", "google"),
        ("Search Talha Anjum on YouTube", "talha anjum", "", "youtube"),
        ("Open YouTube and search Kaavish", "kaavish", "", "youtube"),
        ("Google Pakistan cricket score", "pakistan cricket score", "", "google"),
        ("Search for best biryani in Peshawar in Chrome", "best biryani in peshawar", "chrome", "google"),
    ],
)
async def test_browser_searches_are_instant(monkeypatch, spoken, query, browser, site):
    seen = {}

    async def fake_search(q, b="", s="google"):
        seen.update(query=q, browser=b, site=s)
        return Result(True, "Searching.")

    monkeypatch.setattr(fastpath.system, "search_in_browser", fake_search)
    command = match_command(spoken)
    assert command is not None and command.name == "search"
    await command.run()
    assert seen == {"query": query, "browser": browser, "site": site}


@pytest.mark.parametrize("spoken", ["Search my files for the invoice", "google chrome"])
def test_non_web_searches_go_to_the_brain(spoken):
    assert match_command(spoken) is None


@pytest.mark.parametrize(
    "spoken, name",
    [
        ("Set a timer for 10 minutes", "timer"),
        ("Hey Jarvis, 5 minute timer", "timer"),
        ("Timer for half an hour please", "timer"),
        ("Set an alarm for 7:30 a.m.", "alarm"),
        ("Wake me up at 6 in the morning", "alarm"),
        ("Cancel the timer", "cancel_timers"),
        ("How much time is left?", "timer_status"),
        ("Turn on dark mode", "dark_mode"),
        ("Switch to light mode", "light_mode"),
        ("Make it brighter", "brighter"),
        ("Dim the screen", "dimmer"),
        ("Take a screenshot", "screenshot"),
        ("Put the Mac to sleep", "sleep_mac"),
        ("Go to sleep", "dismiss"),
    ],
)
def test_timer_and_system_commands(spoken, name):
    command = match_command(spoken)
    assert command is not None, spoken
    assert command.name == name


@pytest.mark.parametrize("spoken", ["Set a timer", "Set an alarm for whenever", "Is dark mode better for my eyes?"])
def test_vague_timer_and_system_requests_go_to_the_brain(spoken):
    assert match_command(spoken) is None


@pytest.mark.parametrize("spoken", ["", "Um.", "Thank you.", "Thanks for watching!", "Jarvis?", "you"])
def test_noise(spoken):
    assert fastpath.is_noise(spoken)


@pytest.mark.parametrize("spoken", ["Pause", "What's the weather?"])
def test_not_noise(spoken):
    assert not fastpath.is_noise(spoken)


@pytest.mark.parametrize("spoken", ["Close Jarvis", "Jarvis, shut down.", "Quit Jarvis", "Turn yourself off"])
def test_shutdown(spoken):
    assert match_command(spoken).name == "shutdown"


@pytest.mark.parametrize(
    "spoken, site, browser",
    [
        ("Open Brave and search YouTube", "youtube", "brave"),
        ("Open YouTube in Brave", "youtube", "brave"),
        ("Open YouTube", "youtube", ""),
        ("Go to GitHub", "github", ""),
        ("Open Brave and go to Gmail", "gmail", "brave"),
    ],
)
async def test_site_names_open_the_site(monkeypatch, spoken, site, browser):
    seen = {}

    async def fake_open_site(s, b=""):
        seen.update(site=s, browser=b)
        return Result(True, "Opening.")

    monkeypatch.setattr(fastpath.system, "open_site", fake_open_site)
    command = match_command(spoken)
    assert command.name == "site"
    await command.run()
    assert seen == {"site": site, "browser": browser}


async def test_search_youtube_for(monkeypatch):
    seen = {}

    async def fake_search(q, b="", s="google"):
        seen.update(query=q, browser=b, site=s)
        return Result(True, "Searching.")

    monkeypatch.setattr(fastpath.system, "search_in_browser", fake_search)
    await match_command("Search YouTube for Kaavish in Brave").run()
    assert seen == {"query": "kaavish", "browser": "brave", "site": "youtube"}


@pytest.mark.parametrize(
    "spoken, query, browser, mood",
    [
        ("Play some music", "", "", ""),
        ("Hey Jarvis, play music.", "", "", ""),
        ("Play something chill", "", "", "chill"),
        ("Play some music for studying", "", "", "studying"),
        ("Play some music in Brave", "", "brave", ""),
        ("Play Talha Anjum", "talha anjum", "", ""),
        ("Play Talha Anjum on YouTube in Brave", "talha anjum", "brave", ""),
        ("Play some relaxing jazz", "relaxing jazz", "", ""),
        ("Put on some workout music", "workout music", "", ""),
        ("Play Blinding Lights by The Weeknd on YouTube in Brave", "blinding lights by the weeknd", "brave", ""),
    ],
)
async def test_play_requests_play_instead_of_searching(monkeypatch, spoken, query, browser, mood):
    seen = {}

    async def fake_play(q="", b="", m=""):
        seen.update(query=q, browser=b, mood=m)
        return Result(True, "Playing.")

    monkeypatch.setattr(fastpath.system, "play_youtube", fake_play)
    command = match_command(spoken)
    assert command is not None and command.name == "play_music", spoken
    await command.run()
    assert seen == {"query": query, "browser": browser, "mood": mood}


@pytest.mark.parametrize("spoken, name", [("Play", "play"), ("Resume", "play"), ("Pause", "pause"),
                                          ("Search Talha Anjum on YouTube", "search")])
def test_resume_and_search_are_unchanged(spoken, name):
    assert match_command(spoken).name == name
