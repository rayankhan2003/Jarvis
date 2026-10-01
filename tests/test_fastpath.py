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
        "Play some relaxing jazz",
        "What's the weather like in Peshawar?",
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
