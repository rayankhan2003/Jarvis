import asyncio
from datetime import datetime

import pytest

from jarvis import timers
from jarvis.timers import Timers, parse_clock, parse_duration, spoken_duration


@pytest.mark.parametrize(
    "text, seconds",
    [("10 minutes", 600), ("1 hour 30 minutes", 5400), ("half an hour", 1800), ("five minutes", 300),
     ("90 seconds", 90), ("2 and a half minutes", 150), ("an hour", 3600), ("10 min", 600),
     ("ten", None), ("pizza", None), ("10 minutes of pizza", None)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


NOW = datetime(2026, 10, 1, 21, 20)


@pytest.mark.parametrize(
    "text, expected",
    [("7", datetime(2026, 10, 2, 7, 0)), ("7 30 am", datetime(2026, 10, 2, 7, 30)),
     ("7 30 a m", datetime(2026, 10, 2, 7, 30)), ("19 00", datetime(2026, 10, 2, 19, 0)),
     ("730 pm", datetime(2026, 10, 2, 19, 30)), ("11", datetime(2026, 10, 1, 23, 0)),
     ("6 in the morning", datetime(2026, 10, 2, 6, 0)), ("25", None), ("13 pm", None)],
)
def test_parse_clock(text, expected):
    assert parse_clock(text, NOW) == expected


def test_spoken_duration():
    assert spoken_duration(90) == "1 minute and 30 seconds"
    assert spoken_duration(5400) == "1 hour and 30 minutes"


async def test_timer_fires_and_announces(monkeypatch):
    alerts, spoken = [], []

    async def fake_alert(message):
        alerts.append(message)

    async def on_fire(message):
        spoken.append(message)

    monkeypatch.setattr(timers, "alert", fake_alert)
    t = Timers()
    t.on_fire = on_fire
    result = await t.start_timer(1)
    assert result.say == "Timer set for 1 second."
    t._entries[0].task.cancel()  # swap the real 1 s wait for an immediate one
    t._entries = []
    await t._fire_after(0, "Your 1 second timer is done.")
    assert alerts == spoken == ["Your 1 second timer is done."]


async def test_status_and_cancel():
    clock = [100.0]
    t = Timers(clock=lambda: clock[0])
    await t.start_timer(600)
    clock[0] += 150
    assert (await t.status()).say == "7 minutes and 30 seconds left on the 10 minutes timer."
    assert (await t.cancel()).say == "Cancelled."
    assert (await t.status()).say == "No timers running."
    assert (await t.cancel()).say == "There's nothing to cancel."
    await asyncio.sleep(0)


async def test_alarm_says_tomorrow_when_needed():
    t = Timers()
    result = await t.set_alarm(datetime(2026, 10, 2, 7, 0), now=NOW)
    assert result.say == "Alarm set for 7:00 am tomorrow."
    await t.cancel()
