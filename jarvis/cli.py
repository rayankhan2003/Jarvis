"""Command line: `jarvis` to run, `jarvis doctor` to check the setup, `jarvis say` to test commands."""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from jarvis import __version__
from jarvis.config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="A voice-first assistant for your Mac.")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="start Jarvis (default)")
    doctor = sub.add_parser("doctor", help="check this Mac, your keys and the response time")
    doctor.add_argument("--no-mic", action="store_true", help="skip the microphone recording")
    sub.add_parser("usage", help="show today's free-tier usage")
    voices = sub.add_parser("voices", help="hear the available voices and pick one")
    voices.add_argument("names", nargs="*", help="only these voices, e.g. af_heart bf_emma")
    say = sub.add_parser("say", help="run a typed command through the instant-command path")
    say.add_argument("text", nargs="+")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.command == "doctor":
        from jarvis.doctor import main as doctor_main

        return doctor_main(skip_mic=args.no_mic)

    if args.command == "voices":
        from jarvis.voices import audition

        return audition(args.names, current=Config.load().voice)

    if args.command == "usage":
        from jarvis.usage import Usage, report

        print(report(Usage()))
        return 0

    if args.command == "say":
        return asyncio.run(_say(" ".join(args.text)))

    from jarvis.bot import SetupError, run_jarvis

    try:
        asyncio.run(run_jarvis(Config.load()))
    except SetupError as e:
        print(f"\n{e}\nRun `jarvis doctor` to check your setup.\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


async def _say(text: str) -> int:
    from jarvis.fastpath import match_command

    command = match_command(text)
    if command is None:
        print("Not an instant command; Jarvis would send this to the brain.")
        return 0
    result = await command.run()
    print(f"[{command.name}] {result.say}")
    return 0 if result.ok else 1
