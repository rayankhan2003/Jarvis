"""Pick the cheapest model that can handle a request, without asking an LLM.

Most requests ("what's the weather", "remind me at six") are easy and go to a
small, fast model. Requests that chain several steps or ask for real thinking
go to a larger model. The decision is a few string checks on the Mac, so it
costs no API call.
"""

from __future__ import annotations

import re

from pipecat.frames.frames import LLMUpdateSettingsFrame

CHAIN = re.compile(r"\b(?:and then|then|after that|afterwards|followed by|once that's done)\b")
THINKING = re.compile(
    r"\b(?:plan|compare|explain why|analy[sz]e|step by step|write|draft|summari[sz]e|debug|"
    r"pros and cons|figure out|work out|research|recommend)\b"
)
ACTIONS = re.compile(
    r"\b(?:open|close|quit|play|pause|search|set|turn|remind|send|create|find|start|lock|check|tell)\b"
)
LONG_REQUEST_WORDS = 25


def is_complex(text: str) -> bool:
    t = text.lower()
    if CHAIN.search(t) or THINKING.search(t):
        return True
    if len(t.split()) >= LONG_REQUEST_WORDS:
        return True
    # "open spotify, set the volume to 30 and play something" = three actions
    return len(ACTIONS.findall(t)) >= 3


class ModelRouter:
    """Switches one LLM service between a small and a large model, per request.

    Args:
        service: The LLM service to switch (only it applies the update).
        settings_cls: That service's ``Settings`` class.
        small: Model for everyday requests.
        large: Model for complex requests.
    """

    def __init__(self, service, settings_cls, small: str, large: str):
        self._service = service
        self._settings_cls = settings_cls
        self._small = small
        self._large = large
        self._current = small

    @property
    def current(self) -> str:
        return self._current

    def frames_for(self, text: str) -> list[LLMUpdateSettingsFrame]:
        """Settings frames to push ahead of this request, if the model must change."""
        wanted = self._large if is_complex(text) else self._small
        if wanted == self._current:
            return []
        self._current = wanted
        return [LLMUpdateSettingsFrame(delta=self._settings_cls(model=wanted), service=self._service)]
