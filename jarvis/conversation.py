"""Keep the conversation sent to the LLM small.

Every request carries the whole conversation, so a long session makes each
request bigger and burns free-tier tokens. Jarvis clears it when it goes back
to sleep (each "Hey Jarvis" starts fresh) and caps it while awake.
"""

from __future__ import annotations

from pipecat.processors.aggregators.llm_context import LLMContext

MAX_USER_TURNS = 6


def _is_user_turn(message) -> bool:
    # Tool results are role "tool"; only real user messages start a turn.
    return isinstance(message, dict) and message.get("role") == "user"


def trim(context: LLMContext, max_user_turns: int = MAX_USER_TURNS) -> int:
    """Keep only the last ``max_user_turns`` exchanges. Returns messages dropped.

    Cuts only at the start of a user turn, so a tool call is never separated
    from its result.
    """
    messages = context.get_messages()
    starts = [i for i, m in enumerate(messages) if _is_user_turn(m)]
    if len(starts) <= max_user_turns:
        return 0
    cut = starts[-max_user_turns]
    context.set_messages(messages[cut:])
    return cut


def clear(context: LLMContext) -> int:
    dropped = len(context.get_messages())
    if dropped:
        context.set_messages([])
    return dropped
