from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


_REASON_MESSAGES = {
    "no_availability": (
        "We don't have any openings right now, but I'll have someone "
        "call you to help find a time."
    ),
    "tool_error": (
        "I'm experiencing a technical issue. Let me have someone follow up with you."
    ),
    "budget_exceeded": (
        "We're experiencing high demand right now. "
        "Let me take your number and have someone call you back."
    ),
    "auth_required": (
        "I'll need to verify your identity for that request. "
        "Let me have someone call you back to assist."
    ),
}

_DEFAULT_MESSAGE = "I'm having some difficulty. Let me have someone call you back."


def _format_phone_for_speech(phone: str) -> str:
    """Format a phone number so TTS reads it digit-by-digit."""
    digits = "".join(c for c in phone if c.isdigit())
    return " ".join(digits)


class CallbackCaptureState(BaseState):
    name = CallState.CALLBACK_CAPTURE

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._unclear_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._unclear_count = 0
        reason_msg = _REASON_MESSAGES.get(context.fallback_reason, _DEFAULT_MESSAGE)
        spoken_phone = _format_phone_for_speech(context.caller_phone)
        return Action(
            type=ActionType.ASK,
            text=f"{reason_msg} Can I confirm your callback number is {spoken_phone}?",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = event.text or ""
        if await self.deps.nlu.is_affirmative(text):
            return Action(
                type=ActionType.SPEAK,
                text="Great, we'll call you back shortly.",
                next_state=CallState.CLOSE,
            )

        self._unclear_count += 1
        if self._unclear_count >= 2:
            spoken_phone = _format_phone_for_speech(context.caller_phone)
            return Action(
                type=ActionType.SPEAK,
                text=f"I'll use {spoken_phone} for the callback. We'll be in touch shortly.",
                next_state=CallState.CLOSE,
            )

        return Action(
            type=ActionType.ASK,
            text="Could you confirm — is this the right number to call you back on?",
        )
