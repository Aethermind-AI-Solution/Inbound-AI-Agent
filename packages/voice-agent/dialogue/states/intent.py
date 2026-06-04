from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class IntentState(BaseState):
    name = CallState.INTENT

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._reprompt_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        return Action(
            type=ActionType.ASK,
            text=(
                "How can I help you today? I can help with booking an appointment, "
                "checking an existing booking, rescheduling, or cancelling."
            ),
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        result = await self.deps.nlu.classify_intent(event.text or "", [])

        if result.intent == "new_booking":
            context.intent = "new_booking"
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_SERVICE)

        if result.intent in ("cancel", "reschedule", "status"):
            context.intent = result.intent
            return Action(type=ActionType.TRANSITION, next_state=CallState.LOOKUP_BOOKINGS)

        self._reprompt_count += 1
        if self._reprompt_count >= 3:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        return Action(
            type=ActionType.ASK,
            text=(
                "I didn't quite catch that. Would you like to book a new appointment, "
                "or check on an existing one?"
            ),
        )
