from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class ReadBackState(BaseState):
    name = CallState.READ_BACK

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._unclear_count = 0
        self._awaiting_change = False

    async def enter(self, context: CallContext) -> Action:
        self._unclear_count = 0
        self._awaiting_change = False
        s = context.slots
        dt = s.datetime_ist
        date_str = dt.strftime("%A, %B %d") if dt else "the requested date"
        time_str = dt.strftime("%I:%M %p") if dt else "the requested time"

        text = (
            f"Let me confirm your booking: {s.service_name} with {s.resource_name} "
            f"on {date_str} at {time_str}."
        )
        if context.intent == "reschedule" and s.old_booking_date:
            text += f" This will replace your existing booking on {s.old_booking_date}."
        if s.custom_fields:
            for key, val in s.custom_fields.items():
                text += f" {key}: {val}."
        text += " Shall I go ahead and confirm this?"
        return Action(type=ActionType.ASK, text=text)

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = event.text or ""

        if self._awaiting_change:
            self._awaiting_change = False
            lower = text.lower()
            if any(w in lower for w in ["service", "what"]):
                context.slots.clear_service()
                return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_SERVICE)
            if any(w in lower for w in ["time", "date", "when"]):
                context.slots.clear_datetime()
                return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)
            context.slots.clear_datetime()
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)

        if await self.deps.nlu.is_affirmative(text):
            return Action(type=ActionType.TRANSITION, next_state=CallState.CONFIRM)

        if await self.deps.nlu.is_negative(text):
            self._awaiting_change = True
            return Action(
                type=ActionType.ASK,
                text="What would you like to change — the service, the time, or something else?",
            )

        self._unclear_count += 1
        if self._unclear_count >= 2:
            context.fallback_reason = "unclear_confirmation"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        return Action(
            type=ActionType.ASK,
            text="I just need a yes or no — shall I confirm this booking?",
        )
