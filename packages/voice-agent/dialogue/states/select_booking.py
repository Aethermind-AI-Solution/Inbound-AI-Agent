from __future__ import annotations

import re

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class SelectBookingState(BaseState):
    name = CallState.SELECT_BOOKING

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._bookings: list = []
        self._reprompt_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        items = []
        for i, b in enumerate(self._bookings, 1):
            items.append(f"{i}) {b.get('service_id', 'booking')} on {b.get('start_ts', 'unknown')}")
        listing = ", ".join(items)
        return Action(
            type=ActionType.ASK,
            text=f"I found {len(self._bookings)} bookings: {listing}. Which one?",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = (event.text or "").strip()
        match = re.search(r"\d+", text)
        idx = None
        if match:
            idx = int(match.group()) - 1

        if idx is not None and 0 <= idx < len(self._bookings):
            context.slots.booking_id = self._bookings[idx]["id"]
            if context.intent == "cancel":
                return Action(type=ActionType.TRANSITION, next_state=CallState.CONFIRM_CANCEL)
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)

        self._reprompt_count += 1
        if self._reprompt_count >= 2:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        return Action(
            type=ActionType.ASK,
            text="Could you tell me the number of the booking you'd like to select?",
        )
