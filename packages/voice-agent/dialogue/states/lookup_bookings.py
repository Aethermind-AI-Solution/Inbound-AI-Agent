from __future__ import annotations

import asyncio

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class LookupBookingsState(BaseState):
    name = CallState.LOOKUP_BOOKINGS

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._bookings: list = []

    async def enter(self, context: CallContext) -> Action:
        tenant_id = context.tenant_config.meta.tenant_id
        self._bookings = await asyncio.to_thread(
            self.deps.data_adapter.lookup_bookings,
            tenant_id,
            context.caller.id,
        )

        if not self._bookings:
            return Action(
                type=ActionType.SPEAK,
                text="I don't see any bookings for your number.",
                next_state=CallState.CLOSE,
            )

        if context.intent == "status":
            return Action(
                type=ActionType.TRANSITION,
                next_state=CallState.READ_STATUS,
            )

        if len(self._bookings) == 1:
            context.slots.booking_id = self._bookings[0]["id"]
            if context.intent == "cancel":
                return Action(type=ActionType.TRANSITION, next_state=CallState.CONFIRM_CANCEL)
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)

        return Action(type=ActionType.TRANSITION, next_state=CallState.SELECT_BOOKING)

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        return await self.enter(context)
