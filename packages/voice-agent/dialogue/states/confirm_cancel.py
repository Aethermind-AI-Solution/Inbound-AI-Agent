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


class ConfirmCancelState(BaseState):
    name = CallState.CONFIRM_CANCEL

    async def enter(self, context: CallContext) -> Action:
        svc_name = context.slots.service_name or "your"
        return Action(
            type=ActionType.ASK,
            text=f"You'd like to cancel your {svc_name} appointment. Are you sure?",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = event.text or ""

        if await self.deps.nlu.is_affirmative(text):
            await asyncio.to_thread(
                self.deps.data_adapter.cancel_booking,
                context.tenant_config.meta.tenant_id,
                context.slots.booking_id,
                context.caller.id,
            )
            return Action(
                type=ActionType.SPEAK,
                text="Your booking has been cancelled.",
                next_state=CallState.CLOSE,
            )

        if await self.deps.nlu.is_negative(text):
            return Action(
                type=ActionType.SPEAK,
                text="Okay, your booking is still active.",
                next_state=CallState.CLOSE,
            )

        return Action(
            type=ActionType.ASK,
            text="I just need a yes or no — would you like to cancel this booking?",
        )
