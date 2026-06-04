from __future__ import annotations

import asyncio
from datetime import timedelta

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class ConfirmState(BaseState):
    name = CallState.CONFIRM

    async def enter(self, context: CallContext) -> Action:
        tenant_id = context.tenant_config.meta.tenant_id
        s = context.slots

        if context.intent == "reschedule" and s.booking_id:
            await asyncio.to_thread(
                self.deps.data_adapter.cancel_booking,
                tenant_id,
                s.booking_id,
                context.caller.id,
            )

        service = next(
            sv for sv in context.tenant_config.booking_model.services
            if sv.id == s.service_id
        )
        end_ts = s.datetime_ist + timedelta(minutes=service.duration_min)

        result = await asyncio.to_thread(
            self.deps.data_adapter.confirm_booking,
            tenant_id,
            s.hold_id,
            s.service_id,
            s.resource_id,
            context.caller.id,
            s.datetime_ist,
            end_ts,
            s.custom_fields,
        )

        return Action(
            type=ActionType.SPEAK,
            text=(
                f"Your booking is confirmed. Your reference number is {result.booking_id}."
            ),
            next_state=CallState.CLOSE,
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        return await self.enter(context)
