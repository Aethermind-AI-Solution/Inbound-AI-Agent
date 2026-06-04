from __future__ import annotations

import asyncio
from typing import Any

from packages.voice_agent.data.adapter import SlotConflictError
from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class OfferSlotsState(BaseState):
    name = CallState.OFFER_SLOTS

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._available: list[dict[str, Any]] = []
        self._reprompt_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        service = next(
            s for s in context.tenant_config.booking_model.services
            if s.id == context.slots.service_id
        )
        availability = await asyncio.to_thread(
            self.deps.data_adapter.check_availability,
            context.tenant_config.meta.tenant_id,
            context.slots.service_id,
            service.resource_type,
            context.slots.datetime_ist,
            context.slots.datetime_ist,
        )
        self._available = [
            r for r in availability if not r.get("blocked_slots")
        ]

        if not self._available:
            context.fallback_reason = "no_availability"
            return Action(
                type=ActionType.SPEAK,
                text="Unfortunately there's nothing available at that time.",
                next_state=CallState.CALLBACK_CAPTURE,
            )

        names = ", ".join(r["resource_name"] for r in self._available[:3])
        return Action(
            type=ActionType.ASK,
            text=f"I have {names} available. Which would you prefer?",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = (event.text or "").lower()
        matched = None
        for r in self._available:
            if r["resource_name"].lower() in text or text in r["resource_name"].lower():
                matched = r
                break

        if matched is None:
            self._reprompt_count += 1
            if self._reprompt_count >= 2:
                context.fallback_reason = "repeated_failure"
                return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)
            names = ", ".join(r["resource_name"] for r in self._available[:3])
            return Action(type=ActionType.ASK, text=f"Could you pick from: {names}?")

        try:
            hold = await asyncio.to_thread(
                self.deps.data_adapter.hold_slot,
                context.tenant_config.meta.tenant_id,
                matched["resource_id"],
                context.slots.datetime_ist,
                context.caller.id,
            )
        except SlotConflictError:
            self._available = [r for r in self._available if r["resource_id"] != matched["resource_id"]]
            if not self._available:
                context.fallback_reason = "no_availability"
                return Action(
                    type=ActionType.SPEAK,
                    text="All those slots were just taken. Let me have someone help you.",
                    next_state=CallState.CALLBACK_CAPTURE,
                )
            names = ", ".join(r["resource_name"] for r in self._available[:3])
            return Action(
                type=ActionType.ASK,
                text=f"That slot was just taken. I still have {names}. Which would you prefer?",
            )

        context.slots.hold_id = hold.hold_id
        context.slots.resource_id = matched["resource_id"]
        context.slots.resource_name = matched["resource_name"]
        context.slots.hold_expires_at = hold.expires_at

        service = next(
            s for s in context.tenant_config.booking_model.services
            if s.id == context.slots.service_id
        )
        if service.custom_fields:
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_CUSTOM)
        return Action(type=ActionType.TRANSITION, next_state=CallState.READ_BACK)
