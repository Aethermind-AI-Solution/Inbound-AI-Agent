from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)
from packages.voice_agent.resolver.date_resolver import ResolutionStatus


class CollectDatetimeState(BaseState):
    name = CallState.COLLECT_DATETIME

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._reprompt_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        return Action(
            type=ActionType.ASK,
            text=(
                "When would you like to come in? You can say something like "
                "'tomorrow at 3 PM' or 'next Saturday morning'."
            ),
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        resolver = self.deps.date_resolver_factory(context.tenant_config)
        result = resolver.resolve(event.text or "")

        if result.status == ResolutionStatus.RESOLVED:
            context.slots.datetime_ist = result.slot.datetime_ist
            return Action(type=ActionType.TRANSITION, next_state=CallState.OFFER_SLOTS)

        self._reprompt_count += 1
        if self._reprompt_count >= 3:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        if result.status == ResolutionStatus.AMBIGUOUS:
            return Action(type=ActionType.ASK, text=result.clarification_prompt)

        bm = context.tenant_config.booking_model
        messages = {
            ResolutionStatus.PAST_DATE: "That date is in the past. Could you pick a future date?",
            ResolutionStatus.OUT_OF_WINDOW: (
                f"We can only book up to {bm.booking_window_days} days ahead. "
                "Could you pick a closer date?"
            ),
            ResolutionStatus.TOO_SOON: (
                f"We need at least {bm.min_notice_min} minutes notice. "
                "Could you pick a later time?"
            ),
            ResolutionStatus.OUTSIDE_HOURS: (
                "We're not open at that time. What other time works for you?"
            ),
        }
        msg = messages.get(result.status, "I didn't understand that date. Could you try again?")
        return Action(type=ActionType.ASK, text=msg)
