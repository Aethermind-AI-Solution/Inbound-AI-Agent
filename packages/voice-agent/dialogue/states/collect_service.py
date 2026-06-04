from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class CollectServiceState(BaseState):
    name = CallState.COLLECT_SERVICE

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._reprompt_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        services = context.tenant_config.booking_model.services
        names = ", ".join(s.name for s in services)
        return Action(
            type=ActionType.ASK,
            text=f"What service would you like? We offer {names}.",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        services = context.tenant_config.booking_model.services
        result = await self.deps.nlu.extract_service(event.text or "", services)

        if result.service_id is not None:
            svc = next(s for s in services if s.id == result.service_id)
            context.slots.service_id = svc.id
            context.slots.service_name = svc.name
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)

        self._reprompt_count += 1
        if self._reprompt_count >= 2:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        names = ", ".join(s.name for s in services)
        return Action(
            type=ActionType.ASK,
            text=f"I'm not sure which service you mean. Could you pick from: {names}?",
        )
