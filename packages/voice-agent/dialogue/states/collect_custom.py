from __future__ import annotations

from packages.voice_agent.config.models import CustomField
from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class CollectCustomState(BaseState):
    name = CallState.COLLECT_CUSTOM

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._fields: list[CustomField] = []
        self._current_idx: int = 0

    async def enter(self, context: CallContext) -> Action:
        service = next(
            s for s in context.tenant_config.booking_model.services
            if s.id == context.slots.service_id
        )
        self._fields = list(service.custom_fields)
        self._current_idx = 0
        if not self._fields:
            return Action(type=ActionType.TRANSITION, next_state=CallState.READ_BACK)
        return Action(type=ActionType.ASK, text=self._fields[0].prompt)

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        field = self._fields[self._current_idx]
        context.slots.custom_fields[field.key] = event.text or ""
        self._current_idx += 1

        if self._current_idx >= len(self._fields):
            return Action(type=ActionType.TRANSITION, next_state=CallState.READ_BACK)

        return Action(type=ActionType.ASK, text=self._fields[self._current_idx].prompt)
