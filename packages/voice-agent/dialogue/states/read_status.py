from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class ReadStatusState(BaseState):
    name = CallState.READ_STATUS

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._unclear_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._unclear_count = 0
        return Action(
            type=ActionType.ASK,
            text="Here are your bookings. Is there anything else I can help with?",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        text = event.text or ""
        if await self.deps.nlu.is_affirmative(text):
            return Action(type=ActionType.TRANSITION, next_state=CallState.INTENT)
        if await self.deps.nlu.is_negative(text):
            return Action(type=ActionType.TRANSITION, next_state=CallState.CLOSE)

        self._unclear_count += 1
        if self._unclear_count >= 2:
            return Action(type=ActionType.TRANSITION, next_state=CallState.CLOSE)

        return Action(type=ActionType.ASK, text="Would you like help with anything else?")
