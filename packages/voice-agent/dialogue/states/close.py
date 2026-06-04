from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class CloseState(BaseState):
    name = CallState.CLOSE

    async def enter(self, context: CallContext) -> Action:
        business = context.tenant_config.persona.business_name
        return Action(
            type=ActionType.END_CALL,
            text=f"Thank you for calling {business}. Have a great day!",
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        return await self.enter(context)
