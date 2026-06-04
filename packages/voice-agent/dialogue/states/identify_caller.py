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


class IdentifyCallerState(BaseState):
    name = CallState.IDENTIFY_CALLER

    async def enter(self, context: CallContext) -> Action:
        tenant_id = context.tenant_config.meta.tenant_id
        caller = await asyncio.to_thread(
            self.deps.data_adapter.resolve_or_create_caller,
            tenant_id,
            context.caller_phone,
        )
        context.caller = caller
        return Action(type=ActionType.TRANSITION, next_state=CallState.INTENT)

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        return Action(type=ActionType.TRANSITION, next_state=CallState.INTENT)
