from __future__ import annotations

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class GreetingState(BaseState):
    name = CallState.GREETING

    async def enter(self, context: CallContext) -> Action:
        persona = context.tenant_config.persona
        fallback = persona.fallback_language
        greeting = persona.greeting.get(fallback, "")
        disclosure = persona.ai_disclosure.get(fallback, "")
        text = f"{greeting}. {disclosure}"
        return Action(
            type=ActionType.SPEAK,
            text=text,
            next_state=CallState.IDENTIFY_CALLER,
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        return Action(type=ActionType.TRANSITION, next_state=CallState.IDENTIFY_CALLER)
