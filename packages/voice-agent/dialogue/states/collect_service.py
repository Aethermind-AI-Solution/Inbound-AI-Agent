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
        text = event.text or ""
        services = context.tenant_config.booking_model.services
        result = await self.deps.nlu.extract_service(text, services)

        if result.service_id is not None:
            svc = next(s for s in services if s.id == result.service_id)
            context.slots.service_id = svc.id
            context.slots.service_name = svc.name
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_DATETIME)

        if self._is_asking_about_services(text):
            names = self._format_service_list(services)
            return Action(
                type=ActionType.ASK,
                text=f"Sure! We offer {names}. Which one would you like?",
            )

        self._reprompt_count += 1
        if self._reprompt_count >= 3:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        names = self._format_service_list(services)
        return Action(
            type=ActionType.ASK,
            text=f"I'm not sure which service you mean. We have {names}. Which would you like?",
        )

    @staticmethod
    def _is_asking_about_services(text: str) -> bool:
        lower = text.lower()
        question_words = ("what", "which", "tell me", "list", "options", "available", "offer", "do you have")
        service_words = ("service", "option", "offer", "available", "menu", "do you do", "you have", "else")
        return any(q in lower for q in question_words) and any(s in lower for s in service_words)

    @staticmethod
    def _format_service_list(services) -> str:
        if len(services) == 1:
            return services[0].name
        if len(services) == 2:
            return f"{services[0].name} and {services[1].name}"
        return ", ".join(s.name for s in services[:-1]) + f", and {services[-1].name}"
