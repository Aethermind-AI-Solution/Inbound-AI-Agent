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
        self._fail_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._fail_count = 0
        services = context.tenant_config.booking_model.services
        names = self._format_service_list(services)
        return Action(
            type=ActionType.ASK,
            text=f"We offer {names}. Which one would you like?",
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

        if await self.deps.nlu.is_negative(text):
            names = self._format_service_list(services)
            return Action(
                type=ActionType.ASK,
                text=(
                    f"I understand. Right now we only offer {names}. "
                    "Would any of those work for you, or would you like us to call you back?"
                ),
            )

        if self._is_question_or_exploration(text):
            names = self._format_service_list(services)
            return Action(
                type=ActionType.ASK,
                text=f"Those are all the services we have right now — {names}. Which one interests you?",
            )

        self._fail_count += 1
        if self._fail_count >= 4:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        names = self._format_service_list(services)
        return Action(
            type=ActionType.ASK,
            text=f"Sorry, I didn't catch that. We have {names}. Which would you like?",
        )

    @staticmethod
    def _is_question_or_exploration(text: str) -> bool:
        lower = text.lower()
        patterns = (
            "what else", "anything else", "other service", "other option",
            "what do you", "what service", "what are", "tell me",
            "is there", "do you have", "do you offer", "third",
            "more option", "more service", "something else",
            "what about", "any other", "besides",
        )
        return any(p in lower for p in patterns)

    @staticmethod
    def _format_service_list(services) -> str:
        if len(services) == 1:
            return services[0].name
        if len(services) == 2:
            return f"{services[0].name} and {services[1].name}"
        return ", ".join(s.name for s in services[:-1]) + f", and {services[-1].name}"
