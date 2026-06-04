from __future__ import annotations

from typing import TYPE_CHECKING

from packages.voice_agent.dialogue.nlu.base import (
    IntentResult,
    NLUService,
    ServiceResult,
)

if TYPE_CHECKING:
    from packages.voice_agent.config.models import Service


class StubNLUService(NLUService):
    def __init__(self) -> None:
        self.intent_map: dict[str, str] = {}
        self.service_map: dict[str, str] = {}
        self.affirmative_words = {
            "yes", "yeah", "yep", "sure", "confirm",
            "go ahead", "ok", "okay",
        }
        self.negative_words = {"no", "nope", "nah", "don't", "stop"}

    async def classify_intent(
        self, text: str, available_intents: list[str]
    ) -> IntentResult:
        lower = text.lower().strip()
        if lower in self.intent_map:
            return IntentResult(intent=self.intent_map[lower], confidence=1.0)
        if "cancel" in lower:
            return IntentResult(intent="cancel", confidence=0.8)
        if any(w in lower for w in ["reschedule", "change", "move"]):
            return IntentResult(intent="reschedule", confidence=0.8)
        if any(w in lower for w in ["status", "check", "existing", "upcoming"]):
            return IntentResult(intent="status", confidence=0.8)
        if any(w in lower for w in ["book", "appointment", "schedule"]):
            return IntentResult(intent="new_booking", confidence=0.8)
        return IntentResult(intent="unknown", confidence=0.0)

    async def extract_service(
        self, text: str, services: list[Service]
    ) -> ServiceResult:
        lower = text.lower().strip()
        if lower in self.service_map:
            return ServiceResult(
                service_id=self.service_map[lower], confidence=1.0, alternatives=[]
            )
        for svc in services:
            if svc.name.lower() in lower or lower in svc.name.lower():
                return ServiceResult(
                    service_id=svc.id, confidence=0.8, alternatives=[]
                )
        return ServiceResult(
            service_id=None,
            confidence=0.0,
            alternatives=[s.id for s in services[:3]],
        )

    async def is_affirmative(self, text: str) -> bool:
        return text.lower().strip() in self.affirmative_words

    async def is_negative(self, text: str) -> bool:
        return text.lower().strip() in self.negative_words
