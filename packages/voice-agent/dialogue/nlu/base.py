from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.voice_agent.config.models import Service


@dataclass
class IntentResult:
    intent: str
    confidence: float


@dataclass
class ServiceResult:
    service_id: str | None
    confidence: float
    alternatives: list[str] = field(default_factory=list)


class NLUService(ABC):
    @abstractmethod
    async def classify_intent(
        self, text: str, available_intents: list[str]
    ) -> IntentResult: ...

    @abstractmethod
    async def extract_service(
        self, text: str, services: list[Service]
    ) -> ServiceResult: ...

    @abstractmethod
    async def is_affirmative(self, text: str) -> bool: ...

    @abstractmethod
    async def is_negative(self, text: str) -> bool: ...
