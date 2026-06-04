from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.voice_agent.dialogue.models import Action, CallContext, CallEvent, StateDeps


class BaseState(ABC):
    name: str

    def __init__(self, deps: StateDeps) -> None:
        self.deps = deps

    @abstractmethod
    async def enter(self, context: CallContext) -> Action:
        ...

    @abstractmethod
    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        ...
