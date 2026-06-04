from abc import ABC, abstractmethod


class CheckpointStore(ABC):
    @abstractmethod
    async def save(
        self, phone: str, data: dict, ttl_seconds: int = 900
    ) -> None: ...

    @abstractmethod
    async def load(self, phone: str) -> dict | None: ...

    @abstractmethod
    async def delete(self, phone: str) -> None: ...
