import time

from packages.voice_agent.dialogue.checkpoint.base import CheckpointStore


class InMemoryCheckpointStore(CheckpointStore):
    def __init__(self) -> None:
        self._store: dict[str, tuple[dict, float]] = {}

    async def save(
        self, phone: str, data: dict, ttl_seconds: int = 900
    ) -> None:
        self._store[phone] = (data, time.monotonic() + ttl_seconds)

    async def load(self, phone: str) -> dict | None:
        entry = self._store.get(phone)
        if entry is None:
            return None
        data, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[phone]
            return None
        return data

    async def delete(self, phone: str) -> None:
        self._store.pop(phone, None)
