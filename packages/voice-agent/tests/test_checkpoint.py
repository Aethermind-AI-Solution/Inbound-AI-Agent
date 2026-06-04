import time

import pytest

from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore


@pytest.fixture
def store():
    return InMemoryCheckpointStore()


class TestInMemoryCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        await store.save("+919876543210", {"state": "intent", "turn_count": 3})
        result = await store.load("+919876543210")
        assert result == {"state": "intent", "turn_count": 3}

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, store):
        result = await store.load("+910000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.save("+919876543210", {"state": "greeting"})
        await store.delete("+919876543210")
        result = await store.load("+919876543210")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_missing_does_not_raise(self, store):
        await store.delete("+910000000000")

    @pytest.mark.asyncio
    async def test_overwrite(self, store):
        await store.save("+919876543210", {"state": "greeting"})
        await store.save("+919876543210", {"state": "intent"})
        result = await store.load("+919876543210")
        assert result["state"] == "intent"

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self, store, monkeypatch):
        await store.save("+919876543210", {"state": "intent"}, ttl_seconds=1)
        # Simulate time passing
        phone = "+919876543210"
        data, _ = store._store[phone]
        store._store[phone] = (data, time.monotonic() - 1)
        result = await store.load("+919876543210")
        assert result is None
