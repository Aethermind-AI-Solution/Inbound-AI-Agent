from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from packages.voice_agent.config.models import Service
from packages.voice_agent.dialogue.nlu.base import IntentResult, ServiceResult


class MockLLMClient:
    def __init__(self, response: str = "") -> None:
        self._response = response
        self.complete = AsyncMock(side_effect=self._complete)
        self._call_count = 0

    async def _complete(self, system: str, user: str) -> str:
        self._call_count += 1
        return self._response

    @property
    def call_count(self) -> int:
        return self._call_count


def make_services() -> list[Service]:
    return [
        Service(
            id="s1", name="Haircut", resource_type="stylist",
            duration_min=30, booking_mode="exclusive", custom_fields=[],
        ),
        Service(
            id="s2", name="Hair Color", resource_type="stylist",
            duration_min=60, booking_mode="exclusive", custom_fields=[],
        ),
    ]


@pytest.fixture
def tenant_config():
    from packages.voice_agent.tests.test_states.conftest import make_tenant_config
    return make_tenant_config()


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_booking_intent(self, tenant_config):
        client = MockLLMClient(response="new_booking")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("I want to book an appointment", [])
        assert result.intent == "new_booking"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_cancel_intent(self, tenant_config):
        client = MockLLMClient(response="cancel")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("cancel my appointment", [])
        assert result.intent == "cancel"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_unknown_intent(self, tenant_config):
        client = MockLLMClient(response="unknown")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("the weather is nice", [])
        assert result.intent == "unknown"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_respects_available_intents(self, tenant_config):
        client = MockLLMClient(response="cancel")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent(
            "cancel my appointment", ["new_booking", "status"]
        )
        assert result.intent == "unknown"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_returns_unknown(self, tenant_config):
        client = MockLLMClient()
        client.complete.side_effect = RuntimeError("API down")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("book something", [])
        assert result.intent == "unknown"
        assert result.confidence == 0.0


class TestExtractService:
    @pytest.mark.asyncio
    async def test_service_match(self, tenant_config):
        client = MockLLMClient(response="s1")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        services = make_services()
        result = await nlu.extract_service("haircut please", services)
        assert result.service_id == "s1"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_service_no_match(self, tenant_config):
        client = MockLLMClient(response="none")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        services = make_services()
        result = await nlu.extract_service("something else", services)
        assert result.service_id is None
        assert result.confidence == 0.0
        assert result.alternatives == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_service_llm_error(self, tenant_config):
        client = MockLLMClient()
        client.complete.side_effect = RuntimeError("timeout")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        services = make_services()
        result = await nlu.extract_service("haircut", services)
        assert result.service_id is None
        assert result.alternatives == ["s1", "s2"]


class TestTemplateSafety:
    @pytest.mark.asyncio
    async def test_text_with_curly_braces_does_not_crash(self, tenant_config):
        client = MockLLMClient(response="new_booking")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("I want {something} booked", [])
        assert result.intent == "new_booking"

    @pytest.mark.asyncio
    async def test_text_with_dollar_sign_does_not_crash(self, tenant_config):
        client = MockLLMClient(response="new_booking")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        result = await nlu.classify_intent("costs $50 to book", [])
        assert result.intent == "new_booking"
