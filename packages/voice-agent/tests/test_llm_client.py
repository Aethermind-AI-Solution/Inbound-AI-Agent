from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_anthropic_client():
    client = AsyncMock()
    response = MagicMock()
    response.content = [MagicMock(text="new_booking")]
    client.messages.create = AsyncMock(return_value=response)
    return client


class TestAnthropicLLMClient:
    @pytest.mark.asyncio
    async def test_correct_api_call_shape(self, mock_anthropic_client):
        with patch(
            "packages.voice_agent.dialogue.nlu.llm_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            from packages.voice_agent.dialogue.nlu.llm_client import (
                AnthropicLLMClient,
            )

            llm = AnthropicLLMClient(api_key="test-key")
            await llm.complete("system prompt", "user prompt")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 50
        assert call_kwargs["temperature"] == 0

    @pytest.mark.asyncio
    async def test_prompt_caching_header(self, mock_anthropic_client):
        with patch(
            "packages.voice_agent.dialogue.nlu.llm_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            from packages.voice_agent.dialogue.nlu.llm_client import (
                AnthropicLLMClient,
            )

            llm = AnthropicLLMClient(api_key="test-key")
            await llm.complete("system prompt", "user prompt")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_response_extraction(self, mock_anthropic_client):
        with patch(
            "packages.voice_agent.dialogue.nlu.llm_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            from packages.voice_agent.dialogue.nlu.llm_client import (
                AnthropicLLMClient,
            )

            llm = AnthropicLLMClient(api_key="test-key")
            result = await llm.complete("system", "user")

        assert result == "new_booking"

    @pytest.mark.asyncio
    async def test_timeout_raises(self, mock_anthropic_client):
        import anthropic

        mock_anthropic_client.messages.create.side_effect = anthropic.APITimeoutError(
            request=MagicMock()
        )
        with patch(
            "packages.voice_agent.dialogue.nlu.llm_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            from packages.voice_agent.dialogue.nlu.llm_client import (
                AnthropicLLMClient,
            )

            llm = AnthropicLLMClient(api_key="test-key")
            with pytest.raises(anthropic.APITimeoutError):
                await llm.complete("system", "user")

    @pytest.mark.asyncio
    async def test_api_error_raises(self, mock_anthropic_client):
        import anthropic

        mock_anthropic_client.messages.create.side_effect = anthropic.APIError(
            message="rate limited",
            request=MagicMock(),
            body=None,
        )
        with patch(
            "packages.voice_agent.dialogue.nlu.llm_client.anthropic.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            from packages.voice_agent.dialogue.nlu.llm_client import (
                AnthropicLLMClient,
            )

            llm = AnthropicLLMClient(api_key="test-key")
            with pytest.raises(anthropic.APIError):
                await llm.complete("system", "user")
