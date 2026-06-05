from __future__ import annotations

import logging
from typing import Protocol

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class AnthropicLLMClient:
    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=50,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
            timeout=3.0,
        )
        return response.content[0].text
