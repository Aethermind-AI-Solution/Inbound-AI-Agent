# Claude NLU Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production NLU service backed by Claude Haiku 4.5, replacing StubNLUService with real intent classification, service extraction, and yes/no detection via prompt templates and a generic LLM client.

**Architecture:** `LLMClient` protocol with `AnthropicLLMClient` implementation (prompt caching, 3s timeout). `ClaudeNLUService(NLUService)` loads `.txt` prompt templates, formats with `string.Template.safe_substitute()`, and parses LLM responses. Keyword-first yes/no with LLM fallback and single-call caching.

**Tech Stack:** Anthropic Python SDK (`anthropic`), `string.Template`, pytest + pytest-asyncio

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/voice-agent/dialogue/nlu/prompts/system.txt` | Per-tenant system prompt template |
| `packages/voice-agent/dialogue/nlu/prompts/classify_intent.txt` | Intent classification user prompt |
| `packages/voice-agent/dialogue/nlu/prompts/extract_service.txt` | Service extraction user prompt |
| `packages/voice-agent/dialogue/nlu/prompts/yes_no.txt` | Yes/no fallback user prompt |
| `packages/voice-agent/dialogue/nlu/llm_client.py` | `LLMClient` protocol + `AnthropicLLMClient` |
| `packages/voice-agent/dialogue/nlu/claude_nlu.py` | `ClaudeNLUService(NLUService)` |
| `packages/voice-agent/tests/test_llm_client.py` | AnthropicLLMClient tests (mocked SDK) |
| `packages/voice-agent/tests/test_claude_nlu.py` | ClaudeNLUService tests (mock LLMClient) |

---

### Task 1: Prompt Template Files

Create the 4 prompt template files. No Python code — just the text files that `ClaudeNLUService` will load.

**Files:**
- Create: `packages/voice-agent/dialogue/nlu/prompts/system.txt`
- Create: `packages/voice-agent/dialogue/nlu/prompts/classify_intent.txt`
- Create: `packages/voice-agent/dialogue/nlu/prompts/extract_service.txt`
- Create: `packages/voice-agent/dialogue/nlu/prompts/yes_no.txt`

- [ ] **Step 1: Create the prompts directory and system.txt**

Create `packages/voice-agent/dialogue/nlu/prompts/system.txt`:

```
You are an NLU classifier for $business_name, a $sector business.

Available services: $services_list

You extract structured information from caller speech. Respond ONLY with the requested format — no explanations, no extra text.
```

- [ ] **Step 2: Create classify_intent.txt**

Create `packages/voice-agent/dialogue/nlu/prompts/classify_intent.txt`:

```
Classify the caller's intent from this transcript:
"$text"

Respond with exactly one of: $available_intents
```

- [ ] **Step 3: Create extract_service.txt**

Create `packages/voice-agent/dialogue/nlu/prompts/extract_service.txt`:

```
The caller said: "$text"

Which service are they requesting? Available services:
$services_list

Respond with the service ID only (e.g. "s1"). If unclear, respond "none".
```

- [ ] **Step 4: Create yes_no.txt**

Create `packages/voice-agent/dialogue/nlu/prompts/yes_no.txt`:

```
The caller said: "$text"

Is this affirmative (yes/agree), negative (no/disagree), or unclear?
Respond with exactly one of: yes, no, unclear
```

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/dialogue/nlu/prompts/
git commit -m "feat: add NLU prompt templates for Claude Haiku"
```

---

### Task 2: LLMClient Protocol + AnthropicLLMClient

Build the generic `LLMClient` protocol and its Anthropic implementation with prompt caching and 3s timeout.

**Files:**
- Create: `packages/voice-agent/dialogue/nlu/llm_client.py`
- Create: `packages/voice-agent/tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests for AnthropicLLMClient**

Create `packages/voice-agent/tests/test_llm_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.dialogue.nlu.llm_client'`

- [ ] **Step 3: Install anthropic SDK**

Run: `.venv/bin/pip install anthropic`

- [ ] **Step 4: Implement LLMClient protocol and AnthropicLLMClient**

Create `packages/voice-agent/dialogue/nlu/llm_client.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_llm_client.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add packages/voice-agent/dialogue/nlu/llm_client.py packages/voice-agent/tests/test_llm_client.py
git commit -m "feat: add LLMClient protocol and AnthropicLLMClient with prompt caching"
```

---

### Task 3: ClaudeNLUService — Intent Classification + Service Extraction

Build the core `ClaudeNLUService` with template loading, `classify_intent`, and `extract_service`. Yes/no methods are in Task 4.

**Files:**
- Create: `packages/voice-agent/dialogue/nlu/claude_nlu.py`
- Create: `packages/voice-agent/tests/test_claude_nlu.py`

- [ ] **Step 1: Write failing tests for intent classification and service extraction**

Create `packages/voice-agent/tests/test_claude_nlu.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_claude_nlu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.dialogue.nlu.claude_nlu'`

- [ ] **Step 3: Implement ClaudeNLUService (intent + service only)**

Create `packages/voice-agent/dialogue/nlu/claude_nlu.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from packages.voice_agent.dialogue.nlu.base import (
    IntentResult,
    NLUService,
    ServiceResult,
)

if TYPE_CHECKING:
    from packages.voice_agent.config.models import Service, TenantConfig
    from packages.voice_agent.dialogue.nlu.llm_client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_INTENTS = "new_booking, cancel, reschedule, status, unknown"

_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "sure", "confirm",
    "go ahead", "ok", "okay",
}
_NEGATIVE_WORDS = {"no", "nope", "nah", "don't", "stop"}


class ClaudeNLUService(NLUService):
    def __init__(self, client: LLMClient, tenant_config: TenantConfig) -> None:
        self._client = client
        self._tpl_intent = Template((_PROMPTS_DIR / "classify_intent.txt").read_text())
        self._tpl_service = Template((_PROMPTS_DIR / "extract_service.txt").read_text())
        self._tpl_yes_no = Template((_PROMPTS_DIR / "yes_no.txt").read_text())

        services_list = ", ".join(
            f"{s.id}:{s.name}"
            for s in tenant_config.booking_model.services
        )
        system_tpl = Template((_PROMPTS_DIR / "system.txt").read_text())
        self._system_prompt = system_tpl.safe_substitute(
            business_name=tenant_config.persona.business_name,
            sector=tenant_config.meta.sector,
            services_list=services_list,
        )
        self._last_yes_no: tuple[str, str] | None = None

    async def classify_intent(
        self, text: str, available_intents: list[str]
    ) -> IntentResult:
        intents_str = ", ".join(available_intents) if available_intents else _DEFAULT_INTENTS
        valid_intents = set(available_intents) if available_intents else {
            "new_booking", "cancel", "reschedule", "status", "unknown"
        }
        user_prompt = self._tpl_intent.safe_substitute(
            text=text, available_intents=intents_str
        )
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            intent = response.strip().lower()
            if intent in valid_intents:
                return IntentResult(intent=intent, confidence=0.9)
            return IntentResult(intent="unknown", confidence=0.0)
        except Exception:
            logger.exception("LLM error in classify_intent")
            return IntentResult(intent="unknown", confidence=0.0)

    async def extract_service(
        self, text: str, services: list[Service]
    ) -> ServiceResult:
        services_list = ", ".join(f"{s.id}: {s.name}" for s in services)
        valid_ids = {s.id for s in services}
        user_prompt = self._tpl_service.safe_substitute(
            text=text, services_list=services_list
        )
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            service_id = response.strip()
            if service_id in valid_ids:
                return ServiceResult(service_id=service_id, confidence=0.9)
            return ServiceResult(
                service_id=None, confidence=0.0,
                alternatives=[s.id for s in services[:3]],
            )
        except Exception:
            logger.exception("LLM error in extract_service")
            return ServiceResult(
                service_id=None, confidence=0.0,
                alternatives=[s.id for s in services[:3]],
            )

    async def is_affirmative(self, text: str) -> bool:
        lower = text.lower().strip()
        if lower in _AFFIRMATIVE_WORDS:
            return True
        if lower in _NEGATIVE_WORDS:
            return False
        return await self._classify_yes_no(text) == "yes"

    async def is_negative(self, text: str) -> bool:
        lower = text.lower().strip()
        if lower in _NEGATIVE_WORDS:
            return True
        if lower in _AFFIRMATIVE_WORDS:
            return False
        return await self._classify_yes_no(text) == "no"

    async def _classify_yes_no(self, text: str) -> str:
        if self._last_yes_no and self._last_yes_no[0] == text:
            return self._last_yes_no[1]
        user_prompt = self._tpl_yes_no.safe_substitute(text=text)
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            result = response.strip().lower()
            if result not in ("yes", "no", "unclear"):
                result = "unclear"
            self._last_yes_no = (text, result)
            return result
        except Exception:
            logger.exception("LLM error in _classify_yes_no")
            self._last_yes_no = (text, "unclear")
            return "unclear"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_claude_nlu.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/dialogue/nlu/claude_nlu.py packages/voice-agent/tests/test_claude_nlu.py
git commit -m "feat: add ClaudeNLUService with intent classification and service extraction"
```

---

### Task 4: ClaudeNLUService — Yes/No Methods + Caching

Add the yes/no tests covering keyword path, LLM fallback, and the single-call caching behavior.

**Files:**
- Modify: `packages/voice-agent/tests/test_claude_nlu.py`

- [ ] **Step 1: Add yes/no tests**

Append to `packages/voice-agent/tests/test_claude_nlu.py`:

```python
class TestIsAffirmative:
    @pytest.mark.asyncio
    async def test_keyword_affirmative(self, tenant_config):
        client = MockLLMClient(response="should not be called")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_affirmative("yes") is True
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_keyword_negative_returns_false(self, tenant_config):
        client = MockLLMClient(response="should not be called")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_affirmative("no") is False
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_llm_fallback_affirmative(self, tenant_config):
        client = MockLLMClient(response="yes")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_affirmative("I suppose so") is True
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_error_returns_false(self, tenant_config):
        client = MockLLMClient()
        client.complete.side_effect = RuntimeError("timeout")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_affirmative("I suppose so") is False


class TestIsNegative:
    @pytest.mark.asyncio
    async def test_keyword_negative(self, tenant_config):
        client = MockLLMClient(response="should not be called")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_negative("no") is True
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_llm_fallback_negative(self, tenant_config):
        client = MockLLMClient(response="no")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        assert await nlu.is_negative("not really") is True
        assert client.call_count == 1


class TestYesNoCaching:
    @pytest.mark.asyncio
    async def test_cache_prevents_double_llm_call(self, tenant_config):
        client = MockLLMClient(response="yes")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        await nlu.is_affirmative("I think so")
        await nlu.is_negative("I think so")
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_different_text_makes_new_llm_call(self, tenant_config):
        client = MockLLMClient(response="yes")
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        nlu = ClaudeNLUService(client, tenant_config)
        await nlu.is_affirmative("I think so")
        await nlu.is_affirmative("maybe later")
        assert client.call_count == 2
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_claude_nlu.py -v`
Expected: All 18 tests PASS (10 from Task 3 + 8 new)

- [ ] **Step 3: Commit**

```bash
git add packages/voice-agent/tests/test_claude_nlu.py
git commit -m "test: add yes/no keyword, LLM fallback, and caching tests for ClaudeNLUService"
```

---

### Task 5: Update NLU Package Exports + Wire into run_local.py

Export the new classes from the nlu package and update `run_local.py` to optionally use `ClaudeNLUService` when `ANTHROPIC_API_KEY` is set (falling back to `StubNLUService` when it's not).

**Files:**
- Modify: `packages/voice-agent/dialogue/nlu/__init__.py`
- Modify: `packages/voice-agent/run_local.py`

- [ ] **Step 1: Update nlu __init__.py exports**

Replace the contents of `packages/voice-agent/dialogue/nlu/__init__.py` with:

```python
from packages.voice_agent.dialogue.nlu.base import (
    IntentResult,
    NLUService,
    ServiceResult,
)
from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
from packages.voice_agent.dialogue.nlu.llm_client import (
    AnthropicLLMClient,
    LLMClient,
)
from packages.voice_agent.dialogue.nlu.stub import StubNLUService

__all__ = [
    "AnthropicLLMClient",
    "ClaudeNLUService",
    "IntentResult",
    "LLMClient",
    "NLUService",
    "ServiceResult",
    "StubNLUService",
]
```

- [ ] **Step 2: Update run_local.py to use ClaudeNLU when API key available**

In `packages/voice-agent/run_local.py`, find the NLU construction block (the line that creates `StubNLUService`) and replace it. The relevant section currently reads:

```python
    from packages.voice_agent.dialogue.nlu.stub import StubNLUService
    ...
    nlu = StubNLUService()
```

Replace the NLU import and construction with:

```python
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        from packages.voice_agent.dialogue.nlu.llm_client import AnthropicLLMClient
        llm_client = AnthropicLLMClient(api_key=anthropic_key)
        nlu = ClaudeNLUService(llm_client, config)
        logger.info("Using Claude Haiku NLU")
    else:
        from packages.voice_agent.dialogue.nlu.stub import StubNLUService
        nlu = StubNLUService()
        logger.info("Using StubNLU (set ANTHROPIC_API_KEY for Claude NLU)")
```

- [ ] **Step 3: Verify run_local.py parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('packages/voice-agent/run_local.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/voice-agent/dialogue/nlu/__init__.py packages/voice-agent/run_local.py
git commit -m "feat: export ClaudeNLU from package, wire into run_local.py with env-based toggle"
```

---

### Task 6: Full Test Suite Verification

Run all tests to verify nothing is broken.

**Files:**
- No changes — verification only

- [ ] **Step 1: Run entire test suite**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: All tests PASS (~285 total: 262 existing + ~23 new)

- [ ] **Step 2: Fix any failures and re-run**

Fix any issues, re-run until green.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test issues from Claude NLU integration"
```

(Skip if no fixes needed.)
