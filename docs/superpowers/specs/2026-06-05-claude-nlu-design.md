# Claude NLU Service Design Spec

**Date:** 2026-06-05
**Status:** Approved

## Goal

Replace the keyword-based `StubNLUService` with a production `ClaudeNLUService` backed by Claude Haiku 4.5, using prompt caching and a generic `LLMClient` protocol for provider swappability. Prompt templates live in external `.txt` files for easy tuning without code changes.

## Architecture

Two layers:

1. **`LLMClient` protocol** — async interface with one method (`complete(system, user) -> str`). The `AnthropicLLMClient` implementation wraps the Anthropic Python SDK with prompt caching and a 3s timeout. This is the swap point for OpenAI or other providers.

2. **`ClaudeNLUService(NLUService)`** — loads prompt templates from `prompts/` directory, formats them with tenant/call context using `string.Template.safe_substitute()`, calls the `LLMClient`, and parses responses. Keyword-first yes/no with LLM fallback and single-call caching for the affirmative/negative pair.

## LLMClient Protocol

```python
class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...
```

### AnthropicLLMClient

- Constructor: `api_key: str`, `model: str = "claude-haiku-4-5-20251001"`
- Creates `anthropic.AsyncAnthropic(api_key=api_key)`
- `complete()` calls `client.messages.create()` with:
  - System prompt marked with `cache_control: {"type": "ephemeral"}` for Anthropic prompt caching
  - `max_tokens=50` (all NLU responses are short)
  - `temperature=0` (deterministic classification)
  - `timeout=3.0` seconds
- Extracts `response.content[0].text` and returns it
- On `TimeoutError` or `APIError`: logs and re-raises so `ClaudeNLUService` handles the fallback

### Prompt Caching Behavior

Anthropic caches system prompts marked with `cache_control` for 5 minutes. Since all NLU calls within one phone call share the same tenant system prompt, every call after the first hits cache. Cache read cost is 10x cheaper than uncached ($0.08/M vs $0.80/M input tokens).

## Prompt Templates

All templates use `$variable` syntax with `string.Template.safe_substitute()`. This prevents crashes when user speech contains `{`, `}`, or `$` characters.

### `prompts/system.txt` — Per-tenant cached prefix

```
You are an NLU classifier for $business_name, a $sector business.

Available services: $services_list

You extract structured information from caller speech. Respond ONLY with the requested format — no explanations, no extra text.
```

Variables: `$business_name` from `tenant_config.persona.business_name`, `$sector` from `tenant_config.meta.sector`, `$services_list` formatted as "s1:Haircut, s2:Hair Color" from `tenant_config.booking_model.services`.

Formatted once at `__init__` and reused for all calls — this is the cached prefix.

### `prompts/classify_intent.txt` — Intent classification

```
Classify the caller's intent from this transcript:
"$text"

Respond with exactly one of: $available_intents
```

Variables: `$text` (caller speech), `$available_intents` (comma-separated, from the method parameter or default "new_booking, cancel, reschedule, status, unknown").

### `prompts/extract_service.txt` — Service extraction

```
The caller said: "$text"

Which service are they requesting? Available services:
$services_list

Respond with the service ID only (e.g. "s1"). If unclear, respond "none".
```

Variables: `$text` (caller speech), `$services_list` formatted as "s1: Haircut, s2: Hair Color".

### `prompts/yes_no.txt` — Yes/no fallback

```
The caller said: "$text"

Is this affirmative (yes/agree), negative (no/disagree), or unclear?
Respond with exactly one of: yes, no, unclear
```

Only called when keyword matching fails. Variable: `$text` (caller speech).

## ClaudeNLUService Logic

### Constructor

```python
def __init__(self, client: LLMClient, tenant_config: TenantConfig) -> None:
```

- Loads all 4 template files from `prompts/` relative to the module
- Formats `system.txt` with tenant config → stores as `self._system_prompt`
- Initializes keyword sets for yes/no (same words as StubNLU)
- Initializes `self._last_yes_no: tuple[str, str] | None = None` for caching

### `classify_intent(text, available_intents) -> IntentResult`

1. Build `$available_intents` string: if list is non-empty use it, else default to `"new_booking, cancel, reschedule, status, unknown"`
2. Format `classify_intent.txt` with `$text` and `$available_intents`
3. Call `self._client.complete(self._system_prompt, user_prompt)`
4. Strip and lowercase response
5. If response is in the intents list → `IntentResult(intent=response, confidence=0.9)`
6. Else → `IntentResult(intent="unknown", confidence=0.0)`
7. On LLM error → `IntentResult(intent="unknown", confidence=0.0)`

### `extract_service(text, services) -> ServiceResult`

1. Build `$services_list` from the services parameter
2. Format `extract_service.txt` with `$text` and `$services_list`
3. Call `self._client.complete(self._system_prompt, user_prompt)`
4. Strip response
5. If response matches a service ID → `ServiceResult(service_id=id, confidence=0.9)`
6. Else → `ServiceResult(service_id=None, confidence=0.0, alternatives=[first 3 IDs])`
7. On LLM error → same fallback as step 6

### `is_affirmative(text) -> bool`

1. Try keyword set: `{"yes", "yeah", "yep", "sure", "confirm", "go ahead", "ok", "okay"}`
2. If keyword matches → return `True` immediately (no LLM call)
3. Check negative keywords too — if match, return `False` immediately
4. Check cache: if `self._last_yes_no` and `self._last_yes_no[0] == text`, use cached result
5. Otherwise call LLM with `yes_no.txt`, cache result as `self._last_yes_no = (text, response)`
6. Return `True` if response == "yes", else `False`
7. On LLM error → return `False`

### `is_negative(text) -> bool`

Same as `is_affirmative` but:
- Step 2: keyword check for negative words → return `True`
- Step 3: keyword check for affirmative words → return `False`
- Step 6: return `True` if cached/fresh response == "no", else `False`

The cache from step 4/5 is shared — if `is_affirmative` was just called with the same text, `is_negative` reuses the result. One LLM call covers both.

## Error Handling

Every `ClaudeNLUService` method wraps the LLM call in try/except. On any error:

- **`classify_intent`** → returns `IntentResult(intent="unknown", confidence=0.0)` — dialogue reprompts
- **`extract_service`** → returns `ServiceResult(service_id=None, ...)` — dialogue reprompts
- **`is_affirmative` / `is_negative`** → returns `False` — dialogue reprompts with "I didn't catch that"

This matches the existing dialogue fallback behavior. The DialogueManager's reprompt counters handle repeated failures → callback capture. No call is ever dropped.

## File Layout

| File | Purpose | ~Lines |
|---|---|---|
| `packages/voice-agent/dialogue/nlu/llm_client.py` | `LLMClient` protocol + `AnthropicLLMClient` | ~40 |
| `packages/voice-agent/dialogue/nlu/claude_nlu.py` | `ClaudeNLUService(NLUService)` | ~100 |
| `packages/voice-agent/dialogue/nlu/prompts/system.txt` | Per-tenant system prompt template | ~5 lines |
| `packages/voice-agent/dialogue/nlu/prompts/classify_intent.txt` | Intent classification template | ~4 lines |
| `packages/voice-agent/dialogue/nlu/prompts/extract_service.txt` | Service extraction template | ~6 lines |
| `packages/voice-agent/dialogue/nlu/prompts/yes_no.txt` | Yes/no fallback template | ~4 lines |
| `packages/voice-agent/tests/test_claude_nlu.py` | Tests with mock LLMClient | ~15 tests |
| `packages/voice-agent/tests/test_llm_client.py` | Tests for AnthropicLLMClient | ~5 tests |

## Testing Strategy

All tests use a `MockLLMClient` returning canned responses — no real API calls.

### ClaudeNLUService tests (~15):

1. **Intent — booking**: "I want to book" → LLM returns "new_booking" → correct IntentResult
2. **Intent — cancel**: "cancel my appointment" → LLM returns "cancel" → correct IntentResult
3. **Intent — unknown**: ambiguous text → LLM returns "unknown" → confidence=0.0
4. **Intent — respects available_intents**: passes param into template, LLM returns intent not in list → returns unknown
5. **Intent — LLM error**: LLM raises → returns unknown
6. **Service — match**: "haircut please" → LLM returns "s1" → correct ServiceResult
7. **Service — no match**: unclear text → LLM returns "none" → service_id=None with alternatives
8. **Service — LLM error**: LLM raises → returns None with alternatives
9. **Yes/no — keyword affirmative**: "yes" → True (LLM not called)
10. **Yes/no — keyword negative**: "no" → True for is_negative (LLM not called)
11. **Yes/no — LLM fallback affirmative**: "I suppose so" → LLM returns "yes" → is_affirmative=True
12. **Yes/no — LLM fallback negative**: "not really" → LLM returns "no" → is_negative=True
13. **Yes/no — caching**: call is_affirmative then is_negative with same text → LLM called once
14. **Yes/no — LLM error**: LLM raises → returns False
15. **Template safety**: text with `{curly}` and `$dollar` → no crash

### AnthropicLLMClient tests (~5):

1. **Correct API call shape**: verify `messages.create` called with right model, max_tokens, temperature
2. **Prompt caching header**: verify system message has `cache_control`
3. **Response extraction**: mock API response → returns `content[0].text`
4. **Timeout**: mock API timeout → raises
5. **API error**: mock API error → raises

## Cost Analysis

| Scenario | LLM calls | Latency added | Cost |
|---|---|---|---|
| Happy path booking | 2-3 | ~400-600ms | ~$0.00006 |
| Cancel/status | 1-2 | ~200-400ms | ~$0.00004 |
| Worst case (reprompts) | 5-6 | ~1-1.2s | ~$0.00012 |
| Yes/no (keyword hit) | 0 | 0ms | $0 |
| Yes/no (LLM fallback, pair) | 1 | ~200ms | ~$0.00002 |

At scale (1000 calls/day): ~$0.06-0.12/day NLU cost. Negligible.

## Design Decisions & Rationale

1. **`string.Template` over `str.format()`**: Prevents crashes from user speech containing `{`/`}`. `safe_substitute()` leaves unmatched variables as-is instead of raising.

2. **Keyword-first yes/no**: 90%+ of yes/no responses are simple words. Saves ~5 LLM calls per booking flow (~$0.0001 and ~1s latency).

3. **Single-call yes/no caching**: States always call `is_affirmative` then `is_negative` on the same text. Caching the LLM result halves fallback calls.

4. **Dynamic `available_intents`**: Prevents the model from classifying into intents the tenant doesn't support.

5. **3s LLM timeout**: Voice calls are real-time. Anything over 3s of silence feels broken. On timeout, the dialogue reprompts — better than hanging.

6. **Generic `LLMClient` protocol**: One-method interface makes it trivial to swap providers. No framework lock-in.

## Non-Goals

- **No conversation history / multi-turn context**: Each NLU call is stateless. The dialogue state machine already tracks conversation flow.
- **No date/time extraction**: Handled by the deterministic `DateResolver` (CLAUDE.md invariant #5).
- **No streaming**: NLU responses are <10 tokens. Streaming adds complexity for no benefit.
- **No retry logic**: On failure, return fallback. The dialogue's reprompt loop is the retry mechanism.
