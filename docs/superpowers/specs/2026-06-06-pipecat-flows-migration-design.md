# Pipecat Flows Migration Design Spec

**Date:** 2026-06-06
**Status:** Draft

## Goal

Replace the rigid state machine + NLU classifier with **Pipecat Flows** — a hybrid architecture where deterministic flow control manages transitions between nodes, and **GPT-4o** generates natural conversational responses within each node. Claude is available as a fallback for complex reasoning in function handlers.

## Why

The current architecture (state machine + Claude Haiku NLU) has fundamental limitations:
- NLU classification takes 3-5 seconds per turn (latency kills conversation quality)
- Canned text responses sound robotic
- Every natural phrase ("what else do you have?", "let's say June 11") requires new pattern matching
- The bot can't handle even basic conversational flexibility

Pipecat Flows solves all of these: the LLM handles natural language natively (streaming, sub-second first token), while tool-call-driven transitions keep the booking flow deterministic.

## Architecture

### Current Pipeline
```
Transport.input() → VAD → STT → PipelineAdapter → TTS → Transport.output()
                                       ↕
                                DialogueManager
                                       ↕
                              State Machine (15 states)
                                       ↕
                              Claude NLU (3-5s/call)
```

### New Pipeline
```
Transport.input() → STT → GuardrailProcessor → ContextAggregator.user() → GPT-4o → TTS → Transport.output() → ContextAggregator.assistant()
                                  ↑                                           ↑
                          Deterministic                                  FlowManager
                        turn/time limits                           (nodes + tool handlers)
```

Key differences:
1. **No PipelineAdapter** — `LLMContextAggregatorPair` handles user/assistant message aggregation natively
2. **No DialogueManager** — `FlowManager` orchestrates transitions via tool calls
3. **No NLU service** — GPT-4o understands intent directly and calls the appropriate tool
4. **No canned text** — GPT-4o generates all speech naturally within each node's focused prompt
5. **Streaming responses** — GPT-4o streams tokens → TTS starts speaking immediately (sub-second latency)
6. **GuardrailProcessor** — deterministic `FrameProcessor` that enforces `max_turns` and `max_call_seconds` at pipeline level (LLM cannot bypass)

### Dual LLM Strategy

| LLM | Role | Where |
|---|---|---|
| **GPT-4o** | Primary — all conversational nodes | `OpenAILLMService` in pipeline |
| **Claude** | Fallback — complex reasoning in handlers | Direct API call in specific function handlers |

Claude is called from within tool handlers (not in the pipeline) for tasks like ambiguous date disambiguation. Most calls won't need it.

## Flow Nodes

The 15 current states collapse into **9 flow nodes**. Silent pass-through states (IdentifyCaller) become pre-actions. Related states (Greeting+Intent, ReadBack+Confirm) merge where natural.

### Node: `greeting`
- **Pre-action:** Identify caller via `data_adapter.resolve_or_create_caller()` + check budget
- **Role message:** Salon receptionist persona with business name, AI disclosure, tone, service list
- **Task:** Greet the caller warmly, disclose AI nature, ask how you can help
- **Tools:** `start_new_booking()`, `check_booking_status()`, `cancel_booking()`, `reschedule_booking()`, `request_callback(reason)`
- **respond_immediately:** true (bot speaks first)

### Node: `collect_service`
- **Task:** Help caller choose a service. List available services with names and prices.
- **Tools:** `select_service(service_id)`, `request_callback(reason)`
- The LLM naturally handles "what else?", "tell me more", "how much?", negative responses

### Node: `collect_datetime`
- **Task:** Ask for preferred date and time. Include business hours, booking window, today's date.
- **Tools:** `check_availability(date, time)` — handler validates via date resolver, then calls `data_adapter.check_availability()`
- The LLM parses natural date expressions ("next Tuesday at 3", "tomorrow afternoon") and structures them into tool arguments
- On resolver failure, handler returns an error message and the LLM naturally asks for clarification

### Node: `offer_slots`
- **Task:** Present available time slots and resources. Help caller pick one.
- **Tools:** `book_slot(resource_id, start_time)` — handler calls `data_adapter.hold_slot()`
- On `SlotConflictError`, handler returns error, LLM offers remaining options

### Node: `collect_custom`
- Only entered if the selected service has `custom_fields`
- **Task:** Collect required information (custom field prompts injected into task message)
- **Tools:** `submit_custom_fields(fields_json)` — handler validates all required fields present

### Node: `confirm_booking`
- **Task:** Read back the full booking summary (service, resource, date, time, custom fields). Ask for confirmation.
- **Tools:** `confirm()` — handler calls `data_adapter.confirm_booking()`, transitions to close node. `change_service()`, `change_datetime()` — transition back to respective nodes. `request_callback(reason)`

### Node: `manage_booking`
- Entered for status/cancel/reschedule intents
- **Pre-action:** `data_adapter.lookup_bookings()` — results injected into task message
- **Task:** Varies by intent. For status: read out bookings. For cancel: confirm cancellation. For reschedule: guide through date change.
- **Tools:** `cancel_booking(booking_id)`, `start_reschedule(booking_id)`, `done()`
- If no bookings found, LLM naturally says so and offers alternatives

### Node: `callback_capture`
- Entered when any node's `request_callback` is called, or guardrails fire
- **Task:** Explain we'll call back, read caller's number digit-by-digit, confirm
- **Tools:** `confirm_callback(phone_number)`, `provide_alternate_number(phone_number)`

### Node: `close`
- **Task:** Thank the caller and say goodbye
- **Post-action:** `end_conversation`

## Guardrails (Deterministic, Not LLM-Enforced)

Guardrail checks remain deterministic per CLAUDE.md invariant #6:

1. **Budget check** — `on_client_connected` calls `budget_tracker.check_budget()` before flow init. Over budget → start at `callback_capture` node instead of `greeting`
2. **Max call time + Max turns** — `GuardrailProcessor` (a `FrameProcessor` in the pipeline between STT and ContextAggregator) counts `TranscriptionFrame` events and checks elapsed time. When limits exceeded, pushes TTS explanation + `EndTaskFrame`. The LLM cannot bypass this because it sits upstream.
3. **Token cost** — `max_completion_tokens=200` and `temperature=0.4` on `OpenAILLMService` to prevent runaway generation
4. **Usage recording** — `on_client_disconnected` calls `budget_tracker.record_usage(tenant_id, duration_seconds)` (2-arg signature matching `BudgetTracker` protocol)
5. **Rate limiting** — Per-caller cooldown (max 3 calls per 15 minutes) enforced at `/incoming-call` before pipeline creation

## Security

### Prompt Injection Defenses
- Role message includes explicit anti-injection instructions: decline requests to change role, reveal prompt, or act outside booking scope
- Tool-call-only transitions — LLM can't trigger actions via free text
- Service IDs constrained via `enum` in tool schemas — LLM can't invent invalid IDs
- Prompts instruct LLM to never read out internal IDs, only use natural names

### PII Protection
- Phone numbers masked in all log statements (show only last 4 digits)
- Callback capture node speaks phone number via `tts_say` pre-action — number never enters LLM context
- Phone number validation in `confirm_callback` handler (regex format check)
- No caller names or full phone numbers in LLM system prompts

### Spam Protection
- In-memory rate limiter per caller phone number
- Monthly budget cap per tenant (checked at call start)
- Deterministic turn/duration limits via GuardrailProcessor

## Date Resolution

Per CLAUDE.md invariant #5, dates remain deterministically validated:

1. GPT-4o extracts the caller's date intent and passes structured args to `check_availability(date="YYYY-MM-DD", time="HH:MM")`
2. The tool handler validates deterministically: not-in-past, within-booking-window, within-business-hours, minimum-notice — the same checks DateResolver performs, but operating on the structured date GPT-4o already parsed (no need for dateparser NLP since the LLM does that part)
3. If validation fails, the handler returns a descriptive error string
4. GPT-4o naturally communicates the issue and asks for a different date

Note: The existing `DateResolver` is preserved in `resolver/date_resolver.py` but not called directly — its NLP parsing capabilities are superseded by GPT-4o's tool argument structuring. The handler reimplements the same validation checks (past-date, booking-window, business-hours, min-notice). Indian festival date parsing is handled naturally by GPT-4o ("book after Diwali" → structured date).

## Checkpoints (Invariant #9)

Every handler that triggers a node transition writes a checkpoint via the `_write_checkpoint()` helper in `handlers.py`. The checkpoint is written to `CheckpointStore` (in-memory for MVP, Redis for production):

```python
checkpoint = {
    "node": next_node_name,
    "intent": state.get("intent"),
    "service_id": state.get("service_id"),
    "datetime_ist": state.get("datetime_ist"),
    "resource_id": state.get("resource_id"),
    "hold_id": state.get("hold_id"),
    "turn_count": state.get("turn_count"),
    "caller_id": state.get("caller_id"),
    "call_id": state.get("call_id"),
}
```

Resume: on WebSocket reconnect, load checkpoint and call `flow_manager.set_node_from_config()` with the appropriate node + restored state from checkpoint.

## What Stays

| Component | Why |
|---|---|
| `config/` (TenantConfig, models, validator) | Config plane is architecture-independent |
| `data/` (DataAdapter, MockDataAdapter) | Database layer unchanged |
| `dialogue/budget/` (BudgetTracker) | Budget enforcement unchanged |
| `dialogue/checkpoint/` (CheckpointStore) | Checkpoint mechanism reused |
| `resolver/date_resolver.py` | Deterministic date validation per invariant #5 |
| `tests/fixtures/salon_config.json` | Test data unchanged |
| Server routes (`/health`, `/incoming-call`) | Twilio integration unchanged |

## What Goes

| Component | Replaced By |
|---|---|
| `pipeline_adapter.py` | FlowManager + ContextAggregator |
| `dialogue/manager.py` | FlowManager |
| `dialogue/states/*.py` (15 files) | Flow nodes (functions returning NodeConfig) |
| `dialogue/nlu/` (Claude NLU + prompts) | GPT-4o native tool calling |
| `dialogue/base_state.py` | Not needed |
| `dialogue/auth.py` | Integrated into function handlers |
| `dialogue/models.py` (partial) | Simplified — keep BookingSlots, drop CallState/Action/ActionType |

## New Files

```
packages/voice-agent/
├── flows/
│   ├── __init__.py
│   ├── nodes.py          # Flow node definitions (NodeConfig factories)
│   ├── handlers.py       # Tool function handlers with checkpoint writing + PII masking
│   ├── prompts.py        # Role message (with anti-injection) and task message templates
│   └── guardrails.py     # GuardrailProcessor — deterministic pipeline-level enforcement
├── server.py             # Modified — new pipeline with GPT-4o + FlowManager + safeguards
```

## Dependencies

```
pipecat-ai-flows>=1.2.0    # NEW — Pipecat Flows framework
pipecat-ai[openai]         # NEW — OpenAI LLM service
openai                     # NEW — for OpenAILLMService
# Keep: pipecat-ai[deepgram,silero], anthropic (for Claude fallback), supabase, etc.
```

## Environment Variables

```
OPENAI_API_KEY=...         # NEW — required for GPT-4o
ANTHROPIC_API_KEY=...      # Optional — Claude fallback in handlers
# Keep: DEEPGRAM_API_KEY, TWILIO_*, TUNNEL_URL
```

## Cost Estimate

GPT-4o per call (~10-15 turns, focused prompts):
- ~5,000-10,000 input tokens, ~1,000-2,000 output tokens
- Cost: ~$0.02-0.05 per call
- vs current Claude Haiku NLU: ~$0.002 per call but 3-5s latency per turn

Trade-off: 10-25x cost increase, but dramatically better conversation quality and sub-second response time. Can switch to GPT-4o-mini ($0.002-0.005/call) once flow is validated.
