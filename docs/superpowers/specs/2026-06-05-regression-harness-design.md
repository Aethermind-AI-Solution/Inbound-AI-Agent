# Regression Harness Design Spec

**Date:** 2026-06-05
**Status:** Approved

## Goal

Build a pytest-based scenario runner that replays scripted conversations through the DialogueManager and asserts on state transitions, action types, slot values, and fallback reasons. Zero external dependencies. Scenarios are YAML data files — adding a regression test means adding a YAML file, not writing Python.

## Scope

- **In scope:** Scenario YAML format, loader, runner, 14 seed scenarios covering all 15 states and fallback paths, dual-NLU mode (StubNLU for CI, ClaudeNLU for nightly).
- **Not in scope:** Coval integration (can layer on later), semantic evaluation (LLM-as-judge), audio/TTS testing, Twilio E2E, production monitoring.

## Scenario Format

Each scenario is a YAML document representing one complete conversation:

```yaml
name: happy_path_new_booking
description: Caller books a haircut with Priya tomorrow at 10am
nlu: stub
caller_phone: "+919876543210"

setup:
  budget_used_inr: 0
  caller_is_returning: false

turns:
  - user: "I want to book an appointment"
    expect_state: collect_service
    expect_intent: new_booking

  - user: "haircut please"
    expect_state: collect_datetime
    expect_slot_service: Haircut

  - user: "tomorrow at 10am"
    expect_state: offer_slots

  - user: "Priya"
    expect_state: read_back

  - user: "yes"
    expect_state: confirm

  - user: "yes"
    expect_state: close
    expect_action: end_call

expect_fallback_reason: null
expect_min_turns: 6
expect_max_turns: 8
```

### YAML Fields

**Top-level (per scenario):**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Unique identifier, used as pytest test ID |
| `description` | no | Human-readable explanation |
| `nlu` | no | `"stub"` (default) or `"claude"`. Claude scenarios skipped unless `EVAL_NLU=claude` |
| `caller_phone` | no | Default `"+919876543210"` |
| `setup` | no | Pre-scenario configuration (see Setup section) |
| `turns` | yes | List of turn objects |
| `expect_fallback_reason` | no | Assert `context.fallback_reason` at end of scenario |
| `expect_min_turns` | no | Assert `context.turn_count >= N` at end |
| `expect_max_turns` | no | Assert `context.turn_count <= N` at end |

**Per-turn fields:**

| Field | Required | Description |
|---|---|---|
| `user` | yes | Text the caller says (fed as `CallEvent(TRANSCRIPTION, text=...)`) |
| `expect_state` | yes | `CallState` value the machine should be in after this turn |
| `expect_action` | no | `ActionType` value: `ask`, `speak`, `end_call` |
| `expect_intent` | no | `context.intent` after this turn |
| `expect_slot_service` | no | `context.slots.service_name` after this turn |
| `expect_text_contains` | no | Substring that must appear in `action.text` |
| `expect_fallback_reason` | no | `context.fallback_reason` after this turn |

**Special turn types:**

For silence events, use `user: "__SILENCE__"` — the runner sends `CallEvent(type=EventType.SILENCE)` instead of a transcription.

### Setup Fields

Optional per-scenario configuration applied before `dm.start()`:

| Field | Default | Description |
|---|---|---|
| `budget_used_inr` | `0` | Pre-load budget tracker with this amount (triggers budget gate if >= monthly_budget_inr) |
| `caller_is_returning` | `false` | If `true`, call `resolve_or_create_caller` twice so `is_new=False` on the actual test call |
| `action_policy` | from tenant config | Override `auth.action_policy` dict (e.g., `{"cancel": "soft"}`) |

## Runner Architecture

### Components

**`ScenarioLoader`** — Reads YAML files from a directory, parses into `Scenario` dataclass objects. Validates required fields and enum values on load.

```python
@dataclass
class Turn:
    user: str
    expect_state: str
    expect_action: str | None = None
    expect_intent: str | None = None
    expect_slot_service: str | None = None
    expect_text_contains: str | None = None
    expect_fallback_reason: str | None = None

@dataclass
class ScenarioSetup:
    budget_used_inr: float = 0.0
    caller_is_returning: bool = False
    action_policy: dict[str, str] | None = None

@dataclass
class Scenario:
    name: str
    turns: list[Turn]
    description: str = ""
    nlu: str = "stub"
    caller_phone: str = "+919876543210"
    setup: ScenarioSetup = field(default_factory=ScenarioSetup)
    expect_fallback_reason: str | None = None
    expect_min_turns: int | None = None
    expect_max_turns: int | None = None
```

**`run_scenario(scenario: Scenario) -> None`** — Async function that:

1. Creates `MockDataAdapter`, `InMemoryCheckpointStore`, `InMemoryBudgetTracker`
2. Applies `setup` overrides (pre-load budget, configure auth policy, simulate returning caller)
3. Selects NLU based on `scenario.nlu` field
4. Creates `DialogueManager` with all dependencies
5. Calls `dm.start()`
6. For each turn: sends `CallEvent` to `dm.handle_event()`, asserts expectations
7. After all turns: asserts end-of-scenario expectations (fallback_reason, turn counts)

On assertion failure, the error message includes: scenario name, turn number, user text, expected vs actual values.

**`conftest.py`** — Loads all scenarios via `ScenarioLoader`, uses `pytest.mark.parametrize` to create one test per scenario. Skips `nlu: claude` scenarios unless `EVAL_NLU=claude` environment variable is set.

**`test_scenarios.py`** — Single parametrized test function:

```python
@pytest.mark.asyncio
async def test_scenario(scenario):
    await run_scenario(scenario)
```

### File Layout

```
packages/eval/
  __init__.py
  runner.py            # Scenario, Turn, ScenarioSetup dataclasses + run_scenario + load_scenarios
  conftest.py          # pytest parametrize wiring
  test_scenarios.py    # parametrized test function
  scenarios/
    booking_happy.yaml
    booking_fallbacks.yaml
    cancel_flows.yaml
    auth_gating.yaml
    budget_guard.yaml
    guardrails.yaml
```

### How to Run

```bash
# Fast — StubNLU, all scenarios, runs in CI (<1s)
.venv/bin/python -m pytest packages/eval/ -v

# Realistic — include ClaudeNLU scenarios (needs ANTHROPIC_API_KEY)
EVAL_NLU=claude .venv/bin/python -m pytest packages/eval/ -v

# Single scenario
.venv/bin/python -m pytest packages/eval/ -v -k "happy_path_booking"
```

## Scenario Coverage

### Booking Flow (5 scenarios)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 1 | `happy_path_booking` | intent → service → datetime → slots → readback → confirm → close | `expect_action: end_call`, no fallback |
| 2 | `booking_unknown_service` | service × 2 misses → callback | `expect_fallback_reason: repeated_failure` |
| 3 | `booking_no_availability` | datetime resolves, no slots → callback | `expect_state: callback_capture` |
| 4 | `booking_with_custom_fields` | service → datetime → slots → custom fields → readback → confirm | Custom field state visited |
| 5 | `booking_edit_at_readback` | readback → "no" → back to edit | State goes backward then forward |

### Cancel Flow (2 scenarios)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 6 | `cancel_happy_path` | intent → lookup → select → confirm cancel → close | Booking cancelled |
| 7 | `cancel_no_bookings` | intent → lookup → no results → close | Graceful close, no callback |

### Status Flow (1 scenario)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 8 | `status_check` | intent → lookup → read status → close | Status read, clean exit |

### Auth Gating (2 scenarios)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 9 | `auth_blocks_new_caller` | new caller + cancel (soft) → blocked × 2 → callback | `expect_fallback_reason: auth_required` |
| 10 | `auth_allows_existing_caller` | returning caller + cancel → proceeds | `setup.caller_is_returning: true` |

### Budget (1 scenario)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 11 | `budget_exceeded` | budget pre-loaded → start → callback | `expect_fallback_reason: budget_exceeded` |

### Guardrails (2 scenarios)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 12 | `max_turns_exceeded` | Feed many turns → guardrail fires → callback | `expect_fallback_reason: max_turns_exceeded` |
| 13 | `repeated_silence` | Two `__SILENCE__` turns → callback | `expect_fallback_reason: repeated_silence` |

### Fallback Floor (1 scenario)

| # | Name | Path | Key Assertion |
|---|---|---|---|
| 14 | `unrecognized_intents` | 3 gibberish inputs at intent → callback | `expect_fallback_reason: repeated_failure` |

**Total: 14 scenarios** covering all 15 states, all fallback paths, auth gating, and budget guardrails.

## Failure Messages

When a scenario fails, the error message is actionable:

```
ScenarioError: Scenario 'happy_path_booking' turn 3:
  User said: "tomorrow at 10am"
  Expected state: offer_slots
  Actual state:   collect_datetime
```

End-of-scenario assertion:

```
ScenarioError: Scenario 'auth_blocks_new_caller' end:
  Expected fallback_reason: auth_required
  Actual fallback_reason:   None
```

## Validation

The scenario loader validates on load:

- `name` is present and non-empty
- `turns` is present and non-empty
- Each `expect_state` is a valid `CallState` value
- Each `expect_action` (if present) is a valid `ActionType` value
- `nlu` is either `"stub"` or `"claude"`
- No duplicate scenario names within a YAML file

Invalid YAML fails fast with a clear `ValueError`, not a runtime crash mid-test.

## Design Decisions

1. **YAML over Python test code:** Scenarios are data, not code. Anyone (including non-developers) can read and write them. Adding a regression test is a 30-second YAML edit, not a pytest function.

2. **StubNLU as default:** CI runs must be fast, free, and deterministic. StubNLU gives exact keyword matching — perfect for regression tests where you control the input. ClaudeNLU is opt-in for realistic validation.

3. **`__SILENCE__` sentinel:** Silence events are a real part of the dialogue flow (guardrail triggers). Rather than a separate event type in YAML, a magic string keeps the format simple.

4. **Setup overrides, not mock customization:** The `setup` block covers the 3 things that vary between scenarios (budget, caller identity, auth policy). Not a general-purpose mock configuration — YAGNI.

5. **One test function, parametrized:** Each scenario is a named pytest parameter. This gives you `pytest -k "scenario_name"` for free, clean test output, and no test code to maintain per scenario.

6. **No Coval dependency:** The harness is self-contained. Coval can be layered on later as a reporting/monitoring tool without changing the scenario format or runner.

## Non-Goals

- **Semantic evaluation:** No LLM-as-judge for response quality. Regression tests check structure (states, actions, slots), not prose.
- **Audio testing:** No STT/TTS/VAD testing. That's a different layer (Twilio E2E in Step 8).
- **Production monitoring:** This is a dev/CI tool. Production call monitoring is a separate concern.
- **Performance benchmarks:** Not measuring latency or throughput. Pure correctness testing.
