# Auth + Guardrails Design Spec

**Date:** 2026-06-05
**Status:** Approved

## Goal

Add two overlays to the existing dialogue state machine: (1) per-intent auth gating using ANI soft-auth and the tenant's `action_policy`, and (2) a Redis-backed budget guardrail with hybrid gate-at-start + duration-tracking-at-end. Both are deterministic — never LLM-enforced (CLAUDE.md invariant #6).

## Scope

- **Auth:** ANI soft-auth only (no OTP). Check `action_policy` after intent classification. Soft = caller must be pre-existing in the system.
- **Budget:** Redis counter tracking per-minute call cost. Gate at call start, record at call end. Over-budget calls flow to callback capture.
- **Not in scope:** OTP auth, notification system for budget warnings, Twilio call transfer.

## Auth Gating

### How It Works

After `IdentifyCallerState` resolves the caller and `IntentState` classifies the intent, an auth check runs. It reads `tenant_config.auth.action_policy` and checks whether the intent requires the caller to be pre-existing.

### action_policy Values

- `"none"` — no auth required, anyone can do this (e.g., `new_booking`)
- `"soft"` — caller must have been *found* (not auto-created) in `resolve_or_create_caller`

If an intent is not listed in `action_policy`, it defaults to `"none"`.

### Caller Identity: is_new Flag

`CallerInfo` gets a new `is_new: bool` field:
- `True` when the caller was just created by `resolve_or_create_caller`
- `False` when the caller was found in the database

`IdentifyCallerState` already stores `context.caller`. The auth check reads `context.caller.is_new`.

### check_auth Function

```python
def check_auth(context: CallContext, intent: str) -> bool:
```

Lives in `packages/voice-agent/dialogue/auth.py`. Pure function — reads `context.caller.is_new` and `context.tenant_config.auth.action_policy`. No external calls, no failure modes beyond code bugs.

Returns `True` if authorized, `False` if not.

### IntentState Integration

After `nlu.classify_intent()` returns an intent, `IntentState.handle()` calls `check_auth(context, intent)`:
- Authorized → proceed as normal (transition to the appropriate state)
- Not authorized → "I can help with new bookings, but I'll need to verify your identity for that request. Would you like to book an appointment instead?"
- Not authorized + 2 rejections → callback capture with `fallback_reason = "auth_required"`

### Auth Error Handling

`check_auth` never touches external systems. It reads in-memory data only. No try/except needed — the existing `_safe_handle` wrapper in `DialogueManager` catches any unexpected exceptions.

## BudgetTracker

### Protocol

```python
class BudgetTracker(Protocol):
    async def check_budget(self, tenant_id: str, monthly_budget_inr: float) -> bool: ...
    async def record_usage(self, tenant_id: str, duration_seconds: float) -> None: ...
```

- `check_budget` — called at call start. Returns `True` if under budget, `False` if over.
- `record_usage` — called at call end. Converts duration to cost and accumulates.

### RedisBudgetTracker

Constructor: `redis: redis.asyncio.Redis`, `cost_per_minute_inr: float`

**Redis key:** `budget:{tenant_id}:{YYYY-MM}` (e.g., `budget:t1:2026-06`)

**Value:** Accumulated cost in paisa (integer). 1 INR = 100 paisa. Integer avoids float precision issues.

**TTL:** End-of-month + 7 days, set on first write. Auto-cleans old data, keeps current month for auditing.

**check_budget:**
1. `GET budget:{tenant_id}:{month}`
2. Parse as int (default 0 if key missing)
3. Return `accumulated_paisa < monthly_budget_inr * 100`

**record_usage:**
1. Compute `cost_paisa = int(duration_seconds / 60 * cost_per_minute_inr * 100)`
2. `INCRBY budget:{tenant_id}:{month} cost_paisa`
3. If key is new (INCRBY result == cost_paisa), set TTL to end-of-month + 7 days

### InMemoryBudgetTracker

Dict-based implementation for tests. `dict[str, int]` keyed by `"{tenant_id}:{YYYY-MM}"`, values in paisa. Same interface as `RedisBudgetTracker`. No Redis dependency.

### Cost Rate Config

Add `cost_per_minute_inr: float = 2.0` to `GuardrailsConfig`. This covers STT + NLU + TTS + Twilio amortized. At 5000 INR/month budget with 2 INR/min rate = 2500 minutes/month.

## Integration Points

### DialogueManager

New constructor parameter: `budget_tracker: BudgetTracker`.

**start():** After `_register_states()`, call `budget_tracker.check_budget(tenant_id, monthly_budget_inr)`. If over budget:
- Set `context.fallback_reason = "budget_exceeded"`
- Set `context.budget_checked = True`
- Enter `CallbackCaptureState` directly (skip greeting)

If under budget:
- Set `context.budget_checked = True`
- Proceed normally

**record_call_usage(duration_seconds):** New public method. Calls `budget_tracker.record_usage(tenant_id, duration_seconds)`. Called externally by PipelineAdapter.

### PipelineAdapter

When the adapter pushes an `EndTaskFrame` (call ending), it computes `elapsed = time.monotonic() - dm.context.call_start` and calls `dm.record_call_usage(elapsed)`. Also on disconnect/error paths.

### CallContext

Add `budget_checked: bool = False` to `CallContext`. Prevents re-checking budget on resumed calls (a resumed call was already allowed when it started).

### Checkpoint

Add `budget_checked` to checkpoint data in `_write_checkpoint()` and `_restore_context()`.

## Error Handling

### Redis Down — Fail Open

Budget tracker failures must NEVER block calls. The guardrail is a cost optimization, not a safety-critical system.

- `check_budget` on Redis error → return `True` (allow the call). Log warning.
- `record_usage` on Redis error → log and swallow. Usage is lost but call completes.
- Redis outage causes budget under-counting, not service disruption.

### Auth Failure

`check_auth` is a pure in-memory function. No external failure modes. Unexpected exceptions caught by `DialogueManager._safe_handle()`.

### Over Budget → Callback Capture

When budget gate triggers, the caller hears: "We're experiencing high demand right now. Let me take your number and have someone call you back." Flows into existing `CallbackCaptureState` with `fallback_reason = "budget_exceeded"`. No dead-ends (CLAUDE.md invariant #4).

## File Layout

### New Files

| File | Purpose | ~Lines |
|---|---|---|
| `packages/voice-agent/dialogue/auth.py` | `check_auth(context, intent) -> bool` | ~25 |
| `packages/voice-agent/dialogue/budget/base.py` | `BudgetTracker` protocol | ~15 |
| `packages/voice-agent/dialogue/budget/redis_tracker.py` | `RedisBudgetTracker` | ~60 |
| `packages/voice-agent/dialogue/budget/memory_tracker.py` | `InMemoryBudgetTracker` | ~30 |
| `packages/voice-agent/dialogue/budget/__init__.py` | Package exports | ~10 |
| `packages/voice-agent/tests/test_auth.py` | Auth gating tests | ~8 tests |
| `packages/voice-agent/tests/test_budget_tracker.py` | Budget tracker tests | ~10 tests |

### Modified Files

| File | Change |
|---|---|
| `packages/voice-agent/data/adapter.py` | Add `is_new: bool` to `CallerInfo` |
| `packages/voice-agent/data/mock_adapter.py` | Set `is_new` in `resolve_or_create_caller` |
| `packages/voice-agent/config/models.py` | Add `cost_per_minute_inr: float = 2.0` to `GuardrailsConfig` |
| `packages/voice-agent/dialogue/manager.py` | Add `budget_tracker` param, budget check in `start()`, `record_call_usage()` |
| `packages/voice-agent/dialogue/states/intent.py` | Call `check_auth()` after classify_intent |
| `packages/voice-agent/dialogue/models.py` | Add `budget_checked: bool` to `CallContext` |
| `packages/voice-agent/pipeline_adapter.py` | Call `record_call_usage()` on call end |
| `packages/voice-agent/run_local.py` | Wire `InMemoryBudgetTracker` (or Redis if `REDIS_URL` set) |
| `packages/voice-agent/tests/test_states/conftest.py` | Add `InMemoryBudgetTracker` to test fixtures |

## Testing Strategy

All tests use `InMemoryBudgetTracker` and `MockDataAdapter` — no real Redis.

### Auth Tests (~8)

1. `action_policy` "none" → always authorized
2. `action_policy` "soft" + existing caller (`is_new=False`) → authorized
3. `action_policy` "soft" + new caller (`is_new=True`) → not authorized
4. Intent not in `action_policy` → defaults to "none" (authorized)
5. IntentState: authorized intent proceeds normally
6. IntentState: unauthorized intent offers alternative
7. IntentState: unauthorized + 2 rejections → callback capture
8. `is_new` flag set correctly by MockDataAdapter (found vs created)

### Budget Tests (~10)

1. `InMemoryBudgetTracker.check_budget` — under budget returns True
2. `InMemoryBudgetTracker.check_budget` — over budget returns False
3. `InMemoryBudgetTracker.record_usage` — accumulates correctly
4. `RedisBudgetTracker` — correct Redis key format (`budget:{tenant}:{month}`)
5. `RedisBudgetTracker` — paisa conversion correct
6. `RedisBudgetTracker` — TTL set on first write
7. `RedisBudgetTracker` — Redis error in `check_budget` returns True (fail-open)
8. `RedisBudgetTracker` — Redis error in `record_usage` swallows error
9. `DialogueManager.start()` — over budget → callback capture
10. `PipelineAdapter` — records usage on call end

## Design Decisions & Rationale

1. **Auth is a pure function, not a state:** Auth doesn't need its own dialogue state. It's a gate within `IntentState` — one function call, no async, no external deps. Adding a state would complicate the FSM for no benefit.

2. **`is_new` flag over `verified_at`:** `verified_at` is a timestamp that implies OTP or some external verification. `is_new` is a simple boolean that means "was this caller just created?" — exactly what soft-auth needs. No semantic overloading.

3. **Paisa integers over float INR:** Float arithmetic causes precision drift (0.1 + 0.2 ≠ 0.3). Budget tracking accumulates thousands of small increments. Paisa integers avoid this entirely.

4. **Fail-open on Redis errors:** A voice call in progress is more valuable than budget accuracy. Blocking calls because Redis is down would be a worse outcome than slightly exceeding the budget.

5. **Gate at start + track at end:** Checking at start prevents over-budget calls from consuming resources. Tracking at end gives accurate duration-based cost. The gap (current call's cost not counted until it ends) is acceptable — it's at most one call's worth of overshoot.

6. **Callback capture on budget exceeded:** Respects CLAUDE.md invariant #4 (no dead-ends). The salon owner doesn't lose the lead. The caller feels taken care of.

7. **cost_per_minute_inr in config:** Different tenants may have different cost profiles (e.g., Hindi STT costs different from English). Making it configurable per-tenant allows accurate budget tracking as you scale to multiple sectors.

## Non-Goals

- **OTP auth:** MVP is soft-auth only. OTP requires DTMF handling (Twilio) which isn't built.
- **Budget notifications:** No webhook/SMS when budget hits 80% or 100%. Requires N8N integration (Step 8).
- **Soft budget ceiling:** No "warn at 100%, hard-stop at 120%" — single threshold for MVP.
- **Per-intent budget:** All intents cost the same rate. No "cancel is cheaper than booking."
- **Billing dashboard:** Budget data in Redis is for the guardrail only, not for tenant-facing reporting.
