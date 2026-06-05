# Auth + Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-intent auth gating (ANI soft-auth) and a Redis-backed budget guardrail with hybrid gate-at-start + duration-tracking-at-end to the existing dialogue state machine.

**Architecture:** `check_auth()` function gates intents based on `action_policy` config and `CallerInfo.is_new`. `BudgetTracker` protocol with `RedisBudgetTracker` (shared async Redis client) and `InMemoryBudgetTracker` for tests. Budget checked at call start in `DialogueManager.start()`, usage recorded at call end via `PipelineAdapter`.

**Tech Stack:** redis-py (`redis.asyncio`), pytest + pytest-asyncio

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/voice-agent/dialogue/auth.py` | `check_auth(context, intent) -> bool` pure function |
| `packages/voice-agent/dialogue/budget/__init__.py` | Package exports |
| `packages/voice-agent/dialogue/budget/base.py` | `BudgetTracker` protocol |
| `packages/voice-agent/dialogue/budget/memory_tracker.py` | `InMemoryBudgetTracker` for tests |
| `packages/voice-agent/dialogue/budget/redis_tracker.py` | `RedisBudgetTracker` with shared Redis client |
| `packages/voice-agent/tests/test_auth.py` | Auth gating tests (~8) |
| `packages/voice-agent/tests/test_budget_tracker.py` | Budget tracker tests (~10) |

---

### Task 1: Add `is_new` flag to CallerInfo + update MockDataAdapter

Add the `is_new: bool` field to `CallerInfo` so auth can distinguish found vs created callers. Update `MockDataAdapter` to set it, and fix all existing test references.

**Files:**
- Modify: `packages/voice-agent/data/adapter.py`
- Modify: `packages/voice-agent/data/mock_adapter.py`
- Modify: `packages/voice-agent/tests/test_mock_adapter.py`
- Modify: `packages/voice-agent/tests/test_states/test_identify_caller.py`
- Modify: `packages/voice-agent/tests/test_dialogue_manager.py`

- [ ] **Step 1: Write a test for is_new flag in MockDataAdapter**

Add to `packages/voice-agent/tests/test_mock_adapter.py`:

```python
class TestIsNewFlag:
    def test_first_call_creates_new_caller(self):
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+919999999999")
        assert caller.is_new is True

    def test_second_call_finds_existing_caller(self):
        adapter = MockDataAdapter()
        adapter.resolve_or_create_caller("t1", "+919999999999")
        caller = adapter.resolve_or_create_caller("t1", "+919999999999")
        assert caller.is_new is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_mock_adapter.py::TestIsNewFlag -v`
Expected: FAIL — `TypeError: CallerInfo.__init__() got an unexpected keyword argument 'is_new'`

- [ ] **Step 3: Add `is_new` to CallerInfo dataclass**

In `packages/voice-agent/data/adapter.py`, add `is_new` field to `CallerInfo`:

```python
@dataclass
class CallerInfo:
    """Identifies a caller within a tenant."""

    id: str
    phone: str
    tenant_id: str
    verified_at: datetime | None
    is_new: bool = False
```

- [ ] **Step 4: Update MockDataAdapter to track callers and set is_new**

Replace the `resolve_or_create_caller` method in `packages/voice-agent/data/mock_adapter.py`:

```python
class MockDataAdapter:
    def __init__(self) -> None:
        self._bookings: dict[str, dict[str, Any]] = {
            "booking-1": {
                "id": "booking-1",
                "tenant_id": "t1",
                "service_id": "s1",
                "resource_id": "r1",
                "caller_id": "caller-1",
                "start_ts": (datetime.now(tz=UTC) + timedelta(days=2)).isoformat(),
                "status": "confirmed",
                "idempotency_key": "hold-existing",
            },
        }
        self._confirmed_holds: dict[str, str] = {}
        self._callers: dict[str, CallerInfo] = {}

    def resolve_or_create_caller(self, tenant_id: str, phone: str) -> CallerInfo:
        key = f"{tenant_id}:{phone}"
        if key in self._callers:
            existing = self._callers[key]
            return CallerInfo(
                id=existing.id,
                phone=existing.phone,
                tenant_id=existing.tenant_id,
                verified_at=existing.verified_at,
                is_new=False,
            )
        caller = CallerInfo(
            id=f"caller-{len(self._callers) + 1}",
            phone=phone,
            tenant_id=tenant_id,
            verified_at=None,
            is_new=True,
        )
        self._callers[key] = caller
        return caller
```

- [ ] **Step 5: Fix existing test references that construct CallerInfo without is_new**

In `packages/voice-agent/tests/test_states/test_identify_caller.py`, update the CallerInfo construction:

```python
caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None, is_new=False)
```

In `packages/voice-agent/tests/test_dialogue_manager.py`, update the fixture:

```python
adapter.resolve_or_create_caller.return_value = CallerInfo(
    id="c1", phone="+919876543210", tenant_id="t1", verified_at=None, is_new=False
)
```

- [ ] **Step 6: Run all tests to verify everything passes**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: All tests PASS (is_new defaults to False, so existing tests are unaffected beyond the explicit constructions)

- [ ] **Step 7: Commit**

```bash
git add packages/voice-agent/data/adapter.py packages/voice-agent/data/mock_adapter.py packages/voice-agent/tests/test_mock_adapter.py packages/voice-agent/tests/test_states/test_identify_caller.py packages/voice-agent/tests/test_dialogue_manager.py
git commit -m "feat: add is_new flag to CallerInfo for soft-auth"
```

---

### Task 2: Auth gating — `check_auth` function + tests

Build the `check_auth` pure function and its test suite.

**Files:**
- Create: `packages/voice-agent/dialogue/auth.py`
- Create: `packages/voice-agent/tests/test_auth.py`

- [ ] **Step 1: Write failing tests**

Create `packages/voice-agent/tests/test_auth.py`:

```python
from __future__ import annotations

import pytest

from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.dialogue.models import CallContext
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


def make_context(is_new: bool, action_policy: dict[str, str] | None = None) -> CallContext:
    overrides = {}
    if action_policy is not None:
        from packages.voice_agent.config.models import AuthConfig
        overrides["auth"] = AuthConfig(
            levels=["soft"],
            action_policy=action_policy,
            soft_match_fields=["caller_number"],
        )
    ctx = CallContext(
        tenant_config=make_tenant_config(**overrides),
        caller_phone="+919876543210",
        call_id="call-001",
    )
    ctx.caller = CallerInfo(
        id="c1", phone="+919876543210", tenant_id="t1",
        verified_at=None, is_new=is_new,
    )
    return ctx


class TestCheckAuth:
    def test_action_policy_none_always_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"new_booking": "none"})
        assert check_auth(ctx, "new_booking") is True

    def test_soft_with_existing_caller_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=False, action_policy={"cancel": "soft"})
        assert check_auth(ctx, "cancel") is True

    def test_soft_with_new_caller_not_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"cancel": "soft"})
        assert check_auth(ctx, "cancel") is False

    def test_intent_not_in_policy_defaults_to_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"new_booking": "none"})
        assert check_auth(ctx, "status") is True

    def test_no_caller_defaults_to_not_authorized_for_soft(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"cancel": "soft"})
        ctx.caller = None
        assert check_auth(ctx, "cancel") is False

    def test_new_booking_none_cancel_soft_combo(self):
        from packages.voice_agent.dialogue.auth import check_auth
        policy = {"new_booking": "none", "cancel": "soft", "status": "soft"}
        ctx_new = make_context(is_new=True, action_policy=policy)
        ctx_existing = make_context(is_new=False, action_policy=policy)
        assert check_auth(ctx_new, "new_booking") is True
        assert check_auth(ctx_new, "cancel") is False
        assert check_auth(ctx_existing, "cancel") is True
        assert check_auth(ctx_existing, "status") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.dialogue.auth'`

- [ ] **Step 3: Implement check_auth**

Create `packages/voice-agent/dialogue/auth.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.voice_agent.dialogue.models import CallContext


def check_auth(context: CallContext, intent: str) -> bool:
    policy = context.tenant_config.auth.action_policy
    required_level = policy.get(intent, "none")
    if required_level == "none":
        return True
    if required_level == "soft":
        if context.caller is None:
            return False
        return not context.caller.is_new
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_auth.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/dialogue/auth.py packages/voice-agent/tests/test_auth.py
git commit -m "feat: add check_auth function for per-intent soft-auth gating"
```

---

### Task 3: Integrate auth into IntentState

Wire `check_auth` into `IntentState.handle()` so unauthorized intents get a helpful redirect or callback capture after 2 rejections.

**Files:**
- Modify: `packages/voice-agent/dialogue/states/intent.py`
- Modify: `packages/voice-agent/tests/test_states/test_intent.py`

- [ ] **Step 1: Write failing tests for auth integration in IntentState**

Append to `packages/voice-agent/tests/test_states/test_intent.py`:

```python
from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.config.models import AuthConfig


class TestIntentAuthGating:
    @pytest.mark.asyncio
    async def test_authorized_intent_proceeds(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=False,
        )
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("cancel my booking"), context)
        assert action.next_state == CallState.LOOKUP_BOOKINGS

    @pytest.mark.asyncio
    async def test_unauthorized_intent_offers_alternative(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft", "new_booking": "none"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=True,
        )
        state = IntentState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("cancel my booking"), context)
        assert action.type == ActionType.ASK
        assert action.next_state is None

    @pytest.mark.asyncio
    async def test_unauthorized_twice_goes_to_callback(self, deps, context):
        context.tenant_config.auth.action_policy = {"cancel": "soft", "status": "soft"}
        context.caller = CallerInfo(
            id="c1", phone="+919876543210", tenant_id="t1",
            verified_at=None, is_new=True,
        )
        state = IntentState(deps)
        await state.enter(context)
        await state.handle(make_transcription("cancel my booking"), context)
        action = await state.handle(make_transcription("check my status"), context)
        assert action.next_state == CallState.CALLBACK_CAPTURE
        assert context.fallback_reason == "auth_required"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_states/test_intent.py::TestIntentAuthGating -v`
Expected: FAIL — the current IntentState doesn't call check_auth

- [ ] **Step 3: Update IntentState to call check_auth**

Replace `packages/voice-agent/dialogue/states/intent.py`:

```python
from __future__ import annotations

from packages.voice_agent.dialogue.auth import check_auth
from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
)


class IntentState(BaseState):
    name = CallState.INTENT

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._reprompt_count = 0
        self._auth_fail_count = 0

    async def enter(self, context: CallContext) -> Action:
        self._reprompt_count = 0
        self._auth_fail_count = 0
        return Action(
            type=ActionType.ASK,
            text=(
                "How can I help you today? I can help with booking an appointment, "
                "checking an existing booking, rescheduling, or cancelling."
            ),
        )

    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        result = await self.deps.nlu.classify_intent(event.text or "", [])

        if result.intent == "new_booking":
            if not check_auth(context, "new_booking"):
                return self._handle_auth_failure(context)
            context.intent = "new_booking"
            return Action(type=ActionType.TRANSITION, next_state=CallState.COLLECT_SERVICE)

        if result.intent in ("cancel", "reschedule", "status"):
            if not check_auth(context, result.intent):
                return self._handle_auth_failure(context)
            context.intent = result.intent
            return Action(type=ActionType.TRANSITION, next_state=CallState.LOOKUP_BOOKINGS)

        self._reprompt_count += 1
        if self._reprompt_count >= 3:
            context.fallback_reason = "repeated_failure"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)

        return Action(
            type=ActionType.ASK,
            text=(
                "I didn't quite catch that. Would you like to book a new appointment, "
                "or check on an existing one?"
            ),
        )

    def _handle_auth_failure(self, context: CallContext) -> Action:
        self._auth_fail_count += 1
        if self._auth_fail_count >= 2:
            context.fallback_reason = "auth_required"
            return Action(type=ActionType.TRANSITION, next_state=CallState.CALLBACK_CAPTURE)
        return Action(
            type=ActionType.ASK,
            text=(
                "I can help with new bookings, but I'll need to verify your identity "
                "for that request. Would you like to book an appointment instead?"
            ),
        )
```

- [ ] **Step 4: Run all intent tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_states/test_intent.py -v`
Expected: All tests PASS (existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/dialogue/states/intent.py packages/voice-agent/tests/test_states/test_intent.py
git commit -m "feat: integrate auth gating into IntentState"
```

---

### Task 4: BudgetTracker protocol + InMemoryBudgetTracker

Build the `BudgetTracker` protocol and in-memory implementation for tests.

**Files:**
- Create: `packages/voice-agent/dialogue/budget/__init__.py`
- Create: `packages/voice-agent/dialogue/budget/base.py`
- Create: `packages/voice-agent/dialogue/budget/memory_tracker.py`
- Create: `packages/voice-agent/tests/test_budget_tracker.py`

- [ ] **Step 1: Write failing tests for InMemoryBudgetTracker**

Create `packages/voice-agent/tests/test_budget_tracker.py`:

```python
from __future__ import annotations

import pytest


class TestInMemoryBudgetTracker:
    @pytest.mark.asyncio
    async def test_under_budget_returns_true(self):
        from packages.voice_agent.dialogue.budget.memory_tracker import (
            InMemoryBudgetTracker,
        )
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        assert await tracker.check_budget("t1", 5000.0) is True

    @pytest.mark.asyncio
    async def test_over_budget_returns_false(self):
        from packages.voice_agent.dialogue.budget.memory_tracker import (
            InMemoryBudgetTracker,
        )
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 150000.0)  # 2500 min = 5000 INR
        assert await tracker.check_budget("t1", 5000.0) is False

    @pytest.mark.asyncio
    async def test_accumulates_correctly(self):
        from packages.voice_agent.dialogue.budget.memory_tracker import (
            InMemoryBudgetTracker,
        )
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 60.0)  # 1 min = 200 paisa
        await tracker.record_usage("t1", 120.0)  # 2 min = 400 paisa
        # total = 600 paisa = 6 INR
        assert await tracker.check_budget("t1", 5.0) is False
        assert await tracker.check_budget("t1", 7.0) is True

    @pytest.mark.asyncio
    async def test_different_tenants_tracked_separately(self):
        from packages.voice_agent.dialogue.budget.memory_tracker import (
            InMemoryBudgetTracker,
        )
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 150000.0)  # t1 over budget
        assert await tracker.check_budget("t1", 5000.0) is False
        assert await tracker.check_budget("t2", 5000.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_budget_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create the budget package with protocol and in-memory implementation**

Create `packages/voice-agent/dialogue/budget/base.py`:

```python
from __future__ import annotations

from typing import Protocol


class BudgetTracker(Protocol):
    async def check_budget(self, tenant_id: str, monthly_budget_inr: float) -> bool: ...

    async def record_usage(self, tenant_id: str, duration_seconds: float) -> None: ...
```

Create `packages/voice-agent/dialogue/budget/memory_tracker.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone


class InMemoryBudgetTracker:
    def __init__(self, cost_per_minute_inr: float = 2.0) -> None:
        self._cost_per_minute_inr = cost_per_minute_inr
        self._usage_paisa: dict[str, int] = {}

    def _key(self, tenant_id: str) -> str:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return f"{tenant_id}:{month}"

    async def check_budget(self, tenant_id: str, monthly_budget_inr: float) -> bool:
        key = self._key(tenant_id)
        accumulated = self._usage_paisa.get(key, 0)
        return accumulated < int(monthly_budget_inr * 100)

    async def record_usage(self, tenant_id: str, duration_seconds: float) -> None:
        cost_paisa = int(duration_seconds / 60 * self._cost_per_minute_inr * 100)
        key = self._key(tenant_id)
        self._usage_paisa[key] = self._usage_paisa.get(key, 0) + cost_paisa
```

Create `packages/voice-agent/dialogue/budget/__init__.py`:

```python
from packages.voice_agent.dialogue.budget.base import BudgetTracker
from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker

__all__ = ["BudgetTracker", "InMemoryBudgetTracker"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_budget_tracker.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/dialogue/budget/
git add packages/voice-agent/tests/test_budget_tracker.py
git commit -m "feat: add BudgetTracker protocol and InMemoryBudgetTracker"
```

---

### Task 5: RedisBudgetTracker + tests

Build the Redis-backed implementation with paisa integers, monthly keys, TTL, and fail-open error handling.

**Files:**
- Create: `packages/voice-agent/dialogue/budget/redis_tracker.py`
- Modify: `packages/voice-agent/dialogue/budget/__init__.py`
- Modify: `packages/voice-agent/tests/test_budget_tracker.py`

- [ ] **Step 1: Install redis-py**

Run: `.venv/bin/pip install redis`

- [ ] **Step 2: Write failing tests for RedisBudgetTracker**

Append to `packages/voice-agent/tests/test_budget_tracker.py`:

```python
from unittest.mock import AsyncMock, MagicMock


class TestRedisBudgetTracker:
    @pytest.mark.asyncio
    async def test_correct_key_format(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        await tracker.check_budget("t1", 5000.0)
        call_args = mock_redis.get.call_args[0][0]
        assert call_args.startswith("budget:t1:")
        assert len(call_args.split(":")) == 3  # budget:t1:2026-06

    @pytest.mark.asyncio
    async def test_under_budget(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"100000")  # 1000 INR
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        assert await tracker.check_budget("t1", 5000.0) is True

    @pytest.mark.asyncio
    async def test_over_budget(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"500000")  # 5000 INR
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        assert await tracker.check_budget("t1", 5000.0) is False

    @pytest.mark.asyncio
    async def test_paisa_conversion_on_record(self):
        mock_redis = AsyncMock()
        mock_redis.incrby = AsyncMock(return_value=200)
        mock_redis.ttl = AsyncMock(return_value=-1)
        mock_redis.expireat = AsyncMock()
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 60.0)  # 1 min * 2 INR * 100 = 200 paisa
        call_args = mock_redis.incrby.call_args
        assert call_args[0][1] == 200

    @pytest.mark.asyncio
    async def test_ttl_set_on_new_key(self):
        mock_redis = AsyncMock()
        mock_redis.incrby = AsyncMock(return_value=200)  # first write
        mock_redis.ttl = AsyncMock(return_value=-1)  # no TTL yet
        mock_redis.expireat = AsyncMock()
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 60.0)
        mock_redis.expireat.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_budget_redis_error_returns_true(self):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        assert await tracker.check_budget("t1", 5000.0) is True

    @pytest.mark.asyncio
    async def test_record_usage_redis_error_swallows(self):
        mock_redis = AsyncMock()
        mock_redis.incrby = AsyncMock(side_effect=ConnectionError("Redis down"))
        from packages.voice_agent.dialogue.budget.redis_tracker import (
            RedisBudgetTracker,
        )
        tracker = RedisBudgetTracker(redis=mock_redis, cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 60.0)  # should not raise
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_budget_tracker.py::TestRedisBudgetTracker -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement RedisBudgetTracker**

Create `packages/voice-agent/dialogue/budget/redis_tracker.py`:

```python
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisBudgetTracker:
    def __init__(self, redis: redis.Redis, cost_per_minute_inr: float = 2.0) -> None:
        self._redis = redis
        self._cost_per_minute_inr = cost_per_minute_inr

    def _key(self, tenant_id: str) -> str:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return f"budget:{tenant_id}:{month}"

    def _end_of_month_ts(self) -> int:
        now = datetime.now(timezone.utc)
        last_day = calendar.monthrange(now.year, now.month)[1]
        eom = now.replace(day=last_day, hour=23, minute=59, second=59)
        return int(eom.timestamp()) + 7 * 86400  # +7 days for auditing

    async def check_budget(self, tenant_id: str, monthly_budget_inr: float) -> bool:
        try:
            value = await self._redis.get(self._key(tenant_id))
            accumulated = int(value) if value else 0
            return accumulated < int(monthly_budget_inr * 100)
        except Exception:
            logger.exception("Redis error in check_budget, failing open")
            return True

    async def record_usage(self, tenant_id: str, duration_seconds: float) -> None:
        cost_paisa = int(duration_seconds / 60 * self._cost_per_minute_inr * 100)
        if cost_paisa <= 0:
            return
        key = self._key(tenant_id)
        try:
            await self._redis.incrby(key, cost_paisa)
            ttl = await self._redis.ttl(key)
            if ttl == -1:
                await self._redis.expireat(key, self._end_of_month_ts())
        except Exception:
            logger.exception("Redis error in record_usage, swallowing")
```

- [ ] **Step 5: Update budget __init__.py exports**

Replace `packages/voice-agent/dialogue/budget/__init__.py`:

```python
from packages.voice_agent.dialogue.budget.base import BudgetTracker
from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
from packages.voice_agent.dialogue.budget.redis_tracker import RedisBudgetTracker

__all__ = ["BudgetTracker", "InMemoryBudgetTracker", "RedisBudgetTracker"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_budget_tracker.py -v`
Expected: All 11 tests PASS (4 InMemory + 7 Redis)

- [ ] **Step 7: Commit**

```bash
git add packages/voice-agent/dialogue/budget/
git add packages/voice-agent/tests/test_budget_tracker.py
git commit -m "feat: add RedisBudgetTracker with fail-open error handling"
```

---

### Task 6: Add `cost_per_minute_inr` to GuardrailsConfig + `budget_checked` to CallContext

Update config and context models for budget integration.

**Files:**
- Modify: `packages/voice-agent/config/models.py`
- Modify: `packages/voice-agent/dialogue/models.py`

- [ ] **Step 1: Add `cost_per_minute_inr` to GuardrailsConfig**

In `packages/voice-agent/config/models.py`, update `GuardrailsConfig`:

```python
class GuardrailsConfig(BaseModel):
    max_call_seconds: int = PydanticField(gt=0)
    max_turns: int = PydanticField(gt=0)
    monthly_budget_inr: float = PydanticField(gt=0)
    cost_per_minute_inr: float = PydanticField(gt=0, default=2.0)
    scope: str
    recording_consent: bool
    retention_days: int = PydanticField(gt=0)
```

- [ ] **Step 2: Add `budget_checked` to CallContext**

In `packages/voice-agent/dialogue/models.py`, add to `CallContext`:

```python
@dataclass
class CallContext:
    tenant_config: Any  # TenantConfig
    caller_phone: str
    call_id: str
    caller: Any | None = None  # CallerInfo
    intent: str | None = None
    language: str = "en-IN"
    slots: BookingSlots = field(default_factory=BookingSlots)
    turn_count: int = 0
    silence_count: int = 0
    call_start: float = field(default_factory=time.monotonic)
    fallback_reason: str | None = None
    budget_checked: bool = False
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: All tests PASS (both fields have defaults)

- [ ] **Step 4: Commit**

```bash
git add packages/voice-agent/config/models.py packages/voice-agent/dialogue/models.py
git commit -m "feat: add cost_per_minute_inr to config, budget_checked to CallContext"
```

---

### Task 7: Integrate budget into DialogueManager

Wire `BudgetTracker` into `DialogueManager.start()` for gate-at-start, add `record_call_usage()` method, and update checkpoint.

**Files:**
- Modify: `packages/voice-agent/dialogue/manager.py`
- Modify: `packages/voice-agent/tests/test_dialogue_manager.py`

- [ ] **Step 1: Write failing tests for budget integration**

Append to `packages/voice-agent/tests/test_dialogue_manager.py`:

```python
from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker


class TestBudgetGate:
    @pytest.mark.asyncio
    async def test_under_budget_proceeds_normally(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        m._make_date_resolver = MagicMock(return_value=resolver)
        action = await m.start()
        assert action.type == ActionType.ASK
        assert m.current_state.name == CallState.INTENT

    @pytest.mark.asyncio
    async def test_over_budget_goes_to_callback(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        await tracker.record_usage("t1", 150000.0)  # 5000 INR
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        m._make_date_resolver = MagicMock(return_value=resolver)
        action = await m.start()
        assert m.current_state.name == CallState.CALLBACK_CAPTURE
        assert m.context.fallback_reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_record_call_usage(self, config, data_adapter, resolver):
        tracker = InMemoryBudgetTracker(cost_per_minute_inr=2.0)
        nlu = StubNLUService()
        checkpoint = InMemoryCheckpointStore()
        m = DialogueManager(
            config=config, data_adapter=data_adapter, nlu=nlu,
            checkpoint_store=checkpoint, caller_phone="+919876543210",
            call_id="call-001", budget_tracker=tracker,
        )
        await m.record_call_usage(300.0)  # 5 min = 10 INR = 1000 paisa
        assert await tracker.check_budget("t1", 9.0) is False
        assert await tracker.check_budget("t1", 11.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_dialogue_manager.py::TestBudgetGate -v`
Expected: FAIL — `TypeError: DialogueManager.__init__() got an unexpected keyword argument 'budget_tracker'`

- [ ] **Step 3: Update DialogueManager**

In `packages/voice-agent/dialogue/manager.py`, make these changes:

Add to imports at the top:

```python
if TYPE_CHECKING:
    from packages.voice_agent.config.models import TenantConfig
    from packages.voice_agent.data.adapter import DataAdapter
    from packages.voice_agent.dialogue.budget.base import BudgetTracker
    from packages.voice_agent.dialogue.checkpoint.base import CheckpointStore
    from packages.voice_agent.dialogue.nlu.base import NLUService
```

Update `__init__`:

```python
def __init__(
    self,
    config: TenantConfig,
    data_adapter: DataAdapter,
    nlu: NLUService,
    checkpoint_store: CheckpointStore,
    caller_phone: str,
    call_id: str,
    budget_tracker: BudgetTracker | None = None,
) -> None:
    self.context = CallContext(
        tenant_config=config,
        caller_phone=caller_phone,
        call_id=call_id,
    )
    self.data_adapter = data_adapter
    self.nlu = nlu
    self.checkpoint_store = checkpoint_store
    self.budget_tracker = budget_tracker
    self.states: dict[str, BaseState] = {}
    self.current_state: BaseState | None = None
```

Update `start()`:

```python
async def start(self) -> Action:
    self._register_states()
    if self.budget_tracker and not self.context.budget_checked:
        tenant_id = self.context.tenant_config.meta.tenant_id
        budget = self.context.tenant_config.guardrails.monthly_budget_inr
        within_budget = await self.budget_tracker.check_budget(tenant_id, budget)
        self.context.budget_checked = True
        if not within_budget:
            self.context.fallback_reason = "budget_exceeded"
            return await self._enter_state(CallState.CALLBACK_CAPTURE)
    return await self._enter_state(CallState.GREETING)
```

Add `record_call_usage` method:

```python
async def record_call_usage(self, duration_seconds: float) -> None:
    if self.budget_tracker:
        tenant_id = self.context.tenant_config.meta.tenant_id
        await self.budget_tracker.record_usage(tenant_id, duration_seconds)
```

Update `_write_checkpoint` to include `budget_checked`:

```python
async def _write_checkpoint(self) -> None:
    data = {
        "state": self.current_state.name,
        "intent": self.context.intent,
        "slots": asdict(self.context.slots),
        "turn_count": self.context.turn_count,
        "caller_id": self.context.caller.id if self.context.caller else None,
        "language": self.context.language,
        "call_id": self.context.call_id,
        "budget_checked": self.context.budget_checked,
    }
    await self.checkpoint_store.save(
        self.context.caller_phone, data, ttl_seconds=900
    )
```

Update `_restore_context` to restore `budget_checked`:

```python
def _restore_context(self, checkpoint: dict) -> None:
    self.context.intent = checkpoint.get("intent")
    self.context.turn_count = checkpoint.get("turn_count", 0)
    self.context.language = checkpoint.get("language", "en-IN")
    self.context.budget_checked = checkpoint.get("budget_checked", False)
    slots_data = checkpoint.get("slots", {})
    self.context.slots = BookingSlots(**slots_data)
```

- [ ] **Step 4: Update existing manager fixture to include budget_tracker=None**

The existing `manager` fixture in `test_dialogue_manager.py` doesn't pass `budget_tracker`, so it will default to `None` via the optional parameter. No change needed — but verify existing tests still pass.

- [ ] **Step 5: Run all manager tests to verify they pass**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_dialogue_manager.py -v`
Expected: All tests PASS (existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add packages/voice-agent/dialogue/manager.py packages/voice-agent/tests/test_dialogue_manager.py
git commit -m "feat: integrate BudgetTracker into DialogueManager with gate-at-start"
```

---

### Task 8: Integrate budget recording into PipelineAdapter

Wire `record_call_usage()` into PipelineAdapter so duration is recorded when the call ends.

**Files:**
- Modify: `packages/voice-agent/pipeline_adapter.py`
- Modify: `packages/voice-agent/tests/test_pipeline_adapter.py`

- [ ] **Step 1: Write failing test for budget recording on call end**

Append to `packages/voice-agent/tests/test_pipeline_adapter.py`:

```python
class TestBudgetRecording:
    @pytest.mark.asyncio
    async def test_records_usage_on_end_call(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.return_value = Action(
            type=ActionType.END_CALL, text="Goodbye!"
        )
        mock_dm.record_call_usage = AsyncMock()

        frame = make_transcription_frame("bye", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        mock_dm.record_call_usage.assert_called_once()
        duration = mock_dm.record_call_usage.call_args[0][0]
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_records_usage_on_error(self, adapter, mock_dm):
        adapter._started = True
        pushed_frames = []
        adapter.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed_frames.append(f))

        mock_dm.handle_event.side_effect = RuntimeError("boom")
        mock_dm.record_call_usage = AsyncMock()

        frame = make_transcription_frame("hello", finalized=True)
        await adapter.process_frame(frame, MagicMock())

        mock_dm.record_call_usage.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_pipeline_adapter.py::TestBudgetRecording -v`
Expected: FAIL — `record_call_usage` not called in current code

- [ ] **Step 3: Update PipelineAdapter to record usage on call end**

In `packages/voice-agent/pipeline_adapter.py`, add a helper method and call it from the end-call and error paths:

Add `import time` to the imports.

Update `_push_action` to call `_record_usage()` before pushing EndTaskFrame:

```python
async def _push_action(self, action: Action) -> None:
    if action.type in (ActionType.ASK, ActionType.SPEAK):
        if action.text:
            await self.push_frame(
                TextFrame(text=action.text), FrameDirection.DOWNSTREAM
            )
        self._reset_silence_timer(action.timeout_s)
    elif action.type == ActionType.END_CALL:
        self._cancel_silence_timer()
        if action.text:
            await self.push_frame(
                TextFrame(text=action.text), FrameDirection.DOWNSTREAM
            )
        await self._record_usage()
        await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)
```

Update `_handle_transcription` error path:

```python
async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
    self._cancel_silence_timer()
    event = CallEvent(type=EventType.TRANSCRIPTION, text=frame.text)
    try:
        action = await self._dm.handle_event(event)
        await self._push_action(action)
    except Exception:
        logger.exception("Error in DialogueManager.handle_event()")
        await self._record_usage()
        await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)
```

Update `process_frame` error path for start():

```python
if not self._started:
    self._started = True
    try:
        action = await self._dm.start()
        await self._push_action(action)
    except Exception:
        logger.exception("Error in DialogueManager.start()")
        await self._record_usage()
        await self.push_frame(EndTaskFrame(), FrameDirection.DOWNSTREAM)
        return
```

Add the `_record_usage` helper:

```python
async def _record_usage(self) -> None:
    try:
        elapsed = time.monotonic() - self._dm.context.call_start
        await self._dm.record_call_usage(elapsed)
    except Exception:
        logger.exception("Error recording call usage")
```

- [ ] **Step 4: Run all pipeline adapter tests**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/test_pipeline_adapter.py -v`
Expected: All tests PASS (existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/pipeline_adapter.py packages/voice-agent/tests/test_pipeline_adapter.py
git commit -m "feat: record call usage in PipelineAdapter on call end"
```

---

### Task 9: Update CallbackCaptureState + run_local.py + add budget_exceeded reason message

Add the budget-exceeded reason message to CallbackCaptureState, wire budget tracker into run_local.py, and update test fixtures.

**Files:**
- Modify: `packages/voice-agent/dialogue/states/callback_capture.py`
- Modify: `packages/voice-agent/run_local.py`
- Modify: `packages/voice-agent/tests/test_states/conftest.py`

- [ ] **Step 1: Add budget_exceeded and auth_required to _REASON_MESSAGES**

In `packages/voice-agent/dialogue/states/callback_capture.py`, add to `_REASON_MESSAGES`:

```python
_REASON_MESSAGES = {
    "no_availability": (
        "We don't have any openings right now, but I'll have someone "
        "call you to help find a time."
    ),
    "tool_error": (
        "I'm experiencing a technical issue. Let me have someone follow up with you."
    ),
    "budget_exceeded": (
        "We're experiencing high demand right now. "
        "Let me take your number and have someone call you back."
    ),
    "auth_required": (
        "I'll need to verify your identity for that request. "
        "Let me have someone call you back to assist."
    ),
}
```

- [ ] **Step 2: Update run_local.py to wire InMemoryBudgetTracker**

In `packages/voice-agent/run_local.py`, add budget tracker wiring after the NLU setup and before the DialogueManager construction:

Add import and construction:

```python
    from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
    budget_tracker: Any = InMemoryBudgetTracker()

    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as aioredis
            from packages.voice_agent.dialogue.budget.redis_tracker import RedisBudgetTracker
            redis_client = aioredis.from_url(redis_url)
            budget_tracker = RedisBudgetTracker(
                redis=redis_client,
                cost_per_minute_inr=config.guardrails.cost_per_minute_inr,
            )
            logger.info("Using Redis budget tracker")
        except (ImportError, Exception):
            logger.info("Using in-memory budget tracker (Redis unavailable)")
```

Update the DialogueManager construction to include `budget_tracker`:

```python
    dm = DialogueManager(
        config=config,
        data_adapter=data,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone="+919876543210",
        call_id="local-test-001",
        budget_tracker=budget_tracker,
    )
```

- [ ] **Step 3: Verify run_local.py parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('packages/voice-agent/run_local.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/voice-agent/dialogue/states/callback_capture.py packages/voice-agent/run_local.py packages/voice-agent/tests/test_states/conftest.py
git commit -m "feat: add budget/auth reason messages, wire budget tracker into run_local"
```

---

### Task 10: Full test suite verification

Run all tests to verify nothing is broken.

**Files:**
- No changes — verification only

- [ ] **Step 1: Run entire test suite**

Run: `.venv/bin/python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: All tests PASS (~305 total: 285 existing + ~20 new)

- [ ] **Step 2: Fix any failures and re-run**

Fix any issues, re-run until green.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix: resolve test issues from auth+guardrails integration"
```

(Skip if no fixes needed.)
