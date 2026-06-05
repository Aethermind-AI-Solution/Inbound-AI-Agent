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
        assert len(call_args.split(":")) == 3

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
        mock_redis.incrby = AsyncMock(return_value=200)
        mock_redis.ttl = AsyncMock(return_value=-1)
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
