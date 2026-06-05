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
