from __future__ import annotations

from typing import Protocol


class BudgetTracker(Protocol):
    async def check_budget(self, tenant_id: str, monthly_budget_inr: float) -> bool: ...

    async def record_usage(self, tenant_id: str, duration_seconds: float) -> None: ...
