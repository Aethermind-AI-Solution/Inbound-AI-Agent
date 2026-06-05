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
