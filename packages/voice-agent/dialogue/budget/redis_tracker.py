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
        return int(eom.timestamp()) + 7 * 86400

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
