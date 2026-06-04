from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from packages.voice_agent.data.adapter import BookingResult, CallerInfo, HoldResult


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

    def resolve_or_create_caller(self, tenant_id: str, phone: str) -> CallerInfo:
        return CallerInfo(
            id="caller-1",
            phone=phone,
            tenant_id=tenant_id,
            verified_at=None,
        )

    def check_availability(
        self,
        tenant_id: str,
        service_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        return [
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
            {"resource_id": "r2", "resource_name": "Rahul", "blocked_slots": []},
        ]

    def hold_slot(
        self,
        tenant_id: str,
        resource_id: str,
        start_ts: datetime,
        caller_id: str,
        hold_ttl_seconds: int = 180,
    ) -> HoldResult:
        hold_id = f"hold-{uuid.uuid4().hex[:8]}"
        return HoldResult(
            hold_id=hold_id,
            resource_id=resource_id,
            start_ts=start_ts,
            expires_at=start_ts + timedelta(seconds=hold_ttl_seconds),
        )

    def confirm_booking(
        self,
        tenant_id: str,
        hold_id: str,
        service_id: str,
        resource_id: str,
        caller_id: str,
        start_ts: datetime,
        end_ts: datetime,
        custom_values: dict[str, Any],
    ) -> BookingResult:
        if hold_id in self._confirmed_holds:
            return BookingResult(
                booking_id=self._confirmed_holds[hold_id],
                idempotency_key=hold_id,
                status="confirmed",
            )
        booking_id = f"bk-{uuid.uuid4().hex[:8]}"
        self._confirmed_holds[hold_id] = booking_id
        self._bookings[booking_id] = {
            "id": booking_id,
            "tenant_id": tenant_id,
            "service_id": service_id,
            "resource_id": resource_id,
            "caller_id": caller_id,
            "start_ts": start_ts.isoformat(),
            "status": "confirmed",
            "idempotency_key": hold_id,
        }
        return BookingResult(
            booking_id=booking_id,
            idempotency_key=hold_id,
            status="confirmed",
        )

    def lookup_bookings(self, tenant_id: str, caller_id: str) -> list[dict[str, Any]]:
        return [
            b for b in self._bookings.values()
            if b.get("tenant_id") == tenant_id and b["caller_id"] == caller_id
        ]

    def cancel_booking(self, tenant_id: str, booking_id: str, caller_id: str) -> bool:
        b = self._bookings.get(booking_id)
        if b and b["caller_id"] == caller_id:
            b["status"] = "cancelled"
            return True
        return False
