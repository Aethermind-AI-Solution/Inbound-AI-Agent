"""Python data adapter for the Voice Booking Agent.

Wraps the Supabase Python client with typed methods for the dialogue state
machine to use. Key invariant: idempotency_key = hold_id (one hold → one
confirm → one booking).

All methods are synchronous — the Supabase Python SDK uses synchronous
execute() calls. Wrap in asyncio.to_thread if called from an async context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional postgrest import — create a minimal substitute if not installed
# ---------------------------------------------------------------------------
try:
    from postgrest.exceptions import APIError
except ImportError:

    class APIError(Exception):  # type: ignore[no-redef]
        def __init__(self, details):
            self.details = details
            super().__init__(str(details))


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class SlotConflictError(Exception):
    """Raised when a slot is already held (unique constraint violation)."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CallerInfo:
    """Identifies a caller within a tenant."""

    id: str
    phone: str
    tenant_id: str
    verified_at: datetime | None


@dataclass
class HoldResult:
    """Represents a successful slot_lock row."""

    hold_id: str
    resource_id: str
    start_ts: datetime
    expires_at: datetime


@dataclass
class BookingResult:
    """Represents a confirmed (or idempotently returned) booking."""

    booking_id: str
    idempotency_key: str
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: str | datetime | None) -> datetime | None:
    """Parse an ISO 8601 string to a timezone-aware datetime, or return None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _is_conflict_error(exc: Exception) -> bool:
    """Return True if the exception represents a unique constraint violation."""
    details = getattr(exc, "details", None)
    if details is not None:
        # details may be a dict or a string
        if isinstance(details, dict):
            code = str(details.get("code", ""))
            message = str(details.get("message", "")).lower()
        else:
            code = str(details)
            message = str(details).lower()

        if code == "23505":
            return True
        if "unique" in message:
            return True

    # Fallback: check string representation
    err_str = str(exc).lower()
    if "23505" in err_str or "unique" in err_str:
        return True

    return False


# ---------------------------------------------------------------------------
# DataAdapter
# ---------------------------------------------------------------------------


class DataAdapter:
    """Synchronous Supabase-backed data access layer for the voice agent.

    Parameters
    ----------
    client:
        A configured Supabase client (``supabase.create_client(...)``).
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Callers
    # ------------------------------------------------------------------

    def resolve_or_create_caller(self, tenant_id: str, phone: str) -> CallerInfo:
        """Return CallerInfo for the given phone/tenant, creating if absent.

        The callers table has a UNIQUE(tenant_id, phone) constraint so a
        concurrent insert race is safe — the second insert would fail and
        a re-fetch would succeed.
        """
        result = (
            self._client.table("callers")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("phone", phone)
            .maybe_single()
            .execute()
        )

        if result.data:
            row = result.data
            return CallerInfo(
                id=row["id"],
                phone=row["phone"],
                tenant_id=row["tenant_id"],
                verified_at=_parse_dt(row.get("verified_at")),
            )

        # Caller not found — insert a new one
        insert_result = (
            self._client.table("callers")
            .insert({"tenant_id": tenant_id, "phone": phone})
            .maybe_single()
            .execute()
        )

        # insert returns a list; grab first element
        data = insert_result.data
        if isinstance(data, list):
            row = data[0]
        else:
            row = data

        return CallerInfo(
            id=row["id"],
            phone=row["phone"],
            tenant_id=row["tenant_id"],
            verified_at=_parse_dt(row.get("verified_at")),
        )

    # ------------------------------------------------------------------
    # Slot locks (holds)
    # ------------------------------------------------------------------

    def hold_slot(
        self,
        tenant_id: str,
        resource_id: str,
        start_ts: datetime,
        caller_id: str,
        hold_ttl_seconds: int = 180,
    ) -> HoldResult:
        """Insert a slot_lock row and return HoldResult.

        Raises
        ------
        SlotConflictError
            If the slot is already locked (unique constraint on
            (resource_id, start_ts)).
        """
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=UTC)

        expires_at = start_ts + timedelta(seconds=hold_ttl_seconds)

        payload = {
            "resource_id": resource_id,
            "tenant_id": tenant_id,
            "start_ts": start_ts.isoformat(),
            "expires_at": expires_at.isoformat(),
            "caller_id": caller_id,
        }

        try:
            result = self._client.table("slot_locks").insert(payload).maybe_single().execute()
        except Exception as exc:
            if _is_conflict_error(exc):
                raise SlotConflictError(
                    f"Slot already held: resource={resource_id} start={start_ts}"
                ) from exc
            raise

        data = result.data
        if isinstance(data, list):
            row = data[0]
        else:
            row = data

        return HoldResult(
            hold_id=row["id"],
            resource_id=row["resource_id"],
            start_ts=_parse_dt(row["start_ts"]) or start_ts,
            expires_at=_parse_dt(row["expires_at"]) or expires_at,
        )

    # ------------------------------------------------------------------
    # Bookings
    # ------------------------------------------------------------------

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
        """Confirm a booking, idempotent on hold_id.

        First checks for an existing booking with idempotency_key=hold_id.
        If found, returns it (idempotent). Otherwise inserts a new row.
        """
        # Idempotency check: look for existing booking
        existing = (
            self._client.table("bookings")
            .select("id, idempotency_key, status")
            .eq("tenant_id", tenant_id)
            .eq("idempotency_key", hold_id)
            .maybe_single()
            .execute()
        )

        if existing.data:
            row = existing.data
            return BookingResult(
                booking_id=row["id"],
                idempotency_key=row["idempotency_key"],
                status=row["status"],
            )

        # No existing booking — insert new one
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=UTC)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=UTC)

        payload = {
            "tenant_id": tenant_id,
            "service_id": service_id,
            "resource_id": resource_id,
            "caller_id": caller_id,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "status": "confirmed",
            "custom_values": custom_values,
            "idempotency_key": hold_id,
        }

        insert_result = self._client.table("bookings").insert(payload).maybe_single().execute()

        data = insert_result.data
        if isinstance(data, list):
            row = data[0]
        else:
            row = data

        return BookingResult(
            booking_id=row["id"],
            idempotency_key=row["idempotency_key"],
            status=row["status"],
        )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def check_availability(
        self,
        tenant_id: str,
        service_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Return resources with their blocked slots in the given date range.

        Each item in the returned list has the shape::

            {
                "resource_id": str,
                "resource_name": str,
                "blocked_slots": [
                    {"start_ts": str, "end_ts": str | None, "reason": str},
                    ...
                ],
            }
        """
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        # 1. Fetch all resources of the requested type for this tenant
        resources_result = (
            self._client.table("resources")
            .select("id, name, type, tenant_id")
            .eq("tenant_id", tenant_id)
            .eq("type", resource_type)
            .execute()
        )
        resources = resources_result.data or []

        # 2. Fetch confirmed bookings in the date range
        bookings_result = (
            self._client.table("bookings")
            .select("resource_id, start_ts, end_ts, status")
            .eq("tenant_id", tenant_id)
            .eq("status", "confirmed")
            .gte("start_ts", start_date.isoformat())
            .lte("start_ts", end_date.isoformat())
            .execute()
        )
        bookings = bookings_result.data or []

        # 3. Fetch active locks in the date range (not yet expired)
        now_iso = datetime.now(tz=UTC).isoformat()
        locks_result = (
            self._client.table("slot_locks")
            .select("resource_id, start_ts, expires_at")
            .eq("tenant_id", tenant_id)
            .gte("start_ts", start_date.isoformat())
            .gt("expires_at", now_iso)
            .execute()
        )
        locks = locks_result.data or []

        # 4. Build index of blocked slots per resource
        blocked: dict[str, list[dict[str, Any]]] = {}
        for b in bookings:
            rid = b["resource_id"]
            blocked.setdefault(rid, []).append(
                {"start_ts": b["start_ts"], "end_ts": b.get("end_ts"), "reason": "booking"}
            )
        for lock in locks:
            rid = lock["resource_id"]
            blocked.setdefault(rid, []).append(
                {
                    "start_ts": lock["start_ts"],
                    "end_ts": lock.get("expires_at"),
                    "reason": "hold",
                }
            )

        # 5. Build result list
        output: list[dict[str, Any]] = []
        for r in resources:
            rid = r["id"]
            output.append(
                {
                    "resource_id": rid,
                    "resource_name": r.get("name", ""),
                    "blocked_slots": blocked.get(rid, []),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Booking lookup
    # ------------------------------------------------------------------

    def lookup_bookings(self, tenant_id: str, caller_id: str) -> list[dict[str, Any]]:
        """Return confirmed bookings for the caller, ordered by start_ts."""
        result = (
            self._client.table("bookings")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("caller_id", caller_id)
            .order("start_ts")
            .execute()
        )
        return result.data or []

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_booking(self, tenant_id: str, booking_id: str, caller_id: str) -> bool:
        """Cancel a confirmed booking.

        Returns True if the booking was found and updated, False otherwise.
        """
        result = (
            self._client.table("bookings")
            .update({"status": "cancelled"})
            .eq("id", booking_id)
            .eq("tenant_id", tenant_id)
            .eq("caller_id", caller_id)
            .eq("status", "confirmed")
            .execute()
        )
        return bool(result.data)
