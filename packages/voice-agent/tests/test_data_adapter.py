"""Tests for DataAdapter — Python data access layer wrapping Supabase (TDD).

Mock strategy: Supabase's fluent builder is a chain of method calls each
returning the same or next mock. We fully control the chain via MagicMock.

Supabase Python SDK query chain (synchronous):
    client.table(name)
        .select(cols)  / .insert(data) / .update(data) / .upsert(data)
        .eq(col, val)
        ...
        .maybe_single()   (returns one row or None)
        / .execute()      (returns result list)
    result.data  -> dict | list[dict] | None
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

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


from packages.voice_agent.data.adapter import (
    BookingResult,
    CallerInfo,
    DataAdapter,
    HoldResult,
    SlotConflictError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENANT_ID = "tenant-uuid-1"
CALLER_ID = "caller-uuid-1"
RESOURCE_ID = "stylist_01"
SERVICE_ID = "svc_haircut"
HOLD_ID = "lock-uuid-1"
BOOKING_ID = "booking-uuid-1"
PHONE = "+919876543210"
NOW = datetime(2026, 6, 3, 10, 0, 0, tzinfo=UTC)
END_TS = datetime(2026, 6, 3, 10, 30, 0, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 6, 3, 10, 3, 0, tzinfo=UTC)


def _make_adapter() -> tuple[DataAdapter, MagicMock]:
    """Return (adapter, mock_client)."""
    mock_client = MagicMock()
    adapter = DataAdapter(mock_client)
    return adapter, mock_client


def _chain(*return_values):
    """Build a mock chain where each call returns the next value.

    Last value is typically a MagicMock with `.execute.return_value` set.
    """
    mocks = [MagicMock() for _ in return_values]
    for i, mock in enumerate(mocks[:-1]):
        mock.return_value = mocks[i + 1]
    return mocks[0], mocks


# ---------------------------------------------------------------------------
# SlotConflictError
# ---------------------------------------------------------------------------


class TestSlotConflictError:
    def test_is_exception(self):
        err = SlotConflictError("slot taken")
        assert isinstance(err, Exception)

    def test_message(self):
        err = SlotConflictError("slot taken")
        assert "slot taken" in str(err)


# ---------------------------------------------------------------------------
# resolve_or_create_caller
# ---------------------------------------------------------------------------


class TestResolveOrCreateCaller:
    def test_returns_existing_caller(self):
        """When caller exists, return CallerInfo from DB row."""
        adapter, client = _make_adapter()

        existing_row = {
            "id": CALLER_ID,
            "phone": PHONE,
            "tenant_id": TENANT_ID,
            "verified_at": None,
        }

        # Build the mock chain for:
        # client.table("callers").select("*").eq("tenant_id", ...).eq("phone", ...)
        # .maybe_single().execute()
        exec_result = MagicMock()
        exec_result.data = existing_row

        maybe_single = MagicMock()
        maybe_single.execute.return_value = exec_result

        eq_phone = MagicMock()
        eq_phone.maybe_single.return_value = maybe_single

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_phone

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock

        client.table.return_value = table_mock

        result = adapter.resolve_or_create_caller(TENANT_ID, PHONE)

        assert isinstance(result, CallerInfo)
        assert result.id == CALLER_ID
        assert result.phone == PHONE
        assert result.tenant_id == TENANT_ID
        assert result.verified_at is None

        # Should only read, not write
        client.table.assert_called_once_with("callers")

    def test_creates_new_caller_when_not_found(self):
        """When no caller found, insert new row and return CallerInfo."""
        adapter, client = _make_adapter()

        new_row = {
            "id": CALLER_ID,
            "phone": PHONE,
            "tenant_id": TENANT_ID,
            "verified_at": None,
        }

        # SELECT returns None (not found)
        exec_select = MagicMock()
        exec_select.data = None

        maybe_single = MagicMock()
        maybe_single.execute.return_value = exec_select

        eq_phone = MagicMock()
        eq_phone.maybe_single.return_value = maybe_single

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_phone

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        # INSERT returns the new row
        exec_insert = MagicMock()
        exec_insert.data = [new_row]

        maybe_single_insert = MagicMock()
        maybe_single_insert.execute.return_value = exec_insert

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single_insert

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        result = adapter.resolve_or_create_caller(TENANT_ID, PHONE)

        assert isinstance(result, CallerInfo)
        assert result.id == CALLER_ID
        assert result.phone == PHONE
        assert result.tenant_id == TENANT_ID

        # Insert should have been called with correct data
        table_mock.insert.assert_called_once()
        insert_args = table_mock.insert.call_args[0][0]
        assert insert_args["phone"] == PHONE
        assert insert_args["tenant_id"] == TENANT_ID


# ---------------------------------------------------------------------------
# hold_slot
# ---------------------------------------------------------------------------


class TestHoldSlot:
    def test_successful_hold_returns_hold_result(self):
        """Successful INSERT into slot_locks returns HoldResult."""
        adapter, client = _make_adapter()

        lock_row = {
            "id": HOLD_ID,
            "resource_id": RESOURCE_ID,
            "start_ts": NOW.isoformat(),
            "expires_at": EXPIRES_AT.isoformat(),
        }

        exec_result = MagicMock()
        exec_result.data = [lock_row]

        maybe_single = MagicMock()
        maybe_single.execute.return_value = exec_result

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        result = adapter.hold_slot(TENANT_ID, RESOURCE_ID, NOW, CALLER_ID, hold_ttl_seconds=180)

        assert isinstance(result, HoldResult)
        assert result.hold_id == HOLD_ID
        assert result.resource_id == RESOURCE_ID

        # Verify insert was called
        table_mock.insert.assert_called_once()
        insert_payload = table_mock.insert.call_args[0][0]
        assert insert_payload["resource_id"] == RESOURCE_ID
        assert insert_payload["tenant_id"] == TENANT_ID
        assert insert_payload["caller_id"] == CALLER_ID

    def test_conflict_raises_slot_conflict_error_code_23505(self):
        """Unique constraint violation (code 23505) raises SlotConflictError."""
        adapter, client = _make_adapter()

        # APIError with details dict containing 'code': '23505'
        api_error = APIError({"code": "23505", "message": "duplicate key value"})

        maybe_single = MagicMock()
        maybe_single.execute.side_effect = api_error

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        with pytest.raises(SlotConflictError):
            adapter.hold_slot(TENANT_ID, RESOURCE_ID, NOW, CALLER_ID)

    def test_conflict_raises_slot_conflict_error_unique_string(self):
        """Error with 'unique' in message string raises SlotConflictError."""
        adapter, client = _make_adapter()

        api_error = APIError({"code": "P0001", "message": "unique constraint violated"})

        maybe_single = MagicMock()
        maybe_single.execute.side_effect = api_error

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        with pytest.raises(SlotConflictError):
            adapter.hold_slot(TENANT_ID, RESOURCE_ID, NOW, CALLER_ID)

    def test_non_conflict_error_propagates(self):
        """Non-constraint errors are re-raised as-is."""
        adapter, client = _make_adapter()

        api_error = APIError({"code": "42P01", "message": "relation does not exist"})

        maybe_single = MagicMock()
        maybe_single.execute.side_effect = api_error

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        with pytest.raises(APIError):
            adapter.hold_slot(TENANT_ID, RESOURCE_ID, NOW, CALLER_ID)


# ---------------------------------------------------------------------------
# confirm_booking
# ---------------------------------------------------------------------------


class TestConfirmBooking:
    def _setup_idempotent_existing(self, client):
        """Return a mock client that has an existing booking for idempotency check."""
        existing_booking = {
            "id": BOOKING_ID,
            "idempotency_key": HOLD_ID,
            "status": "confirmed",
        }

        exec_result = MagicMock()
        exec_result.data = existing_booking

        maybe_single = MagicMock()
        maybe_single.execute.return_value = exec_result

        eq_key = MagicMock()
        eq_key.maybe_single.return_value = maybe_single

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_key

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock

        client.table.return_value = table_mock

    def test_idempotent_returns_existing_booking(self):
        """If booking with idempotency_key=hold_id already exists, return it."""
        adapter, client = _make_adapter()
        self._setup_idempotent_existing(client)

        result = adapter.confirm_booking(
            tenant_id=TENANT_ID,
            hold_id=HOLD_ID,
            service_id=SERVICE_ID,
            resource_id=RESOURCE_ID,
            caller_id=CALLER_ID,
            start_ts=NOW,
            end_ts=END_TS,
            custom_values={},
        )

        assert isinstance(result, BookingResult)
        assert result.booking_id == BOOKING_ID
        assert result.idempotency_key == HOLD_ID
        assert result.status == "confirmed"

        # Must NOT have inserted a new row
        table_mock = client.table.return_value
        table_mock.insert.assert_not_called()

    def test_new_booking_inserted(self):
        """If no existing booking, insert new one with idempotency_key=hold_id."""
        adapter, client = _make_adapter()

        new_booking_row = {
            "id": BOOKING_ID,
            "idempotency_key": HOLD_ID,
            "status": "confirmed",
        }

        # First call: SELECT returns None (no existing booking)
        exec_select = MagicMock()
        exec_select.data = None

        maybe_single_select = MagicMock()
        maybe_single_select.execute.return_value = exec_select

        eq_key = MagicMock()
        eq_key.maybe_single.return_value = maybe_single_select

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_key

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        # INSERT returns new booking
        exec_insert = MagicMock()
        exec_insert.data = [new_booking_row]

        maybe_single_insert = MagicMock()
        maybe_single_insert.execute.return_value = exec_insert

        insert_mock = MagicMock()
        insert_mock.maybe_single.return_value = maybe_single_insert

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock
        table_mock.insert.return_value = insert_mock

        client.table.return_value = table_mock

        result = adapter.confirm_booking(
            tenant_id=TENANT_ID,
            hold_id=HOLD_ID,
            service_id=SERVICE_ID,
            resource_id=RESOURCE_ID,
            caller_id=CALLER_ID,
            start_ts=NOW,
            end_ts=END_TS,
            custom_values={"occasion": "birthday"},
        )

        assert isinstance(result, BookingResult)
        assert result.booking_id == BOOKING_ID
        assert result.idempotency_key == HOLD_ID
        assert result.status == "confirmed"

        # Verify insert payload
        table_mock.insert.assert_called_once()
        payload = table_mock.insert.call_args[0][0]
        assert payload["idempotency_key"] == HOLD_ID
        assert payload["status"] == "confirmed"
        assert payload["tenant_id"] == TENANT_ID
        assert payload["caller_id"] == CALLER_ID
        assert payload["service_id"] == SERVICE_ID
        assert payload["resource_id"] == RESOURCE_ID
        assert payload["custom_values"] == {"occasion": "birthday"}


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


class TestCheckAvailability:
    def test_returns_resources_with_blocked_slots(self):
        """Returns list of resources with their blocked time slots."""
        adapter, client = _make_adapter()

        resources_rows = [
            {"id": RESOURCE_ID, "name": "Priya", "type": "stylist", "tenant_id": TENANT_ID},
        ]
        bookings_rows = [
            {
                "resource_id": RESOURCE_ID,
                "start_ts": NOW.isoformat(),
                "end_ts": END_TS.isoformat(),
                "status": "confirmed",
            }
        ]
        locks_rows: list[dict] = []

        call_count = 0

        def table_side_effect(table_name):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()

            if table_name == "resources":
                exec_r = MagicMock()
                exec_r.data = resources_rows
                chain = mock.select.return_value.eq.return_value.eq.return_value
                chain.execute.return_value = exec_r

            elif table_name == "bookings":
                exec_b = MagicMock()
                exec_b.data = bookings_rows
                chain = (
                    mock.select.return_value.eq.return_value.eq.return_value.gte.return_value.lte.return_value
                )
                chain.execute.return_value = exec_b

            elif table_name == "slot_locks":
                exec_l = MagicMock()
                exec_l.data = locks_rows
                chain = mock.select.return_value.eq.return_value.gte.return_value.gt.return_value
                chain.execute.return_value = exec_l

            return mock

        client.table.side_effect = table_side_effect

        start_date = datetime(2026, 6, 3, tzinfo=UTC)
        end_date = datetime(2026, 6, 4, tzinfo=UTC)

        results = adapter.check_availability(TENANT_ID, SERVICE_ID, "stylist", start_date, end_date)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["resource_id"] == RESOURCE_ID
        assert "blocked_slots" in results[0]


# ---------------------------------------------------------------------------
# lookup_bookings
# ---------------------------------------------------------------------------


class TestLookupBookings:
    def test_returns_list_of_bookings(self):
        """Returns confirmed bookings for the caller ordered by start_ts."""
        adapter, client = _make_adapter()

        booking_rows = [
            {
                "id": BOOKING_ID,
                "tenant_id": TENANT_ID,
                "caller_id": CALLER_ID,
                "start_ts": NOW.isoformat(),
                "status": "confirmed",
            }
        ]

        exec_result = MagicMock()
        exec_result.data = booking_rows

        # chain: .select().eq().eq().order().execute()
        order_mock = MagicMock()
        order_mock.execute.return_value = exec_result

        eq_caller = MagicMock()
        eq_caller.order.return_value = order_mock

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_caller

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock

        client.table.return_value = table_mock

        results = adapter.lookup_bookings(TENANT_ID, CALLER_ID)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["id"] == BOOKING_ID

    def test_returns_empty_list_when_no_bookings(self):
        """Returns empty list when caller has no bookings."""
        adapter, client = _make_adapter()

        exec_result = MagicMock()
        exec_result.data = []

        order_mock = MagicMock()
        order_mock.execute.return_value = exec_result

        eq_caller = MagicMock()
        eq_caller.order.return_value = order_mock

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_caller

        select_mock = MagicMock()
        select_mock.eq.return_value = eq_tenant

        table_mock = MagicMock()
        table_mock.select.return_value = select_mock

        client.table.return_value = table_mock

        results = adapter.lookup_bookings(TENANT_ID, CALLER_ID)

        assert results == []


# ---------------------------------------------------------------------------
# cancel_booking
# ---------------------------------------------------------------------------


class TestCancelBooking:
    def test_cancel_returns_true_when_row_updated(self):
        """Returns True when a confirmed booking is cancelled."""
        adapter, client = _make_adapter()

        updated_rows = [{"id": BOOKING_ID, "status": "cancelled"}]

        exec_result = MagicMock()
        exec_result.data = updated_rows

        # chain: .update().eq().eq().eq().eq().execute()
        eq_status = MagicMock()
        eq_status.execute.return_value = exec_result

        eq_caller = MagicMock()
        eq_caller.eq.return_value = eq_status

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_caller

        eq_id = MagicMock()
        eq_id.eq.return_value = eq_tenant

        update_mock = MagicMock()
        update_mock.eq.return_value = eq_id

        table_mock = MagicMock()
        table_mock.update.return_value = update_mock

        client.table.return_value = table_mock

        result = adapter.cancel_booking(TENANT_ID, BOOKING_ID, CALLER_ID)

        assert result is True

        # Verify the update payload
        table_mock.update.assert_called_once_with({"status": "cancelled"})

    def test_cancel_returns_false_when_no_row_updated(self):
        """Returns False when no matching confirmed booking found."""
        adapter, client = _make_adapter()

        exec_result = MagicMock()
        exec_result.data = []

        eq_status = MagicMock()
        eq_status.execute.return_value = exec_result

        eq_caller = MagicMock()
        eq_caller.eq.return_value = eq_status

        eq_tenant = MagicMock()
        eq_tenant.eq.return_value = eq_caller

        eq_id = MagicMock()
        eq_id.eq.return_value = eq_tenant

        update_mock = MagicMock()
        update_mock.eq.return_value = eq_id

        table_mock = MagicMock()
        table_mock.update.return_value = update_mock

        client.table.return_value = table_mock

        result = adapter.cancel_booking(TENANT_ID, BOOKING_ID, CALLER_ID)

        assert result is False
