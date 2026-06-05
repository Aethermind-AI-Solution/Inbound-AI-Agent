from __future__ import annotations

from datetime import UTC, datetime

from packages.voice_agent.data.mock_adapter import MockDataAdapter


class TestMockResolveOrCreateCaller:
    def test_returns_caller_info(self):
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+919876543210")
        assert caller.id == "caller-1"
        assert caller.phone == "+919876543210"
        assert caller.tenant_id == "t1"

    def test_uses_provided_phone(self):
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+911111111111")
        assert caller.phone == "+911111111111"


class TestMockCheckAvailability:
    def test_returns_resources_with_no_blocked_slots(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        result = adapter.check_availability("t1", "s1", "stylist", now, now)
        assert len(result) >= 1
        assert result[0]["resource_id"]
        assert result[0]["resource_name"]
        assert result[0]["blocked_slots"] == []


class TestMockHoldSlot:
    def test_returns_hold_result(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        hold = adapter.hold_slot("t1", "r1", now, "caller-1")
        assert hold.hold_id
        assert hold.resource_id == "r1"
        assert hold.expires_at > hold.start_ts


class TestMockConfirmBooking:
    def test_returns_booking_result(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        result = adapter.confirm_booking(
            "t1", "hold-1", "s1", "r1", "caller-1", now, now, {}
        )
        assert result.booking_id
        assert result.status == "confirmed"

    def test_idempotent_on_same_hold_id(self):
        adapter = MockDataAdapter()
        now = datetime.now(tz=UTC)
        r1 = adapter.confirm_booking("t1", "hold-1", "s1", "r1", "caller-1", now, now, {})
        r2 = adapter.confirm_booking("t1", "hold-1", "s1", "r1", "caller-1", now, now, {})
        assert r1.booking_id == r2.booking_id


class TestMockLookupBookings:
    def test_returns_list_of_bookings(self):
        adapter = MockDataAdapter()
        bookings = adapter.lookup_bookings("t1", "caller-1")
        assert isinstance(bookings, list)
        assert len(bookings) >= 1
        assert "id" in bookings[0]
        assert bookings[0]["status"] == "confirmed"


class TestMockCancelBooking:
    def test_returns_true(self):
        adapter = MockDataAdapter()
        assert adapter.cancel_booking("t1", "booking-1", "caller-1") is True

    def test_lookup_after_cancel_shows_cancelled(self):
        adapter = MockDataAdapter()
        adapter.cancel_booking("t1", "booking-1", "caller-1")
        bookings = adapter.lookup_bookings("t1", "caller-1")
        cancelled = [b for b in bookings if b["id"] == "booking-1"]
        assert len(cancelled) == 1
        assert cancelled[0]["status"] == "cancelled"

    def test_cancel_wrong_caller_returns_false(self):
        adapter = MockDataAdapter()
        assert adapter.cancel_booking("t1", "booking-1", "wrong-caller") is False


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
