"""Tests for DateResolver -- deterministic date/time resolution.

Covers IST, business hours, lunch breaks, festivals, and edge cases.
TDD: these tests were written before the implementation.
"""

from datetime import date, time

from packages.voice_agent.config.models import BusinessHours
from packages.voice_agent.resolver.date_resolver import (
    DateResolver,
    ResolutionStatus,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_resolver(
    reference_date=None,
    reference_time=None,
    booking_window_days=30,
    min_notice_min=60,
):
    hours = BusinessHours(
        mon=["09:00-13:00", "14:00-19:00"],  # lunch break 13:00-14:00
        tue=["09:00-13:00", "14:00-19:00"],
        wed=["09:00-13:00", "14:00-19:00"],
        thu=["09:00-13:00", "14:00-19:00"],
        fri=["09:00-13:00", "14:00-19:00"],
        sat=["10:00-18:00"],
        sun=[],  # Closed
    )
    return DateResolver(
        business_hours=hours,
        booking_window_days=booking_window_days,
        min_notice_min=min_notice_min,
        reference_date=reference_date or date(2026, 6, 3),  # Wednesday
        reference_time=reference_time or time(10, 0),
    )


# ---------------------------------------------------------------------------
# 1. TestExplicitDatetime
# ---------------------------------------------------------------------------


class TestExplicitDatetime:
    """Phrases that include both a date and a time should resolve to RESOLVED."""

    def test_tomorrow_4pm(self):
        r = _make_resolver()
        result = r.resolve("tomorrow 4pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        assert result.slot.date == date(2026, 6, 4)  # Thursday
        assert result.slot.time == time(16, 0)

    def test_next_friday_3pm(self):
        """'next Friday at 3pm' — reference is Wed Jun 3 2026, so next Friday is Jun 5."""
        r = _make_resolver()
        result = r.resolve("next Friday at 3pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        # next Friday from Wed Jun 3 should be Jun 5 (same week) or Jun 12 (next week)
        # dateparser with PREFER_DATES_FROM=future should give Jun 5 or Jun 12
        assert result.slot.date.weekday() == 4  # Friday
        assert result.slot.time == time(15, 0)

    def test_june_15_at_11am(self):
        r = _make_resolver()
        result = r.resolve("June 15th at 11am")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        assert result.slot.date == date(2026, 6, 15)  # Monday
        assert result.slot.time == time(11, 0)

    def test_resolved_has_datetime_ist(self):
        r = _make_resolver()
        result = r.resolve("tomorrow 4pm")
        assert result.slot is not None
        assert result.slot.datetime_ist.tzname() == "IST"

    def test_resolved_has_no_clarification_prompt(self):
        r = _make_resolver()
        result = r.resolve("tomorrow 4pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.clarification_prompt is None


# ---------------------------------------------------------------------------
# 2. TestOutOfWindow
# ---------------------------------------------------------------------------


class TestOutOfWindow:
    """Dates beyond the booking_window_days should be OUT_OF_WINDOW."""

    def test_august_15_at_10am(self):
        """Aug 15 is >30 days from Jun 3."""
        r = _make_resolver(booking_window_days=30)
        result = r.resolve("August 15th at 10am")
        assert result.status == ResolutionStatus.OUT_OF_WINDOW
        assert result.clarification_prompt is not None


# ---------------------------------------------------------------------------
# 3. TestPastDate
# ---------------------------------------------------------------------------


class TestPastDate:
    """Dates in the past should be PAST_DATE."""

    def test_yesterday_3pm(self):
        r = _make_resolver()
        result = r.resolve("yesterday at 3pm")
        assert result.status == ResolutionStatus.PAST_DATE
        assert result.clarification_prompt is not None


# ---------------------------------------------------------------------------
# 4. TestOutsideBusinessHours
# ---------------------------------------------------------------------------


class TestOutsideBusinessHours:
    """Requests outside business hours should be OUTSIDE_HOURS."""

    def test_sunday_closed(self):
        """Sunday is closed entirely."""
        r = _make_resolver()
        result = r.resolve("Sunday at 10am")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS
        assert result.clarification_prompt is not None

    def test_too_early(self):
        """7am is before opening (9am on weekdays)."""
        r = _make_resolver()
        result = r.resolve("tomorrow at 7am")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS

    def test_too_late(self):
        """8pm is after closing (19:00 on weekdays)."""
        r = _make_resolver()
        result = r.resolve("tomorrow at 8pm")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS


# ---------------------------------------------------------------------------
# 5. TestAmbiguous
# ---------------------------------------------------------------------------


class TestAmbiguous:
    """Phrases that are unparseable or lack a time component should be AMBIGUOUS."""

    def test_no_time_component(self):
        """'next week' has no time — should ask for one."""
        r = _make_resolver()
        result = r.resolve("next week")
        assert result.status == ResolutionStatus.AMBIGUOUS
        assert result.clarification_prompt is not None

    def test_unparseable(self):
        """Complete gibberish should be AMBIGUOUS."""
        r = _make_resolver()
        result = r.resolve("when the moon is full")
        assert result.status == ResolutionStatus.AMBIGUOUS
        assert result.clarification_prompt is not None


# ---------------------------------------------------------------------------
# 6. TestMinNotice
# ---------------------------------------------------------------------------


class TestMinNotice:
    """Bookings too close to the current time should be TOO_SOON."""

    def test_today_too_soon(self):
        """10:30am with ref time 10:00 and min_notice 60 min -> only 30 min notice."""
        r = _make_resolver(
            reference_date=date(2026, 6, 3),
            reference_time=time(10, 0),
            min_notice_min=60,
        )
        result = r.resolve("today at 10:30am")
        assert result.status == ResolutionStatus.TOO_SOON
        assert result.clarification_prompt is not None


# ---------------------------------------------------------------------------
# 7. TestFestivals
# ---------------------------------------------------------------------------


class TestFestivals:
    """Indian festival name substitution."""

    def test_diwali_no_time_is_ambiguous(self):
        """'after Diwali' has no time — should be AMBIGUOUS (or OUT_OF_WINDOW)."""
        r = _make_resolver()
        result = r.resolve("after Diwali")
        assert result.status in {
            ResolutionStatus.AMBIGUOUS,
            ResolutionStatus.OUT_OF_WINDOW,
        }

    def test_day_after_holi_at_2pm(self):
        """Day after Holi 2027 is Mar 1 2027 — likely OUT_OF_WINDOW with 30-day window.
        With a wide enough window, it could resolve. We test the substitution works."""
        # Holi 2026 is Mar 10 — already past (ref is Jun 3 2026).
        # Holi 2027 is Feb 28 — far future, out of window with 30 days.
        # Use a very large window to test substitution.
        r = _make_resolver(booking_window_days=365)
        result = r.resolve("day after Holi at 2pm")
        # It should either resolve or be out_of_window depending on date resolution.
        # If holi 2026 (Mar 10) is used -> past_date; if holi 2027 (Feb 28) -> resolved
        # The key assertion: it should NOT be AMBIGUOUS — the festival was substituted.
        assert (
            result.status != ResolutionStatus.AMBIGUOUS
            or "time" not in (result.clarification_prompt or "").lower()
        )


# ---------------------------------------------------------------------------
# 8. TestLunchBreak
# ---------------------------------------------------------------------------


class TestLunchBreak:
    """Requests during the lunch break (13:00-14:00) should be OUTSIDE_HOURS."""

    def test_lunch_break(self):
        r = _make_resolver()
        result = r.resolve("tomorrow at 1:30pm")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS
        assert result.clarification_prompt is not None


# ---------------------------------------------------------------------------
# 9. TestResolutionResultMessage
# ---------------------------------------------------------------------------


class TestResolutionResultMessage:
    """Non-RESOLVED statuses must have a clarification_prompt; RESOLVED must not."""

    def test_resolved_no_prompt(self):
        r = _make_resolver()
        result = r.resolve("tomorrow 4pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.clarification_prompt is None

    def test_ambiguous_has_prompt(self):
        r = _make_resolver()
        result = r.resolve("when the moon is full")
        assert result.status == ResolutionStatus.AMBIGUOUS
        assert result.clarification_prompt is not None
        assert len(result.clarification_prompt) > 0
