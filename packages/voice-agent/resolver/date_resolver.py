"""Deterministic date/time resolver for the Voice Booking Agent.

Takes a natural-language phrase and resolves it to a concrete date/time
against business_hours and booking_window_days, in IST.

Architecture invariant: the LLM proposes a phrase; the resolver decides the
date -- never the reverse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

import dateparser

from packages.voice_agent.config.models import BusinessHours

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Indian festival dates (manually curated — no reliable API for these)
# ---------------------------------------------------------------------------

INDIAN_FESTIVALS: dict[str, dict[int, tuple[int, int]]] = {
    "diwali": {2025: (10, 20), 2026: (10, 9), 2027: (10, 29)},
    "holi": {2025: (3, 14), 2026: (3, 10), 2027: (2, 28)},
    "dussehra": {2025: (10, 2), 2026: (9, 21), 2027: (10, 11)},
    "eid": {2025: (3, 31), 2026: (3, 20), 2027: (3, 10)},
    "raksha bandhan": {2025: (8, 9), 2026: (8, 28), 2027: (8, 17)},
    "ganesh chaturthi": {2025: (8, 27), 2026: (8, 17), 2027: (9, 5)},
}

# ---------------------------------------------------------------------------
# Regex for detecting time indicators in a phrase
# ---------------------------------------------------------------------------

_TIME_PATTERN = re.compile(
    r"""
    \d{1,2}:\d{2}             # HH:MM
    | \d{1,2}\s*(?:am|pm)     # 4pm, 4 pm
    | \b(?:morning|afternoon|evening|night)\b
    | \bo'?\s*clock\b         # o'clock / oclock
    """,
    re.IGNORECASE | re.VERBOSE,
)

# dateparser quirk: "next <day>" fails to parse, but "<day>" with
# PREFER_DATES_FROM=future works correctly (picks the upcoming occurrence).
_NEXT_DAY_PATTERN = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# Day-name lookup for weekday index (0=Mon .. 6=Sun)
_DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    PAST_DATE = "past_date"
    OUT_OF_WINDOW = "out_of_window"
    OUTSIDE_HOURS = "outside_hours"
    TOO_SOON = "too_soon"


@dataclass
class ResolvedSlot:
    date: date
    time: time
    datetime_ist: datetime


@dataclass
class ResolutionResult:
    status: ResolutionStatus
    slot: ResolvedSlot | None = None
    clarification_prompt: str | None = None


# ---------------------------------------------------------------------------
# DateResolver
# ---------------------------------------------------------------------------


class DateResolver:
    """Pure, deterministic resolver: phrase -> concrete IST datetime."""

    def __init__(
        self,
        business_hours: BusinessHours,
        booking_window_days: int,
        min_notice_min: int,
        reference_date: date,
        reference_time: time,
    ) -> None:
        self.business_hours = business_hours
        self.booking_window_days = booking_window_days
        self.min_notice_min = min_notice_min
        self.reference_date = reference_date
        self.reference_time = reference_time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, phrase: str) -> ResolutionResult:
        """Resolve a natural-language phrase to a concrete IST date/time."""
        # 1. Normalize and substitute festivals
        normalized = phrase.strip().lower()
        # dateparser quirk: "next <day>" fails; strip "next" since
        # PREFER_DATES_FROM=future already picks the upcoming occurrence.
        normalized = _NEXT_DAY_PATTERN.sub(r"\1", normalized)
        substituted = self._substitute_festivals(normalized)

        # 2. Check for time indicator BEFORE parsing (use original substituted phrase)
        has_time = self._phrase_has_time(substituted)

        # 3. Parse with dateparser
        reference_dt = datetime.combine(self.reference_date, self.reference_time, tzinfo=IST)

        parsed = dateparser.parse(
            substituted,
            settings={
                "TIMEZONE": "Asia/Kolkata",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": reference_dt.replace(tzinfo=None),
            },
        )

        # 3a. Parse failure -> AMBIGUOUS
        if parsed is None:
            return ResolutionResult(
                status=ResolutionStatus.AMBIGUOUS,
                clarification_prompt=(
                    "I wasn't able to understand that date. "
                    "Could you say something like 'tomorrow at 3pm' or 'June 10th at 11am'?"
                ),
            )

        # Ensure IST
        parsed_ist = parsed.astimezone(IST)

        # 4. No time indicator -> AMBIGUOUS (ask for time)
        if not has_time:
            return ResolutionResult(
                status=ResolutionStatus.AMBIGUOUS,
                clarification_prompt=(
                    f"I got the date as {parsed_ist.strftime('%A, %B %d')}. "
                    "What time would you like to book?"
                ),
            )

        resolved_date = parsed_ist.date()
        resolved_time = parsed_ist.timetz().replace(tzinfo=None)

        # 5. Past date check
        if parsed_ist < reference_dt:
            return ResolutionResult(
                status=ResolutionStatus.PAST_DATE,
                clarification_prompt=(
                    "That time is in the past. Could you pick a future date and time?"
                ),
            )

        # 6. Booking window check
        max_date = self.reference_date + timedelta(days=self.booking_window_days)
        if resolved_date > max_date:
            return ResolutionResult(
                status=ResolutionStatus.OUT_OF_WINDOW,
                clarification_prompt=(
                    f"We can only book up to {self.booking_window_days} days in advance. "
                    f"Could you choose a date before {max_date.strftime('%B %d, %Y')}?"
                ),
            )

        # 7. Too soon check (within min_notice_min)
        min_notice_dt = reference_dt + timedelta(minutes=self.min_notice_min)
        if parsed_ist < min_notice_dt:
            return ResolutionResult(
                status=ResolutionStatus.TOO_SOON,
                clarification_prompt=(
                    f"We need at least {self.min_notice_min} minutes notice for bookings. "
                    "Could you pick a later time?"
                ),
            )

        # 8. Business closed that day?
        day_key = _DAY_KEYS[resolved_date.weekday()]
        ranges = self.business_hours.get_hours_for_day(day_key)
        if not ranges:
            return ResolutionResult(
                status=ResolutionStatus.OUTSIDE_HOURS,
                clarification_prompt=(
                    f"We're closed on {resolved_date.strftime('%A')}s. "
                    "Could you pick a different day?"
                ),
            )

        # 9. Time within business hour ranges?
        if not self._time_in_ranges(resolved_time, ranges):
            return ResolutionResult(
                status=ResolutionStatus.OUTSIDE_HOURS,
                clarification_prompt=(
                    f"Our hours on {resolved_date.strftime('%A')} are "
                    f"{', '.join(ranges)}. Could you pick a time within those hours?"
                ),
            )

        # 10. All checks passed -> RESOLVED
        slot = ResolvedSlot(
            date=resolved_date,
            time=resolved_time,
            datetime_ist=parsed_ist,
        )
        return ResolutionResult(status=ResolutionStatus.RESOLVED, slot=slot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _substitute_festivals(self, phrase: str) -> str:
        """Replace festival names with their actual date strings.

        Picks the nearest future occurrence for the reference year, falling
        back to the next year if this year's date has passed.
        """
        result = phrase
        for festival_name, year_map in INDIAN_FESTIVALS.items():
            if festival_name in result:
                # Find the best year: prefer current year if future, else next year
                fest_date = self._pick_festival_date(year_map)
                if fest_date is not None:
                    date_str = fest_date.strftime("%B %d, %Y")
                    result = result.replace(festival_name, date_str)
        return result

    def _pick_festival_date(self, year_map: dict[int, tuple[int, int]]) -> date | None:
        """Pick the nearest future festival date from the year_map."""
        ref = self.reference_date
        candidates: list[date] = []
        for year, (month, day) in year_map.items():
            try:
                d = date(year, month, day)
                candidates.append(d)
            except ValueError:
                continue

        # Filter to future dates, pick the closest
        future = [d for d in candidates if d >= ref]
        if future:
            return min(future)

        # All in the past — return the most recent (dateparser will flag as past)
        if candidates:
            return max(candidates)

        return None

    def _phrase_has_time(self, phrase: str) -> bool:
        """Check whether the phrase contains a time indicator."""
        return bool(_TIME_PATTERN.search(phrase))

    @staticmethod
    def _time_in_ranges(t: time, ranges: list[str]) -> bool:
        """Check if time *t* falls within any 'HH:MM-HH:MM' range."""
        for r in ranges:
            start_str, end_str = r.split("-")
            start = time.fromisoformat(start_str)
            end = time.fromisoformat(end_str)
            if start <= t < end:
                return True
        return False
