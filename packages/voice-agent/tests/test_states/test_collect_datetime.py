from datetime import date, time
from unittest.mock import MagicMock

import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.collect_datetime import CollectDatetimeState
from packages.voice_agent.resolver.date_resolver import (
    ResolutionResult,
    ResolutionStatus,
    ResolvedSlot,
)
from packages.voice_agent.tests.test_states.conftest import make_transcription


@pytest.fixture
def resolver():
    return MagicMock()


@pytest.fixture
def deps_with_resolver(deps, resolver):
    deps.date_resolver_factory = MagicMock(return_value=resolver)
    return deps


class TestCollectDatetimeState:
    @pytest.mark.asyncio
    async def test_enter_asks_for_date(self, deps_with_resolver, context):
        state = CollectDatetimeState(deps_with_resolver)
        action = await state.enter(context)
        assert action.type == ActionType.ASK
        assert "when" in action.text.lower()

    @pytest.mark.asyncio
    async def test_resolved_date_transitions_to_offer_slots(
        self, deps_with_resolver, context, resolver
    ):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        dt = datetime(2026, 6, 10, 10, 0, tzinfo=ist)
        resolver.resolve.return_value = ResolutionResult(
            status=ResolutionStatus.RESOLVED,
            slot=ResolvedSlot(date=dt.date(), time=dt.time(), datetime_ist=dt),
        )
        state = CollectDatetimeState(deps_with_resolver)
        await state.enter(context)
        action = await state.handle(make_transcription("tomorrow at 10am"), context)
        assert action.next_state == CallState.OFFER_SLOTS
        assert context.slots.datetime_ist == dt

    @pytest.mark.asyncio
    async def test_ambiguous_reprompts_with_clarification(
        self, deps_with_resolver, context, resolver
    ):
        resolver.resolve.return_value = ResolutionResult(
            status=ResolutionStatus.AMBIGUOUS,
            clarification_prompt="Did you mean morning or afternoon?",
        )
        state = CollectDatetimeState(deps_with_resolver)
        await state.enter(context)
        action = await state.handle(make_transcription("tomorrow"), context)
        assert action.type == ActionType.ASK
        assert "morning or afternoon" in action.text
        assert action.next_state is None

    @pytest.mark.asyncio
    async def test_past_date_explains(self, deps_with_resolver, context, resolver):
        resolver.resolve.return_value = ResolutionResult(
            status=ResolutionStatus.PAST_DATE
        )
        state = CollectDatetimeState(deps_with_resolver)
        await state.enter(context)
        action = await state.handle(make_transcription("yesterday"), context)
        assert action.type == ActionType.ASK
        assert "past" in action.text.lower()

    @pytest.mark.asyncio
    async def test_four_failures_goes_to_callback(
        self, deps_with_resolver, context, resolver
    ):
        resolver.resolve.return_value = ResolutionResult(
            status=ResolutionStatus.PAST_DATE
        )
        state = CollectDatetimeState(deps_with_resolver)
        await state.enter(context)
        await state.handle(make_transcription("fail1"), context)
        await state.handle(make_transcription("fail2"), context)
        await state.handle(make_transcription("fail3"), context)
        action = await state.handle(make_transcription("fail4"), context)
        assert action.next_state == CallState.CALLBACK_CAPTURE
