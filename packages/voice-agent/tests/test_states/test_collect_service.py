import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.collect_service import CollectServiceState
from packages.voice_agent.tests.test_states.conftest import make_transcription


class TestCollectServiceState:
    @pytest.mark.asyncio
    async def test_enter_lists_services(self, deps, context):
        state = CollectServiceState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.ASK
        assert "Haircut" in action.text
        assert "Hair Color" in action.text

    @pytest.mark.asyncio
    async def test_match_service(self, deps, context):
        state = CollectServiceState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("I want a haircut"), context)
        assert action.next_state == CallState.COLLECT_DATETIME
        assert context.slots.service_id == "s1"
        assert context.slots.service_name == "Haircut"

    @pytest.mark.asyncio
    async def test_no_match_reprompts(self, deps, context):
        state = CollectServiceState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("something weird"), context)
        assert action.type == ActionType.ASK
        assert action.next_state is None

    @pytest.mark.asyncio
    async def test_asking_about_services_re_offers_list(self, deps, context):
        state = CollectServiceState(deps)
        await state.enter(context)
        action = await state.handle(
            make_transcription("what services do you offer?"), context
        )
        assert action.type == ActionType.ASK
        assert action.next_state is None
        assert "haircut" in action.text.lower()

    @pytest.mark.asyncio
    async def test_three_failures_goes_to_callback(self, deps, context):
        state = CollectServiceState(deps)
        await state.enter(context)
        await state.handle(make_transcription("xyz"), context)
        await state.handle(make_transcription("xyz"), context)
        action = await state.handle(make_transcription("xyz"), context)
        assert action.next_state == CallState.CALLBACK_CAPTURE
