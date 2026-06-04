import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.callback_capture import CallbackCaptureState
from packages.voice_agent.tests.test_states.conftest import make_transcription


class TestCallbackCaptureState:
    @pytest.mark.asyncio
    async def test_enter_default_message(self, deps, context):
        state = CallbackCaptureState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.ASK
        assert "+919876543210" in action.text

    @pytest.mark.asyncio
    async def test_enter_with_no_availability_reason(self, deps, context):
        context.fallback_reason = "no_availability"
        state = CallbackCaptureState(deps)
        action = await state.enter(context)
        assert "opening" in action.text.lower()

    @pytest.mark.asyncio
    async def test_enter_with_tool_error_reason(self, deps, context):
        context.fallback_reason = "tool_error"
        state = CallbackCaptureState(deps)
        action = await state.enter(context)
        assert "technical" in action.text.lower()

    @pytest.mark.asyncio
    async def test_affirmative_transitions_to_close(self, deps, context):
        state = CallbackCaptureState(deps)
        await state.enter(context)
        action = await state.handle(make_transcription("yes"), context)
        assert action.next_state == CallState.CLOSE

    @pytest.mark.asyncio
    async def test_unclear_twice_assumes_correct_and_closes(self, deps, context):
        state = CallbackCaptureState(deps)
        await state.enter(context)
        await state.handle(make_transcription("hmm what"), context)
        action = await state.handle(make_transcription("hmm what again"), context)
        assert action.next_state == CallState.CLOSE
