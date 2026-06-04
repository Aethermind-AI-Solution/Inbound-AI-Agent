import pytest

from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.greeting import GreetingState


class TestGreetingState:
    @pytest.mark.asyncio
    async def test_enter_speaks_greeting_and_disclosure(self, deps, context):
        state = GreetingState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.SPEAK
        assert "Welcome to Glamour Salon" in action.text
        assert "AI assistant" in action.text
        assert action.next_state == CallState.IDENTIFY_CALLER

    @pytest.mark.asyncio
    async def test_name(self, deps):
        state = GreetingState(deps)
        assert state.name == CallState.GREETING
