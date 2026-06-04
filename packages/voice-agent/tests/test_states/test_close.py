import pytest

from packages.voice_agent.dialogue.models import ActionType
from packages.voice_agent.dialogue.states.close import CloseState


class TestCloseState:
    @pytest.mark.asyncio
    async def test_enter_speaks_closing_and_ends_call(self, deps, context):
        state = CloseState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.END_CALL
        assert "Glamour Salon" in action.text
