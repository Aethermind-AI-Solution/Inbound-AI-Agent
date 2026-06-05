import pytest

from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.identify_caller import IdentifyCallerState


class TestIdentifyCallerState:
    @pytest.mark.asyncio
    async def test_enter_resolves_caller_and_transitions(self, deps, context):
        caller = CallerInfo(id="c1", phone="+919876543210", tenant_id="t1", verified_at=None, is_new=False)
        deps.data_adapter.resolve_or_create_caller.return_value = caller
        state = IdentifyCallerState(deps)
        action = await state.enter(context)
        assert action.type == ActionType.TRANSITION
        assert action.next_state == CallState.INTENT
        assert context.caller == caller
        deps.data_adapter.resolve_or_create_caller.assert_called_once_with("t1", "+919876543210")

    @pytest.mark.asyncio
    async def test_enter_data_adapter_failure_raises(self, deps, context):
        deps.data_adapter.resolve_or_create_caller.side_effect = Exception("DB down")
        state = IdentifyCallerState(deps)
        with pytest.raises(Exception, match="DB down"):
            await state.enter(context)
