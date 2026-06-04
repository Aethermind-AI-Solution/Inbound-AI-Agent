import pytest

from packages.voice_agent.config.models import CustomField, Service
from packages.voice_agent.dialogue.models import ActionType, CallState
from packages.voice_agent.dialogue.states.collect_custom import CollectCustomState
from packages.voice_agent.tests.test_states.conftest import make_tenant_config, make_transcription


@pytest.fixture
def context_with_custom_fields(context):
    config = make_tenant_config(
        booking_model=context.tenant_config.booking_model.model_copy(
            update={
                "services": [
                    Service(
                        id="s1", name="Haircut", resource_type="stylist",
                        duration_min=30, booking_mode="exclusive",
                        custom_fields=[
                            CustomField(key="notes", type="str", required=False, prompt="Any special requests?"),
                            CustomField(key="length", type="str", required=False, prompt="What length would you like?"),
                        ],
                    )
                ]
            }
        )
    )
    context.tenant_config = config
    context.slots.service_id = "s1"
    return context


class TestCollectCustomState:
    @pytest.mark.asyncio
    async def test_enter_asks_first_field(self, deps, context_with_custom_fields):
        state = CollectCustomState(deps)
        action = await state.enter(context_with_custom_fields)
        assert action.type == ActionType.ASK
        assert "special requests" in action.text.lower()

    @pytest.mark.asyncio
    async def test_collect_first_then_asks_second(self, deps, context_with_custom_fields):
        state = CollectCustomState(deps)
        await state.enter(context_with_custom_fields)
        action = await state.handle(make_transcription("keep it short"), context_with_custom_fields)
        assert context_with_custom_fields.slots.custom_fields["notes"] == "keep it short"
        assert action.type == ActionType.ASK
        assert "length" in action.text.lower()

    @pytest.mark.asyncio
    async def test_all_fields_collected_transitions_to_read_back(self, deps, context_with_custom_fields):
        state = CollectCustomState(deps)
        await state.enter(context_with_custom_fields)
        await state.handle(make_transcription("keep it short"), context_with_custom_fields)
        action = await state.handle(make_transcription("medium"), context_with_custom_fields)
        assert action.next_state == CallState.READ_BACK
