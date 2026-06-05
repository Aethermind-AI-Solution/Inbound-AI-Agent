import time

import pytest

from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    BookingSlots,
    CallContext,
    CallEvent,
    CallState,
    EventType,
    StateDeps,
)


class TestCallState:
    def test_all_15_states_exist(self):
        assert len(CallState) == 15

    def test_greeting_value(self):
        assert CallState.GREETING == "greeting"

    def test_close_value(self):
        assert CallState.CLOSE == "close"

    def test_states_are_strings(self):
        for state in CallState:
            assert isinstance(state.value, str)


class TestEventType:
    def test_transcription(self):
        assert EventType.TRANSCRIPTION == "transcription"

    def test_silence(self):
        assert EventType.SILENCE == "silence"

    def test_hangup(self):
        assert EventType.HANGUP == "hangup"


class TestCallEvent:
    def test_transcription_event(self):
        event = CallEvent(type=EventType.TRANSCRIPTION, text="hello")
        assert event.text == "hello"
        assert event.confidence is None

    def test_silence_event(self):
        event = CallEvent(type=EventType.SILENCE)
        assert event.text is None

    def test_metadata_defaults_to_empty_dict(self):
        event = CallEvent(type=EventType.HANGUP)
        assert event.metadata == {}


class TestAction:
    def test_ask_action(self):
        action = Action(type=ActionType.ASK, text="How can I help?")
        assert action.type == ActionType.ASK
        assert action.text == "How can I help?"
        assert action.next_state is None
        assert action.timeout_s == 15.0

    def test_speak_with_transition(self):
        action = Action(
            type=ActionType.SPEAK,
            text="Hello",
            next_state=CallState.IDENTIFY_CALLER,
        )
        assert action.next_state == CallState.IDENTIFY_CALLER

    def test_end_call(self):
        action = Action(type=ActionType.END_CALL, text="Goodbye")
        assert action.type == ActionType.END_CALL

    def test_transition_silent(self):
        action = Action(type=ActionType.TRANSITION, next_state=CallState.INTENT)
        assert action.text is None


class TestBookingSlots:
    def test_defaults_are_none(self):
        slots = BookingSlots()
        assert slots.service_id is None
        assert slots.resource_id is None
        assert slots.datetime_ist is None
        assert slots.hold_id is None
        assert slots.booking_id is None
        assert slots.custom_fields == {}

    def test_clear_datetime(self):
        slots = BookingSlots(
            datetime_ist="2026-06-10T10:00:00",
            hold_id="h1",
            hold_expires_at="2026-06-10T10:03:00",
            resource_id="r1",
            resource_name="Priya",
        )
        slots.clear_datetime()
        assert slots.datetime_ist is None
        assert slots.hold_id is None
        assert slots.hold_expires_at is None
        assert slots.resource_id is None
        assert slots.resource_name is None
        assert slots.service_id is None  # was never set

    def test_clear_service_clears_everything_downstream(self):
        slots = BookingSlots(
            service_id="s1",
            service_name="Haircut",
            datetime_ist="2026-06-10T10:00:00",
            hold_id="h1",
            resource_id="r1",
            resource_name="Priya",
            custom_fields={"notes": "short"},
        )
        slots.clear_service()
        assert slots.service_id is None
        assert slots.service_name is None
        assert slots.datetime_ist is None
        assert slots.hold_id is None
        assert slots.resource_id is None
        assert slots.custom_fields == {}


class TestCallContext:
    def test_defaults(self):
        from packages.voice_agent.config.models import (
            AuthConfig,
            BookingModel,
            BusinessHours,
            EdgeProfile,
            EscalationChain,
            EscalationConfig,
            GuardrailsConfig,
            IntegrationConfig,
            LanguagePolicy,
            MetaConfig,
            PersonaConfig,
            Resource,
            Service,
            TenantConfig,
        )

        config = TenantConfig(
            meta=MetaConfig(
                tenant_id="t1", sector="salon", config_version="1.0.0", status="live"
            ),
            persona=PersonaConfig(
                business_name="Test Salon",
                greeting="Hello",
                ai_disclosure="I am an AI",
                tone="warm",
                languages=["en-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=False),
            ),
            integration=IntegrationConfig(
                calendar_provider="native_supabase",
                write_back=False,
                system_of_record="native",
                external_unreachable_policy="capture",
            ),
            auth=AuthConfig(
                levels=["soft"],
                action_policy={"new_booking": "none"},
                soft_match_fields=["caller_number"],
            ),
            edge_profile=EdgeProfile(
                caller_demographic="general",
                endpointing_ms=500,
                barge_in=True,
                asr_lexicon=[],
                noise_profile="quiet",
                dtmf_fallback=False,
            ),
            escalation=EscalationConfig(
                chain=[EscalationChain(type="callback_queue")],
                trigger_on=["repeated_failure"],
            ),
            booking_model=BookingModel(
                resources=[Resource(id="r1", name="Priya", type="stylist", capacity=1, tags=[])],
                services=[
                    Service(
                        id="s1",
                        name="Haircut",
                        resource_type="stylist",
                        duration_min=30,
                        booking_mode="exclusive",
                        custom_fields=[],
                    )
                ],
                business_hours=BusinessHours(
                    mon=["09:00-18:00"],
                    tue=["09:00-18:00"],
                    wed=["09:00-18:00"],
                    thu=["09:00-18:00"],
                    fri=["09:00-18:00"],
                    sat=["10:00-16:00"],
                ),
                booking_window_days=14,
                min_notice_min=30,
            ),
            guardrails=GuardrailsConfig(
                max_call_seconds=300,
                max_turns=20,
                monthly_budget_inr=5000.0,
                scope="booking_only",
                recording_consent=True,
                retention_days=90,
            ),
        )
        ctx = CallContext(
            tenant_config=config,
            caller_phone="+919876543210",
            call_id="call-001",
        )
        assert ctx.caller is None
        assert ctx.intent is None
        assert ctx.turn_count == 0
        assert ctx.silence_count == 0
        assert isinstance(ctx.slots, BookingSlots)
