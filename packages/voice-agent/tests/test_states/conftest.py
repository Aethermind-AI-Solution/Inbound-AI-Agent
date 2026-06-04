from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
from packages.voice_agent.dialogue.models import (
    CallContext,
    CallEvent,
    EventType,
    StateDeps,
)
from packages.voice_agent.dialogue.nlu.stub import StubNLUService


def make_tenant_config(**overrides) -> TenantConfig:
    defaults = dict(
        meta=MetaConfig(
            tenant_id="t1", sector="salon", config_version="1.0.0", status="live"
        ),
        persona=PersonaConfig(
            business_name="Glamour Salon",
            greeting="Welcome to Glamour Salon",
            ai_disclosure="I'm an AI assistant.",
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
            resources=[
                Resource(id="r1", name="Priya", type="stylist", capacity=1, tags=[]),
                Resource(id="r2", name="Rahul", type="stylist", capacity=1, tags=[]),
            ],
            services=[
                Service(
                    id="s1", name="Haircut", resource_type="stylist",
                    duration_min=30, booking_mode="exclusive", custom_fields=[],
                ),
                Service(
                    id="s2", name="Hair Color", resource_type="stylist",
                    duration_min=60, booking_mode="exclusive", custom_fields=[],
                ),
            ],
            business_hours=BusinessHours(
                mon=["09:00-18:00"], tue=["09:00-18:00"],
                wed=["09:00-18:00"], thu=["09:00-18:00"],
                fri=["09:00-18:00"], sat=["10:00-16:00"],
            ),
            booking_window_days=14,
            min_notice_min=30,
        ),
        guardrails=GuardrailsConfig(
            max_call_seconds=300, max_turns=20,
            monthly_budget_inr=5000.0, scope="booking_only",
            recording_consent=True, retention_days=90,
        ),
    )
    defaults.update(overrides)
    return TenantConfig(**defaults)


@pytest.fixture
def tenant_config():
    return make_tenant_config()


@pytest.fixture
def nlu():
    return StubNLUService()


@pytest.fixture
def mock_data_adapter():
    adapter = MagicMock()
    adapter.resolve_or_create_caller = MagicMock()
    adapter.hold_slot = MagicMock()
    adapter.confirm_booking = MagicMock()
    adapter.check_availability = MagicMock()
    adapter.lookup_bookings = MagicMock()
    adapter.cancel_booking = MagicMock()
    return adapter


@pytest.fixture
def deps(mock_data_adapter, nlu):
    return StateDeps(
        data_adapter=mock_data_adapter,
        nlu=nlu,
        date_resolver_factory=MagicMock(),
    )


@pytest.fixture
def context(tenant_config):
    return CallContext(
        tenant_config=tenant_config,
        caller_phone="+919876543210",
        call_id="call-001",
    )


def make_transcription(text: str) -> CallEvent:
    return CallEvent(type=EventType.TRANSCRIPTION, text=text)
