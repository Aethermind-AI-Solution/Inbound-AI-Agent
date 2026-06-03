"""Tests for TenantConfig Pydantic v2 models (TDD — written before implementation)."""

import pytest
from pydantic import ValidationError

from packages.voice_agent.config.models import (
    AuthConfig,
    BookingModel,
    BusinessHours,
    CustomField,
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta(**kwargs):
    base = {
        "tenant_id": "tenant_001",
        "sector": "salon",
        "config_version": 1,
        "status": "draft",
    }
    base.update(kwargs)
    return MetaConfig(**base)


def _language_policy(**kwargs):
    base = {"greeting": "default", "match_caller": True}
    base.update(kwargs)
    return LanguagePolicy(**base)


def _persona(**kwargs):
    base = {
        "business_name": "Glow Salon",
        "greeting": "Namaste, Glow Salon mein aapka swagat hai.",
        "ai_disclosure": "Aap ek AI assistant se baat kar rahe hain.",
        "tone": "warm",
        "languages": ["hi", "en"],
        "fallback_language": "hi",
        "language_policy": _language_policy(),
    }
    base.update(kwargs)
    return PersonaConfig(**base)


def _integration(**kwargs):
    base = {
        "calendar_provider": "google_calendar",
        "write_back": True,
        "system_of_record": "external",
        "external_unreachable_policy": "trust_mirror",
    }
    base.update(kwargs)
    return IntegrationConfig(**base)


def _auth(**kwargs):
    base = {
        "levels": ["none"],
        "action_policy": {"book": "none", "cancel": "otp"},
        "soft_match_fields": [],
    }
    base.update(kwargs)
    return AuthConfig(**base)


def _edge_profile(**kwargs):
    base = {
        "caller_demographic": "general",
        "endpointing_ms": 500,
        "barge_in": True,
        "asr_lexicon": [],
        "noise_profile": "quiet",
        "dtmf_fallback": False,
    }
    base.update(kwargs)
    return EdgeProfile(**base)


def _escalation_chain(**kwargs):
    base = {"type": "transfer", "number": "+919876543210"}
    base.update(kwargs)
    return EscalationChain(**base)


def _escalation(**kwargs):
    base = {
        "chain": [_escalation_chain()],
        "trigger_on": ["agent_unavailable", "user_request"],
    }
    base.update(kwargs)
    return EscalationConfig(**base)


def _resource(**kwargs):
    base = {
        "id": "stylist_01",
        "name": "Priya",
        "type": "stylist",
        "capacity": 1,
        "tags": ["hair", "colour"],
    }
    base.update(kwargs)
    return Resource(**base)


def _service(**kwargs):
    base = {
        "id": "svc_haircut",
        "name": "Haircut",
        "resource_type": "stylist",
        "duration_min": 30,
        "booking_mode": "exclusive",
        "custom_fields": [],
    }
    base.update(kwargs)
    return Service(**base)


def _business_hours(**kwargs):
    base = {
        "mon": ["09:00-19:00"],
        "tue": ["09:00-19:00"],
        "wed": ["09:00-19:00"],
        "thu": ["09:00-19:00"],
        "fri": ["09:00-19:00"],
        "sat": ["09:00-21:00"],
        "sun": [],
    }
    base.update(kwargs)
    return BusinessHours(**base)


def _booking_model(**kwargs):
    base = {
        "resources": [_resource()],
        "services": [_service()],
        "business_hours": _business_hours(),
        "booking_window_days": 30,
        "min_notice_min": 60,
    }
    base.update(kwargs)
    return BookingModel(**base)


def _guardrails(**kwargs):
    base = {
        "max_call_seconds": 300,
        "max_turns": 20,
        "monthly_budget_inr": 5000,
        "scope": "global",
        "recording_consent": True,
        "retention_days": 90,
    }
    base.update(kwargs)
    return GuardrailsConfig(**base)


def _full_tenant_config():
    return TenantConfig(
        meta=_meta(),
        persona=_persona(),
        integration=_integration(),
        auth=_auth(),
        edge_profile=_edge_profile(),
        escalation=_escalation(),
        booking_model=_booking_model(),
        guardrails=_guardrails(),
    )


# ---------------------------------------------------------------------------
# MetaConfig
# ---------------------------------------------------------------------------


class TestMetaConfig:
    def test_valid(self):
        m = _meta()
        assert m.tenant_id == "tenant_001"
        assert m.sector == "salon"
        assert m.config_version == 1
        assert m.status == "draft"

    def test_all_sectors(self):
        for sector in ("salon", "clinic", "restaurant", "gym", "home_service", "real_estate"):
            assert _meta(sector=sector).sector == sector

    def test_invalid_sector(self):
        with pytest.raises(ValidationError):
            _meta(sector="hospital")

    def test_all_statuses(self):
        for status in ("draft", "staging", "live"):
            assert _meta(status=status).status == status

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            _meta(status="published")

    def test_tenant_id_required(self):
        with pytest.raises(ValidationError):
            MetaConfig(sector="salon", config_version=1, status="draft")


# ---------------------------------------------------------------------------
# LanguagePolicy
# ---------------------------------------------------------------------------


class TestLanguagePolicy:
    def test_valid_default(self):
        lp = _language_policy()
        assert lp.greeting == "default"
        assert lp.match_caller is True

    def test_bilingual_greeting(self):
        lp = _language_policy(greeting="bilingual")
        assert lp.greeting == "bilingual"

    def test_invalid_greeting(self):
        with pytest.raises(ValidationError):
            _language_policy(greeting="multilingual")


# ---------------------------------------------------------------------------
# PersonaConfig
# ---------------------------------------------------------------------------


class TestPersonaConfig:
    def test_valid(self):
        p = _persona()
        assert p.business_name == "Glow Salon"
        assert p.tone == "warm"
        assert len(p.languages) == 2

    def test_all_tones(self):
        for tone in ("warm", "professional", "casual"):
            assert _persona(tone=tone).tone == tone

    def test_invalid_tone(self):
        with pytest.raises(ValidationError):
            _persona(tone="friendly")

    def test_languages_min_one(self):
        with pytest.raises(ValidationError):
            _persona(languages=[])

    def test_language_policy_embedded(self):
        p = _persona()
        assert isinstance(p.language_policy, LanguagePolicy)


# ---------------------------------------------------------------------------
# IntegrationConfig
# ---------------------------------------------------------------------------


class TestIntegrationConfig:
    def test_valid(self):
        i = _integration()
        assert i.calendar_provider == "google_calendar"
        assert i.write_back is True

    def test_all_providers(self):
        for provider in ("google_calendar", "vendor_api", "native_supabase"):
            assert _integration(calendar_provider=provider).calendar_provider == provider

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            _integration(calendar_provider="outlook")

    def test_invalid_system_of_record(self):
        with pytest.raises(ValidationError):
            _integration(system_of_record="cloud")

    def test_invalid_unreachable_policy(self):
        with pytest.raises(ValidationError):
            _integration(external_unreachable_policy="fail")

    def test_optional_fields_default_none(self):
        i = _integration()
        assert i.adapter_id is None
        assert i.credentials_ref is None

    def test_optional_fields_set(self):
        i = _integration(adapter_id="adapter_xyz", credentials_ref="secret/cal")
        assert i.adapter_id == "adapter_xyz"
        assert i.credentials_ref == "secret/cal"


# ---------------------------------------------------------------------------
# AuthConfig
# ---------------------------------------------------------------------------


class TestAuthConfig:
    def test_valid_none_level(self):
        a = _auth()
        assert a.levels == ["none"]

    def test_all_auth_levels(self):
        a = _auth(levels=["none", "soft", "otp"])
        assert a.levels == ["none", "soft", "otp"]

    def test_invalid_auth_level(self):
        with pytest.raises(ValidationError):
            _auth(levels=["pin"])

    def test_otp_channel_optional(self):
        a = _auth()
        assert a.otp_channel is None

    def test_otp_entry_default_dtmf(self):
        a = _auth()
        assert a.otp_entry == "dtmf"

    def test_otp_entry_invalid(self):
        with pytest.raises(ValidationError):
            _auth(otp_entry="voice")


# ---------------------------------------------------------------------------
# EdgeProfile
# ---------------------------------------------------------------------------


class TestEdgeProfile:
    def test_valid(self):
        e = _edge_profile()
        assert e.endpointing_ms == 500
        assert e.barge_in is True

    def test_all_demographics(self):
        for demo in ("general", "elderly", "youth"):
            assert _edge_profile(caller_demographic=demo).caller_demographic == demo

    def test_invalid_demographic(self):
        with pytest.raises(ValidationError):
            _edge_profile(caller_demographic="child")

    def test_endpointing_ms_min(self):
        with pytest.raises(ValidationError):
            _edge_profile(endpointing_ms=99)

    def test_endpointing_ms_max(self):
        with pytest.raises(ValidationError):
            _edge_profile(endpointing_ms=2001)

    def test_endpointing_ms_boundary_min(self):
        e = _edge_profile(endpointing_ms=100)
        assert e.endpointing_ms == 100

    def test_endpointing_ms_boundary_max(self):
        e = _edge_profile(endpointing_ms=2000)
        assert e.endpointing_ms == 2000

    def test_all_noise_profiles(self):
        for np in ("quiet", "moderate", "noisy"):
            assert _edge_profile(noise_profile=np).noise_profile == np

    def test_invalid_noise_profile(self):
        with pytest.raises(ValidationError):
            _edge_profile(noise_profile="loud")


# ---------------------------------------------------------------------------
# EscalationChain
# ---------------------------------------------------------------------------


class TestEscalationChain:
    def test_valid_transfer(self):
        ec = _escalation_chain()
        assert ec.type == "transfer"
        assert ec.number == "+919876543210"

    def test_all_types(self):
        for t in ("transfer", "callback_queue", "sms_owner"):
            assert _escalation_chain(type=t).type == t

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            _escalation_chain(type="voicemail")

    def test_number_optional(self):
        ec = _escalation_chain(number=None)
        assert ec.number is None


# ---------------------------------------------------------------------------
# EscalationConfig
# ---------------------------------------------------------------------------


class TestEscalationConfig:
    def test_valid(self):
        ec = _escalation()
        assert len(ec.chain) == 1
        assert "agent_unavailable" in ec.trigger_on

    def test_chain_min_one(self):
        with pytest.raises(ValidationError):
            _escalation(chain=[])

    def test_multiple_chains(self):
        chain = [_escalation_chain(), _escalation_chain(type="callback_queue", number=None)]
        ec = _escalation(chain=chain)
        assert len(ec.chain) == 2


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class TestResource:
    def test_valid(self):
        r = _resource()
        assert r.id == "stylist_01"
        assert r.capacity == 1

    def test_capacity_min_one(self):
        with pytest.raises(ValidationError):
            _resource(capacity=0)

    def test_calendar_ref_optional(self):
        r = _resource()
        assert r.calendar_ref is None

    def test_calendar_ref_set(self):
        r = _resource(calendar_ref="cal_priya@salon.com")
        assert r.calendar_ref == "cal_priya@salon.com"

    def test_tags_list(self):
        r = _resource(tags=["hair", "colour", "bridal"])
        assert len(r.tags) == 3


# ---------------------------------------------------------------------------
# CustomField
# ---------------------------------------------------------------------------


class TestCustomField:
    def test_valid_str_field(self):
        cf = CustomField(key="occasion", type="str", required=False, prompt="Kya occasion hai?")
        assert cf.key == "occasion"
        assert cf.type == "str"

    def test_all_types(self):
        for t in ("str", "int", "float", "bool"):
            cf = CustomField(key="f", type=t, required=False, prompt="p")
            assert cf.type == t

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            CustomField(key="f", type="date", required=False, prompt="p")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TestService:
    def test_valid(self):
        s = _service()
        assert s.duration_min == 30
        assert s.booking_mode == "exclusive"

    def test_duration_min_positive(self):
        with pytest.raises(ValidationError):
            _service(duration_min=0)

    def test_all_booking_modes(self):
        for mode in ("exclusive", "shared"):
            assert _service(booking_mode=mode).booking_mode == mode

    def test_invalid_booking_mode(self):
        with pytest.raises(ValidationError):
            _service(booking_mode="group")

    def test_price_optional(self):
        s = _service()
        assert s.price is None

    def test_price_set(self):
        s = _service(price=500.0)
        assert s.price == 500.0

    def test_deposit_optional(self):
        s = _service()
        assert s.deposit is None

    def test_custom_fields(self):
        cf = CustomField(key="hair_length", type="str", required=True, prompt="Baal ki length?")
        s = _service(custom_fields=[cf])
        assert len(s.custom_fields) == 1


# ---------------------------------------------------------------------------
# BusinessHours
# ---------------------------------------------------------------------------


class TestBusinessHours:
    def test_valid(self):
        bh = _business_hours()
        assert bh.mon == ["09:00-19:00"]
        assert bh.sun == []

    def test_get_hours_for_day(self):
        bh = _business_hours()
        assert bh.get_hours_for_day("mon") == ["09:00-19:00"]
        assert bh.get_hours_for_day("sun") == []

    def test_get_hours_for_day_case_insensitive(self):
        bh = _business_hours()
        assert bh.get_hours_for_day("MON") == ["09:00-19:00"]
        assert bh.get_hours_for_day("Mon") == ["09:00-19:00"]

    def test_is_open_on_open_day(self):
        bh = _business_hours()
        assert bh.is_open_on("mon") is True

    def test_is_open_on_closed_day(self):
        bh = _business_hours()
        assert bh.is_open_on("sun") is False

    def test_is_open_on_case_insensitive(self):
        bh = _business_hours()
        assert bh.is_open_on("MON") is True
        assert bh.is_open_on("SUN") is False

    def test_multiple_slots_per_day(self):
        bh = _business_hours(mon=["09:00-13:00", "14:00-19:00"])
        assert len(bh.get_hours_for_day("mon")) == 2
        assert bh.is_open_on("mon") is True

    def test_invalid_day_key(self):
        with pytest.raises((ValidationError, ValueError)):
            bh = _business_hours()
            bh.get_hours_for_day("holiday")


# ---------------------------------------------------------------------------
# BookingModel
# ---------------------------------------------------------------------------


class TestBookingModel:
    def test_valid(self):
        bm = _booking_model()
        assert bm.booking_window_days == 30
        assert bm.min_notice_min == 60

    def test_resources_min_one(self):
        with pytest.raises(ValidationError):
            _booking_model(resources=[])

    def test_services_min_one(self):
        with pytest.raises(ValidationError):
            _booking_model(services=[])

    def test_booking_window_days_positive(self):
        with pytest.raises(ValidationError):
            _booking_model(booking_window_days=0)

    def test_min_notice_min_zero_allowed(self):
        bm = _booking_model(min_notice_min=0)
        assert bm.min_notice_min == 0

    def test_min_notice_min_negative_rejected(self):
        with pytest.raises(ValidationError):
            _booking_model(min_notice_min=-1)


# ---------------------------------------------------------------------------
# GuardrailsConfig
# ---------------------------------------------------------------------------


class TestGuardrailsConfig:
    def test_valid(self):
        g = _guardrails()
        assert g.max_call_seconds == 300
        assert g.retention_days == 90

    def test_max_call_seconds_positive(self):
        with pytest.raises(ValidationError):
            _guardrails(max_call_seconds=0)

    def test_max_turns_positive(self):
        with pytest.raises(ValidationError):
            _guardrails(max_turns=0)

    def test_monthly_budget_inr_positive(self):
        with pytest.raises(ValidationError):
            _guardrails(monthly_budget_inr=0)

    def test_retention_days_positive(self):
        with pytest.raises(ValidationError):
            _guardrails(retention_days=0)


# ---------------------------------------------------------------------------
# TenantConfig (full integration)
# ---------------------------------------------------------------------------


class TestTenantConfig:
    def test_full_salon_config(self):
        tc = _full_tenant_config()
        assert tc.meta.sector == "salon"
        assert tc.persona.business_name == "Glow Salon"
        assert tc.integration.calendar_provider == "google_calendar"
        assert tc.auth.levels == ["none"]
        assert tc.edge_profile.endpointing_ms == 500
        assert len(tc.escalation.chain) == 1
        assert len(tc.booking_model.resources) == 1
        assert len(tc.booking_model.services) == 1
        assert tc.guardrails.max_call_seconds == 300

    def test_meta_required(self):
        with pytest.raises((ValidationError, TypeError)):
            TenantConfig(
                persona=_persona(),
                integration=_integration(),
                auth=_auth(),
                edge_profile=_edge_profile(),
                escalation=_escalation(),
                booking_model=_booking_model(),
                guardrails=_guardrails(),
            )

    def test_json_round_trip(self):
        tc = _full_tenant_config()
        json_str = tc.model_dump_json()
        tc2 = TenantConfig.model_validate_json(json_str)
        assert tc2.meta.tenant_id == tc.meta.tenant_id
        assert tc2.persona.tone == tc.persona.tone
        assert tc2.booking_model.business_hours.mon == tc.booking_model.business_hours.mon

    def test_dict_round_trip(self):
        tc = _full_tenant_config()
        d = tc.model_dump()
        tc2 = TenantConfig.model_validate(d)
        assert tc2.meta.sector == tc.meta.sector

    def test_full_config_with_all_optional_fields(self):
        """Exercise optional fields that have None defaults."""
        auth = _auth(otp_channel="sms", soft_match_fields=["phone", "name"])
        integration = _integration(adapter_id="adp_001", credentials_ref="secret/goog")
        resource = _resource(calendar_ref="priya@salon.com")
        cf = CustomField(key="occasion", type="str", required=False, prompt="Kya occasion?")
        service = _service(price=300.0, deposit=100.0, custom_fields=[cf])
        booking = _booking_model(resources=[resource], services=[service])

        tc = TenantConfig(
            meta=_meta(status="live"),
            persona=_persona(),
            integration=integration,
            auth=auth,
            edge_profile=_edge_profile(asr_lexicon=["balayage", "keratin"]),
            escalation=_escalation(),
            booking_model=booking,
            guardrails=_guardrails(),
        )
        assert tc.meta.status == "live"
        assert tc.integration.credentials_ref == "secret/goog"
        assert tc.auth.otp_channel == "sms"
        assert tc.booking_model.resources[0].calendar_ref == "priya@salon.com"
        assert tc.booking_model.services[0].custom_fields[0].key == "occasion"
