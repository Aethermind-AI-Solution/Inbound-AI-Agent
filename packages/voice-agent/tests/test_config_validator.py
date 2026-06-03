"""Tests for cross-field config validator (spec §7.4). Written before implementation (TDD)."""

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
from packages.voice_agent.config.validator import (
    ConfigValidationError,
    validate_or_raise,
    validate_tenant_config,
)

# ---------------------------------------------------------------------------
# Builder helpers — produce valid sub-configs; override individual fields via kwargs
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
        "languages": ["en-IN", "hi-IN"],
        "fallback_language": "hi-IN",
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
        "action_policy": {"book": "none", "cancel": "none"},
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


def _make_valid_config(**overrides) -> TenantConfig:
    """Return a fully valid TenantConfig. Pass section instances to override."""
    return TenantConfig(
        meta=overrides.get("meta", _meta()),
        persona=overrides.get("persona", _persona()),
        integration=overrides.get("integration", _integration()),
        auth=overrides.get("auth", _auth()),
        edge_profile=overrides.get("edge_profile", _edge_profile()),
        escalation=overrides.get("escalation", _escalation()),
        booking_model=overrides.get("booking_model", _booking_model()),
        guardrails=overrides.get("guardrails", _guardrails()),
    )


# ---------------------------------------------------------------------------
# Tests: validate_tenant_config (returns list[str])
# ---------------------------------------------------------------------------


class TestValidateTenantConfigReturnsErrors:
    def test_valid_config_returns_empty_list(self):
        """A fully correct config produces no errors."""
        config = _make_valid_config()
        errors = validate_tenant_config(config)
        assert errors == []

    # Rule 1: service.resource_type must match a resource.type
    def test_resource_type_mismatch_detected(self):
        """Service references a resource type that does not exist in resources."""
        service = _service(resource_type="therapist")  # no therapist resource
        bm = _booking_model(services=[service])  # resource is "stylist"
        config = _make_valid_config(booking_model=bm)
        errors = validate_tenant_config(config)
        assert len(errors) >= 1
        assert any("therapist" in e for e in errors)

    def test_resource_type_match_passes(self):
        """Service resource_type that exists in resources produces no error."""
        resource = _resource(type="chair")
        service = _service(resource_type="chair")
        bm = _booking_model(resources=[resource], services=[service])
        config = _make_valid_config(booking_model=bm)
        errors = validate_tenant_config(config)
        assert errors == []

    def test_multiple_resource_type_mismatches(self):
        """Each bad service resource_type generates its own error."""
        svc1 = _service(id="svc1", resource_type="ghost")
        svc2 = _service(id="svc2", resource_type="phantom")
        resource = _resource(type="stylist")
        bm = _booking_model(resources=[resource], services=[svc1, svc2])
        config = _make_valid_config(booking_model=bm)
        errors = validate_tenant_config(config)
        # Both bad types should appear in error messages
        types_mentioned = " ".join(errors)
        assert "ghost" in types_mentioned
        assert "phantom" in types_mentioned

    # Rule 2: OTP in action_policy requires otp_channel to be set
    def test_otp_action_without_channel_detected(self):
        """action_policy has an 'otp' value but otp_channel is None."""
        auth = _auth(
            action_policy={"book": "none", "cancel": "otp"},
            otp_channel=None,
        )
        config = _make_valid_config(auth=auth)
        errors = validate_tenant_config(config)
        assert len(errors) >= 1
        assert any("otp" in e.lower() for e in errors)

    def test_otp_action_with_channel_passes(self):
        """action_policy has 'otp' and otp_channel is set — no error."""
        auth = _auth(
            action_policy={"book": "none", "cancel": "otp"},
            otp_channel="sms",
        )
        config = _make_valid_config(auth=auth)
        errors = validate_tenant_config(config)
        assert errors == []

    def test_no_otp_action_without_channel_passes(self):
        """No 'otp' value in action_policy, otp_channel is None — no error."""
        auth = _auth(
            action_policy={"book": "none", "cancel": "none"},
            otp_channel=None,
        )
        config = _make_valid_config(auth=auth)
        errors = validate_tenant_config(config)
        assert errors == []

    # Rule 3: business_hours must have at least one day with hours
    def test_empty_business_hours_detected(self):
        """All days empty means no hours defined at all."""
        bh = _business_hours(mon=[], tue=[], wed=[], thu=[], fri=[], sat=[], sun=[])
        bm = _booking_model(business_hours=bh)
        config = _make_valid_config(booking_model=bm)
        errors = validate_tenant_config(config)
        assert len(errors) >= 1
        assert any("business_hours" in e.lower() or "hours" in e.lower() for e in errors)

    def test_one_day_with_hours_passes(self):
        """Even if only one day has hours defined, rule passes."""
        bh = _business_hours(mon=["10:00-18:00"], tue=[], wed=[], thu=[], fri=[], sat=[], sun=[])
        bm = _booking_model(business_hours=bh)
        config = _make_valid_config(booking_model=bm)
        errors = validate_tenant_config(config)
        assert errors == []

    # Rule 4: All languages must be in the supported set
    def test_unsupported_language_detected(self):
        """A language code outside the valid set is flagged."""
        persona = _persona(languages=["en-IN", "fr-FR"], fallback_language="en-IN")
        config = _make_valid_config(persona=persona)
        errors = validate_tenant_config(config)
        assert len(errors) >= 1
        assert any("fr-FR" in e for e in errors)

    def test_all_supported_languages_pass(self):
        """All six supported codes are individually valid."""
        supported = ["en-IN", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN"]
        for lang in supported:
            persona = _persona(languages=[lang], fallback_language=lang)
            config = _make_valid_config(persona=persona)
            errors = validate_tenant_config(config)
            assert errors == [], f"Expected no errors for language {lang}, got {errors}"

    def test_multiple_unsupported_languages_each_flagged(self):
        """Every unsupported language code generates its own error."""
        persona = _persona(languages=["xx-XX", "yy-YY"], fallback_language="xx-XX")
        config = _make_valid_config(persona=persona)
        errors = validate_tenant_config(config)
        all_text = " ".join(errors)
        assert "xx-XX" in all_text
        assert "yy-YY" in all_text

    # Rule 5: fallback_language must be in persona.languages
    def test_fallback_language_not_in_languages_detected(self):
        """fallback_language is a valid code but not in the languages list."""
        persona = _persona(languages=["en-IN"], fallback_language="hi-IN")
        config = _make_valid_config(persona=persona)
        errors = validate_tenant_config(config)
        assert len(errors) >= 1
        assert any("fallback" in e.lower() for e in errors)

    def test_fallback_language_in_languages_passes(self):
        """fallback_language present in languages list — no error."""
        persona = _persona(languages=["en-IN", "hi-IN"], fallback_language="hi-IN")
        config = _make_valid_config(persona=persona)
        errors = validate_tenant_config(config)
        assert errors == []

    def test_multiple_rules_failing_simultaneously(self):
        """Config that violates several rules reports all errors at once."""
        # Violate: resource type mismatch + OTP without channel + empty hours + bad language
        service = _service(resource_type="ghost")
        bh = _business_hours(mon=[], tue=[], wed=[], thu=[], fri=[], sat=[], sun=[])
        bm = _booking_model(services=[service], business_hours=bh)
        auth = _auth(action_policy={"book": "otp"}, otp_channel=None)
        persona = _persona(languages=["xx-XX"], fallback_language="xx-XX")
        config = _make_valid_config(booking_model=bm, auth=auth, persona=persona)
        errors = validate_tenant_config(config)
        assert len(errors) >= 3  # at least resource, otp, hours — language error too


# ---------------------------------------------------------------------------
# Tests: validate_or_raise
# ---------------------------------------------------------------------------


class TestValidateOrRaise:
    def test_valid_config_does_not_raise(self):
        config = _make_valid_config()
        validate_or_raise(config)  # must not raise

    def test_invalid_config_raises_config_validation_error(self):
        service = _service(resource_type="nonexistent")
        bm = _booking_model(services=[service])
        config = _make_valid_config(booking_model=bm)
        with pytest.raises(ConfigValidationError):
            validate_or_raise(config)

    def test_raised_error_has_errors_attribute(self):
        service = _service(resource_type="nonexistent")
        bm = _booking_model(services=[service])
        config = _make_valid_config(booking_model=bm)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_or_raise(config)
        assert hasattr(exc_info.value, "errors")
        assert isinstance(exc_info.value.errors, list)
        assert len(exc_info.value.errors) >= 1

    def test_error_messages_in_errors_attribute(self):
        service = _service(resource_type="nonexistent")
        bm = _booking_model(services=[service])
        config = _make_valid_config(booking_model=bm)
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_or_raise(config)
        assert any("nonexistent" in e for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# Tests: ConfigValidationError
# ---------------------------------------------------------------------------


class TestConfigValidationError:
    def test_is_exception(self):
        err = ConfigValidationError(["error one", "error two"])
        assert isinstance(err, Exception)

    def test_has_errors_attribute(self):
        messages = ["error one", "error two"]
        err = ConfigValidationError(messages)
        assert err.errors == messages

    def test_str_representation_contains_errors(self):
        err = ConfigValidationError(["something went wrong"])
        assert "something went wrong" in str(err)
