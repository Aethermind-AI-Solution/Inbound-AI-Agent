"""Pydantic v2 config models for the Voice Booking Agent tenant configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator
from pydantic import Field as PydanticField

# ---------------------------------------------------------------------------
# Valid value tuples (used in validators)
# ---------------------------------------------------------------------------

VALID_SECTORS = ("salon", "clinic", "restaurant", "gym", "home_service", "real_estate")
VALID_STATUSES = ("draft", "staging", "live")
VALID_GREETING_MODES = ("default", "bilingual")
VALID_TONES = ("warm", "professional", "casual")
VALID_CALENDAR_PROVIDERS = ("google_calendar", "vendor_api", "native_supabase")
VALID_SYSTEM_OF_RECORD = ("external", "native")
VALID_UNREACHABLE_POLICIES = ("trust_mirror", "capture")
VALID_AUTH_LEVELS = ("none", "soft", "otp")
VALID_OTP_ENTRY = ("dtmf",)
VALID_CALLER_DEMOGRAPHICS = ("general", "elderly", "youth")
VALID_NOISE_PROFILES = ("quiet", "moderate", "noisy")
VALID_ESCALATION_TYPES = ("transfer", "callback_queue", "sms_owner")
VALID_CUSTOM_FIELD_TYPES = ("str", "int", "float", "bool")
VALID_BOOKING_MODES = ("exclusive", "shared")

VALID_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


# ---------------------------------------------------------------------------
# 1. MetaConfig
# ---------------------------------------------------------------------------


class MetaConfig(BaseModel):
    tenant_id: str
    sector: str
    config_version: int
    status: str

    @field_validator("sector")
    @classmethod
    def validate_sector(cls, v: str) -> str:
        if v not in VALID_SECTORS:
            raise ValueError(f"sector must be one of {VALID_SECTORS}, got '{v}'")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 2. LanguagePolicy
# ---------------------------------------------------------------------------


class LanguagePolicy(BaseModel):
    greeting: str
    match_caller: bool

    @field_validator("greeting")
    @classmethod
    def validate_greeting(cls, v: str) -> str:
        if v not in VALID_GREETING_MODES:
            raise ValueError(f"greeting must be one of {VALID_GREETING_MODES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 3. PersonaConfig
# ---------------------------------------------------------------------------


class PersonaConfig(BaseModel):
    business_name: str
    greeting: str
    ai_disclosure: str
    tone: str
    languages: list[str] = PydanticField(min_length=1)
    fallback_language: str
    language_policy: LanguagePolicy

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in VALID_TONES:
            raise ValueError(f"tone must be one of {VALID_TONES}, got '{v}'")
        return v

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("languages must have at least 1 entry")
        return v


# ---------------------------------------------------------------------------
# 4. IntegrationConfig
# ---------------------------------------------------------------------------


class IntegrationConfig(BaseModel):
    calendar_provider: str
    adapter_id: str | None = None
    credentials_ref: str | None = None
    write_back: bool
    system_of_record: str
    external_unreachable_policy: str

    @field_validator("calendar_provider")
    @classmethod
    def validate_calendar_provider(cls, v: str) -> str:
        if v not in VALID_CALENDAR_PROVIDERS:
            raise ValueError(
                f"calendar_provider must be one of {VALID_CALENDAR_PROVIDERS}, got '{v}'"
            )
        return v

    @field_validator("system_of_record")
    @classmethod
    def validate_system_of_record(cls, v: str) -> str:
        if v not in VALID_SYSTEM_OF_RECORD:
            raise ValueError(f"system_of_record must be one of {VALID_SYSTEM_OF_RECORD}, got '{v}'")
        return v

    @field_validator("external_unreachable_policy")
    @classmethod
    def validate_unreachable_policy(cls, v: str) -> str:
        if v not in VALID_UNREACHABLE_POLICIES:
            raise ValueError(
                f"external_unreachable_policy must be one of "
                f"{VALID_UNREACHABLE_POLICIES}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# 5. AuthConfig
# ---------------------------------------------------------------------------


class AuthConfig(BaseModel):
    levels: list[str]
    action_policy: dict[str, str]
    otp_channel: str | None = None
    otp_entry: str = "dtmf"
    soft_match_fields: list[str]

    @field_validator("levels", mode="before")
    @classmethod
    def validate_levels(cls, v: list[Any]) -> list[Any]:
        for level in v:
            if level not in VALID_AUTH_LEVELS:
                raise ValueError(
                    f"each auth level must be one of {VALID_AUTH_LEVELS}, got '{level}'"
                )
        return v

    @field_validator("otp_entry")
    @classmethod
    def validate_otp_entry(cls, v: str) -> str:
        if v not in VALID_OTP_ENTRY:
            raise ValueError(f"otp_entry must be one of {VALID_OTP_ENTRY}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 6. EdgeProfile
# ---------------------------------------------------------------------------


class EdgeProfile(BaseModel):
    caller_demographic: str
    endpointing_ms: int = PydanticField(ge=100, le=2000)
    barge_in: bool
    asr_lexicon: list[str]
    noise_profile: str
    dtmf_fallback: bool

    @field_validator("caller_demographic")
    @classmethod
    def validate_caller_demographic(cls, v: str) -> str:
        if v not in VALID_CALLER_DEMOGRAPHICS:
            raise ValueError(
                f"caller_demographic must be one of {VALID_CALLER_DEMOGRAPHICS}, got '{v}'"
            )
        return v

    @field_validator("noise_profile")
    @classmethod
    def validate_noise_profile(cls, v: str) -> str:
        if v not in VALID_NOISE_PROFILES:
            raise ValueError(f"noise_profile must be one of {VALID_NOISE_PROFILES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 7. EscalationChain
# ---------------------------------------------------------------------------


class EscalationChain(BaseModel):
    type: str
    number: str | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_ESCALATION_TYPES:
            raise ValueError(f"escalation type must be one of {VALID_ESCALATION_TYPES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 8. EscalationConfig
# ---------------------------------------------------------------------------


class EscalationConfig(BaseModel):
    chain: list[EscalationChain] = PydanticField(min_length=1)
    trigger_on: list[str]

    @field_validator("chain")
    @classmethod
    def validate_chain(cls, v: list[EscalationChain]) -> list[EscalationChain]:
        if len(v) < 1:
            raise ValueError("chain must have at least 1 entry")
        return v


# ---------------------------------------------------------------------------
# 9. Resource
# ---------------------------------------------------------------------------


class Resource(BaseModel):
    id: str
    name: str
    type: str
    capacity: int = PydanticField(ge=1)
    tags: list[str]
    calendar_ref: str | None = None


# ---------------------------------------------------------------------------
# 10. CustomField
# ---------------------------------------------------------------------------


class CustomField(BaseModel):
    key: str
    type: str
    required: bool
    prompt: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_CUSTOM_FIELD_TYPES:
            raise ValueError(
                f"CustomField type must be one of {VALID_CUSTOM_FIELD_TYPES}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# 11. Service
# ---------------------------------------------------------------------------


class Service(BaseModel):
    id: str
    name: str
    resource_type: str
    duration_min: int = PydanticField(gt=0)
    booking_mode: str
    price: float | None = None
    deposit: float | None = None
    custom_fields: list[CustomField]

    @field_validator("booking_mode")
    @classmethod
    def validate_booking_mode(cls, v: str) -> str:
        if v not in VALID_BOOKING_MODES:
            raise ValueError(f"booking_mode must be one of {VALID_BOOKING_MODES}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# 12. BusinessHours
# ---------------------------------------------------------------------------


class BusinessHours(BaseModel):
    mon: list[str] = PydanticField(default_factory=list)
    tue: list[str] = PydanticField(default_factory=list)
    wed: list[str] = PydanticField(default_factory=list)
    thu: list[str] = PydanticField(default_factory=list)
    fri: list[str] = PydanticField(default_factory=list)
    sat: list[str] = PydanticField(default_factory=list)
    sun: list[str] = PydanticField(default_factory=list)

    def get_hours_for_day(self, day: str) -> list[str]:
        """Return the list of time-range strings for the given day (case-insensitive)."""
        key = day.lower()
        if key not in VALID_DAYS:
            raise ValueError(f"Invalid day '{day}'. Must be one of {VALID_DAYS}.")
        return getattr(self, key)

    def is_open_on(self, day: str) -> bool:
        """Return True if the business has at least one time range on the given day."""
        return len(self.get_hours_for_day(day)) > 0


# ---------------------------------------------------------------------------
# 13. BookingModel
# ---------------------------------------------------------------------------


class BookingModel(BaseModel):
    resources: list[Resource] = PydanticField(min_length=1)
    services: list[Service] = PydanticField(min_length=1)
    business_hours: BusinessHours
    booking_window_days: int = PydanticField(gt=0)
    min_notice_min: int = PydanticField(ge=0)

    @field_validator("resources")
    @classmethod
    def validate_resources(cls, v: list[Resource]) -> list[Resource]:
        if len(v) < 1:
            raise ValueError("resources must have at least 1 entry")
        return v

    @field_validator("services")
    @classmethod
    def validate_services(cls, v: list[Service]) -> list[Service]:
        if len(v) < 1:
            raise ValueError("services must have at least 1 entry")
        return v


# ---------------------------------------------------------------------------
# 14. GuardrailsConfig
# ---------------------------------------------------------------------------


class GuardrailsConfig(BaseModel):
    max_call_seconds: int = PydanticField(gt=0)
    max_turns: int = PydanticField(gt=0)
    monthly_budget_inr: float = PydanticField(gt=0)
    scope: str
    recording_consent: bool
    retention_days: int = PydanticField(gt=0)


# ---------------------------------------------------------------------------
# 15. TenantConfig (root)
# ---------------------------------------------------------------------------


class TenantConfig(BaseModel):
    meta: MetaConfig
    persona: PersonaConfig
    integration: IntegrationConfig
    auth: AuthConfig
    edge_profile: EdgeProfile
    escalation: EscalationConfig
    booking_model: BookingModel
    guardrails: GuardrailsConfig
