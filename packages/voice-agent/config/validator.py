"""Cross-field config validator implementing spec §7.4 rules.

Pydantic's per-field validators cannot express relationships between fields in
different sub-models.  This module fills that gap.

Public API
----------
validate_tenant_config(config) -> list[str]
    Returns a list of human-readable error strings.  Empty list means valid.

validate_or_raise(config) -> None
    Raises ConfigValidationError (with a .errors attribute) when errors exist.

ConfigValidationError
    Exception raised by validate_or_raise; carries .errors: list[str].
"""

from __future__ import annotations

from packages.voice_agent.config.models import TenantConfig

# ---------------------------------------------------------------------------
# Supported language codes (spec §7.4, rule 4)
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"en-IN", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN"}
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ConfigValidationError(Exception):
    """Raised by validate_or_raise when cross-field validation fails.

    Attributes
    ----------
    errors : list[str]
        Human-readable error strings, one per violated rule.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors: list[str] = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# Rule implementations (each returns a list of error strings)
# ---------------------------------------------------------------------------


def _check_service_resource_types(config: TenantConfig) -> list[str]:
    """Rule 1: every service.resource_type must match at least one resource.type."""
    resource_types = {r.type for r in config.booking_model.resources}
    errors: list[str] = []
    for service in config.booking_model.services:
        if service.resource_type not in resource_types:
            errors.append(
                f"Service '{service.id}' references resource_type '{service.resource_type}' "
                f"which does not match any resource type in booking_model.resources "
                f"(available: {sorted(resource_types)})"
            )
    return errors


def _check_otp_channel(config: TenantConfig) -> list[str]:
    """Rule 2: any 'otp' value in auth.action_policy requires auth.otp_channel to be set."""
    has_otp_action = any(v == "otp" for v in config.auth.action_policy.values())
    if has_otp_action and config.auth.otp_channel is None:
        return [
            "auth.action_policy contains 'otp' but auth.otp_channel is not set; "
            "set auth.otp_channel (e.g. 'sms') when any action requires otp authentication"
        ]
    return []


def _check_business_hours(config: TenantConfig) -> list[str]:
    """Rule 3: business_hours must have at least one day with hours defined."""
    bh = config.booking_model.business_hours
    days = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    has_any = any(len(getattr(bh, day)) > 0 for day in days)
    if not has_any:
        return [
            "booking_model.business_hours has no hours defined for any day; "
            "at least one day must have at least one time range"
        ]
    return []


def _check_supported_languages(config: TenantConfig) -> list[str]:
    """Rule 4: all languages in persona.languages must be in the supported set."""
    errors: list[str] = []
    for lang in config.persona.languages:
        if lang not in SUPPORTED_LANGUAGES:
            errors.append(
                f"persona.languages contains unsupported language code '{lang}'; "
                f"supported codes are {sorted(SUPPORTED_LANGUAGES)}"
            )
    return errors


def _check_fallback_language(config: TenantConfig) -> list[str]:
    """Rule 5: persona.fallback_language must be in persona.languages."""
    fallback = config.persona.fallback_language
    if fallback not in config.persona.languages:
        return [
            f"persona.fallback_language '{fallback}' is not present in "
            f"persona.languages {config.persona.languages}"
        ]
    return []


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

_RULES = [
    _check_service_resource_types,
    _check_otp_channel,
    _check_business_hours,
    _check_supported_languages,
    _check_fallback_language,
]


def validate_tenant_config(config: TenantConfig) -> list[str]:
    """Run all cross-field validation rules and return a list of error strings.

    Parameters
    ----------
    config:
        A fully-constructed TenantConfig instance (Pydantic per-field
        validation has already passed).

    Returns
    -------
    list[str]
        Empty list when the config is valid; one string per violated rule
        otherwise.
    """
    errors: list[str] = []
    for rule in _RULES:
        errors.extend(rule(config))
    return errors


def validate_or_raise(config: TenantConfig) -> None:
    """Validate config and raise ConfigValidationError if any errors are found.

    Parameters
    ----------
    config:
        A fully-constructed TenantConfig instance.

    Raises
    ------
    ConfigValidationError
        If one or more cross-field rules are violated.  The exception's
        `.errors` attribute contains the full list of error strings.
    """
    errors = validate_tenant_config(config)
    if errors:
        raise ConfigValidationError(errors)
