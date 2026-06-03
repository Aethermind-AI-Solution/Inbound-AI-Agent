"""Config loader for the Voice Booking Agent.

Provides two public functions:

load_config_from_file(path) -> TenantConfig
    Read a JSON file from disk, parse it, validate it, and return a TenantConfig.
    FileNotFoundError and json.JSONDecodeError propagate to the caller unchanged.

load_config_from_dict(data) -> TenantConfig
    Parse a raw dict into a TenantConfig, validate cross-field rules, and return it.
    Raises ConfigValidationError on both Pydantic per-field failures and cross-field
    rule violations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from packages.voice_agent.config.models import TenantConfig
from packages.voice_agent.config.validator import ConfigValidationError, validate_tenant_config


def load_config_from_file(path: Path) -> TenantConfig:
    """Load and validate a tenant config from a JSON file.

    Parameters
    ----------
    path:
        Path to a UTF-8 JSON file containing a tenant config document.

    Returns
    -------
    TenantConfig
        A fully validated tenant config instance.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist (propagated from open()).
    json.JSONDecodeError
        If the file content is not valid JSON (propagated from json.load()).
    ConfigValidationError
        If Pydantic field validation or cross-field validation fails.
    """
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return load_config_from_dict(data)


def load_config_from_dict(data: dict[str, Any]) -> TenantConfig:
    """Parse and validate a tenant config from a dict.

    Parameters
    ----------
    data:
        A raw dictionary (typically parsed from JSON) that should conform to
        the TenantConfig schema.

    Returns
    -------
    TenantConfig
        A fully validated tenant config instance.

    Raises
    ------
    ConfigValidationError
        If Pydantic per-field validation fails, or if any cross-field rule is
        violated.  The exception's ``.errors`` attribute contains a list of
        human-readable error strings.
    """
    try:
        config = TenantConfig(**data)
    except ValidationError as exc:
        # Wrap Pydantic errors in ConfigValidationError for a uniform error surface.
        errors = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ConfigValidationError(errors) from exc

    cross_field_errors = validate_tenant_config(config)
    if cross_field_errors:
        raise ConfigValidationError(cross_field_errors)

    return config
