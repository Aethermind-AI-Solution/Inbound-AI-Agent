"""Tests for config loader (TDD — written before implementation).

Covers:
- Loading a valid JSON fixture from disk.
- FileNotFoundError and json.JSONDecodeError propagating naturally.
- Loading from a raw dict (valid and invalid).
- Cross-field validation errors surfacing via ConfigValidationError.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.voice_agent.config.loader import load_config_from_dict, load_config_from_file
from packages.voice_agent.config.models import TenantConfig
from packages.voice_agent.config.validator import ConfigValidationError

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SALON_FIXTURE = FIXTURES_DIR / "salon_config.json"


# ---------------------------------------------------------------------------
# Helper — load fixture as a dict (reusable across tests)
# ---------------------------------------------------------------------------


def _salon_dict() -> dict:
    with open(SALON_FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# TestLoadFromFile
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_load_valid_salon_config(self):
        """load_config_from_file returns a TenantConfig with correct top-level values."""
        config = load_config_from_file(SALON_FIXTURE)

        assert isinstance(config, TenantConfig)
        assert config.meta.tenant_id == "t-salon-001"
        assert config.meta.sector == "salon"
        assert config.meta.status == "draft"
        assert config.persona.business_name == "Glamour Salon"
        assert config.persona.tone == "warm"
        assert config.persona.languages == ["en-IN", "hi-IN"]
        assert config.integration.calendar_provider == "native_supabase"
        assert len(config.booking_model.resources) == 2
        assert len(config.booking_model.services) == 2
        assert config.booking_model.business_hours.is_open_on("mon") is True
        assert config.booking_model.business_hours.is_open_on("sun") is False
        assert len(config.escalation.chain) == 3

    def test_load_nonexistent_file_raises(self):
        """FileNotFoundError propagates when the path does not exist."""
        missing = FIXTURES_DIR / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_config_from_file(missing)

    def test_load_invalid_json_raises(self, tmp_path):
        """json.JSONDecodeError propagates when the file contains malformed JSON."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_config_from_file(bad_json)


# ---------------------------------------------------------------------------
# TestLoadFromDict
# ---------------------------------------------------------------------------


class TestLoadFromDict:
    def test_load_valid_dict(self):
        """load_config_from_dict returns a TenantConfig for a complete valid dict."""
        data = _salon_dict()
        config = load_config_from_dict(data)

        assert isinstance(config, TenantConfig)
        assert config.meta.tenant_id == "t-salon-001"
        assert config.persona.business_name == "Glamour Salon"

    def test_load_invalid_dict_raises_validation_error(self):
        """A dict missing required fields raises ConfigValidationError."""
        # Deliberately omit most required fields — only include tenant_id
        partial: dict = {"meta": {"tenant_id": "x"}}
        with pytest.raises(ConfigValidationError):
            load_config_from_dict(partial)

    def test_load_dict_with_cross_field_error_raises(self):
        """A per-field valid config that breaks a cross-field rule raises ConfigValidationError."""
        data = _salon_dict()
        # Make every service reference a resource_type that does not exist
        for service in data["booking_model"]["services"]:
            service["resource_type"] = "nonexistent_type"

        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict(data)

        # Error message should mention the bad resource type
        assert any("nonexistent_type" in e for e in exc_info.value.errors)
