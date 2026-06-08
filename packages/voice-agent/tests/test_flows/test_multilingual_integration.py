from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import TranscriptionFrame

from packages.voice_agent.config.models import (
    LanguagePolicy,
    PersonaConfig,
    PipelineConfig,
)
from packages.voice_agent.config.validator import validate_tenant_config
from packages.voice_agent.data.mock_adapter import MockDataAdapter
from packages.voice_agent.flows.language_detector import LanguageDetectorProcessor
from packages.voice_agent.flows.nodes import create_greeting_node, create_collect_service_node
from packages.voice_agent.flows.prompts import build_role_message
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


def _indian_salon_config():
    return make_tenant_config(
        persona=PersonaConfig(
            business_name="Glamour Salon",
            greeting={
                "en-IN": "Welcome to Glamour Salon!",
                "hi-IN": "Glamour Salon mein aapka swagat hai!",
                "ta-IN": "Glamour Salon-kku varavēṟkirōm!",
            },
            ai_disclosure={
                "en-IN": "I'm an AI assistant.",
                "hi-IN": "Aap ek AI assistant se baat kar rahe hain.",
                "ta-IN": "Nāṉ oru AI utaviyāḷar.",
            },
            tone="warm",
            languages=["en-IN", "hi-IN", "ta-IN"],
            fallback_language="en-IN",
            language_policy=LanguagePolicy(greeting="default", match_caller=True),
        ),
        pipeline=PipelineConfig(
            stt_provider="sarvam",
            tts_provider="sarvam",
            tts_voices={
                "en-IN": "anushka",
                "hi-IN": "anushka",
                "ta-IN": "lakshmi",
            },
        ),
    )


def _us_salon_config():
    return make_tenant_config(
        persona=PersonaConfig(
            business_name="Joe's Barbershop",
            greeting={"en-US": "Welcome to Joe's Barbershop!"},
            ai_disclosure={"en-US": "I'm an AI assistant."},
            tone="casual",
            languages=["en-US"],
            fallback_language="en-US",
            language_policy=LanguagePolicy(greeting="default", match_caller=False),
        ),
        pipeline=PipelineConfig(
            stt_provider="deepgram",
            tts_provider="deepgram",
            tts_voices={"en-US": "aura-asteria-en"},
        ),
    )


class TestMultilingualIntegration:
    def test_indian_salon_config_validates(self):
        config = _indian_salon_config()
        errors = validate_tenant_config(config)
        assert errors == [], f"Validation errors: {errors}"

    def test_us_salon_config_validates(self):
        config = _us_salon_config()
        errors = validate_tenant_config(config)
        assert errors == [], f"Validation errors: {errors}"

    def test_us_salon_role_message_is_english(self):
        config = _us_salon_config()
        msg = build_role_message(config, language="en-US")
        assert "Joe's Barbershop" in msg
        assert "Hindi" not in msg

    def test_indian_salon_hindi_role_message(self):
        config = _indian_salon_config()
        msg = build_role_message(config, language="hi-IN")
        assert "Hindi" in msg
        assert "Glamour Salon" in msg
        assert "Aap ek AI assistant se baat kar rahe hain." in msg

    @pytest.mark.asyncio
    async def test_detection_through_node_creation_flow(self):
        config = _indian_salon_config()
        flow_state = {
            "config": config,
            "language": "en-IN",
            "data_adapter": MockDataAdapter(),
        }

        detector = LanguageDetectorProcessor(
            config=config,
            flow_state=flow_state,
            caller_id="c1",
            data_adapter=flow_state["data_adapter"],
        )
        detector.push_frame = AsyncMock()
        worker = MagicMock()
        worker.queue_frame = AsyncMock()
        detector.set_worker(worker)

        with patch.object(
            detector, "_detect_language", return_value=("hi-IN", 0.9)
        ):
            frame = TranscriptionFrame(text="namaste", user_id="u", timestamp="0")
            await detector.process_frame(frame, MagicMock())

        assert flow_state["language"] == "hi-IN"

        fm = MagicMock()
        fm.state = flow_state
        node = create_collect_service_node(fm)
        assert "Hindi" in node["role_message"]

    @pytest.mark.asyncio
    async def test_us_salon_single_language_no_detection(self):
        config = _us_salon_config()
        flow_state = {"config": config, "language": "en-US"}

        detector = LanguageDetectorProcessor(
            config=config,
            flow_state=flow_state,
            caller_id="c1",
            data_adapter=MockDataAdapter(),
        )
        assert detector.is_locked
        assert detector.language == "en-US"

    @pytest.mark.asyncio
    async def test_repeat_caller_skips_detection(self):
        config = _indian_salon_config()
        adapter = MockDataAdapter()
        caller = adapter.resolve_or_create_caller("t1", "+919876543210")
        adapter.update_caller_language(caller.id, "ta-IN")

        caller2 = adapter.resolve_or_create_caller("t1", "+919876543210")
        assert caller2.preferred_language == "ta-IN"

        detector = LanguageDetectorProcessor(
            config=config,
            flow_state={},
            caller_id=caller2.id,
            data_adapter=adapter,
            preferred_language=caller2.preferred_language,
        )
        assert detector.is_locked
        assert detector.language == "ta-IN"
