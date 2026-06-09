from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import (
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)

from packages.voice_agent.config.models import (
    LanguagePolicy,
    PersonaConfig,
    PipelineConfig,
)
from packages.voice_agent.flows.language_detector import LanguageDetectorProcessor
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


def _make_config(
    languages: list[str],
    fallback: str,
    match_caller: bool = True,
) -> MagicMock:
    """Build a TenantConfig tailored for language-detection tests."""
    greeting = {lang: f"Greeting in {lang}" for lang in languages}
    disclosure = {lang: f"Disclosure in {lang}" for lang in languages}
    tts_voices = {lang: f"voice-{lang}" for lang in languages}
    return make_tenant_config(
        persona=PersonaConfig(
            business_name="Test Salon",
            greeting=greeting,
            ai_disclosure=disclosure,
            tone="warm",
            languages=languages,
            fallback_language=fallback,
            language_policy=LanguagePolicy(greeting="default", match_caller=match_caller),
        ),
        pipeline=PipelineConfig(tts_voices=tts_voices),
    )


def _make_proc(
    config=None,
    flow_state=None,
    caller_id=None,
    data_adapter=None,
    preferred_language=None,
):
    """Instantiate a LanguageDetectorProcessor with test defaults."""
    if config is None:
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
    if flow_state is None:
        flow_state = {"language": "en-IN"}
    return LanguageDetectorProcessor(
        config=config,
        flow_state=flow_state,
        caller_id=caller_id,
        data_adapter=data_adapter,
        preferred_language=preferred_language,
    )


class TestLanguageDetectorSingleLanguage:
    def test_single_language_is_passthrough(self):
        config = _make_config(["en-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        assert proc.is_locked is True
        assert proc.language == "en-IN"

    @pytest.mark.asyncio
    async def test_single_language_passes_frames_through(self):
        config = _make_config(["en-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        pushed: list = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        frame = TranscriptionFrame(text="hello", user_id="u", timestamp="0")
        await proc.process_frame(frame, MagicMock())

        assert len(pushed) == 1
        assert pushed[0] is frame


class TestLanguageDetectorMultiLanguage:
    @pytest.mark.asyncio
    async def test_detects_hindi_and_locks_in(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        flow_state: dict = {"language": "en-IN"}
        proc = _make_proc(config=config, flow_state=flow_state)
        proc.push_frame = AsyncMock()

        worker = MagicMock()
        worker.queue_frame = AsyncMock()
        proc.set_worker(worker)

        with patch.object(proc, "_detect_language", return_value=("hi-IN", 0.9)):
            frame = TranscriptionFrame(text="namaste", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        assert proc.is_locked is True
        assert proc.language == "hi-IN"
        assert flow_state["language"] == "hi-IN"

    @pytest.mark.asyncio
    async def test_low_confidence_does_not_lock_on_first_turn(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        proc.push_frame = AsyncMock()

        with patch.object(proc, "_detect_language", return_value=("hi-IN", 0.4)):
            frame = TranscriptionFrame(text="maybe hindi", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        assert proc.is_locked is False

    @pytest.mark.asyncio
    async def test_falls_back_after_two_turns_with_no_detection(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        flow_state: dict = {"language": "en-IN"}
        proc = _make_proc(config=config, flow_state=flow_state)
        proc.push_frame = AsyncMock()

        worker = MagicMock()
        worker.queue_frame = AsyncMock()
        proc.set_worker(worker)

        with patch.object(proc, "_detect_language", return_value=(None, 0.0)):
            for i in range(2):
                frame = TranscriptionFrame(text=f"msg {i}", user_id="u", timestamp="0")
                await proc.process_frame(frame, MagicMock())

        assert proc.is_locked is True
        assert proc.language == "en-IN"

    @pytest.mark.asyncio
    async def test_after_lock_in_becomes_passthrough(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        pushed: list = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        worker = MagicMock()
        worker.queue_frame = AsyncMock()
        proc.set_worker(worker)

        # Lock in on turn 1
        with patch.object(proc, "_detect_language", return_value=("hi-IN", 0.9)):
            frame1 = TranscriptionFrame(text="namaste", user_id="u", timestamp="0")
            await proc.process_frame(frame1, MagicMock())

        assert proc.is_locked is True

        # Turn 2 should pass through without calling _detect_language
        pushed.clear()
        with patch.object(proc, "_detect_language") as mock_detect:
            frame2 = TranscriptionFrame(text="kya haal hai", user_id="u", timestamp="0")
            await proc.process_frame(frame2, MagicMock())
            mock_detect.assert_not_called()

        assert frame2 in pushed

    @pytest.mark.asyncio
    async def test_lock_in_sends_stt_update_upstream(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        pushed: list = []
        proc.push_frame = AsyncMock(side_effect=lambda f, d=None: pushed.append(f))

        worker = MagicMock()
        worker.queue_frame = AsyncMock()
        proc.set_worker(worker)

        with patch.object(proc, "_detect_language", return_value=("hi-IN", 0.9)):
            frame = TranscriptionFrame(text="namaste", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        stt_updates = [f for f in pushed if isinstance(f, STTUpdateSettingsFrame)]
        assert len(stt_updates) == 1
        assert stt_updates[0].delta.language == "hi-IN"

    @pytest.mark.asyncio
    async def test_lock_in_does_not_re_greet(self):
        """Bilingual greeting covers both languages, so lock-in should not re-greet."""
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config)
        proc.push_frame = AsyncMock()

        worker = MagicMock()
        queued: list = []
        worker.queue_frame = AsyncMock(side_effect=lambda f: queued.append(f))
        proc.set_worker(worker)

        with patch.object(proc, "_detect_language", return_value=("hi-IN", 0.9)):
            frame = TranscriptionFrame(text="namaste", user_id="u", timestamp="0")
            await proc.process_frame(frame, MagicMock())

        tts_speaks = [f for f in queued if isinstance(f, TTSSpeakFrame)]
        assert len(tts_speaks) == 0


class TestLanguageDetectorRepeatCaller:
    def test_pre_locked_from_preferred_language(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config, preferred_language="hi-IN")
        assert proc.is_locked is True
        assert proc.language == "hi-IN"

    def test_preferred_language_not_in_tenant_languages_ignored(self):
        config = _make_config(["en-IN", "hi-IN"], fallback="en-IN")
        proc = _make_proc(config=config, preferred_language="fr-FR")
        assert proc.is_locked is False
