from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from lingua import Language, LanguageDetectorBuilder
from pipecat.frames.frames import (
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker

    from packages.voice_agent.config.models import TenantConfig

logger = logging.getLogger(__name__)

LINGUA_LANGUAGE_MAP: dict[Language, str] = {
    Language.HINDI: "hi-IN",
    Language.TAMIL: "ta-IN",
    Language.TELUGU: "te-IN",
    Language.MARATHI: "mr-IN",
    Language.BENGALI: "bn-IN",
}

CONFIDENCE_THRESHOLD = 0.7
MAX_DETECTION_TURNS = 2


def _build_lingua_detector(languages: list[str]) -> Any:
    """Build a lingua detector scoped to the tenant's configured languages.

    Always includes English as a candidate. Returns ``None`` if fewer than
    two lingua languages can be resolved (detection would be meaningless).
    """
    lingua_langs: list[Language] = [Language.ENGLISH]
    for lang_code in languages:
        for lingua_lang, code in LINGUA_LANGUAGE_MAP.items():
            if code == lang_code and lingua_lang not in lingua_langs:
                lingua_langs.append(lingua_lang)
    if len(lingua_langs) < 2:
        return None
    return LanguageDetectorBuilder.from_languages(*lingua_langs).build()


class LanguageDetectorProcessor(FrameProcessor):
    """Detects the caller's language from early transcription frames.

    Behaviour depends on the tenant configuration:

    * **Single-language tenant** (1 language in config) -- pure passthrough,
      locked from init.
    * **Multi-language tenant** with ``match_caller: true`` --
      1. Listens to the first 1-2 ``TranscriptionFrame``s.
      2. Runs lingua-language-detector on the text.
      3. If confidence >= 0.7 AND the detected language is in the tenant's
         languages list, locks in immediately.
      4. After 2 turns with no confident detection, falls back to
         ``fallback_language``.
      5. On lock-in: pushes ``STTUpdateSettingsFrame`` upstream, queues
         ``TTSUpdateSettingsFrame`` via the worker, stores the language in
         ``flow_state``, and re-greets if the language differs from fallback.
      6. After lock-in, all frames pass through without detection.
    """

    def __init__(
        self,
        *,
        config: TenantConfig,
        flow_state: dict[str, Any],
        caller_id: str | None,
        data_adapter: Any,
        preferred_language: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._flow_state = flow_state
        self._caller_id = caller_id
        self._data_adapter = data_adapter
        self._languages = config.persona.languages
        self._fallback = config.persona.fallback_language
        self._locked = False
        self._language = self._fallback
        self._turn_count = 0
        self._worker: PipelineWorker | None = None
        self._detector = None

        # Determine whether we can skip detection entirely.
        if len(self._languages) == 1:
            self._locked = True
            self._language = self._languages[0]
        elif not config.persona.language_policy.match_caller:
            self._locked = True
            self._language = self._fallback
        elif preferred_language and preferred_language in self._languages:
            self._locked = True
            self._language = preferred_language
        else:
            self._detector = _build_lingua_detector(self._languages)
            if self._detector is None:
                self._locked = True
                self._language = self._fallback

    def set_worker(self, worker: PipelineWorker) -> None:
        self._worker = worker

    def set_flow_state(self, flow_state: dict[str, Any]) -> None:
        self._flow_state = flow_state

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def language(self) -> str:
        return self._language

    def _detect_language(self, text: str) -> tuple[str | None, float]:
        """Run lingua on *text* and return ``(lang_code, confidence)``."""
        if self._detector is None:
            return None, 0.0

        result = self._detector.detect_language_of(text)
        if result is None:
            return None, 0.0

        confidence_values = self._detector.compute_language_confidence_values(text)
        confidence = 0.0
        for lang, conf in confidence_values:
            if lang == result:
                confidence = conf
                break

        # Map the lingua result back to a tenant language code.
        if result == Language.ENGLISH:
            for lang_code in self._languages:
                if lang_code.startswith("en-"):
                    return lang_code, confidence
            return None, 0.0

        code = LINGUA_LANGUAGE_MAP.get(result)
        if code and code in self._languages:
            return code, confidence

        return None, 0.0

    async def process_frame(self, frame: Any, direction: Any) -> None:
        await super().process_frame(frame, direction)

        if self._locked:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            self._turn_count += 1
            detected_code, confidence = self._detect_language(frame.text)

            if detected_code and confidence >= CONFIDENCE_THRESHOLD:
                logger.info(
                    "Language detected: %s (confidence=%.2f, turn=%d)",
                    detected_code,
                    confidence,
                    self._turn_count,
                )
                await self._lock_in(detected_code)
            elif self._turn_count >= MAX_DETECTION_TURNS:
                logger.info(
                    "No confident detection after %d turns, falling back to %s",
                    self._turn_count,
                    self._fallback,
                )
                await self._lock_in(self._fallback)

        await self.push_frame(frame, direction)

    async def _lock_in(self, language: str) -> None:
        """Lock in *language* and propagate settings changes."""
        self._locked = True
        self._language = language

        # Tell STT to switch language (upstream toward the transport input).
        await self.push_frame(
            STTUpdateSettingsFrame(settings={"language": language}),
            FrameDirection.UPSTREAM,
        )

        # Tell TTS to switch voice (via worker, downstream toward output).
        tts_voice = self._config.pipeline.tts_voices.get(language)
        if tts_voice and self._worker:
            await self._worker.queue_frame(
                TTSUpdateSettingsFrame(delta=TTSSettings(voice=tts_voice))
            )

        # Persist to flow state.
        self._flow_state["language"] = language

        # Re-greet if the detected language differs from the fallback.
        if language != self._fallback:
            greeting = self._config.persona.greeting.get(language, "")
            if greeting and self._worker:
                await self._worker.queue_frame(TTSSpeakFrame(text=greeting))

        # Persist caller preference for repeat-caller fast-path.
        if self._caller_id and language != self._fallback and self._data_adapter:
            try:
                await asyncio.to_thread(
                    self._data_adapter.update_caller_language,
                    self._caller_id,
                    language,
                )
            except Exception:
                logger.exception("Failed to save caller language preference")
