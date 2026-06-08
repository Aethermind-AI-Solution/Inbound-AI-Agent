from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import (
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker

    from packages.voice_agent.config.models import TenantConfig

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7
MAX_DETECTION_TURNS = 2

# Common Hindi words that appear in romanized (Latin-script) transcripts.
# Sarvam STT transcribes Hindi speech as romanized English text and tags it
# as en-IN, so we need keyword-based detection instead.
_HINDI_KEYWORDS: set[str] = {
    "mujhe", "chahiye", "kya", "hai", "hain", "nahi", "nahin", "ji",
    "aap", "aapka", "aapke", "aapki", "haan", "theek", "accha",
    "karna", "karenge", "karana", "kariye", "karo", "kar",
    "pehle", "baad", "kal", "aaj", "parso", "subah", "dopahar", "shaam",
    "baje", "baj", "bajke",
    "booking", "appointment",  # these are common in both, so not counted alone
    "salon", "parlour",
    "mera", "meri", "mere", "humara", "hamara",
    "kitna", "kitne", "kitni", "kab", "kaise", "kahan", "kaun",
    "dijiye", "batao", "bataiye", "bataye", "boliye", "bolo",
    "namaste", "dhanyavaad", "shukriya", "swagat",
    "lena", "dena", "milna", "milega", "milegi",
    "wala", "wali", "wale",
    "abhi", "phir", "toh", "bhi", "aur", "lekin", "ya",
    "samay", "samajh", "samjha", "samjhi",
    "chahte", "chahti", "chaahiye",
    "rakhiye", "rakh", "rakho",
    "kahinge", "kahunga", "kahungi",
}
_HINDI_KEYWORD_THRESHOLD = 2
_HINDI_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _HINDI_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

_DEVANAGARI_PATTERN = re.compile(r"[ऀ-ॿ]")


def _build_lingua_detector(languages: list[str]) -> Any:
    """Build a lingua detector scoped to the tenant's configured languages.

    Always includes English as a candidate. Returns ``None`` if fewer than
    two lingua languages can be resolved (detection would be meaningless).
    Lazy-imports lingua so that Sarvam-based detection never loads it.
    """
    from lingua import Language, LanguageDetectorBuilder

    lingua_language_map: dict[Language, str] = {
        Language.HINDI: "hi-IN",
        Language.TAMIL: "ta-IN",
        Language.TELUGU: "te-IN",
        Language.MARATHI: "mr-IN",
        Language.BENGALI: "bn-IN",
    }

    lingua_langs: list[Language] = [Language.ENGLISH]
    for lang_code in languages:
        for lingua_lang, code in lingua_language_map.items():
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
        stt_provider: str = "deepgram",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._flow_state = flow_state
        self._caller_id = caller_id
        self._data_adapter = data_adapter
        self._languages = config.persona.languages
        self._fallback = config.persona.fallback_language
        self._stt_provider = stt_provider
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
        elif self._stt_provider == "sarvam":
            # Sarvam STT romanizes Hindi and tags it en-IN, so its
            # language_code is unreliable.  Use keyword detection instead.
            pass
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
        from lingua import Language as LinguaLanguage

        lingua_language_map: dict[LinguaLanguage, str] = {
            LinguaLanguage.HINDI: "hi-IN",
            LinguaLanguage.TAMIL: "ta-IN",
            LinguaLanguage.TELUGU: "te-IN",
            LinguaLanguage.MARATHI: "mr-IN",
            LinguaLanguage.BENGALI: "bn-IN",
        }

        if self._detector is None:
            return None, 0.0

        result = self._detector.detect_language_of(text)
        if result is None:
            return None, 0.0

        confidence_values = self._detector.compute_language_confidence_values(text)
        confidence = 0.0
        for cv in confidence_values:
            if cv.language == result:
                confidence = cv.value
                break

        if result == LinguaLanguage.ENGLISH:
            for lang_code in self._languages:
                if lang_code.startswith("en-"):
                    return lang_code, confidence
            return None, 0.0

        code = lingua_language_map.get(result)
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

            if self._stt_provider == "sarvam":
                detected_code = self._detect_from_stt_frame(frame)
            else:
                detected_code = self._detect_from_text(frame.text)

            if detected_code:
                logger.info(
                    "Language detected: %s (provider=%s, turn=%d)",
                    detected_code,
                    self._stt_provider,
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

    def _detect_from_stt_frame(self, frame: TranscriptionFrame) -> str | None:
        """Detect language from a Sarvam STT frame.

        Strategy depends on how STT is initialized:
        - STT in hi-IN mode: Devanagari text = Hindi, Latin-only = English.
        - STT in en-IN mode: keyword matching on romanized transcript.
        """
        text = frame.text or ""
        if not text.strip():
            return None

        devanagari_chars = len(_DEVANAGARI_PATTERN.findall(text))

        if devanagari_chars >= 2:
            for lang in self._languages:
                if lang.startswith("hi-"):
                    logger.info(
                        "Devanagari script detected (%d chars): %s",
                        devanagari_chars,
                        text[:80],
                    )
                    return lang

        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        if latin_chars > 0 and devanagari_chars == 0:
            for lang in self._languages:
                if lang.startswith("en-"):
                    logger.info(
                        "Latin-only text detected (English): %s",
                        text[:80],
                    )
                    return lang

        matches = _HINDI_KEYWORD_PATTERN.findall(text)
        if len(matches) >= _HINDI_KEYWORD_THRESHOLD:
            for lang in self._languages:
                if lang.startswith("hi-"):
                    logger.info(
                        "Hindi keywords in romanized text: %s (count=%d)",
                        matches,
                        len(matches),
                    )
                    return lang

        return None

    def _detect_from_text(self, text: str) -> str | None:
        """Run lingua on text; return lang code if confidence >= threshold."""
        detected_code, confidence = self._detect_language(text)
        if detected_code and confidence >= CONFIDENCE_THRESHOLD:
            return detected_code
        return None

    async def _lock_in(self, language: str) -> None:
        """Lock in *language* and propagate settings changes."""
        self._locked = True
        self._language = language

        # Tell STT to switch language (upstream toward the transport input).
        await self.push_frame(
            STTUpdateSettingsFrame(delta=STTSettings(language=language)),
            FrameDirection.UPSTREAM,
        )

        # Tell TTS to switch voice AND language (via worker, downstream).
        tts_voice = self._config.pipeline.tts_voices.get(language)
        if self._worker:
            tts_delta = TTSSettings(language=language)
            if tts_voice:
                tts_delta.voice = tts_voice
            await self._worker.queue_frame(
                TTSUpdateSettingsFrame(delta=tts_delta)
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
