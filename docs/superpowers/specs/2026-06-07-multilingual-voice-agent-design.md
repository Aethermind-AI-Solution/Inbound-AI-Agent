# Multilingual Voice Agent — Design Spec

**Date:** 2026-06-07
**Status:** Draft

## Goal

Enable the voice booking agent to handle calls in multiple languages
(en-IN, hi-IN, ta-IN, te-IN, mr-IN, bn-IN, en-US, en-GB, en-AU) using a
single codebase with config-driven regional behavior. No code forks per
region — the tenant config determines everything.

---

## Architecture

One codebase, config-driven regional behavior. The tenant config drives:
which STT/TTS provider to use, which languages are available, whether
language detection runs, and what prompts say.

### Core Flow

1. Caller dials in -> Tenant config loaded
2. If `len(languages) == 1` -> **Single-language mode** (no detection, use
   that language directly)
3. If `len(languages) > 1` and `language_policy.match_caller == true` ->
   **Detection mode:**
   1. Greet in `fallback_language`
   2. Listen to first 1-2 utterances
   3. Detect language from transcription text using
      `lingua-language-detector`
   4. Lock in: switch STT language, TTS voice, prompt language instruction
   5. Re-greet in detected language if different from fallback
   6. Rest of call uses locked-in language

### Provider Strategy

One STT+TTS provider per tenant, not per language.

- **Indian tenants** -> Sarvam STT + Sarvam TTS (handles all 6 Indian
  languages + English)
- **US/global tenants** -> Deepgram STT + Deepgram TTS (English)

This avoids hot-swapping between different providers mid-call.

---

## Config Model Changes

### 1. New PipelineConfig Model

```python
class PipelineConfig(BaseModel):
    stt_provider: Literal["deepgram", "sarvam"] = "deepgram"
    tts_provider: Literal["deepgram", "sarvam"] = "deepgram"
    llm_provider: Literal["openai", "anthropic"] = "openai"
    tts_voices: dict[str, str] = {}
    # Maps language code to TTS voice ID
    # e.g. {"en-IN": "anushka", "hi-IN": "anushka", "ta-IN": "lakshmi"}
```

Added as a new field on `TenantConfig`:

```python
class TenantConfig(BaseModel):
    # ... existing fields ...
    pipeline: PipelineConfig = PipelineConfig()
```

### 2. Per-Language Persona Strings

`greeting` and `ai_disclosure` change from `str` to `dict[str, str]`:

```python
class PersonaConfig(BaseModel):
    # Before: greeting: str
    # After:
    greeting: dict[str, str]
    # e.g. {"en-IN": "Welcome to Glamour Salon!", "hi-IN": "Glamour Salon mein aapka swagat hai!"}

    ai_disclosure: dict[str, str]
    # e.g. {"en-IN": "I'm an AI assistant.", "hi-IN": "Aap ek AI assistant se baat kar rahe hain."}
```

**Validation:** every language in `persona.languages` must have a key in
`greeting` and `ai_disclosure`.

### 3. Example Tenant Configs

**US Salon:**

```yaml
pipeline:
  stt_provider: deepgram
  tts_provider: deepgram
  tts_voices: {"en-US": "aura-asteria-en"}
persona:
  languages: ["en-US"]
  fallback_language: "en-US"
  language_policy: {greeting: "default", match_caller: false}
  greeting: {"en-US": "Welcome to Joe's Barbershop!"}
  ai_disclosure: {"en-US": "I'm an AI assistant."}
```

**Indian Salon:**

```yaml
pipeline:
  stt_provider: sarvam
  tts_provider: sarvam
  tts_voices:
    en-IN: anushka
    hi-IN: anushka
    ta-IN: lakshmi
persona:
  languages: ["en-IN", "hi-IN", "ta-IN"]
  fallback_language: "en-IN"
  language_policy: {greeting: "default", match_caller: true}
  greeting:
    en-IN: "Welcome to Glamour Salon!"
    hi-IN: "Glamour Salon mein aapka swagat hai!"
    ta-IN: "Glamour Salon-kku varavēṟkirōm!"
  ai_disclosure:
    en-IN: "I'm an AI assistant."
    hi-IN: "Aap ek AI assistant se baat kar rahe hain."
    ta-IN: "Nāṉ oru AI utaviyāḷar."
```

---

## Language Detection — LanguageDetectorProcessor

### Position in Pipeline

```
transport.input() -> STT -> LanguageDetector -> Guardrail -> ContextAgg -> LLM -> TTS -> transport.output()
```

### Behavior

New `LanguageDetectorProcessor(FrameProcessor)`:

**Single-language tenant** (`len(languages) == 1`): Pure passthrough. No
detection, no overhead.

**Multi-language tenant** (`match_caller: true`):

1. Counts `TranscriptionFrame`s. For the first 2 transcriptions:
   - Run `lingua-language-detector` on the transcription text
   - `lingua` returns detected language + confidence score
   - Map lingua's language enum to tenant's language codes (e.g.,
     `Language.HINDI` -> `"hi-IN"`)
2. Lock-in decision:
   - If confidence > 0.7 AND detected language is in tenant's `languages`
     list -> lock in immediately (even on turn 1)
   - After turn 2 with no confident detection -> lock in to
     `fallback_language`
3. After lock-in -> `_lock_in(language)` method runs (see Service Switching
   below)
4. Detector becomes passthrough for all remaining frames

**Dependencies:** `lingua-language-detector` Python package. Lightweight,
offline, supports Hindi, Tamil, Telugu, Marathi, Bengali, English, and many
more. No API calls.

### Repeat Caller Optimization

`CallerInfo` gets a new field:

```python
class CallerInfo:
    id: str
    phone: str
    preferred_language: str | None = None
```

On call start in `server.py`:

- If `caller_info.preferred_language` exists AND is in tenant's `languages`
  -> initialize STT/TTS in that language, set LanguageDetector to
  already-locked-in state
- Otherwise -> normal detection flow

On lock-in:

- If detected language differs from `fallback_language` -> write
  `preferred_language` to caller record via data adapter

---

## Service Switching Mechanics

After language lock-in, the `LanguageDetectorProcessor._lock_in()` method:

1. **Update STT language:** Push `STTUpdateSettingsFrame` upstream (toward
   STT) with new language setting
2. **Update TTS voice:** Queue `TTSUpdateSettingsFrame` via worker
   (downstream toward TTS) with the voice from
   `config.pipeline.tts_voices[language]`
3. **Store language in flow state:** Set
   `flow_manager.state["language"] = language`
4. **Re-greet if language changed:** If detected language differs from
   `fallback_language`, queue a `TTSSpeakFrame` with the greeting from
   `config.persona.greeting[language]`

These are pipecat `ControlFrame`s — they flow through the pipeline in order
with no audio gap or reconnection needed. Both Sarvam and Deepgram services
support dynamic settings updates.

```python
async def _lock_in(self, language: str):
    self._locked = True
    self._language = language

    # 1. Switch STT language (upstream to STT service)
    await self.push_frame(
        STTUpdateSettingsFrame(settings={"language": language}),
        FrameDirection.UPSTREAM
    )

    # 2. Switch TTS voice (downstream via worker)
    tts_voice = self._config.pipeline.tts_voices.get(language)
    if tts_voice:
        await self._worker.queue_frame(
            TTSUpdateSettingsFrame(delta=TTSSettings(voice=tts_voice))
        )

    # 3. Store in flow state
    self._flow_state["language"] = language

    # 4. Re-greet in detected language if different from fallback
    if language != self._config.persona.fallback_language:
        greeting = self._config.persona.greeting.get(language, "")
        if greeting:
            await self._worker.queue_frame(TTSSpeakFrame(text=greeting))

    # 5. Store preference for repeat calls
    if self._caller_id:
        await asyncio.to_thread(
            self._data_adapter.update_caller_language,
            self._caller_id, language
        )
```

---

## Prompt Localization

**Approach:** Language instruction in GPT-4o system prompt, NOT
pre-translated prompts for each node.

### Rationale

- 6 languages x 9 node prompts = 54 translated strings to maintain
- Every prompt change needs 6 translations
- GPT-4o speaks all 6 languages fluently — it just needs the instruction
- Natural conversational Hindi/Tamil from GPT-4o is better than manual
  translations
- Tool names and function schemas stay in English (GPT-4o tool-calling
  works best in English)

### What Gets Per-Language Translations (Hand-Written in Config)

- `greeting` — first thing caller hears, must be perfect
- `ai_disclosure` — regulatory requirement, must be precise

**Everything else — GPT-4o generates on the fly based on a language
instruction.**

### Changes to build_role_message()

```python
LANGUAGE_NAMES = {
    "en-IN": "English", "en-US": "English", "en-GB": "English", "en-AU": "English",
    "hi-IN": "Hindi", "ta-IN": "Tamil", "te-IN": "Telugu",
    "mr-IN": "Marathi", "bn-IN": "Bengali",
}

def build_role_message(config: TenantConfig, language: str) -> str:
    persona = config.persona
    lang_name = LANGUAGE_NAMES.get(language, "English")
    disclosure = persona.ai_disclosure.get(
        language, persona.ai_disclosure[persona.fallback_language]
    )

    return (
        f"You are a voice assistant for {persona.business_name}. "
        f"{disclosure} "
        f"Respond in {lang_name}. Use natural, conversational {lang_name} "
        f"— not formal or textbook. "
        # ... rest of role message (anti-injection rules, task constraints, etc.)
    )
```

### Flow Node Integration

All node factory functions (`create_greeting_node`,
`create_collect_service_node`, etc.) read
`flow_manager.state["language"]` and pass it to prompt builders. When the
flow transitions to a new node after lock-in, the new node's prompts
automatically use the detected language.

**Hinglish handling:** If detected language is `hi-IN`, the prompt says
"Respond in Hindi." GPT-4o naturally code-switches — if the caller mixes
English and Hindi, GPT-4o responds in a similar mix. This is the best UX
for Indian callers. No special handling needed.

---

## Pipeline Initialization in server.py

The `_run_pipeline` function creates STT/TTS services based on tenant
config:

```python
# STT
if config.pipeline.stt_provider == "sarvam":
    from pipecat.services.sarvam.stt import SarvamSTTService
    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        settings=SarvamSTTService.Settings(language=initial_language),
    )
else:
    from pipecat.services.deepgram.stt import DeepgramSTTService
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramSTTService.Settings(
            language=initial_language.split("-")[0],  # "en-IN" -> "en"
            model="nova-2-phonecall",
            ...
        ),
    )

# TTS
initial_voice = config.pipeline.tts_voices.get(
    initial_language, "aura-asteria-en"
)
if config.pipeline.tts_provider == "sarvam":
    from pipecat.services.sarvam.tts import SarvamTTSService
    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(voice=initial_voice),
    )
else:
    from pipecat.services.deepgram.tts import DeepgramTTSService
    tts = DeepgramTTSService(
        api_key=DEEPGRAM_API_KEY,
        settings=DeepgramTTSService.Settings(voice=initial_voice),
    )
```

Where `initial_language` is:

- `caller_info.preferred_language` if available and in tenant's languages
- Otherwise `config.persona.fallback_language`

---

## Deployment Strategy

Same Docker image, same codebase. Tenant config is the only thing that
changes between regions.

### Region Mapping

| Aspect | US | India | UK/AU |
|---|---|---|---|
| `languages` | `["en-US"]` | `["en-IN","hi-IN","ta-IN","te-IN","mr-IN","bn-IN"]` | `["en-GB"]` or `["en-AU"]` |
| `stt_provider` | `deepgram` | `sarvam` | `deepgram` |
| `tts_provider` | `deepgram` | `sarvam` | `deepgram` |
| `tts_voices` | `{"en-US":"aura-asteria-en"}` | per-language Sarvam voices | `{"en-GB":"aura-asteria-en"}` |
| Detection active | No (single language) | Yes (multi-language) | No (single language) |
| Greeting | Single English | Per-language map | Single English |

### API Key Management

Both Deepgram and Sarvam API keys present in environment. Config decides
which is used. If deploying region-specific instances, each only needs its
region's keys.

### Tenant Config Storage

Supabase table `tenant_configs` keyed by `tenant_id`. The Twilio phone
number maps to a tenant ID. Each phone number -> one tenant -> one config.

---

## Validation Rules (Config Validator Additions)

1. Every language in `persona.languages` must exist as a key in
   `persona.greeting`
2. Every language in `persona.languages` must exist as a key in
   `persona.ai_disclosure`
3. Every language in `persona.languages` must exist as a key in
   `pipeline.tts_voices`
4. `persona.fallback_language` must be in `persona.languages`
5. If `stt_provider` is `"sarvam"`, `SARVAM_API_KEY` must be set
6. If `tts_provider` is `"sarvam"`, `SARVAM_API_KEY` must be set

---

## Out of Scope

- **Dynamic per-utterance switching** — intentionally excluded; early
  lock-in is sufficient
- **Language-specific STT models** (e.g., different Sarvam model per
  language) — Sarvam uses the same model, just different language param
- **Translated tool/function schemas** — GPT-4o's tool-calling works best
  in English
- **Multi-language within a single utterance** — handled naturally by
  GPT-4o understanding Hinglish
- **Language detection from audio** (as opposed to text) — text-based
  detection via lingua is sufficient and simpler
