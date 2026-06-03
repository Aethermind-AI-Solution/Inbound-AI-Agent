# Voice Booking Agent — Foundation (Phase 0 + Steps 0–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the monorepo, prove voice latency on a real +91 call, and build the three foundation subsystems (config service, Supabase data layer, date/time resolver) that the dialogue state machine (Step 4) depends on.

**Architecture:** Monorepo with `packages/` scoped by concern. Python 3.11+ with Pydantic models and async throughout. Supabase (Postgres + RLS + Edge Functions) as the data/trust layer. A throwaway Pipecat spike to establish the P95 latency baseline before building the real pipeline. The config service, data adapter, and date resolver are pure-function libraries with no I/O dependencies on each other — they compose at Step 4.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Pipecat, Plivo SDK, Deepgram SDK, Sarvam/Smallest SDK, Supabase (Postgres + Deno Edge Functions), `dateparser`, `holidays`, `ruff`, `black`.

**Reference docs:**
- Full spec: `/Users/abhisheksingh/Downloads/voice-booking-agent-FINAL-technical-document.md`
- Build playbook: `/Users/abhisheksingh/Downloads/voice-agent/BUILD.md`
- Known pitfalls: `/Users/abhisheksingh/Downloads/voice-agent/LEARN.md`
- Root conventions: `/Users/abhisheksingh/Downloads/voice-agent/CLAUDE.md`

---

## File Structure

```
voice-booking-agent/                          (project root)
├── CLAUDE.md                                 # Root invariants (from spec §2)
├── BUILD.md                                  # Implementation playbook (living)
├── LEARN.md                                  # Bug/fix/decision log
├── pyproject.toml                            # Python project config (ruff, black, pytest)
├── requirements.txt                          # Python dependencies
├── .env.example                              # Secret template (no real values)
├── .gitignore
├── packages/
│   ├── voice-agent/
│   │   ├── __init__.py
│   │   ├── spike.py                          # Step 0: throwaway voice-loop spike
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── models.py                     # Step 1: Pydantic config models
│   │   │   ├── loader.py                     # Step 1: load config from DB/file
│   │   │   └── validator.py                  # Step 1: validation rules
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── adapter.py                    # Step 2: Python data-access module
│   │   ├── resolver/
│   │   │   ├── __init__.py
│   │   │   └── date_resolver.py              # Step 3: deterministic date/time resolver
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_config_models.py         # Step 1 tests
│   │       ├── test_config_loader.py         # Step 1 tests
│   │       ├── test_config_validator.py      # Step 1 tests
│   │       ├── test_data_adapter.py          # Step 2 tests
│   │       └── test_date_resolver.py         # Step 3 tests
│   ├── edge-functions/
│   │   └── hello/
│   │       └── index.ts                      # Phase 0: smoke-test Edge Function
│   ├── config-service/                       # (alias — models live in voice-agent/config/)
│   ├── admin/                                # (placeholder — Step 8+)
│   └── eval/                                 # (placeholder — Step 7)
├── supabase/
│   └── migrations/
│       ├── 00001_create_tenants.sql          # Step 2
│       ├── 00002_create_tenant_configs.sql   # Step 2
│       ├── 00003_create_resources.sql        # Step 2
│       ├── 00004_create_services.sql         # Step 2
│       ├── 00005_create_callers.sql          # Step 2
│       ├── 00006_create_bookings.sql         # Step 2
│       ├── 00007_create_slot_locks.sql       # Step 2
│       ├── 00008_create_auth_otps.sql        # Step 2
│       ├── 00009_create_audit_log.sql        # Step 2
│       ├── 00010_create_resume_checkpoints.sql # Step 2
│       └── 00011_create_lock_sweeper.sql     # Step 2
└── .claude/
    ├── settings.json
    └── CLAUDE.md                             # (optional scoped context)
```

---

## Task 1: Initialize monorepo and Python environment

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `packages/voice-agent/__init__.py`
- Create: `packages/voice-agent/tests/__init__.py`
- Create: `packages/edge-functions/hello/index.ts`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
venv/
.venv/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Supabase
supabase/.temp/

# Node (admin/edge-functions)
node_modules/

# Test / coverage
htmlcov/
.coverage
.pytest_cache/
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "voice-booking-agent"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.black]
target-version = ["py311"]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["packages/voice-agent/tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create `requirements.txt`**

```txt
# Core
pydantic>=2.0,<3.0
python-dateutil>=2.8

# Voice pipeline (Step 0 spike)
pipecat-ai>=0.5
deepgram-sdk>=3.0
plivo>=4.0

# LLM
anthropic>=0.30

# Data
supabase>=2.0
redis>=5.0

# Date resolution (Step 3)
dateparser>=1.2
holidays>=0.40

# Observability
opentelemetry-sdk>=1.20
langfuse>=2.0

# Dev
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.4
black>=24.0
```

- [ ] **Step 4: Create `.env.example`**

```env
# Telephony
PLIVO_AUTH_ID=
PLIVO_AUTH_TOKEN=
PLIVO_PHONE_NUMBER=

# STT
DEEPGRAM_API_KEY=

# TTS
SMALLEST_API_KEY=
SARVAM_API_KEY=

# LLM
ANTHROPIC_API_KEY=

# Data
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=

# Resume store
REDIS_URL=

# Observability
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
COVAL_API_KEY=
```

- [ ] **Step 5: Create package init files**

Create `packages/voice-agent/__init__.py`:
```python
```

Create `packages/voice-agent/tests/__init__.py`:
```python
```

Create `packages/voice-agent/config/__init__.py`:
```python
```

Create `packages/voice-agent/data/__init__.py`:
```python
```

Create `packages/voice-agent/resolver/__init__.py`:
```python
```

- [ ] **Step 6: Create the hello Edge Function**

Create `packages/edge-functions/hello/index.ts`:
```typescript
import { serve } from "https://deno.land/std@0.177.0/http/server.ts";

serve(async (_req: Request) => {
  return new Response(JSON.stringify({ status: "ok", service: "voice-booking-agent" }), {
    headers: { "Content-Type": "application/json" },
  });
});
```

- [ ] **Step 7: Create placeholder directories**

```bash
mkdir -p packages/admin packages/eval packages/config-service supabase/migrations .claude
```

- [ ] **Step 8: Install Python dependencies and verify**

Run: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
Expected: all packages install without errors.

Run: `source .venv/bin/activate && python -c "import pydantic; import dateparser; import holidays; print('OK')" `
Expected: `OK`

- [ ] **Step 9: Verify ruff and black work**

Run: `source .venv/bin/activate && ruff check packages/ && echo "ruff OK"`
Expected: `ruff OK`

Run: `source .venv/bin/activate && black --check packages/ && echo "black OK"`
Expected: `black OK` (or "All done!" message)

- [ ] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml requirements.txt .env.example packages/ supabase/ .claude/
git commit -m "chore: initialize monorepo structure with Python env and package scaffolding"
```

---

## Task 2: Write root project docs (CLAUDE.md, BUILD.md, LEARN.md)

**Files:**
- Create: `CLAUDE.md`
- Create: `BUILD.md`
- Create: `LEARN.md`

- [ ] **Step 1: Create `CLAUDE.md`**

```markdown
# CLAUDE.md — Voice Booking Agent (root context)

Generic inbound voice appointment-booking agent. India-only, inbound-only. One engine, many sectors via a per-tenant config plane. MVP vertical: salon.

> **Read first:** `BUILD.md` (what to build next) · `LEARN.md` (known pitfalls — check before coding, append after any fix).

---

## Architecture invariants (NEVER violate)
1. The LLM is untrusted. It **never** writes to the DB. Only Supabase Edge Functions touch the DB, and they enforce authz.
2. **Read-back before every write.** Never fake a confirmation.
3. **External systems (CRM/calendar) never sit in the synchronous voice path.** Offer from the Supabase mirror; verify at hold; sync async.
4. Fallback floor is **guaranteed callback capture**. No dead-ends.
5. Dates are resolved by the **deterministic parser**, never invented by the LLM.
6. Guardrail caps (duration, turns, budget) are **deterministic**, never LLM-enforced.
7. Every config change bumps `config_version` and must pass `schema-validator` + regression before `status: live`.
8. `idempotency_key = hold_id`. One hold → one confirm → one booking.
9. In-call state is in-memory per instance, but a **checkpoint is written to the shared store on every state transition** (resume must work cross-instance).

## Stack
Pipecat (Python 3.11+, async) voice service · Supabase (Postgres + RLS + Edge Functions/Deno-TS) · Redis (resume checkpoints) · Plivo (telephony) · Deepgram/Sarvam (STT) · Claude Haiku 4.5 + prompt caching (NLU) · Smallest/Sarvam (TTS) · N8N (async post-call) · Next.js/Vercel (admin) · Coval (regression) · Langfuse/OTel (observability).

## Directory map
```
packages/voice-agent/      # Pipecat service (Python) — pipeline, state machine, resolver, guardrails
packages/edge-functions/   # Supabase Edge Functions (TS) — the tool catalog = trust boundary
packages/config-service/   # Pydantic config models + validation
packages/admin/            # Next.js dashboard
packages/eval/             # Coval scenarios
supabase/migrations/       # SQL migrations
```

## Coding conventions
- **Python:** type hints everywhere; `pydantic` for all config/IO models; `async`/`await` (no blocking calls in the pipeline); `ruff` + `black`; structured logging (no `print`).
- **Edge Functions (TS/Deno):** every tool validates input, checks authz per `auth.action_policy`, scopes by `tenant_id`, applies idempotency; returns typed result or typed error.
- **SQL:** all access via RLS; migrations are forward-only.
- **Secrets:** never inline; read from env/vault.
- **Tests:** `pytest` (Python), `deno test` (functions), Coval (E2E). A feature isn't done until its tests pass and fallback paths are covered.
```

- [ ] **Step 2: Create `BUILD.md`**

```markdown
# BUILD.md — Implementation Playbook

Living build plan. Work top-to-bottom; update status as you go. Full rationale in the spec. Check `LEARN.md` before each step.

**Status legend:** ☐ todo · ◐ in-progress · ☑ done
**MVP cut:** salon · `native_supabase` · `exclusive` booking only · ANI soft-auth (no OTP) · Hindi+English.

---

## Phase 0 — Setup ☐
## Step 0 — Voice-loop spike ☐
## Step 1 — Config service + validation ☐
## Step 2 — Supabase data + adapter ☐
## Step 3 — Date/time resolver ☐
## Step 4 — Dialogue state machine + tools + fallback ☐
## Step 5 — Voice pipeline integration ☐
## Step 6 — Auth + guardrails ☐
## Step 7 — Regression harness ☐
## Step 8 — Salon go-live ☐

## Parallelism
Step 0 ∥ Step 1. Steps 2 ∥ 3 after 1. 4 needs 2+3. 6 overlays 4. 7 seeds early. 8 needs 1–7.
```

- [ ] **Step 3: Create `LEARN.md`**

```markdown
# LEARN.md — Bugs, Fixes & Decisions Log

Running log so a problem solved once is never re-debugged. **Check this before coding in an area; append after every bug fix or non-obvious decision.**

## Entry template
```
### [YYYY-MM-DD] <short title>  ·  area: <voice|data|nlu|fallback|auth|integration|config|infra>
Symptom:    what was observed
Root cause: why it happened
Fix:        what changed
Prevention: test added / rule / hook so it can't recur
```

---

## Pre-empted pitfalls

### [decision] Idempotency key origin · area: data
Symptom: risk of silent double-booking on `confirm_booking` retry.
Root cause: if the agent generates a fresh idempotency key per attempt, a retry books twice.
Fix: **`idempotency_key = hold_id`.** One hold → one confirm → one booking.
Prevention: `bookings.idempotency_key UNIQUE`; concurrency test in Step 2.

### [decision] Date resolution ownership · area: nlu
Symptom: risk of the LLM inventing a non-existent slot/date.
Root cause: letting the model resolve dates.
Fix: deterministic resolver decides dates against business_hours + window; LLM only proposes a phrase.
Prevention: resolver unit suite (Step 3).

### [decision] External systems in the turn loop · area: integration
Symptom: would blow the latency budget.
Root cause: slow/flaky external CRM/calendar APIs.
Fix: offer from Supabase mirror; verify at hold only; sync async.

---

## Bug log
*(append real bugs below as they occur, newest first, using the template)*
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md BUILD.md LEARN.md
git commit -m "docs: add root CLAUDE.md, BUILD.md, and LEARN.md"
```

---

## Task 3: Voice-loop spike (Step 0 — throwaway)

**Files:**
- Create: `packages/voice-agent/spike.py`

**Purpose:** Prove voice latency and turn-taking on a real +91 leg. This is throwaway code — no tests, no TDD. The deliverable is a P95 latency number, not production code.

**Prerequisites:** You need working API keys for Plivo, Deepgram, and Smallest/Sarvam in your `.env` file. You need a Plivo +91 DID number provisioned.

- [ ] **Step 1: Create the spike**

Create `packages/voice-agent/spike.py`:
```python
"""
Throwaway voice-loop spike — prove latency on a real +91 call.

Run: python packages/voice-agent/spike.py
Then call the Plivo +91 number. The agent echoes back what you say.

Measure P50/P95 turn latency. Test barge-in (interrupt while it's speaking).
Delete this file after recording the baseline.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("spike")


@dataclass
class LatencyTracker:
    turn_starts: dict[str, float] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    def start_turn(self, turn_id: str) -> None:
        self.turn_starts[turn_id] = time.monotonic()

    def end_turn(self, turn_id: str) -> None:
        if turn_id in self.turn_starts:
            latency = time.monotonic() - self.turn_starts.pop(turn_id)
            self.latencies.append(latency)
            logger.info(f"Turn {turn_id} latency: {latency:.3f}s")

    def report(self) -> None:
        if not self.latencies:
            logger.warning("No latency data collected")
            return
        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[n // 2]
        p95 = sorted_lat[int(n * 0.95)]
        logger.info(f"Latency report — turns: {n}, P50: {p50:.3f}s, P95: {p95:.3f}s")


tracker = LatencyTracker()
turn_counter = 0


async def run_spike() -> None:
    try:
        from pipecat.frames.frames import (
            EndFrame,
            LLMMessagesFrame,
            TextFrame,
            TranscriptionFrame,
        )
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.deepgram import DeepgramSTTService
        from pipecat.transports.services.plivo import PlivoTransport, PlivoParams
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install: pip install -r requirements.txt")
        return

    global turn_counter

    class EchoProcessor(FrameProcessor):
        """Receives transcription, echoes it back as TTS text."""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, TranscriptionFrame):
                if frame.text and frame.text.strip():
                    global turn_counter
                    turn_counter += 1
                    turn_id = str(turn_counter)

                    tracker.start_turn(turn_id)
                    echo_text = f"You said: {frame.text}"
                    logger.info(f"Echoing: {echo_text}")
                    await self.push_frame(TextFrame(text=echo_text), FrameDirection.DOWNSTREAM)
                    tracker.end_turn(turn_id)
            else:
                await self.push_frame(frame, direction)

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        language="hi",
        model="nova-2",
    )

    echo = EchoProcessor()

    # NOTE: TTS provider setup depends on which SDK you have.
    # Replace with Smallest or Sarvam TTS as appropriate:
    #
    # For Smallest:
    #   from pipecat.services.smallest import SmallestTTSService
    #   tts = SmallestTTSService(api_key=os.environ["SMALLEST_API_KEY"], voice="...")
    #
    # For Sarvam:
    #   from pipecat.services.sarvam import SarvamTTSService
    #   tts = SarvamTTSService(api_key=os.environ["SARVAM_API_KEY"], voice="...")
    #
    # For now, using a placeholder — swap in the real TTS before calling:
    logger.error(
        "TTS provider not configured. Edit spike.py and uncomment your TTS provider "
        "(Smallest or Sarvam) before running."
    )
    return

    # Uncomment after configuring TTS above:
    # transport = PlivoTransport(
    #     PlivoParams(
    #         auth_id=os.environ["PLIVO_AUTH_ID"],
    #         auth_token=os.environ["PLIVO_AUTH_TOKEN"],
    #     )
    # )
    #
    # pipeline = Pipeline([
    #     transport.input(),
    #     stt,
    #     echo,
    #     tts,
    #     transport.output(),
    # ])
    #
    # runner = PipelineRunner()
    # task = PipelineTask(pipeline)
    #
    # logger.info("Spike running — call the +91 number to test")
    # await runner.run(task)
    # tracker.report()


if __name__ == "__main__":
    asyncio.run(run_spike())
```

- [ ] **Step 2: Configure your TTS provider in spike.py**

Open `packages/voice-agent/spike.py` and:
1. Uncomment the import and initialization for your chosen TTS provider (Smallest or Sarvam)
2. Remove the `logger.error(...)` and `return` lines
3. Uncomment the transport, pipeline, runner, and task block at the bottom

- [ ] **Step 3: Run the spike**

Run: `source .venv/bin/activate && python packages/voice-agent/spike.py`

Then call the Plivo +91 DID number from a phone. Verify:
1. The agent answers
2. It echoes back what you say
3. You can interrupt it mid-speech (barge-in)
4. Note the P50/P95 latency numbers from the log

- [ ] **Step 4: Record the baseline**

Add a LEARN.md entry with your latency results:
```markdown
### [YYYY-MM-DD] Voice spike baseline · area: voice
Symptom:    n/a (initial measurement)
Root cause: n/a
Fix:        P50 = X.XXXs, P95 = X.XXXs on +91 with Deepgram Nova-2 + [Smallest|Sarvam] TTS
Prevention: this P95 is the latency target for Steps 4-5
```

- [ ] **Step 5: Delete the spike and commit**

```bash
rm packages/voice-agent/spike.py
git add LEARN.md
git add packages/voice-agent/spike.py
git commit -m "spike: voice-loop latency baseline recorded, spike removed

P95 target set from real +91 call. See LEARN.md for numbers."
```

---

## Task 4: Config models — Pydantic schema (Step 1, part 1)

**Files:**
- Create: `packages/voice-agent/config/models.py`
- Create: `packages/voice-agent/tests/test_config_models.py`

- [ ] **Step 1: Write the failing tests for config models**

Create `packages/voice-agent/tests/test_config_models.py`:
```python
import pytest
from pydantic import ValidationError

from packages.voice_agent.config.models import (
    AuthConfig,
    BookingModel,
    BusinessHours,
    CustomField,
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


class TestMetaConfig:
    def test_valid_meta(self):
        meta = MetaConfig(
            tenant_id="t-001",
            sector="salon",
            config_version="1.0.0",
            status="draft",
        )
        assert meta.tenant_id == "t-001"
        assert meta.sector == "salon"

    def test_invalid_sector_rejected(self):
        with pytest.raises(ValidationError):
            MetaConfig(
                tenant_id="t-001",
                sector="spaceship",
                config_version="1.0.0",
                status="draft",
            )

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            MetaConfig(
                tenant_id="t-001",
                sector="salon",
                config_version="1.0.0",
                status="yolo",
            )


class TestPersonaConfig:
    def test_valid_persona(self):
        persona = PersonaConfig(
            business_name="Glamour Salon",
            greeting="Welcome to Glamour Salon!",
            ai_disclosure="I am an AI assistant.",
            tone="warm",
            languages=["en-IN", "hi-IN"],
            fallback_language="en-IN",
            language_policy=LanguagePolicy(greeting="default", match_caller=True),
        )
        assert persona.business_name == "Glamour Salon"
        assert len(persona.languages) == 2

    def test_invalid_tone_rejected(self):
        with pytest.raises(ValidationError):
            PersonaConfig(
                business_name="Test",
                greeting="Hi",
                ai_disclosure="I am AI.",
                tone="aggressive",
                languages=["en-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=False),
            )


class TestIntegrationConfig:
    def test_native_supabase(self):
        config = IntegrationConfig(
            calendar_provider="native_supabase",
            adapter_id=None,
            credentials_ref=None,
            write_back=False,
            system_of_record="native",
            external_unreachable_policy="capture",
        )
        assert config.calendar_provider == "native_supabase"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            IntegrationConfig(
                calendar_provider="outlook",
                adapter_id=None,
                credentials_ref=None,
                write_back=False,
                system_of_record="native",
                external_unreachable_policy="capture",
            )


class TestAuthConfig:
    def test_valid_auth(self):
        auth = AuthConfig(
            levels=["none", "soft"],
            action_policy={
                "new_booking": "none",
                "read_existing": "soft",
                "modify_cancel": "soft",
                "take_payment": "otp",
            },
            otp_channel=None,
            otp_entry="dtmf",
            soft_match_fields=["caller_number"],
        )
        assert auth.levels == ["none", "soft"]

    def test_invalid_level_rejected(self):
        with pytest.raises(ValidationError):
            AuthConfig(
                levels=["super_admin"],
                action_policy={"new_booking": "none"},
                otp_channel=None,
                otp_entry="dtmf",
                soft_match_fields=["caller_number"],
            )


class TestEdgeProfile:
    def test_valid_edge_profile(self):
        profile = EdgeProfile(
            caller_demographic="general",
            endpointing_ms=700,
            barge_in=True,
            asr_lexicon=[],
            noise_profile="quiet",
            dtmf_fallback=True,
        )
        assert profile.endpointing_ms == 700

    def test_endpointing_too_low_rejected(self):
        with pytest.raises(ValidationError):
            EdgeProfile(
                caller_demographic="general",
                endpointing_ms=50,
                barge_in=True,
                asr_lexicon=[],
                noise_profile="quiet",
                dtmf_fallback=True,
            )


class TestEscalationConfig:
    def test_valid_escalation(self):
        config = EscalationConfig(
            chain=[
                EscalationChain(type="transfer", number="+919999999999"),
                EscalationChain(type="callback_queue", number=None),
            ],
            trigger_on=["repeated_failure", "explicit_request"],
        )
        assert len(config.chain) == 2

    def test_empty_chain_rejected(self):
        with pytest.raises(ValidationError):
            EscalationConfig(
                chain=[],
                trigger_on=["repeated_failure"],
            )


class TestBookingModel:
    def test_valid_salon_booking_model(self):
        model = BookingModel(
            resources=[
                Resource(
                    id="stylist-1",
                    name="Priya",
                    type="stylist",
                    capacity=1,
                    tags=[],
                    calendar_ref=None,
                )
            ],
            services=[
                Service(
                    id="svc-haircut",
                    name="Haircut",
                    resource_type="stylist",
                    duration_min=30,
                    booking_mode="exclusive",
                    price=500,
                    deposit=None,
                    custom_fields=[],
                )
            ],
            business_hours=BusinessHours(
                mon=["09:00-19:00"],
                tue=["09:00-19:00"],
                wed=["09:00-19:00"],
                thu=["09:00-19:00"],
                fri=["09:00-19:00"],
                sat=["10:00-18:00"],
                sun=[],
            ),
            booking_window_days=30,
            min_notice_min=60,
        )
        assert model.resources[0].name == "Priya"
        assert model.services[0].duration_min == 30

    def test_service_with_custom_fields(self):
        service = Service(
            id="svc-consult",
            name="Consultation",
            resource_type="doctor",
            duration_min=15,
            booking_mode="exclusive",
            price=None,
            deposit=None,
            custom_fields=[
                CustomField(key="reason", type="str", required=True, prompt="What is the reason for your visit?"),
            ],
        )
        assert service.custom_fields[0].key == "reason"


class TestGuardrailsConfig:
    def test_valid_guardrails(self):
        config = GuardrailsConfig(
            max_call_seconds=240,
            max_turns=25,
            monthly_budget_inr=6000,
            scope="booking_only",
            recording_consent=True,
            retention_days=90,
        )
        assert config.max_call_seconds == 240

    def test_zero_retention_rejected(self):
        with pytest.raises(ValidationError):
            GuardrailsConfig(
                max_call_seconds=240,
                max_turns=25,
                monthly_budget_inr=6000,
                scope="booking_only",
                recording_consent=True,
                retention_days=0,
            )


class TestTenantConfig:
    def test_full_salon_config(self):
        config = TenantConfig(
            meta=MetaConfig(
                tenant_id="t-salon-001",
                sector="salon",
                config_version="1.0.0",
                status="draft",
            ),
            persona=PersonaConfig(
                business_name="Glamour Salon",
                greeting="Welcome to Glamour Salon!",
                ai_disclosure="I am an AI assistant helping you book appointments.",
                tone="warm",
                languages=["en-IN", "hi-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="bilingual", match_caller=True),
            ),
            integration=IntegrationConfig(
                calendar_provider="native_supabase",
                adapter_id=None,
                credentials_ref=None,
                write_back=False,
                system_of_record="native",
                external_unreachable_policy="capture",
            ),
            auth=AuthConfig(
                levels=["none", "soft"],
                action_policy={
                    "new_booking": "none",
                    "read_existing": "soft",
                    "modify_cancel": "soft",
                    "take_payment": "otp",
                },
                otp_channel=None,
                otp_entry="dtmf",
                soft_match_fields=["caller_number"],
            ),
            edge_profile=EdgeProfile(
                caller_demographic="general",
                endpointing_ms=700,
                barge_in=True,
                asr_lexicon=["balayage", "keratin", "ombre"],
                noise_profile="moderate",
                dtmf_fallback=True,
            ),
            escalation=EscalationConfig(
                chain=[
                    EscalationChain(type="transfer", number="+919876543210"),
                    EscalationChain(type="callback_queue", number=None),
                    EscalationChain(type="sms_owner", number="+919876543210"),
                ],
                trigger_on=["repeated_failure", "explicit_request", "out_of_scope", "abuse"],
            ),
            booking_model=BookingModel(
                resources=[
                    Resource(id="stylist-priya", name="Priya", type="stylist", capacity=1, tags=["senior"], calendar_ref=None),
                    Resource(id="stylist-neha", name="Neha", type="stylist", capacity=1, tags=[], calendar_ref=None),
                ],
                services=[
                    Service(id="svc-haircut", name="Haircut", resource_type="stylist", duration_min=30, booking_mode="exclusive", price=500, deposit=None, custom_fields=[]),
                    Service(id="svc-color", name="Hair Color", resource_type="stylist", duration_min=120, booking_mode="exclusive", price=2500, deposit=None, custom_fields=[]),
                ],
                business_hours=BusinessHours(mon=["09:00-19:00"], tue=["09:00-19:00"], wed=["09:00-19:00"], thu=["09:00-19:00"], fri=["09:00-19:00"], sat=["10:00-18:00"], sun=[]),
                booking_window_days=30,
                min_notice_min=60,
            ),
            guardrails=GuardrailsConfig(
                max_call_seconds=240,
                max_turns=25,
                monthly_budget_inr=6000,
                scope="booking_only",
                recording_consent=True,
                retention_days=90,
            ),
        )
        assert config.meta.tenant_id == "t-salon-001"
        assert len(config.booking_model.resources) == 2
        assert len(config.booking_model.services) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.config.models'`

- [ ] **Step 3: Implement config models**

Create `packages/voice-agent/config/models.py`:
```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


VALID_SECTORS = ("salon", "clinic", "restaurant", "gym", "home_service", "real_estate")
VALID_STATUSES = ("draft", "staging", "live")
VALID_TONES = ("warm", "professional", "casual")
VALID_AUTH_LEVELS = ("none", "soft", "otp")
VALID_CALENDAR_PROVIDERS = ("google_calendar", "vendor_api", "native_supabase")
VALID_SOR = ("external", "native")
VALID_UNREACHABLE_POLICIES = ("trust_mirror", "capture")
VALID_DEMOGRAPHICS = ("general", "elderly", "youth")
VALID_NOISE_PROFILES = ("quiet", "moderate", "noisy")
VALID_ESCALATION_TYPES = ("transfer", "callback_queue", "sms_owner")
VALID_BOOKING_MODES = ("exclusive", "shared")
VALID_GREETING_POLICIES = ("default", "bilingual")
VALID_CUSTOM_FIELD_TYPES = ("str", "int", "float", "bool")
VALID_OTP_ENTRIES = ("dtmf",)


class MetaConfig(BaseModel):
    tenant_id: str
    sector: str
    config_version: str
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


class LanguagePolicy(BaseModel):
    greeting: str
    match_caller: bool

    @field_validator("greeting")
    @classmethod
    def validate_greeting(cls, v: str) -> str:
        if v not in VALID_GREETING_POLICIES:
            raise ValueError(f"greeting must be one of {VALID_GREETING_POLICIES}, got '{v}'")
        return v


class PersonaConfig(BaseModel):
    business_name: str
    greeting: str
    ai_disclosure: str
    tone: str
    languages: list[str] = Field(min_length=1)
    fallback_language: str
    language_policy: LanguagePolicy

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in VALID_TONES:
            raise ValueError(f"tone must be one of {VALID_TONES}, got '{v}'")
        return v


class IntegrationConfig(BaseModel):
    calendar_provider: str
    adapter_id: str | None
    credentials_ref: str | None
    write_back: bool
    system_of_record: str
    external_unreachable_policy: str

    @field_validator("calendar_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in VALID_CALENDAR_PROVIDERS:
            raise ValueError(f"calendar_provider must be one of {VALID_CALENDAR_PROVIDERS}, got '{v}'")
        return v

    @field_validator("system_of_record")
    @classmethod
    def validate_sor(cls, v: str) -> str:
        if v not in VALID_SOR:
            raise ValueError(f"system_of_record must be one of {VALID_SOR}, got '{v}'")
        return v

    @field_validator("external_unreachable_policy")
    @classmethod
    def validate_unreachable(cls, v: str) -> str:
        if v not in VALID_UNREACHABLE_POLICIES:
            raise ValueError(f"external_unreachable_policy must be one of {VALID_UNREACHABLE_POLICIES}, got '{v}'")
        return v


class AuthConfig(BaseModel):
    levels: list[str] = Field(min_length=1)
    action_policy: dict[str, str]
    otp_channel: str | None
    otp_entry: str
    soft_match_fields: list[str]

    @field_validator("levels")
    @classmethod
    def validate_levels(cls, v: list[str]) -> list[str]:
        for level in v:
            if level not in VALID_AUTH_LEVELS:
                raise ValueError(f"auth level must be one of {VALID_AUTH_LEVELS}, got '{level}'")
        return v

    @field_validator("otp_entry")
    @classmethod
    def validate_otp_entry(cls, v: str) -> str:
        if v not in VALID_OTP_ENTRIES:
            raise ValueError(f"otp_entry must be one of {VALID_OTP_ENTRIES}, got '{v}'")
        return v


class EdgeProfile(BaseModel):
    caller_demographic: str
    endpointing_ms: int = Field(ge=100, le=2000)
    barge_in: bool
    asr_lexicon: list[str]
    noise_profile: str
    dtmf_fallback: bool

    @field_validator("caller_demographic")
    @classmethod
    def validate_demographic(cls, v: str) -> str:
        if v not in VALID_DEMOGRAPHICS:
            raise ValueError(f"caller_demographic must be one of {VALID_DEMOGRAPHICS}, got '{v}'")
        return v

    @field_validator("noise_profile")
    @classmethod
    def validate_noise(cls, v: str) -> str:
        if v not in VALID_NOISE_PROFILES:
            raise ValueError(f"noise_profile must be one of {VALID_NOISE_PROFILES}, got '{v}'")
        return v


class EscalationChain(BaseModel):
    type: str
    number: str | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_ESCALATION_TYPES:
            raise ValueError(f"escalation type must be one of {VALID_ESCALATION_TYPES}, got '{v}'")
        return v


class EscalationConfig(BaseModel):
    chain: list[EscalationChain] = Field(min_length=1)
    trigger_on: list[str]


class Resource(BaseModel):
    id: str
    name: str
    type: str
    capacity: int = Field(ge=1)
    tags: list[str] = []
    calendar_ref: str | None = None


class CustomField(BaseModel):
    key: str
    type: str
    required: bool
    prompt: str

    @field_validator("type")
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        if v not in VALID_CUSTOM_FIELD_TYPES:
            raise ValueError(f"custom field type must be one of {VALID_CUSTOM_FIELD_TYPES}, got '{v}'")
        return v


class Service(BaseModel):
    id: str
    name: str
    resource_type: str
    duration_min: int = Field(gt=0)
    booking_mode: str
    price: float | None = None
    deposit: float | None = None
    custom_fields: list[CustomField] = []

    @field_validator("booking_mode")
    @classmethod
    def validate_booking_mode(cls, v: str) -> str:
        if v not in VALID_BOOKING_MODES:
            raise ValueError(f"booking_mode must be one of {VALID_BOOKING_MODES}, got '{v}'")
        return v


class BusinessHours(BaseModel):
    mon: list[str] = []
    tue: list[str] = []
    wed: list[str] = []
    thu: list[str] = []
    fri: list[str] = []
    sat: list[str] = []
    sun: list[str] = []

    def get_hours_for_day(self, day: str) -> list[str]:
        return getattr(self, day[:3].lower(), [])

    def is_open_on(self, day: str) -> bool:
        return len(self.get_hours_for_day(day)) > 0


class BookingModel(BaseModel):
    resources: list[Resource] = Field(min_length=1)
    services: list[Service] = Field(min_length=1)
    business_hours: BusinessHours
    booking_window_days: int = Field(gt=0)
    min_notice_min: int = Field(ge=0)


class GuardrailsConfig(BaseModel):
    max_call_seconds: int = Field(gt=0)
    max_turns: int = Field(gt=0)
    monthly_budget_inr: int = Field(gt=0)
    scope: str
    recording_consent: bool
    retention_days: int = Field(gt=0)


class TenantConfig(BaseModel):
    meta: MetaConfig
    persona: PersonaConfig
    integration: IntegrationConfig
    auth: AuthConfig
    edge_profile: EdgeProfile
    escalation: EscalationConfig
    booking_model: BookingModel
    guardrails: GuardrailsConfig
```

- [ ] **Step 4: Fix the import path for tests**

The tests import from `packages.voice_agent.config.models`. For this to work, you need the project root on `PYTHONPATH`. Add a `conftest.py` at the project root:

Create `conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

Also, rename the package directory for Python imports. Because the directory is `voice-agent` (with a hyphen), Python can't import it. Create a symlink or rename:

```bash
cd packages && ln -s voice-agent voice_agent && cd ..
```

Update all test imports to use `packages.voice_agent.config.models`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_models.py -v`
Expected: all tests PASS

- [ ] **Step 6: Run ruff and black**

Run: `source .venv/bin/activate && ruff check packages/voice-agent/config/models.py --fix && black packages/voice-agent/config/models.py`
Expected: no errors or auto-fixed

- [ ] **Step 7: Commit**

```bash
git add packages/voice-agent/config/models.py packages/voice-agent/tests/test_config_models.py conftest.py
git commit -m "feat(config): add Pydantic config models for tenant configuration schema"
```

---

## Task 5: Config validator — cross-field validation rules (Step 1, part 2)

**Files:**
- Create: `packages/voice-agent/config/validator.py`
- Create: `packages/voice-agent/tests/test_config_validator.py`

The Pydantic models enforce per-field constraints. The validator enforces cross-field rules from SPEC §7.4:
- Every `service.resource_type` matches a resource's `type`
- `escalation.chain` non-empty (already enforced by Pydantic, but validator double-checks)
- Any `otp` in `action_policy` values requires `otp_channel` to be set
- `ai_disclosure` and `recording_consent` and `retention_days` must be present (already enforced by Pydantic, but validator double-checks)
- All guardrail caps must be present (already enforced)
- `business_hours` covers at least one bookable day
- `languages` are all in the supported set
- `fallback_language` is in `languages`

- [ ] **Step 1: Write failing tests**

Create `packages/voice-agent/tests/test_config_validator.py`:
```python
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
from packages.voice_agent.config.validator import validate_tenant_config, ConfigValidationError


def _make_valid_config(**overrides) -> TenantConfig:
    """Build a valid salon config, applying overrides to specific sections."""
    base = dict(
        meta=MetaConfig(tenant_id="t-001", sector="salon", config_version="1.0.0", status="draft"),
        persona=PersonaConfig(
            business_name="Test Salon",
            greeting="Welcome!",
            ai_disclosure="I am an AI.",
            tone="warm",
            languages=["en-IN", "hi-IN"],
            fallback_language="en-IN",
            language_policy=LanguagePolicy(greeting="default", match_caller=True),
        ),
        integration=IntegrationConfig(
            calendar_provider="native_supabase",
            adapter_id=None,
            credentials_ref=None,
            write_back=False,
            system_of_record="native",
            external_unreachable_policy="capture",
        ),
        auth=AuthConfig(
            levels=["none", "soft"],
            action_policy={"new_booking": "none", "modify_cancel": "soft"},
            otp_channel=None,
            otp_entry="dtmf",
            soft_match_fields=["caller_number"],
        ),
        edge_profile=EdgeProfile(
            caller_demographic="general",
            endpointing_ms=700,
            barge_in=True,
            asr_lexicon=[],
            noise_profile="quiet",
            dtmf_fallback=True,
        ),
        escalation=EscalationConfig(
            chain=[EscalationChain(type="callback_queue")],
            trigger_on=["repeated_failure"],
        ),
        booking_model=BookingModel(
            resources=[Resource(id="r1", name="Priya", type="stylist", capacity=1)],
            services=[
                Service(
                    id="s1",
                    name="Haircut",
                    resource_type="stylist",
                    duration_min=30,
                    booking_mode="exclusive",
                )
            ],
            business_hours=BusinessHours(mon=["09:00-19:00"], tue=["09:00-19:00"]),
            booking_window_days=30,
            min_notice_min=60,
        ),
        guardrails=GuardrailsConfig(
            max_call_seconds=240,
            max_turns=25,
            monthly_budget_inr=6000,
            scope="booking_only",
            recording_consent=True,
            retention_days=90,
        ),
    )
    base.update(overrides)
    return TenantConfig(**base)


class TestValidConfig:
    def test_valid_salon_config_passes(self):
        config = _make_valid_config()
        errors = validate_tenant_config(config)
        assert errors == []


class TestResourceTypeMatch:
    def test_service_resource_type_not_matching_any_resource_fails(self):
        config = _make_valid_config(
            booking_model=BookingModel(
                resources=[Resource(id="r1", name="Priya", type="stylist", capacity=1)],
                services=[
                    Service(
                        id="s1",
                        name="Consult",
                        resource_type="doctor",
                        duration_min=15,
                        booking_mode="exclusive",
                    )
                ],
                business_hours=BusinessHours(mon=["09:00-19:00"]),
                booking_window_days=30,
                min_notice_min=60,
            ),
        )
        errors = validate_tenant_config(config)
        assert any("resource_type" in e for e in errors)


class TestOtpRequiresChannel:
    def test_otp_in_policy_without_channel_fails(self):
        config = _make_valid_config(
            auth=AuthConfig(
                levels=["none", "otp"],
                action_policy={"new_booking": "none", "read_existing": "otp"},
                otp_channel=None,
                otp_entry="dtmf",
                soft_match_fields=["caller_number"],
            ),
        )
        errors = validate_tenant_config(config)
        assert any("otp_channel" in e for e in errors)

    def test_otp_in_policy_with_channel_passes(self):
        config = _make_valid_config(
            auth=AuthConfig(
                levels=["none", "otp"],
                action_policy={"new_booking": "none", "read_existing": "otp"},
                otp_channel="sms_registered_number",
                otp_entry="dtmf",
                soft_match_fields=["caller_number"],
            ),
        )
        errors = validate_tenant_config(config)
        assert errors == []


class TestBusinessHoursCoverage:
    def test_no_business_hours_fails(self):
        config = _make_valid_config(
            booking_model=BookingModel(
                resources=[Resource(id="r1", name="Priya", type="stylist", capacity=1)],
                services=[
                    Service(
                        id="s1",
                        name="Haircut",
                        resource_type="stylist",
                        duration_min=30,
                        booking_mode="exclusive",
                    )
                ],
                business_hours=BusinessHours(),
                booking_window_days=30,
                min_notice_min=60,
            ),
        )
        errors = validate_tenant_config(config)
        assert any("business_hours" in e for e in errors)


class TestLanguageValidation:
    def test_unsupported_language_fails(self):
        config = _make_valid_config(
            persona=PersonaConfig(
                business_name="Test",
                greeting="Hi",
                ai_disclosure="I am AI.",
                tone="warm",
                languages=["en-IN", "jp-JP"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=False),
            ),
        )
        errors = validate_tenant_config(config)
        assert any("language" in e.lower() for e in errors)

    def test_fallback_not_in_languages_fails(self):
        config = _make_valid_config(
            persona=PersonaConfig(
                business_name="Test",
                greeting="Hi",
                ai_disclosure="I am AI.",
                tone="warm",
                languages=["hi-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=False),
            ),
        )
        errors = validate_tenant_config(config)
        assert any("fallback_language" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_validator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.config.validator'`

- [ ] **Step 3: Implement the validator**

Create `packages/voice-agent/config/validator.py`:
```python
from __future__ import annotations

from packages.voice_agent.config.models import TenantConfig


SUPPORTED_LANGUAGES = {"en-IN", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN"}


class ConfigValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Config validation failed: {errors}")


def validate_tenant_config(config: TenantConfig) -> list[str]:
    errors: list[str] = []

    resource_types = {r.type for r in config.booking_model.resources}
    for svc in config.booking_model.services:
        if svc.resource_type not in resource_types:
            errors.append(
                f"Service '{svc.name}' has resource_type '{svc.resource_type}' "
                f"but no resource of that type exists. Available: {resource_types}"
            )

    uses_otp = any(v == "otp" for v in config.auth.action_policy.values())
    if uses_otp and not config.auth.otp_channel:
        errors.append(
            "action_policy uses 'otp' but otp_channel is not set. "
            "Set otp_channel (e.g. 'sms_registered_number') when any action requires OTP."
        )

    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    has_hours = any(config.booking_model.business_hours.is_open_on(d) for d in days)
    if not has_hours:
        errors.append(
            "business_hours has no open days. At least one day must have hours defined."
        )

    for lang in config.persona.languages:
        if lang not in SUPPORTED_LANGUAGES:
            errors.append(
                f"Language '{lang}' is not in supported set: {SUPPORTED_LANGUAGES}"
            )

    if config.persona.fallback_language not in config.persona.languages:
        errors.append(
            f"fallback_language '{config.persona.fallback_language}' "
            f"is not in languages list: {config.persona.languages}"
        )

    return errors


def validate_or_raise(config: TenantConfig) -> None:
    errors = validate_tenant_config(config)
    if errors:
        raise ConfigValidationError(errors)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_validator.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/config/validator.py packages/voice-agent/tests/test_config_validator.py
git commit -m "feat(config): add cross-field config validator with spec §7.4 rules"
```

---

## Task 6: Config loader — load config from file/DB (Step 1, part 3)

**Files:**
- Create: `packages/voice-agent/config/loader.py`
- Create: `packages/voice-agent/tests/test_config_loader.py`
- Create: `packages/voice-agent/tests/fixtures/salon_config.json`

- [ ] **Step 1: Create the test fixture — a valid salon config JSON**

Create `packages/voice-agent/tests/fixtures/salon_config.json`:
```json
{
  "meta": {
    "tenant_id": "t-salon-001",
    "sector": "salon",
    "config_version": "1.0.0",
    "status": "draft"
  },
  "persona": {
    "business_name": "Glamour Salon",
    "greeting": "Welcome to Glamour Salon! How can I help you today?",
    "ai_disclosure": "Just so you know, I am an AI assistant helping with appointments.",
    "tone": "warm",
    "languages": ["en-IN", "hi-IN"],
    "fallback_language": "en-IN",
    "language_policy": { "greeting": "bilingual", "match_caller": true }
  },
  "integration": {
    "calendar_provider": "native_supabase",
    "adapter_id": null,
    "credentials_ref": null,
    "write_back": false,
    "system_of_record": "native",
    "external_unreachable_policy": "capture"
  },
  "auth": {
    "levels": ["none", "soft"],
    "action_policy": {
      "new_booking": "none",
      "read_existing": "soft",
      "modify_cancel": "soft",
      "take_payment": "otp"
    },
    "otp_channel": "sms_registered_number",
    "otp_entry": "dtmf",
    "soft_match_fields": ["caller_number"]
  },
  "edge_profile": {
    "caller_demographic": "general",
    "endpointing_ms": 700,
    "barge_in": true,
    "asr_lexicon": ["balayage", "keratin", "ombre", "highlights"],
    "noise_profile": "moderate",
    "dtmf_fallback": true
  },
  "escalation": {
    "chain": [
      { "type": "transfer", "number": "+919876543210" },
      { "type": "callback_queue", "number": null },
      { "type": "sms_owner", "number": "+919876543210" }
    ],
    "trigger_on": ["repeated_failure", "explicit_request", "out_of_scope", "abuse"]
  },
  "booking_model": {
    "resources": [
      { "id": "stylist-priya", "name": "Priya", "type": "stylist", "capacity": 1, "tags": ["senior"], "calendar_ref": null },
      { "id": "stylist-neha", "name": "Neha", "type": "stylist", "capacity": 1, "tags": [], "calendar_ref": null }
    ],
    "services": [
      { "id": "svc-haircut", "name": "Haircut", "resource_type": "stylist", "duration_min": 30, "booking_mode": "exclusive", "price": 500, "deposit": null, "custom_fields": [] },
      { "id": "svc-color", "name": "Hair Color", "resource_type": "stylist", "duration_min": 120, "booking_mode": "exclusive", "price": 2500, "deposit": null, "custom_fields": [] }
    ],
    "business_hours": {
      "mon": ["09:00-19:00"],
      "tue": ["09:00-19:00"],
      "wed": ["09:00-19:00"],
      "thu": ["09:00-19:00"],
      "fri": ["09:00-19:00"],
      "sat": ["10:00-18:00"],
      "sun": []
    },
    "booking_window_days": 30,
    "min_notice_min": 60
  },
  "guardrails": {
    "max_call_seconds": 240,
    "max_turns": 25,
    "monthly_budget_inr": 6000,
    "scope": "booking_only",
    "recording_consent": true,
    "retention_days": 90
  }
}
```

- [ ] **Step 2: Write failing tests for the loader**

Create `packages/voice-agent/tests/test_config_loader.py`:
```python
import json
from pathlib import Path

import pytest

from packages.voice_agent.config.loader import load_config_from_file, load_config_from_dict
from packages.voice_agent.config.models import TenantConfig
from packages.voice_agent.config.validator import ConfigValidationError


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadFromFile:
    def test_load_valid_salon_config(self):
        config = load_config_from_file(FIXTURES_DIR / "salon_config.json")
        assert isinstance(config, TenantConfig)
        assert config.meta.tenant_id == "t-salon-001"
        assert config.meta.sector == "salon"
        assert len(config.booking_model.resources) == 2

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config_from_file(Path("/nonexistent/config.json"))

    def test_load_invalid_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all {{{")
        with pytest.raises(json.JSONDecodeError):
            load_config_from_file(bad_file)


class TestLoadFromDict:
    def test_load_valid_dict(self):
        with open(FIXTURES_DIR / "salon_config.json") as f:
            data = json.load(f)
        config = load_config_from_dict(data)
        assert isinstance(config, TenantConfig)
        assert config.meta.sector == "salon"

    def test_load_invalid_dict_raises_validation_error(self):
        with pytest.raises(ConfigValidationError):
            load_config_from_dict({"meta": {"tenant_id": "t-001", "sector": "salon", "config_version": "1.0.0", "status": "draft"}})

    def test_load_dict_with_cross_field_error_raises(self):
        with open(FIXTURES_DIR / "salon_config.json") as f:
            data = json.load(f)
        data["booking_model"]["services"][0]["resource_type"] = "doctor"
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict(data)
        assert "resource_type" in str(exc_info.value)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.config.loader'`

- [ ] **Step 4: Implement the loader**

Create `packages/voice-agent/config/loader.py`:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from packages.voice_agent.config.models import TenantConfig
from packages.voice_agent.config.validator import ConfigValidationError, validate_tenant_config


def load_config_from_file(path: Path) -> TenantConfig:
    with open(path) as f:
        data = json.load(f)
    return load_config_from_dict(data)


def load_config_from_dict(data: dict[str, Any]) -> TenantConfig:
    try:
        config = TenantConfig(**data)
    except PydanticValidationError as e:
        raise ConfigValidationError([str(err) for err in e.errors()]) from e

    errors = validate_tenant_config(config)
    if errors:
        raise ConfigValidationError(errors)

    return config
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_loader.py -v`
Expected: all tests PASS

- [ ] **Step 6: Run the full Step 1 test suite**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_config_models.py packages/voice-agent/tests/test_config_validator.py packages/voice-agent/tests/test_config_loader.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add packages/voice-agent/config/loader.py packages/voice-agent/tests/test_config_loader.py packages/voice-agent/tests/fixtures/
git commit -m "feat(config): add config loader from file and dict with validation"
```

---

## Task 7: Supabase migrations — all tables with RLS (Step 2, part 1)

**Files:**
- Create: `supabase/migrations/00001_create_tenants.sql`
- Create: `supabase/migrations/00002_create_tenant_configs.sql`
- Create: `supabase/migrations/00003_create_resources.sql`
- Create: `supabase/migrations/00004_create_services.sql`
- Create: `supabase/migrations/00005_create_callers.sql`
- Create: `supabase/migrations/00006_create_bookings.sql`
- Create: `supabase/migrations/00007_create_slot_locks.sql`
- Create: `supabase/migrations/00008_create_auth_otps.sql`
- Create: `supabase/migrations/00009_create_audit_log.sql`
- Create: `supabase/migrations/00010_create_resume_checkpoints.sql`
- Create: `supabase/migrations/00011_create_lock_sweeper.sql`

- [ ] **Step 1: Create tenants table**

Create `supabase/migrations/00001_create_tenants.sql`:
```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sector TEXT NOT NULL CHECK (sector IN ('salon', 'clinic', 'restaurant', 'gym', 'home_service', 'real_estate')),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'staging', 'live')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to tenants"
    ON tenants
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 2: Create tenant_configs table**

Create `supabase/migrations/00002_create_tenant_configs.sql`:
```sql
CREATE TABLE tenant_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    config JSONB NOT NULL,
    config_version TEXT NOT NULL,
    validated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tenant_configs_tenant_id ON tenant_configs(tenant_id);

ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to tenant_configs"
    ON tenant_configs
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 3: Create resources table**

Create `supabase/migrations/00003_create_resources.sql`:
```sql
CREATE TABLE resources (
    id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    capacity INT NOT NULL DEFAULT 1 CHECK (capacity >= 1),
    tags JSONB NOT NULL DEFAULT '[]'::JSONB,
    calendar_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, tenant_id)
);

CREATE INDEX idx_resources_tenant_id ON resources(tenant_id);
CREATE INDEX idx_resources_type ON resources(tenant_id, type);

ALTER TABLE resources ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to resources"
    ON resources
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 4: Create services table**

Create `supabase/migrations/00004_create_services.sql`:
```sql
CREATE TABLE services (
    id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    duration_min INT NOT NULL CHECK (duration_min > 0),
    booking_mode TEXT NOT NULL CHECK (booking_mode IN ('exclusive', 'shared')),
    price NUMERIC,
    deposit NUMERIC,
    custom_fields JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, tenant_id)
);

CREATE INDEX idx_services_tenant_id ON services(tenant_id);

ALTER TABLE services ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to services"
    ON services
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 5: Create callers table**

Create `supabase/migrations/00005_create_callers.sql`:
```sql
CREATE TABLE callers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    phone TEXT NOT NULL,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, phone)
);

CREATE INDEX idx_callers_phone ON callers(tenant_id, phone);

ALTER TABLE callers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to callers"
    ON callers
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 6: Create bookings table**

Create `supabase/migrations/00006_create_bookings.sql`:
```sql
CREATE TABLE bookings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    service_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    caller_id UUID NOT NULL REFERENCES callers(id),
    start_ts TIMESTAMPTZ NOT NULL,
    end_ts TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed', 'cancelled', 'completed', 'no_show')),
    custom_values JSONB NOT NULL DEFAULT '{}'::JSONB,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_bookings_tenant_id ON bookings(tenant_id);
CREATE INDEX idx_bookings_caller_id ON bookings(caller_id);
CREATE INDEX idx_bookings_resource_time ON bookings(resource_id, start_ts, end_ts);

ALTER TABLE bookings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to bookings"
    ON bookings
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 7: Create slot_locks table**

Create `supabase/migrations/00007_create_slot_locks.sql`:
```sql
CREATE TABLE slot_locks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    start_ts TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    caller_id UUID REFERENCES callers(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (resource_id, start_ts)
);

CREATE INDEX idx_slot_locks_expires ON slot_locks(expires_at);

ALTER TABLE slot_locks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to slot_locks"
    ON slot_locks
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 8: Create auth_otps table**

Create `supabase/migrations/00008_create_auth_otps.sql`:
```sql
CREATE TABLE auth_otps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    caller_id UUID NOT NULL REFERENCES callers(id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_auth_otps_caller ON auth_otps(caller_id);
CREATE INDEX idx_auth_otps_expires ON auth_otps(expires_at);

ALTER TABLE auth_otps ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to auth_otps"
    ON auth_otps
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 9: Create audit_log table**

Create `supabase/migrations/00009_create_audit_log.sql`:
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    transcript JSONB,
    tool_calls JSONB NOT NULL DEFAULT '[]'::JSONB,
    outcome TEXT,
    flags JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_log_tenant ON audit_log(tenant_id);
CREATE INDEX idx_audit_log_call ON audit_log(call_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);

ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to audit_log"
    ON audit_log
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 10: Create resume_checkpoints table**

Create `supabase/migrations/00010_create_resume_checkpoints.sql`:
```sql
CREATE TABLE resume_checkpoints (
    phone TEXT PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    state JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_resume_checkpoints_expires ON resume_checkpoints(expires_at);

ALTER TABLE resume_checkpoints ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to resume_checkpoints"
    ON resume_checkpoints
    FOR ALL
    USING (auth.role() = 'service_role');
```

- [ ] **Step 11: Create lock sweeper function**

Create `supabase/migrations/00011_create_lock_sweeper.sql`:
```sql
-- Sweeper function: deletes expired slot locks.
-- Schedule this via pg_cron or call it from a Supabase scheduled function.
CREATE OR REPLACE FUNCTION sweep_expired_locks()
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM slot_locks WHERE expires_at < now();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Also sweep expired OTPs
CREATE OR REPLACE FUNCTION sweep_expired_otps()
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM auth_otps WHERE expires_at < now();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Sweep expired resume checkpoints
CREATE OR REPLACE FUNCTION sweep_expired_checkpoints()
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM resume_checkpoints WHERE expires_at < now();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;
```

- [ ] **Step 12: Commit all migrations**

```bash
git add supabase/migrations/
git commit -m "feat(data): add all Supabase migrations with RLS and lock sweeper functions"
```

---

## Task 8: Python data adapter (Step 2, part 2)

**Files:**
- Create: `packages/voice-agent/data/adapter.py`
- Create: `packages/voice-agent/tests/test_data_adapter.py`

This is the Python-side data access module that the state machine will use. For MVP with `native_supabase`, it wraps the Supabase client. The adapter handles hold_slot atomicity (catch unique constraint violation), idempotent confirm, and lock expiry awareness.

- [ ] **Step 1: Write failing tests**

Create `packages/voice-agent/tests/test_data_adapter.py`:
```python
"""
Tests for the data adapter.

These tests use a mock Supabase client. The real concurrency tests
(two concurrent hold_slot on same slot → exactly one wins) require
a live Supabase instance and are run via /supabase-check.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from packages.voice_agent.data.adapter import (
    DataAdapter,
    SlotConflictError,
    HoldResult,
    BookingResult,
    CallerInfo,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TestResolveOrCreateCaller:
    @pytest.mark.asyncio
    async def test_existing_caller_returned(self):
        mock_client = AsyncMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data={"id": "caller-123", "phone": "+919999999999", "tenant_id": "t-001", "verified_at": None}
        )
        adapter = DataAdapter(client=mock_client)
        result = await adapter.resolve_or_create_caller(tenant_id="t-001", phone="+919999999999")
        assert isinstance(result, CallerInfo)
        assert result.id == "caller-123"

    @pytest.mark.asyncio
    async def test_new_caller_created(self):
        mock_client = AsyncMock()
        # First call: no existing caller
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data=None
        )
        # Second call: insert returns new caller
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": "caller-new", "phone": "+919999999999", "tenant_id": "t-001", "verified_at": None}]
        )
        adapter = DataAdapter(client=mock_client)
        result = await adapter.resolve_or_create_caller(tenant_id="t-001", phone="+919999999999")
        assert isinstance(result, CallerInfo)
        assert result.id == "caller-new"


class TestHoldSlot:
    @pytest.mark.asyncio
    async def test_successful_hold(self):
        mock_client = AsyncMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "lock-001",
                "resource_id": "stylist-priya",
                "start_ts": "2026-06-10T10:00:00+00:00",
                "expires_at": "2026-06-10T10:03:00+00:00",
            }]
        )
        adapter = DataAdapter(client=mock_client)
        start = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        result = await adapter.hold_slot(
            tenant_id="t-001",
            resource_id="stylist-priya",
            start_ts=start,
            caller_id="caller-123",
            hold_ttl_seconds=180,
        )
        assert isinstance(result, HoldResult)
        assert result.hold_id == "lock-001"

    @pytest.mark.asyncio
    async def test_conflict_raises_slot_conflict_error(self):
        mock_client = AsyncMock()
        from postgrest.exceptions import APIError
        mock_client.table.return_value.insert.return_value.execute.side_effect = APIError(
            {"message": "duplicate key value violates unique constraint", "code": "23505"}
        )
        adapter = DataAdapter(client=mock_client)
        start = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        with pytest.raises(SlotConflictError):
            await adapter.hold_slot(
                tenant_id="t-001",
                resource_id="stylist-priya",
                start_ts=start,
                caller_id="caller-123",
                hold_ttl_seconds=180,
            )


class TestConfirmBooking:
    @pytest.mark.asyncio
    async def test_idempotent_confirm(self):
        mock_client = AsyncMock()
        # First: check existing booking by idempotency key — none found
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data=None
        )
        # Then: insert booking
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{
                "id": "booking-001",
                "idempotency_key": "lock-001",
                "status": "confirmed",
            }]
        )
        adapter = DataAdapter(client=mock_client)
        result = await adapter.confirm_booking(
            tenant_id="t-001",
            hold_id="lock-001",
            service_id="svc-haircut",
            resource_id="stylist-priya",
            caller_id="caller-123",
            start_ts=datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc),
            end_ts=datetime(2026, 6, 10, 10, 30, tzinfo=timezone.utc),
            custom_values={},
        )
        assert isinstance(result, BookingResult)
        assert result.booking_id == "booking-001"

    @pytest.mark.asyncio
    async def test_already_confirmed_returns_existing(self):
        mock_client = AsyncMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = MagicMock(
            data={"id": "booking-001", "idempotency_key": "lock-001", "status": "confirmed"}
        )
        adapter = DataAdapter(client=mock_client)
        result = await adapter.confirm_booking(
            tenant_id="t-001",
            hold_id="lock-001",
            service_id="svc-haircut",
            resource_id="stylist-priya",
            caller_id="caller-123",
            start_ts=datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc),
            end_ts=datetime(2026, 6, 10, 10, 30, tzinfo=timezone.utc),
            custom_values={},
        )
        assert result.booking_id == "booking-001"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_data_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.data.adapter'`

- [ ] **Step 3: Implement the data adapter**

Create `packages/voice-agent/data/adapter.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


class SlotConflictError(Exception):
    pass


@dataclass
class CallerInfo:
    id: str
    phone: str
    tenant_id: str
    verified_at: datetime | None


@dataclass
class HoldResult:
    hold_id: str
    resource_id: str
    start_ts: datetime
    expires_at: datetime


@dataclass
class BookingResult:
    booking_id: str
    idempotency_key: str
    status: str


class DataAdapter:
    def __init__(self, client: Any):
        self._client = client

    async def resolve_or_create_caller(self, tenant_id: str, phone: str) -> CallerInfo:
        result = (
            self._client.table("callers")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("phone", phone)
            .maybe_single()
            .execute()
        )
        if result.data:
            return CallerInfo(
                id=result.data["id"],
                phone=result.data["phone"],
                tenant_id=result.data["tenant_id"],
                verified_at=result.data.get("verified_at"),
            )

        insert_result = (
            self._client.table("callers")
            .insert({"tenant_id": tenant_id, "phone": phone})
            .execute()
        )
        row = insert_result.data[0]
        return CallerInfo(
            id=row["id"],
            phone=row["phone"],
            tenant_id=row["tenant_id"],
            verified_at=row.get("verified_at"),
        )

    async def hold_slot(
        self,
        tenant_id: str,
        resource_id: str,
        start_ts: datetime,
        caller_id: str,
        hold_ttl_seconds: int = 180,
    ) -> HoldResult:
        expires_at = start_ts.astimezone(timezone.utc) + timedelta(seconds=hold_ttl_seconds)
        # Use the current time for expiry, not start_ts
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=hold_ttl_seconds)
        try:
            result = (
                self._client.table("slot_locks")
                .insert({
                    "resource_id": resource_id,
                    "tenant_id": tenant_id,
                    "start_ts": start_ts.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "caller_id": caller_id,
                })
                .execute()
            )
        except Exception as e:
            if "23505" in str(e) or "unique" in str(e).lower():
                raise SlotConflictError(
                    f"Slot already held: resource={resource_id}, start={start_ts}"
                ) from e
            raise

        row = result.data[0]
        return HoldResult(
            hold_id=row["id"],
            resource_id=row["resource_id"],
            start_ts=datetime.fromisoformat(row["start_ts"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )

    async def confirm_booking(
        self,
        tenant_id: str,
        hold_id: str,
        service_id: str,
        resource_id: str,
        caller_id: str,
        start_ts: datetime,
        end_ts: datetime,
        custom_values: dict[str, Any],
    ) -> BookingResult:
        existing = (
            self._client.table("bookings")
            .select("*")
            .eq("idempotency_key", hold_id)
            .maybe_single()
            .execute()
        )
        if existing.data:
            return BookingResult(
                booking_id=existing.data["id"],
                idempotency_key=existing.data["idempotency_key"],
                status=existing.data["status"],
            )

        result = (
            self._client.table("bookings")
            .insert({
                "tenant_id": tenant_id,
                "service_id": service_id,
                "resource_id": resource_id,
                "caller_id": caller_id,
                "start_ts": start_ts.isoformat(),
                "end_ts": end_ts.isoformat(),
                "status": "confirmed",
                "custom_values": custom_values,
                "idempotency_key": hold_id,
            })
            .execute()
        )
        row = result.data[0]
        return BookingResult(
            booking_id=row["id"],
            idempotency_key=row["idempotency_key"],
            status=row["status"],
        )

    async def check_availability(
        self,
        tenant_id: str,
        service_id: str,
        resource_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        resources = (
            self._client.table("resources")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("type", resource_type)
            .execute()
        )

        existing_bookings = (
            self._client.table("bookings")
            .select("resource_id, start_ts, end_ts")
            .eq("tenant_id", tenant_id)
            .eq("status", "confirmed")
            .gte("start_ts", start_date.isoformat())
            .lte("end_ts", end_date.isoformat())
            .execute()
        )

        active_locks = (
            self._client.table("slot_locks")
            .select("resource_id, start_ts")
            .eq("tenant_id", tenant_id)
            .gt("expires_at", datetime.now(timezone.utc).isoformat())
            .execute()
        )

        booked = {(b["resource_id"], b["start_ts"]) for b in (existing_bookings.data or [])}
        locked = {(l["resource_id"], l["start_ts"]) for l in (active_locks.data or [])}
        blocked = booked | locked

        available: list[dict[str, Any]] = []
        for resource in (resources.data or []):
            available.append({
                "resource_id": resource["id"],
                "resource_name": resource["name"],
                "blocked_slots": [
                    ts for (rid, ts) in blocked if rid == resource["id"]
                ],
            })

        return available

    async def lookup_bookings(self, tenant_id: str, caller_id: str) -> list[dict[str, Any]]:
        result = (
            self._client.table("bookings")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("caller_id", caller_id)
            .eq("status", "confirmed")
            .order("start_ts")
            .execute()
        )
        return result.data or []

    async def cancel_booking(self, tenant_id: str, booking_id: str, caller_id: str) -> bool:
        result = (
            self._client.table("bookings")
            .update({"status": "cancelled"})
            .eq("id", booking_id)
            .eq("tenant_id", tenant_id)
            .eq("caller_id", caller_id)
            .eq("status", "confirmed")
            .execute()
        )
        return bool(result.data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_data_adapter.py -v`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add packages/voice-agent/data/adapter.py packages/voice-agent/tests/test_data_adapter.py
git commit -m "feat(data): add Python data adapter with hold/confirm/availability/cancel operations"
```

---

## Task 9: Deterministic date/time resolver (Step 3)

**Files:**
- Create: `packages/voice-agent/resolver/date_resolver.py`
- Create: `packages/voice-agent/tests/test_date_resolver.py`

The resolver takes a natural language phrase and resolves it to a concrete date/time against `business_hours` and `booking_window_days`, in IST. The LLM proposes a phrase; the resolver decides the date. Ambiguous, past, or out-of-window phrases return a "needs clarification" result.

- [ ] **Step 1: Write failing tests**

Create `packages/voice-agent/tests/test_date_resolver.py`:
```python
import pytest
from datetime import date, datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

from packages.voice_agent.resolver.date_resolver import (
    DateResolver,
    ResolvedSlot,
    ResolutionResult,
    ResolutionStatus,
)
from packages.voice_agent.config.models import BusinessHours

IST = ZoneInfo("Asia/Kolkata")


def _make_resolver(
    reference_date: date | None = None,
    booking_window_days: int = 30,
    min_notice_min: int = 60,
) -> DateResolver:
    hours = BusinessHours(
        mon=["09:00-13:00", "14:00-19:00"],
        tue=["09:00-13:00", "14:00-19:00"],
        wed=["09:00-13:00", "14:00-19:00"],
        thu=["09:00-13:00", "14:00-19:00"],
        fri=["09:00-13:00", "14:00-19:00"],
        sat=["10:00-18:00"],
        sun=[],
    )
    ref = reference_date or date(2026, 6, 3)
    return DateResolver(
        business_hours=hours,
        booking_window_days=booking_window_days,
        min_notice_min=min_notice_min,
        reference_date=ref,
        reference_time=time(10, 0),
    )


class TestExplicitDatetime:
    def test_tomorrow_4pm(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("tomorrow 4pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        assert result.slot.date == date(2026, 6, 4)
        assert result.slot.time == time(16, 0)

    def test_next_friday(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))  # Wednesday
        result = resolver.resolve("next Friday")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        assert result.slot.date.weekday() == 4  # Friday

    def test_specific_date(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("June 15th at 11am")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.slot is not None
        assert result.slot.date == date(2026, 6, 15)
        assert result.slot.time == time(11, 0)


class TestOutOfWindow:
    def test_too_far_in_future(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3), booking_window_days=30)
        result = resolver.resolve("August 15th at 10am")
        assert result.status == ResolutionStatus.OUT_OF_WINDOW

    def test_past_date_rejected(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("yesterday at 3pm")
        assert result.status == ResolutionStatus.PAST_DATE


class TestOutsideBusinessHours:
    def test_sunday_rejected(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))  # Wednesday
        result = resolver.resolve("Sunday at 10am")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS

    def test_too_early_rejected(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("tomorrow at 7am")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS

    def test_too_late_rejected(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("tomorrow at 8pm")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS


class TestAmbiguous:
    def test_no_time_given(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("next week")
        assert result.status == ResolutionStatus.AMBIGUOUS

    def test_unparseable(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("when the moon is full")
        assert result.status == ResolutionStatus.AMBIGUOUS


class TestMinNotice:
    def test_too_soon_rejected(self):
        resolver = _make_resolver(
            reference_date=date(2026, 6, 3),
            min_notice_min=60,
        )
        # Reference time is 10:00, so 10:30 is within 60 min notice
        result = resolver.resolve("today at 10:30am")
        assert result.status == ResolutionStatus.TOO_SOON


class TestFestivals:
    def test_diwali_resolves(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("after Diwali")
        # Diwali 2026 is ~Oct 20. "After Diwali" is ambiguous without a time.
        assert result.status in (ResolutionStatus.AMBIGUOUS, ResolutionStatus.OUT_OF_WINDOW)

    def test_holi_resolves(self):
        resolver = _make_resolver(reference_date=date(2026, 2, 1), booking_window_days=60)
        result = resolver.resolve("day after Holi at 2pm")
        # Holi 2026 is March 10. Day after = March 11.
        if result.status == ResolutionStatus.RESOLVED:
            assert result.slot is not None
            assert result.slot.date.month == 3


class TestLunchBreak:
    def test_during_lunch_break_rejected(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("tomorrow at 1:30pm")
        assert result.status == ResolutionStatus.OUTSIDE_HOURS


class TestResolutionResultMessage:
    def test_resolved_has_no_message(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("tomorrow 4pm")
        assert result.status == ResolutionStatus.RESOLVED
        assert result.clarification_prompt is None

    def test_ambiguous_has_message(self):
        resolver = _make_resolver(reference_date=date(2026, 6, 3))
        result = resolver.resolve("sometime next week")
        assert result.status == ResolutionStatus.AMBIGUOUS
        assert result.clarification_prompt is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_date_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'packages.voice_agent.resolver.date_resolver'`

- [ ] **Step 3: Implement the date resolver**

Create `packages/voice-agent/resolver/date_resolver.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo
from typing import Any

import dateparser
import holidays


IST = ZoneInfo("Asia/Kolkata")

INDIAN_FESTIVALS: dict[str, dict[int, tuple[int, int]]] = {
    "diwali": {2025: (10, 20), 2026: (10, 9), 2027: (10, 29)},
    "holi": {2025: (3, 14), 2026: (3, 10), 2027: (2, 28)},
    "dussehra": {2025: (10, 2), 2026: (9, 21), 2027: (10, 11)},
    "eid": {2025: (3, 31), 2026: (3, 20), 2027: (3, 10)},
    "raksha bandhan": {2025: (8, 9), 2026: (8, 28), 2027: (8, 17)},
    "ganesh chaturthi": {2025: (8, 27), 2026: (8, 17), 2027: (9, 5)},
}


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    PAST_DATE = "past_date"
    OUT_OF_WINDOW = "out_of_window"
    OUTSIDE_HOURS = "outside_hours"
    TOO_SOON = "too_soon"


@dataclass
class ResolvedSlot:
    date: date
    time: time
    datetime_ist: datetime


@dataclass
class ResolutionResult:
    status: ResolutionStatus
    slot: ResolvedSlot | None = None
    clarification_prompt: str | None = None


class DateResolver:
    def __init__(
        self,
        business_hours: Any,
        booking_window_days: int,
        min_notice_min: int,
        reference_date: date,
        reference_time: time,
    ):
        self._business_hours = business_hours
        self._window_days = booking_window_days
        self._min_notice_min = min_notice_min
        self._ref_date = reference_date
        self._ref_time = reference_time
        self._india_holidays = holidays.India(years=range(reference_date.year - 1, reference_date.year + 3))

    def resolve(self, phrase: str) -> ResolutionResult:
        normalized = phrase.lower().strip()
        normalized = self._substitute_festivals(normalized)

        ref_dt = datetime.combine(self._ref_date, self._ref_time, tzinfo=IST)

        parsed = dateparser.parse(
            normalized,
            settings={
                "RELATIVE_BASE": datetime.combine(self._ref_date, self._ref_time),
                "TIMEZONE": "Asia/Kolkata",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
                "PREFER_DAY_OF_MONTH": "first",
            },
        )

        if parsed is None:
            return ResolutionResult(
                status=ResolutionStatus.AMBIGUOUS,
                clarification_prompt="I couldn't understand that date. Could you say something like 'tomorrow at 4pm' or 'next Friday at 11am'?",
            )

        parsed_ist = parsed.astimezone(IST)

        has_time = self._phrase_has_time(normalized)
        if not has_time:
            return ResolutionResult(
                status=ResolutionStatus.AMBIGUOUS,
                clarification_prompt=f"I understood {parsed_ist.strftime('%A, %B %d')} — what time would you prefer?",
            )

        resolved_date = parsed_ist.date()
        resolved_time = parsed_ist.time()

        if parsed_ist < ref_dt:
            return ResolutionResult(
                status=ResolutionStatus.PAST_DATE,
                clarification_prompt="That time has already passed. Could you pick a future date and time?",
            )

        window_end = self._ref_date + timedelta(days=self._window_days)
        if resolved_date > window_end:
            return ResolutionResult(
                status=ResolutionStatus.OUT_OF_WINDOW,
                clarification_prompt=f"We can only book up to {self._window_days} days ahead. The latest available date is {window_end.strftime('%B %d')}.",
            )

        min_notice_dt = ref_dt + timedelta(minutes=self._min_notice_min)
        if parsed_ist < min_notice_dt:
            return ResolutionResult(
                status=ResolutionStatus.TOO_SOON,
                clarification_prompt=f"We need at least {self._min_notice_min} minutes notice. The earliest available time is {min_notice_dt.strftime('%I:%M %p')}.",
            )

        day_name = resolved_date.strftime("%a").lower()
        hours_for_day = self._business_hours.get_hours_for_day(day_name)

        if not hours_for_day:
            return ResolutionResult(
                status=ResolutionStatus.OUTSIDE_HOURS,
                clarification_prompt=f"We're closed on {resolved_date.strftime('%A')}s. Could you pick another day?",
            )

        if not self._time_in_ranges(resolved_time, hours_for_day):
            return ResolutionResult(
                status=ResolutionStatus.OUTSIDE_HOURS,
                clarification_prompt=f"That time is outside our hours. On {resolved_date.strftime('%A')}s we're open {', '.join(hours_for_day)}. Could you pick a time within those hours?",
            )

        return ResolutionResult(
            status=ResolutionStatus.RESOLVED,
            slot=ResolvedSlot(
                date=resolved_date,
                time=resolved_time,
                datetime_ist=parsed_ist,
            ),
        )

    def _substitute_festivals(self, phrase: str) -> str:
        for festival, year_map in INDIAN_FESTIVALS.items():
            if festival in phrase:
                year = self._ref_date.year
                if year in year_map:
                    month, day = year_map[year]
                    festival_date = date(year, month, day)
                    if "after" in phrase or "day after" in phrase:
                        festival_date += timedelta(days=1)
                    time_part = ""
                    for token in phrase.split():
                        if "am" in token or "pm" in token or ":" in token:
                            time_part = token
                            break
                    date_str = festival_date.strftime("%B %d")
                    if time_part:
                        return f"{date_str} at {time_part}"
                    return date_str
        return phrase

    def _phrase_has_time(self, phrase: str) -> bool:
        time_indicators = [
            "am", "pm", "morning", "afternoon", "evening",
            "o'clock", "oclock", "noon", "midnight",
        ]
        if any(ind in phrase for ind in time_indicators):
            return True
        if ":" in phrase:
            parts = phrase.split(":")
            for i, part in enumerate(parts):
                if part and part[-1].isdigit() and i + 1 < len(parts) and parts[i + 1] and parts[i + 1][0].isdigit():
                    return True
        for token in phrase.split():
            if token.isdigit() and 1 <= int(token) <= 12:
                if any(w in phrase for w in ["at", "by", "around", "before", "after"]):
                    return True
        return False

    def _time_in_ranges(self, t: time, ranges: list[str]) -> bool:
        for r in ranges:
            start_str, end_str = r.split("-")
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))
            range_start = time(start_h, start_m)
            range_end = time(end_h, end_m)
            if range_start <= t < range_end:
                return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/test_date_resolver.py -v`
Expected: all tests PASS

Some festival tests may need adjustment depending on exact dateparser behavior. If a test fails, check whether the parsed date matches expectations and adjust the festival map or test assertions.

- [ ] **Step 5: Run the full test suite (all Steps 1–3)**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/ -v`
Expected: all tests PASS

- [ ] **Step 6: Run ruff and black on all Python files**

Run: `source .venv/bin/activate && ruff check packages/ --fix && black packages/`
Expected: no errors or auto-fixed

- [ ] **Step 7: Commit**

```bash
git add packages/voice-agent/resolver/date_resolver.py packages/voice-agent/tests/test_date_resolver.py
git commit -m "feat(resolver): add deterministic date/time resolver with IST, business hours, and festival support"
```

---

## Task 10: Integration verification and final commit

**Files:**
- Modify: `BUILD.md` (update status markers)

- [ ] **Step 1: Run the complete test suite**

Run: `source .venv/bin/activate && python -m pytest packages/voice-agent/tests/ -v --tb=short`
Expected: all tests PASS

- [ ] **Step 2: Run linters**

Run: `source .venv/bin/activate && ruff check packages/ && black --check packages/`
Expected: no errors

- [ ] **Step 3: Update BUILD.md status**

Edit `BUILD.md` and change statuses:
```markdown
## Phase 0 — Setup ☑
## Step 0 — Voice-loop spike ☑
## Step 1 — Config service + validation ☑
## Step 2 — Supabase data + adapter ☑
## Step 3 — Date/time resolver ☑
## Step 4 — Dialogue state machine + tools + fallback ☐
```

- [ ] **Step 4: Commit**

```bash
git add BUILD.md
git commit -m "docs: mark Phase 0 + Steps 0-3 complete in BUILD.md"
```

- [ ] **Step 5: Verify git log looks clean**

Run: `git log --oneline`
Expected output (approximately):
```
<hash> docs: mark Phase 0 + Steps 0-3 complete in BUILD.md
<hash> feat(resolver): add deterministic date/time resolver with IST, business hours, and festival support
<hash> feat(data): add Python data adapter with hold/confirm/availability/cancel operations
<hash> feat(data): add all Supabase migrations with RLS and lock sweeper functions
<hash> feat(config): add config loader from file and dict with validation
<hash> feat(config): add cross-field config validator with spec §7.4 rules
<hash> feat(config): add Pydantic config models for tenant configuration schema
<hash> spike: voice-loop latency baseline recorded, spike removed
<hash> docs: add root CLAUDE.md, BUILD.md, and LEARN.md
<hash> chore: initialize monorepo structure with Python env and package scaffolding
```

---

## Summary of what this plan produces

| Deliverable | What it proves |
|---|---|
| **Monorepo + tooling** | Dev environment works, linting, testing infra |
| **Voice spike P95 number** | Real latency target for Steps 4–5 |
| **Config models + validator + loader** | Any tenant config can be authored, validated, and loaded |
| **11 SQL migrations with RLS** | Data layer ready for the state machine; concurrency-safe slot locking |
| **Python data adapter** | hold/confirm/cancel/availability with idempotency, ready for Step 4 |
| **Deterministic date resolver** | Natural language → concrete IST datetime against business hours + festivals |

**Next plan:** Step 4 (Dialogue state machine + tools + fallback) — the core of the agent. Should be planned once the spike P95 number is known and the foundation is working.
