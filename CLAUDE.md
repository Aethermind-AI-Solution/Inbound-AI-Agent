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
Pipecat (Python 3.11+, async) voice service · Supabase (Postgres + RLS + Edge Functions/Deno-TS) · Redis (resume checkpoints) · Twilio (telephony) · Deepgram/Sarvam (STT) · Claude Haiku 4.5 + prompt caching (NLU) · Smallest/Sarvam (TTS) · N8N (async post-call) · Next.js/Vercel (admin) · Coval (regression) · Langfuse/OTel (observability).

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
