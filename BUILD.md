# BUILD.md — Implementation Playbook

Living build plan. Work top-to-bottom; update status as you go. Full rationale in the spec. Check `LEARN.md` before each step.

**Status legend:** ☐ todo · ◐ in-progress · ☑ done
**MVP cut:** salon · `native_supabase` · `exclusive` booking only · ANI soft-auth (no OTP) · Hindi+English.

---

## Phase 0 — Setup ☑
## Step 0 — Voice-loop spike ☐ (manual — requires real API keys + phone call)
## Step 1 — Config service + validation ☑
## Step 2 — Supabase data + adapter ☑
## Step 3 — Date/time resolver ☑
## Step 4 — Dialogue state machine + tools + fallback ☑
## Step 5 — Voice pipeline integration ☑
## Step 6 — Auth + guardrails ☐
## Step 7 — Regression harness ☐
## Step 8 — Salon go-live ☐

## Parallelism
Step 0 ∥ Step 1. Steps 2 ∥ 3 after 1. 4 needs 2+3. 6 overlays 4. 7 seeds early. 8 needs 1–7.
