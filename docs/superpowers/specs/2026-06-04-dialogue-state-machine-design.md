# Dialogue State Machine — Design Spec

**Goal:** Build the core conversation engine that drives inbound voice calls through booking, cancellation, rescheduling, and status-check flows using a class-per-state architecture.

**Architecture:** Class-per-state FSM. Each state is a subclass of `BaseState` with `enter()` and `handle()` methods. A `DialogueManager` orchestrates transitions, enforces guardrails, writes checkpoints, and handles errors. The LLM is accessed through a stubbed `NLUService` interface. The state machine is pipeline-agnostic — a thin `PipelineAdapter` bridges it to Pipecat.

**Tech stack:** Python 3.11+ / async / Pydantic v2. Consumes existing `TenantConfig`, `DataAdapter`, `DateResolver`. Stubs for NLU and checkpoint store.

---

## 1. Scope

### In scope
- All four intents: new booking, cancel, reschedule, status check.
- 15 states (see §3).
- Soft auth (caller number match via `resolve_or_create_caller`).
- Fallback: normal flow (T0) + callback capture (T5) as the guaranteed floor.
- Guardrails: deterministic enforcement of `max_call_seconds`, `max_turns`.
- Checkpoint interface with in-memory implementation (write on every state transition).
- NLU interface with deterministic stubs for testing.
- Pipeline adapter (Pipecat `FrameProcessor`) as integration layer.

### Out of scope (future work)
- OTP/DTMF auth flow.
- Fallback tiers T1–T4 (DTMF fallback, simplified capture, human transfer).
- Actual Claude Haiku 4.5 integration (NLU stubs only).
- Redis-backed checkpoint store.
- Static audio caching / pre-synthesis.
- Language detection and switching.
- Monthly budget enforcement (needs billing integration).
- Waitlist flow.

---

## 2. Architecture

### 2.1 Component diagram

```
┌─────────────────────────────────────────────────────────┐
│  Pipecat Pipeline                                       │
│                                                         │
│  mic/twilio → STT → PipelineAdapter → TTS → speaker    │
│                          │                              │
└──────────────────────────┼──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Dialogue   │
                    │  Manager    │
                    ├─────────────┤
                    │ • states{}  │
                    │ • context   │
                    │ • guardrails│
                    └──┬───┬───┬──┘
                       │   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌────────────┐
        │ NLU      │ │ Data     │ │ Checkpoint │
        │ Service  │ │ Adapter  │ │ Store      │
        │ (stub)   │ │ (supabase│ │ (in-memory)│
        └──────────┘ └──────────┘ └────────────┘
                           │
                     ┌─────▼─────┐
                     │ Date      │
                     │ Resolver  │
                     └───────────┘
```

### 2.2 Separation of concerns

| Layer | Responsibility | Touches Pipecat? |
|-------|---------------|------------------|
| `PipelineAdapter` | Translates `TranscriptionFrame` → `CallEvent`, `Action` → `TextFrame`. Thin glue only. | Yes |
| `DialogueManager` | Orchestrates state transitions, enforces guardrails, writes checkpoints, handles tool errors. | No |
| `BaseState` subclasses | Individual state logic: what to say, how to handle input, where to go next. | No |
| `NLUService` | Intent classification, entity extraction. Stubbed for now. | No |
| `CheckpointStore` | Save/load/delete call state. In-memory for now, Redis later. | No |

### 2.3 Design principle: the manager loops

When a state transitions to another state, the manager calls the new state's `enter()`. If `enter()` itself returns an immediate transition (e.g., `IDENTIFY_CALLER` does a lookup and transitions to `INTENT` without user input), the manager keeps looping until it reaches a state that returns `Ask` (needs user input) or `EndCall`. This prevents silent states from stalling the pipeline.

```
handle_event(event):
    result = current_state.handle(event, context)
    while result.next_state is not None:
        transition to result.next_state
        write checkpoint (if result.checkpoint)
        check guardrails → override next_state if cap exceeded
        action = new_state.enter(context)
        if action is Ask or EndCall:
            return action  # stop looping, wait for input
        result = action    # keep looping (silent state)
    return result.action   # stay in current state
```

---

## 3. States

### 3.1 State enum

```python
class CallState(str, Enum):
    GREETING = "greeting"
    IDENTIFY_CALLER = "identify_caller"
    INTENT = "intent"
    COLLECT_SERVICE = "collect_service"
    COLLECT_DATETIME = "collect_datetime"
    OFFER_SLOTS = "offer_slots"
    COLLECT_CUSTOM = "collect_custom"
    READ_BACK = "read_back"
    CONFIRM = "confirm"
    LOOKUP_BOOKINGS = "lookup_bookings"
    SELECT_BOOKING = "select_booking"
    READ_STATUS = "read_status"
    CONFIRM_CANCEL = "confirm_cancel"
    CALLBACK_CAPTURE = "callback_capture"
    CLOSE = "close"
```

### 3.2 Transition map

```
GREETING ──────────────────► IDENTIFY_CALLER (auto)
IDENTIFY_CALLER ───────────► INTENT (auto, after phone lookup)

INTENT ─── new_booking ────► COLLECT_SERVICE
       ├── cancel ─────────► LOOKUP_BOOKINGS
       ├── reschedule ─────► LOOKUP_BOOKINGS
       ├── status ─────────► LOOKUP_BOOKINGS
       └── unrecognized ×3 ► CALLBACK_CAPTURE

COLLECT_SERVICE ───────────► COLLECT_DATETIME (service matched)
                └── fail ×2 ► CALLBACK_CAPTURE

COLLECT_DATETIME ──────────► OFFER_SLOTS (date resolved)
                 ├── ambiguous ► re-ask with clarification
                 ├── invalid ──► explain + re-ask
                 └── fail ×3 ──► CALLBACK_CAPTURE

OFFER_SLOTS ───────────────► COLLECT_CUSTOM (if custom fields)
            ├───────────────► READ_BACK (no custom fields)
            ├── conflict ───► re-offer remaining
            ├── no availability ► suggest alternatives or CALLBACK_CAPTURE
            └── fail ×2 ────► CALLBACK_CAPTURE

COLLECT_CUSTOM ────────────► READ_BACK (all fields collected)

READ_BACK ─── yes ─────────► CONFIRM
          ├── no (change) ─► route back to relevant COLLECT_* state
          └── unclear ×2 ──► CALLBACK_CAPTURE

CONFIRM ───────────────────► CLOSE (auto, after booking confirmed)
        ├── hold expired ──► OFFER_SLOTS (re-try)
        └── tool error ────► CALLBACK_CAPTURE

LOOKUP_BOOKINGS ── found ──► SELECT_BOOKING (cancel/reschedule, multiple)
                ├── found ─► auto-select (cancel/reschedule, single)
                ├── found ─► READ_STATUS (status intent)
                └── none ──► CLOSE (with message)

SELECT_BOOKING ── cancel ──► CONFIRM_CANCEL
               └── reschedule ► COLLECT_DATETIME

READ_STATUS ── "anything else?" ── yes ► INTENT
            └── no ────────────────────► CLOSE

CONFIRM_CANCEL ── yes ─────► CLOSE (after cancel)
               └── no ─────► CLOSE

CALLBACK_CAPTURE ──────────► CLOSE (after phone confirmed)

CLOSE ─────────────────────► (end call, delete checkpoint)
```

### 3.3 Per-state detail

#### GREETING
- **enter():** Speak `"{persona.greeting}. {persona.ai_disclosure}"`. Return immediate transition to `IDENTIFY_CALLER`.
- **handle():** Not called (auto-transition from enter).
- **Checkpoint:** No (first state, nothing to resume).

#### IDENTIFY_CALLER
- **enter():** Call `data_adapter.resolve_or_create_caller(tenant_id, caller_phone)`. Store `CallerInfo` in `context.caller`. Silent — no speech. Return immediate transition to `INTENT`.
- **handle():** Not called.
- **Error:** If data adapter fails → `CALLBACK_CAPTURE`.

#### INTENT
- **enter():** Ask "How can I help you today? I can help with booking an appointment, checking an existing booking, rescheduling, or cancelling."
- **handle():** Pass `event.text` to `nlu.classify_intent()`. On result:
  - `new_booking` → set `context.intent`, transition to `COLLECT_SERVICE`.
  - `cancel` → set `context.intent`, transition to `LOOKUP_BOOKINGS`.
  - `reschedule` → set `context.intent`, transition to `LOOKUP_BOOKINGS`.
  - `status` → set `context.intent`, transition to `LOOKUP_BOOKINGS`.
  - `unknown` → increment reprompt counter. If < 3, reprompt with simpler phrasing. If >= 3, → `CALLBACK_CAPTURE`.
- **Reprompt text:** "I didn't quite catch that. Would you like to book a new appointment, or check on an existing one?"

#### COLLECT_SERVICE
- **enter():** Build service list from `tenant_config.booking_model.services`. Ask: "What service would you like? We offer {service_names}."
- **handle():** Pass `event.text` + service list to `nlu.extract_service()`. On result:
  - Matched → store `context.slots["service_id"]`, transition to `COLLECT_DATETIME`.
  - Ambiguous (multiple matches) → "Did you mean {option_a} or {option_b}?" Stay in state.
  - No match → reprompt. If reprompt count >= 2 → `CALLBACK_CAPTURE`.
- **Reprompt text:** "I'm not sure which service you mean. Could you pick from: {service_names}?"

#### COLLECT_DATETIME
- **enter():** Ask "When would you like to come in? You can say something like 'tomorrow at 3 PM' or 'next Saturday morning'."
- **handle():** Pass `event.text` to `date_resolver.resolve()`. Based on `status`:
  - `RESOLVED` → store `context.slots["datetime_ist"]`, transition to `OFFER_SLOTS`.
  - `AMBIGUOUS` → speak `result.clarification_prompt`. Stay in state.
  - `PAST_DATE` → "That date has already passed. Could you pick a future date?"
  - `OUT_OF_WINDOW` → "We can only book up to {booking_window_days} days ahead. Could you pick a closer date?"
  - `TOO_SOON` → "We need at least {min_notice_min} minutes notice. Could you pick a later time?"
  - `OUTSIDE_HOURS` → "We're not open at that time. Our hours are {hours_for_that_day}. What time works for you?"
  - Reprompt count >= 3 → `CALLBACK_CAPTURE`.

#### OFFER_SLOTS
- **enter():** Call `data_adapter.check_availability(tenant_id, service_id, resource_type, date, date)`.
  - If slots available: present up to 3 options. "I have {resource_name} available at {time_1}, {time_2}, or {time_3}. Which works for you?"
  - If no availability on requested date: "Unfortunately there's nothing available on {date}. Would you like to try {next_available_date}?" Stay in state for response.
  - If no availability at all in booking window → `CALLBACK_CAPTURE` with "no availability" reason.
- **handle():** User picks a slot. Call `data_adapter.hold_slot(tenant_id, resource_id, start_ts, caller_id)`.
  - Success → store `context.slots["hold_id"]` and `context.slots["resource_id"]`.
    - If service has `custom_fields` → `COLLECT_CUSTOM`.
    - Else → `READ_BACK`.
  - `SlotConflictError` → "That slot was just taken. {remaining options}." Stay in state.
  - All options exhausted → suggest next available date or `CALLBACK_CAPTURE`.

#### COLLECT_CUSTOM
- **enter():** Load custom fields from `tenant_config.booking_model.services[service_id].custom_fields`. Track which fields are collected in `context.slots["custom_fields"]`. Ask the first uncollected field's `prompt`.
- **handle():** Store the answer in `context.slots["custom_fields"][field.key]`.
  - More fields remaining → ask next field's prompt. Stay in state.
  - All fields collected → transition to `READ_BACK`.
- **Validation:** For `int`/`float`/`bool` typed fields, validate the response. On invalid → reprompt with "I need a {type} value for {field_name}." After 2 failures per field → skip the field (mark as None if not required, or → `CALLBACK_CAPTURE` if required).

#### READ_BACK (mandatory — architecture invariant #2)
- **enter():** Build confirmation summary from context slots:
  ```
  "Let me confirm your booking:
   {service_name} with {resource_name}
   on {date} at {time}.
   {custom_field_1}: {value_1}.
   Shall I go ahead and confirm this?"
  ```
  For reschedule, include: "This will replace your existing booking on {old_date}."
- **handle():**
  - Affirmative (yes/confirm/go ahead) → `CONFIRM`.
  - Negative (no/change) → ask "What would you like to change — the service, the time, or something else?"
    - Service → clear service/datetime/hold slots, → `COLLECT_SERVICE`.
    - Time/date → clear datetime/hold slots, → `COLLECT_DATETIME`.
    - Resource → clear hold slot, → `OFFER_SLOTS`.
  - Unclear response × 2 → `CALLBACK_CAPTURE`.
- **Important:** The hold is still active during read-back. If the user changes something, the old hold expires naturally (TTL-based) and a new hold is created.

#### CONFIRM
- **enter():** Async operation with error handling:
  1. If `context.intent == "reschedule"`: call `data_adapter.cancel_booking(old_booking_id, caller_id)` first.
  2. Call `data_adapter.confirm_booking(tenant_id, hold_id, service_id, resource_id, caller_id, start_ts, end_ts, custom_values)`.
  3. On success → speak "Your booking is confirmed. Your reference number is {booking_id}." → transition to `CLOSE`.
  4. On failure (hold expired) → speak "I'm sorry, that slot is no longer available. Let me find another option." → transition to `OFFER_SLOTS` (clear hold_id from slots).
  5. On failure (other error) → retry once. If still failing → `CALLBACK_CAPTURE`.
- **handle():** Not called (auto-transition from enter).
- **Idempotency:** `hold_id` is the idempotency key. If `confirm_booking` is retried with the same `hold_id`, it returns the existing booking (no double-book).

#### LOOKUP_BOOKINGS
- **enter():** Call `data_adapter.lookup_bookings(tenant_id, caller_id)`.
  - No bookings found → speak "I don't see any bookings for your number." → `CLOSE`.
  - Bookings found:
    - `intent == "status"` → transition to `READ_STATUS` (with bookings in context).
    - `intent == "cancel"` or `"reschedule"`:
      - Single booking → auto-select, store `booking_id` in slots.
        - `cancel` → `CONFIRM_CANCEL`.
        - `reschedule` → `COLLECT_DATETIME`.
      - Multiple → transition to `SELECT_BOOKING`.
- **handle():** Not called (auto-transition from enter).
- **Error:** Data adapter fails → `CALLBACK_CAPTURE`.

#### SELECT_BOOKING
- **enter():** List bookings: "I found {n} bookings: 1) {service} on {date}, 2) {service} on {date}. Which one?"
- **handle():** Match user's selection to a booking. Store `context.slots["booking_id"]`.
  - `intent == "cancel"` → `CONFIRM_CANCEL`.
  - `intent == "reschedule"` → `COLLECT_DATETIME`.
  - No match → reprompt. After 2 failures → `CALLBACK_CAPTURE`.

#### READ_STATUS
- **enter():** Read out booking details for all bookings:
  ```
  "Here are your bookings:
   1. {service} with {resource} on {date} at {time}. Status: {status}.
   Is there anything else I can help with?"
  ```
- **handle():**
  - Yes / new request → `INTENT`.
  - No → `CLOSE`.
  - Unclear × 2 → `CLOSE` (safe default — status is read-only).

#### CONFIRM_CANCEL
- **enter():** "You'd like to cancel your {service} appointment on {date} at {time}. Are you sure?"
- **handle():**
  - Yes → call `data_adapter.cancel_booking(tenant_id, booking_id, caller_id)`.
    - Success → "Your booking has been cancelled." → `CLOSE`.
    - Failure → retry once, then `CALLBACK_CAPTURE`.
  - No → "Okay, your booking is still active." → `CLOSE`.

#### CALLBACK_CAPTURE (the floor — architecture invariant #4)
- **enter():** Speak based on `context.fallback_reason`:
  - Default: "I'm having some difficulty. Let me have someone call you back."
  - No availability: "We don't have any openings right now, but I'll have someone call you to help find a time."
  - Tool error: "I'm experiencing a technical issue. Let me have someone follow up with you."
  - Then: "Can I confirm your callback number is {caller_phone}?"
- **handle():**
  - Affirmative → log callback request (data adapter or audit log). → `CLOSE`.
  - Different number → update callback number. → `CLOSE`.
  - After 2 unclear responses → assume the number on file is correct, log callback, → `CLOSE`.
- **Logging:** Write to `audit_log` table: `{tenant_id, caller_id, phone, fallback_reason, state_at_fallback, slots_collected, timestamp}`.

#### CLOSE
- **enter():** Speak closing: "Thank you for calling {business_name}. Have a great day!" Call `checkpoint_store.delete(phone)`. Return `EndCall`.
- **handle():** Not called.

---

## 4. Event Model

```python
class EventType(str, Enum):
    TRANSCRIPTION = "transcription"   # user said something
    SILENCE = "silence"               # no speech detected for timeout period
    HANGUP = "hangup"                 # caller disconnected
    TIMEOUT = "timeout"              # state-level timeout (no response in N seconds)
    ERROR = "error"                   # system error propagated as event

@dataclass
class CallEvent:
    type: EventType
    text: str | None = None           # transcription text (only for TRANSCRIPTION)
    confidence: float | None = None   # ASR confidence 0.0–1.0
    metadata: dict = field(default_factory=dict)
```

### 4.1 Global event handling (DialogueManager level)

These events are handled by the manager before dispatching to the current state:

| Event | Behavior |
|-------|----------|
| `HANGUP` | Stop processing. Checkpoint is already written on last transition. Call ends. |
| `SILENCE` × 2 consecutive | → `CALLBACK_CAPTURE` (from any state). |
| `TIMEOUT` | Equivalent to silence. Treated as one silence count. |
| `ERROR` | Log error. → `CALLBACK_CAPTURE`. |

Only `TRANSCRIPTION` events are dispatched to the current state's `handle()` method.

---

## 5. CallContext

```python
@dataclass
class CallContext:
    tenant_config: TenantConfig
    caller_phone: str
    call_id: str                          # unique per call

    # Populated during the call
    caller: CallerInfo | None = None
    intent: str | None = None             # "new_booking" | "cancel" | "reschedule" | "status"
    language: str = "en-IN"               # detected/matched language

    # Progressive slot filling
    slots: BookingSlots = field(default_factory=BookingSlots)

    # Counters
    turn_count: int = 0
    silence_count: int = 0
    call_start: float = field(default_factory=time.monotonic)

    # Fallback tracking
    fallback_reason: str | None = None

@dataclass
class BookingSlots:
    service_id: str | None = None
    service_name: str | None = None
    resource_id: str | None = None
    resource_name: str | None = None
    datetime_ist: datetime | None = None
    hold_id: str | None = None
    hold_expires_at: datetime | None = None
    booking_id: str | None = None         # for cancel/reschedule (old booking)
    custom_fields: dict = field(default_factory=dict)

    # Reschedule: remember old booking details for read-back
    old_booking_date: str | None = None
    old_booking_service: str | None = None

    def clear_datetime(self) -> None:
        """Clear date/time and downstream slots when user changes date."""
        self.datetime_ist = None
        self.hold_id = None
        self.hold_expires_at = None
        self.resource_id = None
        self.resource_name = None

    def clear_service(self) -> None:
        """Clear service and all downstream slots when user changes service."""
        self.service_id = None
        self.service_name = None
        self.clear_datetime()
        self.custom_fields = {}
```

---

## 6. Actions

```python
class ActionType(str, Enum):
    SPEAK = "speak"
    ASK = "ask"
    END_CALL = "end_call"
    TRANSITION = "transition"   # silent transition, no speech

@dataclass
class Action:
    type: ActionType
    text: str | None = None               # what to say (SPEAK/ASK)
    next_state: str | None = None         # state to transition to
    timeout_s: float = 10.0              # how long to wait for response (ASK)
    checkpoint: bool = True               # write checkpoint on this transition
```

### 6.1 Action semantics

| Action | Pipeline behavior |
|--------|-------------------|
| `SPEAK` + `next_state` | Say the text, then immediately transition (no wait for input). |
| `ASK` + no `next_state` | Say the text, wait for user input, deliver to current state's `handle()`. |
| `ASK` + `next_state` | Say the text, wait for input, deliver to the NEXT state's `handle()`. (Rare — used when a prompt belongs to the next state.) |
| `TRANSITION` + `next_state` | Silent transition. No speech. Manager calls next state's `enter()`. |
| `END_CALL` | Say the text (if any), then hang up. |

---

## 7. DialogueManager

```python
class DialogueManager:
    def __init__(
        self,
        config: TenantConfig,
        data_adapter: DataAdapter,
        nlu: NLUService,
        checkpoint_store: CheckpointStore,
        caller_phone: str,
        call_id: str,
    ):
        self.context = CallContext(
            tenant_config=config,
            caller_phone=caller_phone,
            call_id=call_id,
        )
        self.data_adapter = data_adapter
        self.nlu = nlu
        self.checkpoint_store = checkpoint_store
        self.states: dict[str, BaseState] = {}  # populated by register_states()
        self.current_state: BaseState | None = None

    async def start(self) -> Action:
        """Begin a new call. Enter GREETING state."""
        self._register_states()
        return await self._enter_state(CallState.GREETING)

    async def resume(self, checkpoint: dict) -> Action:
        """Resume a dropped call from checkpoint."""
        self._register_states()
        self._restore_context(checkpoint)
        return await self._enter_state(checkpoint["state"])

    async def handle_event(self, event: CallEvent) -> Action:
        """Process an incoming event. Returns the action to execute."""
        # 1. Global event handling
        if event.type == EventType.HANGUP:
            return Action(type=ActionType.END_CALL)

        if event.type == EventType.SILENCE:
            self.context.silence_count += 1
            if self.context.silence_count >= 2:
                self.context.fallback_reason = "repeated_silence"
                return await self._enter_state(CallState.CALLBACK_CAPTURE)
            # Single silence: reprompt current state
            return Action(type=ActionType.ASK, text="Are you still there?")

        if event.type == EventType.ERROR:
            self.context.fallback_reason = "system_error"
            return await self._enter_state(CallState.CALLBACK_CAPTURE)

        # 2. Reset silence counter on any transcription
        if event.type == EventType.TRANSCRIPTION:
            self.context.silence_count = 0
            self.context.turn_count += 1

        # 3. Check guardrails before dispatching
        guardrail_result = self._check_guardrails()
        if guardrail_result is not None:
            return guardrail_result

        # 4. Dispatch to current state
        result = await self._safe_handle(event)
        return await self._process_result(result)

    async def _enter_state(self, state_name: str) -> Action:
        """Transition to a new state. Loops through silent transitions."""
        while True:
            self.current_state = self.states[state_name]

            # Write checkpoint
            await self._write_checkpoint()

            # Check guardrails on entry
            guardrail_result = self._check_guardrails()
            if guardrail_result is not None:
                return guardrail_result

            # Enter the state
            action = await self._safe_enter()

            if action.type == ActionType.END_CALL:
                return action
            if action.type == ActionType.ASK:
                return action
            if action.next_state is not None:
                state_name = action.next_state  # loop — silent transition
                continue
            return action  # SPEAK with no transition — stay and wait

    async def _process_result(self, action: Action) -> Action:
        """After a state's handle() returns, process transitions."""
        if action.next_state is not None:
            return await self._enter_state(action.next_state)
        return action

    async def _safe_handle(self, event: CallEvent) -> Action:
        """Call current state's handle() with tool-error protection."""
        try:
            return await self.current_state.handle(event, self.context)
        except Exception:
            self.context.fallback_reason = "tool_error"
            return Action(
                type=ActionType.TRANSITION,
                next_state=CallState.CALLBACK_CAPTURE,
            )

    async def _safe_enter(self) -> Action:
        """Call current state's enter() with tool-error protection."""
        try:
            return await self.current_state.enter(self.context)
        except Exception:
            self.context.fallback_reason = "tool_error"
            return Action(
                type=ActionType.TRANSITION,
                next_state=CallState.CALLBACK_CAPTURE,
            )

    def _check_guardrails(self) -> Action | None:
        """Deterministic guardrail enforcement. Returns Action if cap exceeded."""
        g = self.context.tenant_config.guardrails
        elapsed = time.monotonic() - self.context.call_start

        # If at READ_BACK or CONFIRM, let them finish (spec rule)
        if self.current_state and self.current_state.name in (
            CallState.READ_BACK, CallState.CONFIRM, CallState.CLOSE
        ):
            return None

        if elapsed > g.max_call_seconds:
            self.context.fallback_reason = "max_duration_exceeded"
            return Action(
                type=ActionType.SPEAK,
                text="I'm sorry, we've been on the call for a while. Let me have someone call you back to finish up.",
                next_state=CallState.CALLBACK_CAPTURE,
            )

        if self.context.turn_count > g.max_turns:
            self.context.fallback_reason = "max_turns_exceeded"
            return Action(
                type=ActionType.SPEAK,
                text="I want to make sure we get this right. Let me have someone call you back.",
                next_state=CallState.CALLBACK_CAPTURE,
            )

        return None

    async def _write_checkpoint(self) -> None:
        """Write current state to checkpoint store."""
        data = {
            "state": self.current_state.name,
            "intent": self.context.intent,
            "slots": asdict(self.context.slots),
            "turn_count": self.context.turn_count,
            "caller_id": self.context.caller.id if self.context.caller else None,
            "language": self.context.language,
            "call_id": self.context.call_id,
        }
        await self.checkpoint_store.save(
            self.context.caller_phone, data, ttl_seconds=900
        )

    def _register_states(self) -> None:
        """Create and register all state instances."""
        # Each state receives references to shared services
        deps = StateDeps(
            data_adapter=self.data_adapter,
            nlu=self.nlu,
            date_resolver_factory=self._make_date_resolver,
        )
        self.states = {
            CallState.GREETING: GreetingState(deps),
            CallState.IDENTIFY_CALLER: IdentifyCallerState(deps),
            CallState.INTENT: IntentState(deps),
            CallState.COLLECT_SERVICE: CollectServiceState(deps),
            CallState.COLLECT_DATETIME: CollectDatetimeState(deps),
            CallState.OFFER_SLOTS: OfferSlotsState(deps),
            CallState.COLLECT_CUSTOM: CollectCustomState(deps),
            CallState.READ_BACK: ReadBackState(deps),
            CallState.CONFIRM: ConfirmState(deps),
            CallState.LOOKUP_BOOKINGS: LookupBookingsState(deps),
            CallState.SELECT_BOOKING: SelectBookingState(deps),
            CallState.READ_STATUS: ReadStatusState(deps),
            CallState.CONFIRM_CANCEL: ConfirmCancelState(deps),
            CallState.CALLBACK_CAPTURE: CallbackCaptureState(deps),
            CallState.CLOSE: CloseState(deps),
        }

    def _restore_context(self, checkpoint: dict) -> None:
        """Restore CallContext from a checkpoint dict."""
        self.context.intent = checkpoint.get("intent")
        self.context.turn_count = checkpoint.get("turn_count", 0)
        self.context.language = checkpoint.get("language", "en-IN")
        slots_data = checkpoint.get("slots", {})
        self.context.slots = BookingSlots(**slots_data)
```

---

## 8. Interfaces

### 8.1 BaseState

```python
class BaseState(ABC):
    name: str  # CallState enum value

    def __init__(self, deps: StateDeps):
        self.deps = deps

    @abstractmethod
    async def enter(self, context: CallContext) -> Action:
        """Called once when entering the state. Returns initial action."""
        ...

    @abstractmethod
    async def handle(self, event: CallEvent, context: CallContext) -> Action:
        """Called for each event while in this state. Returns action (possibly with transition)."""
        ...
```

### 8.2 StateDeps

```python
@dataclass
class StateDeps:
    """Shared dependencies injected into every state."""
    data_adapter: DataAdapter
    nlu: NLUService
    date_resolver_factory: Callable[[TenantConfig], DateResolver]
```

### 8.3 NLUService (stub interface)

```python
@dataclass
class IntentResult:
    intent: str           # "new_booking" | "cancel" | "reschedule" | "status" | "unknown"
    confidence: float

@dataclass
class ServiceResult:
    service_id: str | None
    confidence: float
    alternatives: list[str]   # service IDs if ambiguous

class NLUService(ABC):
    @abstractmethod
    async def classify_intent(self, text: str, available_intents: list[str]) -> IntentResult:
        ...

    @abstractmethod
    async def extract_service(self, text: str, services: list[Service]) -> ServiceResult:
        ...

    @abstractmethod
    async def is_affirmative(self, text: str) -> bool:
        """Is this a 'yes' response? Used for confirmations."""
        ...

    @abstractmethod
    async def is_negative(self, text: str) -> bool:
        """Is this a 'no' response?"""
        ...
```

### 8.4 StubNLUService (for testing)

```python
class StubNLUService(NLUService):
    """Deterministic NLU for testing. Maps exact text to results."""

    def __init__(self):
        self.intent_map: dict[str, str] = {}       # text → intent
        self.service_map: dict[str, str] = {}       # text → service_id
        self.affirmative_words = {"yes", "yeah", "yep", "sure", "confirm", "go ahead", "ok", "okay"}
        self.negative_words = {"no", "nope", "nah", "cancel", "don't", "stop"}

    async def classify_intent(self, text, available_intents):
        lower = text.lower().strip()
        if lower in self.intent_map:
            return IntentResult(intent=self.intent_map[lower], confidence=1.0)
        # Simple keyword matching
        if any(w in lower for w in ["book", "appointment", "schedule"]):
            return IntentResult(intent="new_booking", confidence=0.8)
        if any(w in lower for w in ["cancel"]):
            return IntentResult(intent="cancel", confidence=0.8)
        if any(w in lower for w in ["reschedule", "change", "move"]):
            return IntentResult(intent="reschedule", confidence=0.8)
        if any(w in lower for w in ["status", "check", "existing", "upcoming"]):
            return IntentResult(intent="status", confidence=0.8)
        return IntentResult(intent="unknown", confidence=0.0)

    async def extract_service(self, text, services):
        lower = text.lower().strip()
        if lower in self.service_map:
            return ServiceResult(service_id=self.service_map[lower], confidence=1.0, alternatives=[])
        # Fuzzy match against service names
        for svc in services:
            if svc.name.lower() in lower or lower in svc.name.lower():
                return ServiceResult(service_id=svc.id, confidence=0.8, alternatives=[])
        return ServiceResult(service_id=None, confidence=0.0, alternatives=[s.id for s in services[:3]])

    async def is_affirmative(self, text):
        return text.lower().strip() in self.affirmative_words

    async def is_negative(self, text):
        return text.lower().strip() in self.negative_words
```

### 8.5 CheckpointStore

```python
class CheckpointStore(ABC):
    @abstractmethod
    async def save(self, phone: str, data: dict, ttl_seconds: int = 900) -> None: ...

    @abstractmethod
    async def load(self, phone: str) -> dict | None: ...

    @abstractmethod
    async def delete(self, phone: str) -> None: ...

class InMemoryCheckpointStore(CheckpointStore):
    """Dict-backed checkpoint store for development and testing."""

    def __init__(self):
        self._store: dict[str, tuple[dict, float]] = {}  # phone → (data, expires_at)

    async def save(self, phone, data, ttl_seconds=900):
        self._store[phone] = (data, time.monotonic() + ttl_seconds)

    async def load(self, phone):
        entry = self._store.get(phone)
        if entry is None:
            return None
        data, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[phone]
            return None
        return data

    async def delete(self, phone):
        self._store.pop(phone, None)
```

---

## 9. PipelineAdapter

Thin Pipecat `FrameProcessor` that bridges the state machine to the voice pipeline. This is the only component that imports Pipecat.

```python
class PipelineAdapter(FrameProcessor):
    """Bridges DialogueManager ↔ Pipecat pipeline."""

    def __init__(self, manager: DialogueManager):
        super().__init__()
        self.manager = manager
        self._started = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame) and not self._started:
            self._started = True
            action = await self.manager.start()
            await self._execute_action(action)

        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip() if frame.text else ""
            if not text:
                return
            event = CallEvent(type=EventType.TRANSCRIPTION, text=text)
            action = await self.manager.handle_event(event)
            await self._execute_action(action)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # VAD silence — could use for silence detection
            pass

        else:
            await self.push_frame(frame, direction)

    async def _execute_action(self, action: Action):
        if action.type in (ActionType.SPEAK, ActionType.ASK):
            if action.text:
                await self.push_frame(
                    TextFrame(text=action.text),
                    FrameDirection.DOWNSTREAM,
                )
        elif action.type == ActionType.END_CALL:
            if action.text:
                await self.push_frame(
                    TextFrame(text=action.text),
                    FrameDirection.DOWNSTREAM,
                )
            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
```

---

## 10. File Structure

```
packages/voice-agent/
├── dialogue/
│   ├── __init__.py
│   ├── manager.py              # DialogueManager
│   ├── models.py               # CallEvent, Action, CallContext, BookingSlots, StateDeps
│   ├── base_state.py           # BaseState ABC
│   ├── states/
│   │   ├── __init__.py
│   │   ├── greeting.py         # GreetingState
│   │   ├── identify_caller.py  # IdentifyCallerState
│   │   ├── intent.py           # IntentState
│   │   ├── collect_service.py  # CollectServiceState
│   │   ├── collect_datetime.py # CollectDatetimeState
│   │   ├── offer_slots.py      # OfferSlotsState
│   │   ├── collect_custom.py   # CollectCustomState
│   │   ├── read_back.py        # ReadBackState
│   │   ├── confirm.py          # ConfirmState
│   │   ├── lookup_bookings.py  # LookupBookingsState
│   │   ├── select_booking.py   # SelectBookingState
│   │   ├── read_status.py      # ReadStatusState
│   │   ├── confirm_cancel.py   # ConfirmCancelState
│   │   ├── callback_capture.py # CallbackCaptureState
│   │   └── close.py            # CloseState
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── base.py             # NLUService ABC + result types
│   │   └── stub.py             # StubNLUService
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   ├── base.py             # CheckpointStore ABC
│   │   └── memory.py           # InMemoryCheckpointStore
│   └── pipeline_adapter.py     # PipelineAdapter (Pipecat glue)
├── tests/
│   ├── test_dialogue_manager.py
│   ├── test_states/
│   │   ├── test_greeting.py
│   │   ├── test_intent.py
│   │   ├── test_collect_service.py
│   │   ├── test_collect_datetime.py
│   │   ├── test_offer_slots.py
│   │   ├── test_read_back.py
│   │   ├── test_confirm.py
│   │   ├── test_lookup_bookings.py
│   │   ├── test_confirm_cancel.py
│   │   ├── test_callback_capture.py
│   │   └── test_close.py
│   └── test_checkpoint.py
```

---

## 11. Testing Strategy

### 11.1 Unit tests (per state)

Each state is tested in isolation:
- Construct the state with mock `StateDeps` (mock data adapter, stub NLU).
- Create a `CallContext` with the relevant slots pre-filled.
- Call `enter()` and assert the returned `Action`.
- Call `handle()` with various `CallEvent` inputs and assert transitions + actions.

Example patterns:
- **Happy path:** correct input → expected transition.
- **Reprompt:** wrong input → stays in state, speaks reprompt.
- **Max retries:** wrong input × N → transitions to `CALLBACK_CAPTURE`.
- **Tool error:** data adapter raises → `CALLBACK_CAPTURE`.

### 11.2 Integration tests (DialogueManager)

Test full conversation flows end-to-end through the manager:
- **New booking flow:** GREETING → ... → CONFIRM → CLOSE.
- **Cancel flow:** GREETING → ... → CONFIRM_CANCEL → CLOSE.
- **Reschedule flow:** GREETING → ... → COLLECT_DATETIME → ... → CONFIRM → CLOSE.
- **Fallback flow:** GREETING → INTENT (fail × 3) → CALLBACK_CAPTURE → CLOSE.
- **Guardrail flow:** Exceed max_turns → CALLBACK_CAPTURE.
- **Resume flow:** Checkpoint at COLLECT_DATETIME → resume → continue from COLLECT_DATETIME.

### 11.3 What we DON'T test here

- Pipecat integration (PipelineAdapter) — tested manually via spike or future E2E tests.
- Actual LLM calls — stubbed.
- Actual Supabase calls — mocked data adapter.

---

## 12. Invariant Compliance

| Invariant | How this design satisfies it |
|-----------|------------------------------|
| #1 LLM untrusted, never writes DB | NLU only returns structured results (intent, service_id). All DB writes go through DataAdapter. |
| #2 Read-back before every write | READ_BACK state is mandatory before CONFIRM. No path skips it. |
| #3 External systems not in sync path | DataAdapter reads from Supabase mirror. No external CRM/calendar in the turn loop. |
| #4 Fallback floor = callback capture | Every state's failure path leads to CALLBACK_CAPTURE. Manager catches all tool errors. |
| #5 Dates by deterministic resolver | COLLECT_DATETIME uses DateResolver, not LLM. LLM only provides the text phrase. |
| #6 Guardrails are deterministic | Manager's `_check_guardrails()` enforces max_call_seconds and max_turns before every state dispatch. |
| #8 idempotency_key = hold_id | CONFIRM passes hold_id to confirm_booking. Retries are idempotent. |
| #9 Checkpoint on every transition | Manager's `_enter_state()` writes checkpoint before entering each new state. |
