# PipelineAdapter Design Spec

**Date:** 2026-06-04
**Status:** Approved

## Goal

Build a thin Pipecat `FrameProcessor` that bridges the dialogue state machine (`DialogueManager`) into the Pipecat voice pipeline, translating `TranscriptionFrame` → `CallEvent` and `Action` → `TextFrame`/`EndTaskFrame`. Include a runner script for local mic testing with mock business logic.

## Architecture

Single `FrameProcessor` subclass (`PipelineAdapter`) sitting in the pipeline between STT and TTS — the same position `EchoProcessor` occupies in the spike. The adapter owns three responsibilities: inbound frame-to-event translation, outbound action-to-frame translation, and silence timer management.

The runner script (`run_local.py`) wires the adapter into a real voice pipeline with `LocalAudioTransport`, Deepgram STT, and a TTS service, using mock/stub deps for the business logic layer.

## Pipeline Position

```
transport.input() → stt → PipelineAdapter → tts → transport.output()
```

Same shape as `spike.py`, with `PipelineAdapter` replacing `EchoProcessor`.

## Inbound Translation (Frames → CallEvents)

| Pipecat Frame | Condition | CallEvent |
|---|---|---|
| `TranscriptionFrame` | `finalized=True` | `CallEvent(TRANSCRIPTION, text=frame.text)` |
| `TranscriptionFrame` | `finalized=False` | Ignored (no action) |
| `UserStoppedSpeakingFrame` | Silence timer running | Timer continues; no immediate event |
| Silence timer fires | After `timeout_s` with no finalized transcription | `CallEvent(SILENCE)` |

Interim (non-finalized) transcriptions are dropped. The DialogueManager expects one event per complete utterance.

## Outbound Translation (Actions → Frames)

| Action Type | Frames Pushed | Notes |
|---|---|---|
| `ASK` | `TextFrame(action.text)` | TTS speaks, then silence timer starts with `action.timeout_s` |
| `SPEAK` | `TextFrame(action.text)` | Same as ASK at the adapter level; DM handles the difference internally via `_enter_state` loop |
| `END_CALL` | `TextFrame(action.text)` (if text), then `EndTaskFrame()` | Pipeline shuts down |
| `TRANSITION` | Nothing | Internal to DM; adapter never sees this — `_enter_state` resolves transitions before returning |

## Silence Timer

An `asyncio.Task` inside the adapter that detects caller silence.

### Behavior

- **Start/reset** when the adapter pushes a `TextFrame` downstream (bot finished prompting, now waiting for caller)
- **Reset** on every finalized `TranscriptionFrame` (caller is talking)
- **Timeout** defaults to `action.timeout_s` (10s fallback)
- **On fire:** calls `dialogue_manager.handle_event(CallEvent(SILENCE))` and translates the resulting `Action` (typically "Are you still there?" on first silence, callback capture on second)
- **Cancel** on `END_CALL` or pipeline shutdown

### Implementation

```python
def _reset_silence_timer(self, timeout: float) -> None:
    if self._silence_task and not self._silence_task.done():
        self._silence_task.cancel()
    self._silence_task = asyncio.ensure_future(self._silence_watchdog(timeout))

async def _silence_watchdog(self, timeout: float) -> None:
    await asyncio.sleep(timeout)
    event = CallEvent(type=EventType.SILENCE)
    action = await self._dialogue_manager.handle_event(event)
    await self._push_action(action)
```

## Lifecycle

### Startup

1. Runner constructs `DialogueManager` with mock deps
2. Runner constructs `PipelineAdapter(dialogue_manager)`
3. Pipeline starts, Pipecat delivers frames to `process_frame()`
4. On the first frame received (any type), adapter calls `dialogue_manager.start()` — guarded by `self._started: bool`
5. Greeting `Action(ASK, "Hello, thanks for calling...")` is translated to `TextFrame` → TTS speaks → silence timer starts

### Shutdown

- `END_CALL` from DM → push final `TextFrame` if present → push `EndTaskFrame()`
- Transport hangup → not directly observable as a frame in Pipecat v1.3.0; handled by pipeline teardown. If a `HangupFrame` or equivalent is available, translate to `CallEvent(HANGUP)`.

### Error Boundary

The adapter wraps `dialogue_manager.handle_event()` and `dialogue_manager.start()` in try/except. If an exception escapes (shouldn't — DM catches internally), the adapter pushes `EndTaskFrame()` to avoid crashing the pipeline. No retry logic; the DM's fallback floor (callback capture) is the recovery mechanism.

## File Layout

| File | Purpose | ~Lines |
|---|---|---|
| `packages/voice-agent/pipeline_adapter.py` | `PipelineAdapter(FrameProcessor)` class | 80-100 |
| `packages/voice-agent/run_local.py` | Runner script for local mic testing | ~50 |
| `packages/voice-agent/data/mock_adapter.py` | Mock `DataAdapter` with canned responses | ~60 |
| `packages/voice-agent/tests/test_pipeline_adapter.py` | Unit tests for the adapter | 15-20 tests |

## PipelineAdapter Class API

```python
class PipelineAdapter(FrameProcessor):
    def __init__(self, dialogue_manager: DialogueManager) -> None: ...
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None: ...
```

No public methods beyond `process_frame`. Everything else is internal:

- `_handle_transcription(frame: TranscriptionFrame) -> None`
- `_push_action(action: Action) -> None`
- `_reset_silence_timer(timeout: float) -> None`
- `_silence_watchdog(timeout: float) -> None`

## Mock DataAdapter

An in-memory implementation of the `DataAdapter` protocol with canned data, sufficient to walk through a full booking flow by voice:

- `lookup_caller(phone)` → returns a fake `CallerInfo(id="caller-1", name="Test User")`
- `list_services(tenant_id)` → returns 2-3 fake services (e.g., "Haircut", "Facial")
- `find_slots(service_id, date, tenant_id)` → returns 2-3 fake time slots
- `hold_slot(slot_id)` → returns a fake hold ID
- `confirm_booking(hold_id)` → returns a fake booking ID
- `lookup_bookings(caller_id)` → returns 1 fake existing booking
- `cancel_booking(booking_id)` → returns success

No network calls, no Supabase. Pure in-memory with hardcoded responses.

## Runner Script (`run_local.py`)

```python
# Pseudocode outline
async def main():
    config = make_local_tenant_config()
    data = MockDataAdapter()
    nlu = StubNLUService()
    checkpoint = InMemoryCheckpointStore()

    dm = DialogueManager(config, data, nlu, checkpoint, caller_phone="+911234567890", call_id="local-test")
    adapter = PipelineAdapter(dm)

    transport = LocalAudioTransport(...)
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = ...  # TTS service

    pipeline = Pipeline([transport.input(), stt, adapter, tts, transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)
```

Requires `DEEPGRAM_API_KEY` in environment. TTS service choice is flexible (Smallest, Sarvam, or any Pipecat-compatible TTS).

## Testing Strategy

Unit tests mock the `DialogueManager` — they test the adapter's translation logic, not the state machine:

1. **Finalized transcription → CallEvent**: Verify `handle_event` called with correct `CallEvent(TRANSCRIPTION, text=...)`
2. **Non-finalized transcription ignored**: Verify `handle_event` NOT called
3. **ASK action → TextFrame**: Verify `TextFrame` pushed downstream
4. **END_CALL action → TextFrame + EndTaskFrame**: Verify both frames in order
5. **TRANSITION action → no frame**: Verify nothing pushed (adapter should never see this, but defensive)
6. **Silence timer fires**: Verify `CallEvent(SILENCE)` sent after timeout
7. **Silence timer resets on transcription**: Verify timer cancelled and restarted
8. **Start called once**: Verify `dialogue_manager.start()` called on first frame, not on subsequent frames
9. **Error in DM → EndTaskFrame**: Verify pipeline doesn't crash
10. **Multiple transcriptions**: Verify each gets its own `handle_event` call

No integration tests with real STT/TTS — that's what `run_local.py` is for (manual testing).

## Non-Goals

- **No Twilio transport** — local mic only. Twilio is a transport swap later.
- **No real DataAdapter** — mock only. Supabase integration is a separate task.
- **No barge-in detection** — out of scope. Could use interim transcriptions later if needed.
- **No checkpoint resume** — `dialogue_manager.start()` only, not `resume()`. Resume support is a separate task.
- **No echo suppression** — the spike had this; not needed in the adapter since DialogueManager controls when the bot speaks.
