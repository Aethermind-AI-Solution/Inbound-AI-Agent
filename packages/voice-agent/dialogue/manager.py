from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import date, time as dt_time
from typing import TYPE_CHECKING

from packages.voice_agent.dialogue.base_state import BaseState
from packages.voice_agent.dialogue.models import (
    Action,
    ActionType,
    CallContext,
    CallEvent,
    CallState,
    EventType,
    StateDeps,
    BookingSlots,
)
from packages.voice_agent.dialogue.states import (
    CallbackCaptureState,
    CloseState,
    CollectCustomState,
    CollectDatetimeState,
    CollectServiceState,
    ConfirmCancelState,
    ConfirmState,
    GreetingState,
    IdentifyCallerState,
    IntentState,
    LookupBookingsState,
    OfferSlotsState,
    ReadBackState,
    ReadStatusState,
    SelectBookingState,
)

if TYPE_CHECKING:
    from packages.voice_agent.config.models import TenantConfig
    from packages.voice_agent.data.adapter import DataAdapter
    from packages.voice_agent.dialogue.budget.base import BudgetTracker
    from packages.voice_agent.dialogue.checkpoint.base import CheckpointStore
    from packages.voice_agent.dialogue.nlu.base import NLUService

logger = logging.getLogger(__name__)


class DialogueManager:
    def __init__(
        self,
        config: TenantConfig,
        data_adapter: DataAdapter,
        nlu: NLUService,
        checkpoint_store: CheckpointStore,
        caller_phone: str,
        call_id: str,
        budget_tracker: BudgetTracker | None = None,
    ) -> None:
        self.context = CallContext(
            tenant_config=config,
            caller_phone=caller_phone,
            call_id=call_id,
        )
        self.data_adapter = data_adapter
        self.nlu = nlu
        self.checkpoint_store = checkpoint_store
        self.budget_tracker = budget_tracker
        self.states: dict[str, BaseState] = {}
        self.current_state: BaseState | None = None

    async def start(self) -> Action:
        self._register_states()
        if self.budget_tracker and not self.context.budget_checked:
            tenant_id = self.context.tenant_config.meta.tenant_id
            budget = self.context.tenant_config.guardrails.monthly_budget_inr
            within_budget = await self.budget_tracker.check_budget(tenant_id, budget)
            self.context.budget_checked = True
            if not within_budget:
                self.context.fallback_reason = "budget_exceeded"
                return await self._enter_state(CallState.CALLBACK_CAPTURE)
        return await self._enter_state(CallState.GREETING)

    async def resume(self, checkpoint: dict) -> Action:
        self._register_states()
        self._restore_context(checkpoint)
        return await self._enter_state(checkpoint["state"])

    async def handle_event(self, event: CallEvent) -> Action:
        if event.type == EventType.HANGUP:
            return Action(type=ActionType.END_CALL)

        if event.type == EventType.SILENCE:
            self.context.silence_count += 1
            if self.context.silence_count >= 2:
                self.context.fallback_reason = "repeated_silence"
                return await self._enter_state(CallState.CALLBACK_CAPTURE)
            return Action(type=ActionType.ASK, text="Are you still there?")

        if event.type == EventType.ERROR:
            self.context.fallback_reason = "system_error"
            return await self._enter_state(CallState.CALLBACK_CAPTURE)

        if event.type == EventType.TRANSCRIPTION:
            self.context.silence_count = 0
            self.context.turn_count += 1

        guardrail_result = self._check_guardrails()
        if guardrail_result is not None:
            return await self._process_result(guardrail_result)

        action = await self._safe_handle(event)
        return await self._process_result(action)

    async def record_call_usage(self, duration_seconds: float) -> None:
        if self.budget_tracker:
            tenant_id = self.context.tenant_config.meta.tenant_id
            await self.budget_tracker.record_usage(tenant_id, duration_seconds)

    async def _enter_state(self, state_name: str) -> Action:
        while True:
            self.current_state = self.states[state_name]
            await self._write_checkpoint()

            guardrail_result = self._check_guardrails()
            if guardrail_result is not None:
                return guardrail_result

            action = await self._safe_enter()

            if action.type == ActionType.END_CALL:
                return action
            if action.type == ActionType.ASK:
                return action
            if action.next_state is not None:
                state_name = action.next_state
                continue
            return action

    async def _process_result(self, action: Action) -> Action:
        if action.next_state is not None:
            return await self._enter_state(action.next_state)
        return action

    async def _safe_handle(self, event: CallEvent) -> Action:
        try:
            return await self.current_state.handle(event, self.context)
        except Exception:
            logger.exception("Error in state %s handle()", self.current_state.name)
            self.context.fallback_reason = "tool_error"
            return Action(
                type=ActionType.TRANSITION,
                next_state=CallState.CALLBACK_CAPTURE,
            )

    async def _safe_enter(self) -> Action:
        try:
            return await self.current_state.enter(self.context)
        except Exception:
            logger.exception("Error in state %s enter()", self.current_state.name)
            self.context.fallback_reason = "tool_error"
            return Action(
                type=ActionType.TRANSITION,
                next_state=CallState.CALLBACK_CAPTURE,
            )

    def _check_guardrails(self) -> Action | None:
        g = self.context.tenant_config.guardrails
        elapsed = time.monotonic() - self.context.call_start

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
        data = {
            "state": self.current_state.name,
            "intent": self.context.intent,
            "slots": asdict(self.context.slots),
            "turn_count": self.context.turn_count,
            "caller_id": self.context.caller.id if self.context.caller else None,
            "language": self.context.language,
            "call_id": self.context.call_id,
            "budget_checked": self.context.budget_checked,
        }
        await self.checkpoint_store.save(
            self.context.caller_phone, data, ttl_seconds=900
        )

    def _register_states(self) -> None:
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

    def _make_date_resolver(self, config):
        from packages.voice_agent.resolver.date_resolver import DateResolver
        from datetime import date as d, datetime as dt
        bm = config.booking_model
        now = d.today()
        now_time = dt.now().time()
        return DateResolver(
            business_hours=bm.business_hours,
            booking_window_days=bm.booking_window_days,
            min_notice_min=bm.min_notice_min,
            reference_date=now,
            reference_time=now_time,
        )

    def _restore_context(self, checkpoint: dict) -> None:
        self.context.intent = checkpoint.get("intent")
        self.context.turn_count = checkpoint.get("turn_count", 0)
        self.context.language = checkpoint.get("language", "en-IN")
        slots_data = checkpoint.get("slots", {})
        self.context.slots = BookingSlots(**slots_data)
        self.context.budget_checked = checkpoint.get("budget_checked", False)
