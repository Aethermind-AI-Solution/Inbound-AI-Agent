# packages/voice-agent/dialogue/models.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from packages.voice_agent.config.models import TenantConfig
    from packages.voice_agent.data.adapter import CallerInfo, DataAdapter
    from packages.voice_agent.dialogue.nlu.base import NLUService
    from packages.voice_agent.resolver.date_resolver import DateResolver


class CallState(StrEnum):
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


class EventType(StrEnum):
    TRANSCRIPTION = "transcription"
    SILENCE = "silence"
    HANGUP = "hangup"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class CallEvent:
    type: EventType
    text: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


class ActionType(StrEnum):
    SPEAK = "speak"
    ASK = "ask"
    END_CALL = "end_call"
    TRANSITION = "transition"


@dataclass
class Action:
    type: ActionType
    text: str | None = None
    next_state: str | None = None
    timeout_s: float = 10.0
    checkpoint: bool = True


@dataclass
class BookingSlots:
    service_id: str | None = None
    service_name: str | None = None
    resource_id: str | None = None
    resource_name: str | None = None
    datetime_ist: Any | None = None
    hold_id: str | None = None
    hold_expires_at: Any | None = None
    booking_id: str | None = None
    custom_fields: dict = field(default_factory=dict)
    old_booking_date: str | None = None
    old_booking_service: str | None = None

    def clear_datetime(self) -> None:
        self.datetime_ist = None
        self.hold_id = None
        self.hold_expires_at = None
        self.resource_id = None
        self.resource_name = None

    def clear_service(self) -> None:
        self.service_id = None
        self.service_name = None
        self.clear_datetime()
        self.custom_fields = {}


@dataclass
class CallContext:
    tenant_config: Any  # TenantConfig
    caller_phone: str
    call_id: str
    caller: Any | None = None  # CallerInfo
    intent: str | None = None
    language: str = "en-IN"
    slots: BookingSlots = field(default_factory=BookingSlots)
    turn_count: int = 0
    silence_count: int = 0
    call_start: float = field(default_factory=time.monotonic)
    fallback_reason: str | None = None


@dataclass
class StateDeps:
    data_adapter: Any  # DataAdapter
    nlu: Any  # NLUService
    date_resolver_factory: Callable[..., Any]  # Callable[[TenantConfig], DateResolver]
