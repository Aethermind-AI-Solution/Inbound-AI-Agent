from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from packages.voice_agent.utils import mask_phone

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
FlowArgs = dict[str, Any]

_PHONE_PATTERN = re.compile(r"^\+?\d{7,15}$")

DATA_TIMEOUT = 8.0


async def _write_checkpoint(flow_manager: Any, node_name: str) -> None:
    store = flow_manager.state.get("checkpoint_store")
    if not store:
        return
    phone = flow_manager.state.get("caller_phone", "unknown")
    data = {
        "node": node_name,
        "intent": flow_manager.state.get("intent"),
        "service_id": flow_manager.state.get("service_id"),
        "service_name": flow_manager.state.get("service_name"),
        "datetime_ist": flow_manager.state.get("datetime_ist"),
        "resource_id": flow_manager.state.get("resource_id"),
        "resource_name": flow_manager.state.get("resource_name"),
        "hold_id": flow_manager.state.get("hold_id"),
        "turn_count": flow_manager.state.get("turn_count", 0),
        "caller_id": flow_manager.state.get("caller_id"),
        "call_id": flow_manager.state.get("call_id"),
    }
    await store.save(phone, data, ttl_seconds=900)


async def _transition(flow_manager: Any, node_factory, *args) -> dict:
    node = node_factory(flow_manager, *args) if args else node_factory(flow_manager)
    await _write_checkpoint(flow_manager, node["name"])
    return node


# --- Greeting handlers ---

async def start_new_booking(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    flow_manager.state["intent"] = "new_booking"
    from packages.voice_agent.flows.nodes import create_collect_service_node
    node = await _transition(flow_manager, create_collect_service_node)
    return "Starting new booking.", node


async def check_booking_status(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    flow_manager.state["intent"] = "status"
    return await _lookup_and_transition(flow_manager, "status")


async def cancel_booking_intent(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    flow_manager.state["intent"] = "cancel"
    return await _lookup_and_transition(flow_manager, "cancel")


async def reschedule_booking_intent(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    flow_manager.state["intent"] = "reschedule"
    return await _lookup_and_transition(flow_manager, "reschedule")


async def _lookup_and_transition(flow_manager: Any, intent: str) -> tuple[str, dict | None]:
    data = flow_manager.state["data_adapter"]
    tenant_id = flow_manager.state["tenant_id"]
    caller_id = flow_manager.state.get("caller_id")
    bookings = await asyncio.wait_for(
        asyncio.to_thread(data.lookup_bookings, tenant_id, caller_id),
        timeout=DATA_TIMEOUT,
    )
    flow_manager.state["bookings"] = bookings
    from packages.voice_agent.flows.nodes import create_manage_booking_node
    node = await _transition(flow_manager, create_manage_booking_node, intent)
    return f"Found {len(bookings)} booking(s).", node


async def request_callback(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    reason = args.get("reason", "general")
    flow_manager.state["fallback_reason"] = reason
    from packages.voice_agent.flows.nodes import create_callback_capture_node
    node = await _transition(flow_manager, create_callback_capture_node)
    return "Arranging callback.", node


# --- Collect service handlers ---

async def select_service(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    service_id = args.get("service_id", "")
    config = flow_manager.state["config"]
    services = config.booking_model.services
    svc = next((s for s in services if s.id == service_id), None)
    if not svc:
        valid = [s.id for s in services]
        return {"error": f"Service not found. Valid IDs: {valid}"}, None
    flow_manager.state["service_id"] = svc.id
    flow_manager.state["service_name"] = svc.name
    flow_manager.state["service_duration_min"] = svc.duration_min
    flow_manager.state["resource_type"] = svc.resource_type
    flow_manager.state["custom_fields"] = [
        {"key": f.key, "type": f.type, "required": f.required, "prompt": f.prompt}
        for f in (svc.custom_fields or [])
    ]
    from packages.voice_agent.flows.nodes import create_collect_datetime_node
    node = await _transition(flow_manager, create_collect_datetime_node)
    return f"Selected {svc.name}.", node


# --- Collect datetime handlers ---

async def check_availability(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    date_str = args.get("date", "")
    time_str = args.get("time", "")

    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt = dt.replace(tzinfo=IST)
    except ValueError:
        return {"error": "Invalid format. Use date as YYYY-MM-DD and time as HH:MM."}, None

    now = datetime.now(IST)
    config = flow_manager.state["config"]
    bm = config.booking_model

    if dt <= now:
        return {"error": "That date and time is in the past. Please suggest a future date."}, None

    min_notice = timedelta(minutes=bm.min_notice_min)
    if dt < now + min_notice:
        return {"error": f"We need at least {bm.min_notice_min} minutes notice. Please suggest a later time."}, None

    max_date = now + timedelta(days=bm.booking_window_days)
    if dt > max_date:
        return {"error": f"We can only book up to {bm.booking_window_days} days ahead."}, None

    day_name = dt.strftime("%a").lower()
    day_hours = getattr(bm.business_hours, day_name, [])
    if not day_hours:
        open_days = [d for d in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
                     if getattr(bm.business_hours, d, [])]
        return {"error": f"We're closed on {dt.strftime('%A')}s. We're open on: {', '.join(d.capitalize() for d in open_days)}."}, None

    duration = flow_manager.state.get("service_duration_min", 30)
    service_end_time = (dt + timedelta(minutes=duration)).time()

    within_hours = False
    for slot in day_hours:
        parts = slot.split("-")
        if len(parts) != 2:
            continue
        open_time = datetime.strptime(parts[0], "%H:%M").time()
        close_time = datetime.strptime(parts[1], "%H:%M").time()
        if open_time <= dt.time() and service_end_time <= close_time:
            within_hours = True
            break
    if not within_hours:
        return {"error": f"That time is outside business hours. Hours for {dt.strftime('%A')}: {', '.join(day_hours)}."}, None

    data = flow_manager.state["data_adapter"]
    tenant_id = flow_manager.state["tenant_id"]
    service_id = flow_manager.state["service_id"]
    resource_type = flow_manager.state["resource_type"]
    end_dt = dt + timedelta(minutes=duration)

    try:
        availability = await asyncio.wait_for(
            asyncio.to_thread(data.check_availability, tenant_id, service_id, resource_type, dt, end_dt),
            timeout=DATA_TIMEOUT,
        )
    except (TimeoutError, Exception):
        logger.exception("check_availability failed or timed out")
        return {"error": "Could not check availability right now. Please try a different time."}, None

    available = [r for r in availability if not r.get("blocked_slots")]
    if not available:
        return {"error": "No staff available at that time. Please suggest a different time."}, None

    flow_manager.state["datetime_ist"] = dt.isoformat()
    flow_manager.state["available_resources"] = available
    from packages.voice_agent.flows.nodes import create_offer_slots_node
    node = await _transition(flow_manager, create_offer_slots_node)
    return f"{len(available)} staff available.", node


# --- Offer slots handlers ---

async def book_slot(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    resource_id = args.get("resource_id", "")
    available = flow_manager.state.get("available_resources", [])
    resource = next((r for r in available if r["resource_id"] == resource_id), None)
    if not resource:
        return {"error": f"That staff member is not available. Options: {[r['resource_name'] for r in available]}"}, None

    data = flow_manager.state["data_adapter"]
    tenant_id = flow_manager.state["tenant_id"]
    caller_id = flow_manager.state["caller_id"]
    dt = datetime.fromisoformat(flow_manager.state["datetime_ist"])

    try:
        hold = await asyncio.wait_for(
            asyncio.to_thread(data.hold_slot, tenant_id, resource_id, dt, caller_id),
            timeout=DATA_TIMEOUT,
        )
    except (TimeoutError, Exception):
        logger.exception("hold_slot failed or timed out")
        return {"error": "Could not reserve that slot. Try a different option."}, None

    flow_manager.state["hold_id"] = hold.hold_id
    flow_manager.state["resource_id"] = resource_id
    flow_manager.state["resource_name"] = resource["resource_name"]
    flow_manager.state["hold_expires_at"] = hold.expires_at.isoformat() if hold.expires_at else None

    custom_fields = flow_manager.state.get("custom_fields", [])
    if custom_fields:
        from packages.voice_agent.flows.nodes import create_collect_custom_node
        node = await _transition(flow_manager, create_collect_custom_node)
        return f"Reserved with {resource['resource_name']}.", node

    from packages.voice_agent.flows.nodes import create_confirm_booking_node
    node = await _transition(flow_manager, create_confirm_booking_node)
    return f"Reserved with {resource['resource_name']}.", node


async def try_different_time(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    for key in ("datetime_ist", "available_resources"):
        flow_manager.state.pop(key, None)
    from packages.voice_agent.flows.nodes import create_collect_datetime_node
    node = await _transition(flow_manager, create_collect_datetime_node)
    return "Let's try a different time.", node


# --- Custom fields handlers ---

async def submit_custom_fields(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    fields = args.get("fields", {})
    flow_manager.state["custom_values"] = fields
    from packages.voice_agent.flows.nodes import create_confirm_booking_node
    node = await _transition(flow_manager, create_confirm_booking_node)
    return "Details collected.", node


# --- Confirm booking handlers ---

async def confirm(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    data = flow_manager.state["data_adapter"]
    tenant_id = flow_manager.state["tenant_id"]
    hold_id = flow_manager.state["hold_id"]
    service_id = flow_manager.state["service_id"]
    resource_id = flow_manager.state["resource_id"]
    caller_id = flow_manager.state["caller_id"]
    dt = datetime.fromisoformat(flow_manager.state["datetime_ist"])
    duration = flow_manager.state.get("service_duration_min", 30)
    end_dt = dt + timedelta(minutes=duration)
    custom_values = flow_manager.state.get("custom_values", {})

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                data.confirm_booking,
                tenant_id, hold_id, service_id, resource_id, caller_id, dt, end_dt, custom_values,
            ),
            timeout=DATA_TIMEOUT,
        )
    except (TimeoutError, Exception):
        logger.exception("confirm_booking failed")
        flow_manager.state["fallback_reason"] = "tool_error"
        from packages.voice_agent.flows.nodes import create_callback_capture_node
        node = await _transition(flow_manager, create_callback_capture_node)
        return {"error": "Booking system error."}, node

    flow_manager.state["booking_id"] = result.booking_id
    from packages.voice_agent.flows.nodes import create_close_node
    node = await _transition(flow_manager, create_close_node)
    return (
        f"Booking confirmed! "
        f"{flow_manager.state['service_name']} with {flow_manager.state['resource_name']} "
        f"on {dt.strftime('%A, %B %d at %I:%M %p')}."
    ), node


async def change_service(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    for key in ("service_id", "service_name", "service_duration_min", "resource_type",
                "datetime_ist", "hold_id", "resource_id", "resource_name",
                "available_resources", "custom_values", "custom_fields"):
        flow_manager.state.pop(key, None)
    from packages.voice_agent.flows.nodes import create_collect_service_node
    node = await _transition(flow_manager, create_collect_service_node)
    return "Let's pick a different service.", node


async def change_datetime(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    for key in ("datetime_ist", "hold_id", "resource_id", "resource_name", "available_resources"):
        flow_manager.state.pop(key, None)
    from packages.voice_agent.flows.nodes import create_collect_datetime_node
    node = await _transition(flow_manager, create_collect_datetime_node)
    return "Let's pick a different time.", node


# --- Manage booking handlers ---

async def cancel_booking(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    booking_id = args.get("booking_id", "")
    data = flow_manager.state["data_adapter"]
    tenant_id = flow_manager.state["tenant_id"]
    caller_id = flow_manager.state["caller_id"]

    try:
        success = await asyncio.wait_for(
            asyncio.to_thread(data.cancel_booking, tenant_id, booking_id, caller_id),
            timeout=DATA_TIMEOUT,
        )
    except (TimeoutError, Exception):
        logger.exception("cancel_booking failed or timed out")
        return {"error": "Could not cancel right now. Please try again."}, None
    if not success:
        return {"error": "Could not cancel that booking. It may have already been cancelled."}, None

    from packages.voice_agent.flows.nodes import create_close_node
    node = await _transition(flow_manager, create_close_node)
    return "Booking cancelled.", node


async def start_reschedule(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    booking_id = args.get("booking_id", "")
    flow_manager.state["reschedule_booking_id"] = booking_id
    flow_manager.state["intent"] = "reschedule"
    bookings = flow_manager.state.get("bookings", [])
    booking = next((b for b in bookings if b["id"] == booking_id), None)
    if booking:
        service_id = booking.get("service_id")
        config = flow_manager.state["config"]
        svc = next((s for s in config.booking_model.services if s.id == service_id), None)
        if svc:
            flow_manager.state["service_id"] = svc.id
            flow_manager.state["service_name"] = svc.name
            flow_manager.state["service_duration_min"] = svc.duration_min
            flow_manager.state["resource_type"] = svc.resource_type
    from packages.voice_agent.flows.nodes import create_collect_datetime_node
    node = await _transition(flow_manager, create_collect_datetime_node)
    return "Let's pick a new date and time.", node


async def done(args: FlowArgs, flow_manager: Any) -> tuple[str, dict | None]:
    from packages.voice_agent.flows.nodes import create_close_node
    node = await _transition(flow_manager, create_close_node)
    return "Wrapping up.", node


# --- Callback capture handlers ---

async def confirm_callback(args: FlowArgs, flow_manager: Any) -> tuple[str | dict, dict | None]:
    phone = args.get("phone_number", "")
    if not _PHONE_PATTERN.match(phone.replace(" ", "").replace("-", "")):
        return {"error": "That doesn't look like a valid phone number. Please ask for the number again."}, None
    flow_manager.state["callback_number"] = phone
    logger.info("Callback confirmed for %s", mask_phone(phone))
    from packages.voice_agent.flows.nodes import create_close_node
    node = await _transition(flow_manager, create_close_node)
    return "We'll call back shortly.", node
