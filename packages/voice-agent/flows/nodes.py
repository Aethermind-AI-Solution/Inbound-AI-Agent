from __future__ import annotations

import logging
from typing import Any

from pipecat_flows import FlowsFunctionSchema

from packages.voice_agent.flows import handlers, prompts

logger = logging.getLogger(__name__)


def _tool(name: str, description: str, properties: dict, required: list[str], handler) -> FlowsFunctionSchema:
    return FlowsFunctionSchema(
        name=name,
        description=description,
        properties=properties,
        required=required,
        handler=handler,
    )


def make_request_callback_tool() -> FlowsFunctionSchema:
    return _tool(
        name="request_callback",
        description="Arrange a callback when you cannot help the caller or they request one",
        properties={"reason": {"type": "string", "description": "Why callback is needed", "enum": [
            "no_availability", "repeated_failure", "auth_required", "caller_request", "general"
        ]}},
        required=["reason"],
        handler=handlers.request_callback,
    )


def create_greeting_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    return {
        "name": "greeting",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.greeting_task(config),
        "respond_immediately": True,
        "functions": [
            _tool("start_new_booking", "Start a new appointment booking",
                  {}, [], handlers.start_new_booking),
            _tool("check_booking_status", "Look up existing bookings to check status",
                  {}, [], handlers.check_booking_status),
            _tool("cancel_booking_intent", "Caller wants to cancel a booking",
                  {}, [], handlers.cancel_booking_intent),
            _tool("reschedule_booking_intent", "Caller wants to reschedule a booking",
                  {}, [], handlers.reschedule_booking_intent),
        ],
    }


def create_collect_service_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    return {
        "name": "collect_service",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.collect_service_task(config),
        "functions": [
            _tool("select_service", "Record the caller's chosen service",
                  {"service_id": {
                      "type": "string",
                      "description": "The ID of the chosen service",
                      "enum": [s.id for s in config.booking_model.services],
                  }},
                  ["service_id"], handlers.select_service),
        ],
    }


def create_collect_datetime_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    return {
        "name": "collect_datetime",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.collect_datetime_task(config),
        "functions": [
            _tool("check_availability", "Check if a date and time has available staff",
                  {
                      "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                      "time": {"type": "string", "description": "Time in HH:MM 24-hour format"},
                  },
                  ["date", "time"], handlers.check_availability),
        ],
    }


def create_offer_slots_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    available = flow_manager.state.get("available_resources", [])
    return {
        "name": "offer_slots",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.offer_slots_task(available),
        "functions": [
            _tool("book_slot", "Reserve a slot with the chosen staff member",
                  {"resource_id": {
                      "type": "string",
                      "description": "ID of the chosen staff member",
                      "enum": [r["resource_id"] for r in available],
                  }},
                  ["resource_id"], handlers.book_slot),
            _tool("try_different_time", "Go back and pick a different date/time",
                  {}, [], handlers.try_different_time),
        ],
    }


def create_collect_custom_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    custom_fields = flow_manager.state.get("custom_fields", [])
    properties = {}
    required = []
    for f in custom_fields:
        properties[f["key"]] = {"type": "string", "description": f.get("prompt", f["key"])}
        if f.get("required", False):
            required.append(f["key"])

    return {
        "name": "collect_custom",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.collect_custom_task(custom_fields),
        "functions": [
            _tool("submit_custom_fields", "Submit the collected custom field values",
                  {"fields": {
                      "type": "object",
                      "description": "Key-value pairs of custom field data",
                      "properties": properties,
                      "required": required,
                  }},
                  ["fields"], handlers.submit_custom_fields),
        ],
    }


def create_confirm_booking_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    summary = {
        "service_name": flow_manager.state.get("service_name"),
        "resource_name": flow_manager.state.get("resource_name"),
        "datetime": flow_manager.state.get("datetime_ist"),
        "custom_fields": flow_manager.state.get("custom_values", {}),
    }
    return {
        "name": "confirm_booking",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.confirm_booking_task(summary),
        "functions": [
            _tool("confirm", "Confirm and finalize the booking",
                  {}, [], handlers.confirm),
            _tool("change_service", "Go back and pick a different service",
                  {}, [], handlers.change_service),
            _tool("change_datetime", "Go back and pick a different date/time",
                  {}, [], handlers.change_datetime),
        ],
    }


def create_manage_booking_node(flow_manager: Any, intent: str) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    bookings = flow_manager.state.get("bookings", [])
    functions = [
        _tool("start_new_booking", "Start a new appointment booking",
              {}, [], handlers.start_new_booking),
        _tool("done", "End the call when the caller is satisfied",
              {}, [], handlers.done),
    ]

    if intent == "cancel":
        functions.insert(0, _tool(
            "cancel_booking", "Cancel a specific booking",
            {"booking_id": {"type": "string", "description": "The booking ID to cancel"}},
            ["booking_id"], handlers.cancel_booking))

    if intent == "reschedule":
        functions.insert(0, _tool(
            "start_reschedule", "Start rescheduling a specific booking",
            {"booking_id": {"type": "string", "description": "The booking ID to reschedule"}},
            ["booking_id"], handlers.start_reschedule))

    return {
        "name": "manage_booking",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.manage_booking_task(intent, bookings),
        "functions": functions,
    }


def create_callback_capture_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    caller_phone = flow_manager.state.get("caller_phone", "unknown")
    reason = flow_manager.state.get("fallback_reason")
    digits = " ".join(caller_phone.lstrip("+"))
    tts_text = f"Your number on file is {digits}."

    return {
        "name": "callback_capture",
        "role_message": prompts.build_role_message(config, language=language),
        "pre_actions": [{"type": "tts_say", "text": tts_text}],
        "task_messages": prompts.callback_capture_task(reason),
        "functions": [
            _tool("confirm_callback", "Confirm the callback phone number",
                  {"phone_number": {
                      "type": "string",
                      "description": "The phone number to call back on",
                  }},
                  ["phone_number"], handlers.confirm_callback),
        ],
    }


def create_close_node(flow_manager: Any) -> dict:
    config = flow_manager.state["config"]
    language = flow_manager.state.get("language", config.persona.fallback_language)
    return {
        "name": "close",
        "role_message": prompts.build_role_message(config, language=language),
        "task_messages": prompts.close_task(config),
        "post_actions": [{"type": "end_conversation"}],
    }
