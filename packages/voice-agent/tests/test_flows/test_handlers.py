from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from packages.voice_agent.data.adapter import HoldResult
from packages.voice_agent.tests.test_flows.conftest import *  # noqa: F401,F403


class TestStartNewBooking:
    @pytest.mark.asyncio
    async def test_sets_intent_and_transitions(self, flow_manager):
        from packages.voice_agent.flows.handlers import start_new_booking
        result, node = await start_new_booking({}, flow_manager)
        assert flow_manager.state["intent"] == "new_booking"
        assert node is not None
        assert node["name"] == "collect_service"

    @pytest.mark.asyncio
    async def test_writes_checkpoint(self, flow_manager):
        from packages.voice_agent.flows.handlers import start_new_booking
        _, node = await start_new_booking({}, flow_manager)
        store = flow_manager.state["checkpoint_store"]
        cp = await store.load("+919876543210")
        assert cp is not None
        assert cp["node"] == "collect_service"


class TestSelectService:
    @pytest.mark.asyncio
    async def test_valid_service(self, flow_manager):
        from packages.voice_agent.flows.handlers import select_service
        result, node = await select_service({"service_id": "s1"}, flow_manager)
        assert flow_manager.state["service_id"] == "s1"
        assert flow_manager.state["service_name"] == "Haircut"
        assert node["name"] == "collect_datetime"

    @pytest.mark.asyncio
    async def test_invalid_service_stays(self, flow_manager):
        from packages.voice_agent.flows.handlers import select_service
        result, node = await select_service({"service_id": "bad"}, flow_manager)
        assert node is None
        assert "error" in result


class TestCheckAvailability:
    @pytest.mark.asyncio
    async def test_past_date_rejected(self, flow_manager):
        from packages.voice_agent.flows.handlers import check_availability
        result, node = await check_availability({"date": "2020-01-01", "time": "10:00"}, flow_manager)
        assert node is None
        assert "past" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_closed_day_rejected(self, flow_manager):
        from packages.voice_agent.flows.handlers import check_availability
        result, node = await check_availability({"date": "2026-06-07", "time": "10:00"}, flow_manager)
        assert node is None
        assert "closed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_valid_date_transitions(self, flow_manager):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        from packages.voice_agent.flows.handlers import check_availability

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        days_ahead = (2 - now.weekday()) % 7 or 7  # next Wednesday
        future_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        flow_manager.state["service_id"] = "s1"
        flow_manager.state["service_name"] = "Haircut"
        flow_manager.state["service_duration_min"] = 30
        flow_manager.state["resource_type"] = "stylist"
        flow_manager.state["data_adapter"].check_availability = MagicMock(return_value=[
            {"resource_id": "r1", "resource_name": "Priya", "blocked_slots": []},
        ])
        result, node = await check_availability({"date": future_date, "time": "10:00"}, flow_manager)
        assert node is not None
        assert node["name"] == "offer_slots"


class TestConfirmCallback:
    @pytest.mark.asyncio
    async def test_transitions_to_close(self, flow_manager):
        from packages.voice_agent.flows.handlers import confirm_callback
        result, node = await confirm_callback({"phone_number": "+919876543210"}, flow_manager)
        assert node["name"] == "close"
        assert flow_manager.state["callback_number"] == "+919876543210"

    @pytest.mark.asyncio
    async def test_validates_phone_format(self, flow_manager):
        from packages.voice_agent.flows.handlers import confirm_callback
        result, node = await confirm_callback({"phone_number": "abc"}, flow_manager)
        assert node is None
        assert "error" in result


class TestRecordUsageSignature:
    @pytest.mark.asyncio
    async def test_record_usage_matches_protocol(self):
        from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
        tracker = InMemoryBudgetTracker()
        await tracker.record_usage("t1", 60.0)
