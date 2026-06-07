from __future__ import annotations

import pytest
from packages.voice_agent.tests.test_flows.conftest import *  # noqa: F401,F403


class TestGreetingNode:
    def test_has_name_and_role_message(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_greeting_node
        node = create_greeting_node(flow_manager)
        assert node["name"] == "greeting"
        assert "Glamour Salon" in node["role_message"]

    def test_has_anti_injection_in_role(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_greeting_node
        node = create_greeting_node(flow_manager)
        assert "ignore" in node["role_message"].lower()

    def test_has_intent_tools(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_greeting_node
        node = create_greeting_node(flow_manager)
        names = [f.name for f in node["functions"]]
        assert "start_new_booking" in names
        assert "request_callback" not in names  # request_callback is a global_function

    def test_respond_immediately(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_greeting_node
        node = create_greeting_node(flow_manager)
        assert node.get("respond_immediately", True) is True


class TestCollectServiceNode:
    def test_service_id_enum_constraint(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_collect_service_node
        node = create_collect_service_node(flow_manager)
        select_tool = next(f for f in node["functions"] if f.name == "select_service")
        assert "enum" in select_tool.properties["service_id"]
        assert "s1" in select_tool.properties["service_id"]["enum"]


class TestCallbackCaptureNode:
    def test_has_tts_pre_action_for_phone(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_callback_capture_node
        node = create_callback_capture_node(flow_manager)
        pre_types = [a["type"] for a in node.get("pre_actions", [])]
        assert "tts_say" in pre_types

    def test_phone_not_in_task_messages(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_callback_capture_node
        node = create_callback_capture_node(flow_manager)
        content = node["task_messages"][0]["content"]
        assert "9876" not in content


class TestCloseNode:
    def test_has_end_conversation(self, flow_manager):
        from packages.voice_agent.flows.nodes import create_close_node
        node = create_close_node(flow_manager)
        post_types = [a["type"] for a in node.get("post_actions", [])]
        assert "end_conversation" in post_types
