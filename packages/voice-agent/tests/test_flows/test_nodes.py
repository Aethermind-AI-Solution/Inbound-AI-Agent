from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from packages.voice_agent.tests.test_flows.conftest import *  # noqa: F401,F403
from packages.voice_agent.tests.test_states.conftest import make_tenant_config
from packages.voice_agent.config.models import LanguagePolicy, PersonaConfig, PipelineConfig


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


class TestNodeLanguageAwareness:
    def _make_flow_manager(self, language="en-IN", **extra_state):
        config = make_tenant_config(
            persona=PersonaConfig(
                business_name="Test Salon",
                greeting={"en-IN": "Welcome!", "hi-IN": "Swagat!"},
                ai_disclosure={"en-IN": "I'm AI.", "hi-IN": "Main AI hoon."},
                tone="warm",
                languages=["en-IN", "hi-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=True),
            ),
            pipeline=PipelineConfig(
                tts_voices={"en-IN": "aura-asteria-en", "hi-IN": "anushka"},
            ),
        )
        fm = MagicMock()
        fm.state = {
            "config": config,
            "language": language,
            **extra_state,
        }
        return fm

    def test_greeting_node_hindi_role_message(self):
        from packages.voice_agent.flows.nodes import create_greeting_node
        fm = self._make_flow_manager(language="hi-IN")
        node = create_greeting_node(fm)
        assert "role_message" in node
        assert "Hindi" in node["role_message"]

    def test_greeting_node_english_no_hindi(self):
        from packages.voice_agent.flows.nodes import create_greeting_node
        fm = self._make_flow_manager(language="en-IN")
        node = create_greeting_node(fm)
        assert "Hindi" not in node["role_message"]

    def test_collect_service_node_has_role_message(self):
        from packages.voice_agent.flows.nodes import create_collect_service_node
        fm = self._make_flow_manager(language="hi-IN")
        node = create_collect_service_node(fm)
        assert "role_message" in node
        assert "Hindi" in node["role_message"]

    def test_collect_service_node_english(self):
        from packages.voice_agent.flows.nodes import create_collect_service_node
        fm = self._make_flow_manager(language="en-IN")
        node = create_collect_service_node(fm)
        assert "role_message" in node
        assert "Hindi" not in node["role_message"]
