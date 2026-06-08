import pytest
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


class TestLanguageNames:
    def test_all_supported_languages_have_names(self):
        from packages.voice_agent.flows.prompts import LANGUAGE_NAMES
        expected = {"en-IN", "en-US", "en-GB", "en-AU", "hi-IN", "ta-IN", "te-IN", "mr-IN", "bn-IN"}
        assert expected.issubset(set(LANGUAGE_NAMES.keys()))

    def test_hindi_maps_to_hindi(self):
        from packages.voice_agent.flows.prompts import LANGUAGE_NAMES
        assert LANGUAGE_NAMES["hi-IN"] == "Hindi"

    def test_english_variants_map_to_english(self):
        from packages.voice_agent.flows.prompts import LANGUAGE_NAMES
        for code in ("en-IN", "en-US", "en-GB", "en-AU"):
            assert LANGUAGE_NAMES[code] == "English"


class TestBuildRoleMessageWithLanguage:
    def test_includes_language_instruction_for_hindi(self):
        from packages.voice_agent.flows.prompts import build_role_message
        config = make_tenant_config()
        msg = build_role_message(config, language="hi-IN")
        assert "Hindi" in msg
        assert "Respond in Hindi" in msg

    def test_english_does_not_add_extra_instruction(self):
        from packages.voice_agent.flows.prompts import build_role_message
        config = make_tenant_config()
        msg = build_role_message(config, language="en-IN")
        assert "Hindi" not in msg

    def test_uses_language_specific_disclosure(self):
        from packages.voice_agent.flows.prompts import build_role_message
        from packages.voice_agent.config.models import LanguagePolicy, PersonaConfig
        config = make_tenant_config(
            persona=PersonaConfig(
                business_name="Test Salon",
                greeting={"en-IN": "Welcome!", "hi-IN": "Swagat!"},
                ai_disclosure={"en-IN": "I'm an AI.", "hi-IN": "Main AI hoon."},
                tone="warm",
                languages=["en-IN", "hi-IN"],
                fallback_language="en-IN",
                language_policy=LanguagePolicy(greeting="default", match_caller=True),
            ),
        )
        msg = build_role_message(config, language="hi-IN")
        assert "Main AI hoon." in msg

    def test_defaults_to_fallback_language(self):
        from packages.voice_agent.flows.prompts import build_role_message
        config = make_tenant_config()
        msg_default = build_role_message(config)
        msg_explicit = build_role_message(config, language="en-IN")
        assert msg_default == msg_explicit


class TestRoleMessage:
    def test_contains_business_name(self):
        from packages.voice_agent.flows.prompts import build_role_message
        msg = build_role_message(make_tenant_config())
        assert "Glamour Salon" in msg

    def test_contains_ai_disclosure(self):
        from packages.voice_agent.flows.prompts import build_role_message
        msg = build_role_message(make_tenant_config())
        assert "AI" in msg

    def test_contains_anti_injection(self):
        from packages.voice_agent.flows.prompts import build_role_message
        msg = build_role_message(make_tenant_config())
        assert "ignore" in msg.lower()
        assert "reveal" in msg.lower() or "disclose" in msg.lower()

    def test_contains_voice_instructions(self):
        from packages.voice_agent.flows.prompts import build_role_message
        msg = build_role_message(make_tenant_config())
        assert "concise" in msg.lower() or "short" in msg.lower()


class TestTaskMessages:
    def test_greeting_task(self):
        from packages.voice_agent.flows.prompts import greeting_task
        msgs = greeting_task(make_tenant_config())
        assert len(msgs) >= 1
        assert "greet" in msgs[0]["content"].lower() or "welcome" in msgs[0]["content"].lower()

    def test_collect_service_lists_services(self):
        from packages.voice_agent.flows.prompts import collect_service_task
        msgs = collect_service_task(make_tenant_config())
        assert "Haircut" in msgs[0]["content"]
        assert "Hair Color" in msgs[0]["content"]

    def test_collect_datetime_includes_hours(self):
        from packages.voice_agent.flows.prompts import collect_datetime_task
        msgs = collect_datetime_task(make_tenant_config())
        assert "09:00" in msgs[0]["content"] or "9:00" in msgs[0]["content"]

    def test_callback_task_includes_reason(self):
        from packages.voice_agent.flows.prompts import callback_capture_task
        msgs = callback_capture_task("repeated_failure")
        assert "trouble" in msgs[0]["content"].lower() or "understand" in msgs[0]["content"].lower()

    def test_callback_task_does_not_contain_phone(self):
        from packages.voice_agent.flows.prompts import callback_capture_task
        msgs = callback_capture_task("repeated_failure")
        assert "9876" not in msgs[0]["content"]
