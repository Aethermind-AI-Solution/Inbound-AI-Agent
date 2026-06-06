import textwrap
from pathlib import Path

import pytest
import yaml

from packages.eval.runner import (
    Scenario,
    ScenarioSetup,
    Turn,
    load_scenarios,
)


class TestTurnDataclass:
    def test_required_fields(self):
        t = Turn(user="hello", expect_state="intent")
        assert t.user == "hello"
        assert t.expect_state == "intent"
        assert t.expect_action is None
        assert t.expect_intent is None
        assert t.expect_slot_service is None
        assert t.expect_text_contains is None
        assert t.expect_fallback_reason is None

    def test_all_fields(self):
        t = Turn(
            user="book",
            expect_state="collect_service",
            expect_action="ask",
            expect_intent="new_booking",
            expect_slot_service="Haircut",
            expect_text_contains="service",
            expect_fallback_reason=None,
        )
        assert t.expect_action == "ask"
        assert t.expect_intent == "new_booking"


class TestScenarioSetupDataclass:
    def test_defaults(self):
        s = ScenarioSetup()
        assert s.budget_used_inr == 0.0
        assert s.caller_is_returning is False
        assert s.action_policy is None
        assert s.force_no_availability is False
        assert s.force_no_bookings is False
        assert s.guardrails_max_turns is None
        assert s.add_custom_fields is None


class TestScenarioDataclass:
    def test_defaults(self):
        s = Scenario(
            name="test",
            turns=[Turn(user="hi", expect_state="intent")],
        )
        assert s.description == ""
        assert s.nlu == "stub"
        assert s.caller_phone == "+919876543210"
        assert isinstance(s.setup, ScenarioSetup)
        assert s.expect_fallback_reason is None
        assert s.expect_min_turns is None
        assert s.expect_max_turns is None


class TestLoadScenarios:
    def test_loads_single_scenario(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: simple_test
            description: A simple test
            turns:
              - user: "hello"
                expect_state: intent
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "test.yaml").write_text(yaml_content)

        scenarios = load_scenarios(scenario_dir)
        assert len(scenarios) == 1
        assert scenarios[0].name == "simple_test"
        assert scenarios[0].description == "A simple test"
        assert len(scenarios[0].turns) == 1
        assert scenarios[0].turns[0].user == "hello"

    def test_loads_multi_document_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: first
            turns:
              - user: "a"
                expect_state: intent
            ---
            name: second
            turns:
              - user: "b"
                expect_state: collect_service
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "multi.yaml").write_text(yaml_content)

        scenarios = load_scenarios(scenario_dir)
        assert len(scenarios) == 2
        names = {s.name for s in scenarios}
        assert names == {"first", "second"}

    def test_loads_setup_fields(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: with_setup
            setup:
              budget_used_inr: 1000.0
              caller_is_returning: true
              action_policy:
                cancel: "soft"
              force_no_availability: true
              guardrails_max_turns: 3
            turns:
              - user: "hi"
                expect_state: intent
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "setup.yaml").write_text(yaml_content)

        scenarios = load_scenarios(scenario_dir)
        s = scenarios[0]
        assert s.setup.budget_used_inr == 1000.0
        assert s.setup.caller_is_returning is True
        assert s.setup.action_policy == {"cancel": "soft"}
        assert s.setup.force_no_availability is True
        assert s.setup.guardrails_max_turns == 3

    def test_validates_missing_name(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            turns:
              - user: "hi"
                expect_state: intent
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="name"):
            load_scenarios(scenario_dir)

    def test_validates_missing_turns(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: no_turns
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="turns"):
            load_scenarios(scenario_dir)

    def test_validates_empty_turns(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: empty_turns
            turns: []
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="turns"):
            load_scenarios(scenario_dir)

    def test_validates_invalid_state(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: bad_state
            turns:
              - user: "hi"
                expect_state: nonexistent_state
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="expect_state"):
            load_scenarios(scenario_dir)

    def test_validates_invalid_action(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: bad_action
            turns:
              - user: "hi"
                expect_state: intent
                expect_action: fly
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="expect_action"):
            load_scenarios(scenario_dir)

    def test_validates_invalid_nlu(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: bad_nlu
            nlu: gpt4
            turns:
              - user: "hi"
                expect_state: intent
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "bad.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="nlu"):
            load_scenarios(scenario_dir)

    def test_validates_duplicate_names(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: duplicate
            turns:
              - user: "a"
                expect_state: intent
            ---
            name: duplicate
            turns:
              - user: "b"
                expect_state: intent
        """)
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "dup.yaml").write_text(yaml_content)

        with pytest.raises(ValueError, match="uplicate"):
            load_scenarios(scenario_dir)

    def test_loads_from_multiple_files(self, tmp_path):
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "a.yaml").write_text(
            "name: from_a\nturns:\n  - user: hi\n    expect_state: intent\n"
        )
        (scenario_dir / "b.yaml").write_text(
            "name: from_b\nturns:\n  - user: hi\n    expect_state: intent\n"
        )

        scenarios = load_scenarios(scenario_dir)
        assert len(scenarios) == 2
        names = {s.name for s in scenarios}
        assert names == {"from_a", "from_b"}

    def test_ignores_non_yaml_files(self, tmp_path):
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        (scenario_dir / "readme.md").write_text("# Not a scenario")
        (scenario_dir / "real.yaml").write_text(
            "name: real\nturns:\n  - user: hi\n    expect_state: intent\n"
        )

        scenarios = load_scenarios(scenario_dir)
        assert len(scenarios) == 1


from packages.eval.runner import ScenarioError, run_scenario


class TestRunScenario:
    @pytest.mark.asyncio
    async def test_happy_path_minimal(self):
        """Minimal 1-turn scenario: intent detection."""
        scenario = Scenario(
            name="test_minimal",
            turns=[
                Turn(
                    user="I want to book an appointment",
                    expect_state="collect_service",
                    expect_intent="new_booking",
                ),
            ],
        )
        await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_silence_sentinel(self):
        """__SILENCE__ sends a SILENCE event, not a transcription."""
        scenario = Scenario(
            name="test_silence",
            turns=[
                Turn(user="__SILENCE__", expect_state="intent"),
            ],
        )
        await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_wrong_state_raises_scenario_error(self):
        scenario = Scenario(
            name="test_wrong_state",
            turns=[
                Turn(
                    user="I want to book",
                    expect_state="close",  # wrong — should be collect_service
                ),
            ],
        )
        with pytest.raises(ScenarioError, match="test_wrong_state"):
            await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_wrong_intent_raises_scenario_error(self):
        scenario = Scenario(
            name="test_wrong_intent",
            turns=[
                Turn(
                    user="I want to book",
                    expect_state="collect_service",
                    expect_intent="cancel",  # wrong
                ),
            ],
        )
        with pytest.raises(ScenarioError, match="intent"):
            await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_end_of_scenario_fallback_reason(self):
        """3 gibberish turns → callback → confirm callback → close."""
        scenario = Scenario(
            name="test_fallback_check",
            turns=[
                Turn(user="xyzzy", expect_state="intent"),
                Turn(user="qwerty", expect_state="intent"),
                Turn(user="asdfgh", expect_state="callback_capture"),
                Turn(user="yes", expect_state="close"),
            ],
            expect_fallback_reason="repeated_failure",
        )
        await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_wrong_end_fallback_raises(self):
        scenario = Scenario(
            name="test_wrong_fallback",
            turns=[
                Turn(user="xyzzy", expect_state="intent"),
                Turn(user="qwerty", expect_state="intent"),
                Turn(user="asdfgh", expect_state="callback_capture"),
                Turn(user="yes", expect_state="close"),
            ],
            expect_fallback_reason="budget_exceeded",  # wrong
        )
        with pytest.raises(ScenarioError, match="fallback_reason"):
            await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_turn_count_bounds(self):
        scenario = Scenario(
            name="test_turn_count",
            turns=[
                Turn(user="xyzzy", expect_state="intent"),
                Turn(user="qwerty", expect_state="intent"),
                Turn(user="asdfgh", expect_state="callback_capture"),
                Turn(user="yes", expect_state="close"),
            ],
            expect_min_turns=4,
            expect_max_turns=4,
        )
        await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_setup_budget_exceeded(self):
        scenario = Scenario(
            name="test_budget",
            setup=ScenarioSetup(budget_used_inr=6000.0),
            turns=[
                Turn(user="yes", expect_state="close"),
            ],
            expect_fallback_reason="budget_exceeded",
        )
        await run_scenario(scenario)

    @pytest.mark.asyncio
    async def test_setup_caller_is_returning(self):
        """Returning caller can cancel (with soft auth on cancel)."""
        scenario = Scenario(
            name="test_returning",
            setup=ScenarioSetup(
                caller_is_returning=True,
                action_policy={"cancel": "soft"},
            ),
            turns=[
                Turn(
                    user="cancel my appointment",
                    expect_state="confirm_cancel",
                    expect_intent="cancel",
                ),
            ],
        )
        await run_scenario(scenario)
