# Regression Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest-based scenario runner that replays scripted conversations through the DialogueManager and asserts on state transitions, action types, slot values, and fallback reasons.

**Architecture:** Scenarios are YAML data files loaded by a `load_scenarios()` function into dataclasses. A single `run_scenario()` async function wires up `MockDataAdapter`, `InMemoryCheckpointStore`, `InMemoryBudgetTracker`, and `StubNLU`, then replays each turn through `DialogueManager.handle_event()` with assertions after every turn. Pytest parametrize creates one test per scenario.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio, PyYAML, existing DialogueManager + MockDataAdapter

---

## File Structure

```
packages/eval/
  __init__.py                  # empty
  runner.py                    # Scenario/Turn/ScenarioSetup dataclasses, load_scenarios(), run_scenario()
  conftest.py                  # pytest parametrize wiring
  test_scenarios.py            # single parametrized test function
  test_runner.py               # unit tests for loader + runner
  scenarios/
    booking_happy.yaml          # 3 scenarios: happy_path, custom_fields, edit_at_readback
    booking_fallbacks.yaml      # 2 scenarios: unknown_service, no_availability
    cancel_flows.yaml           # 3 scenarios: cancel_happy, cancel_no_bookings, status_check
    auth_gating.yaml            # 2 scenarios: auth_blocks, auth_allows
    budget_guard.yaml           # 1 scenario: budget_exceeded
    guardrails.yaml             # 3 scenarios: max_turns, repeated_silence, unrecognized_intents
```

**Modified files:**

- `pyproject.toml` — add `"packages/eval"` to `testpaths`

---

### Task 1: Install PyYAML, Create Dataclasses and YAML Loader

**Files:**
- Create: `packages/eval/__init__.py`
- Create: `packages/eval/runner.py`
- Create: `packages/eval/test_runner.py`
- Create: `packages/eval/scenarios/` (directory)

- [ ] **Step 1: Install PyYAML**

```bash
.venv/bin/pip install pyyaml
```

- [ ] **Step 2: Write the failing test for YAML loading**

Create `packages/eval/test_runner.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest packages/eval/test_runner.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'packages.eval'`

- [ ] **Step 4: Create packages/eval/__init__.py**

Create an empty `packages/eval/__init__.py`:

```python
```

- [ ] **Step 5: Write the dataclasses and loader**

Create `packages/eval/runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from packages.voice_agent.dialogue.models import ActionType, CallState


@dataclass
class Turn:
    user: str
    expect_state: str
    expect_action: str | None = None
    expect_intent: str | None = None
    expect_slot_service: str | None = None
    expect_text_contains: str | None = None
    expect_fallback_reason: str | None = None


@dataclass
class ScenarioSetup:
    budget_used_inr: float = 0.0
    caller_is_returning: bool = False
    action_policy: dict[str, str] | None = None
    force_no_availability: bool = False
    force_no_bookings: bool = False
    guardrails_max_turns: int | None = None
    add_custom_fields: list[dict[str, str]] | None = None


@dataclass
class Scenario:
    name: str
    turns: list[Turn]
    description: str = ""
    nlu: str = "stub"
    caller_phone: str = "+919876543210"
    setup: ScenarioSetup = field(default_factory=ScenarioSetup)
    expect_fallback_reason: str | None = None
    expect_min_turns: int | None = None
    expect_max_turns: int | None = None


_VALID_STATES = {s.value for s in CallState}
_VALID_ACTIONS = {a.value for a in ActionType}
_VALID_NLU = {"stub", "claude"}


def load_scenarios(scenario_dir: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen_names: set[str] = set()

    for yaml_path in sorted(scenario_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            docs = list(yaml.safe_load_all(f))

        for doc in docs:
            if doc is None:
                continue
            scenario = _parse_scenario(doc, yaml_path.name)
            if scenario.name in seen_names:
                raise ValueError(
                    f"Duplicate scenario name '{scenario.name}' in {yaml_path.name}"
                )
            seen_names.add(scenario.name)
            scenarios.append(scenario)

    return scenarios


def _parse_scenario(doc: dict, filename: str) -> Scenario:
    name = doc.get("name")
    if not name:
        raise ValueError(f"Scenario in {filename} missing required field 'name'")

    raw_turns = doc.get("turns")
    if not raw_turns:
        raise ValueError(f"Scenario '{name}' in {filename} missing or empty 'turns'")

    nlu = doc.get("nlu", "stub")
    if nlu not in _VALID_NLU:
        raise ValueError(
            f"Scenario '{name}': nlu must be 'stub' or 'claude', got '{nlu}'"
        )

    turns = []
    for i, t in enumerate(raw_turns):
        state = t.get("expect_state", "")
        if state not in _VALID_STATES:
            raise ValueError(
                f"Scenario '{name}' turn {i + 1}: "
                f"expect_state '{state}' is not a valid CallState"
            )
        action = t.get("expect_action")
        if action is not None and action not in _VALID_ACTIONS:
            raise ValueError(
                f"Scenario '{name}' turn {i + 1}: "
                f"expect_action '{action}' is not a valid ActionType"
            )
        turns.append(
            Turn(
                user=t["user"],
                expect_state=state,
                expect_action=action,
                expect_intent=t.get("expect_intent"),
                expect_slot_service=t.get("expect_slot_service"),
                expect_text_contains=t.get("expect_text_contains"),
                expect_fallback_reason=t.get("expect_fallback_reason"),
            )
        )

    setup_raw = doc.get("setup", {})
    setup = ScenarioSetup(
        budget_used_inr=setup_raw.get("budget_used_inr", 0.0),
        caller_is_returning=setup_raw.get("caller_is_returning", False),
        action_policy=setup_raw.get("action_policy"),
        force_no_availability=setup_raw.get("force_no_availability", False),
        force_no_bookings=setup_raw.get("force_no_bookings", False),
        guardrails_max_turns=setup_raw.get("guardrails_max_turns"),
        add_custom_fields=setup_raw.get("add_custom_fields"),
    )

    return Scenario(
        name=name,
        turns=turns,
        description=doc.get("description", ""),
        nlu=nlu,
        caller_phone=doc.get("caller_phone", "+919876543210"),
        setup=setup,
        expect_fallback_reason=doc.get("expect_fallback_reason"),
        expect_min_turns=doc.get("expect_min_turns"),
        expect_max_turns=doc.get("expect_max_turns"),
    )
```

- [ ] **Step 6: Create the scenarios directory**

```bash
mkdir -p "packages/eval/scenarios"
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest packages/eval/test_runner.py -v
```

Expected: All 15 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add packages/eval/__init__.py packages/eval/runner.py packages/eval/test_runner.py packages/eval/scenarios/
git commit -m "feat(eval): add scenario dataclasses and YAML loader"
```

---

### Task 2: Scenario Runner Function

**Files:**
- Modify: `packages/eval/runner.py`
- Modify: `packages/eval/test_runner.py`

- [ ] **Step 1: Write the failing test for run_scenario**

Add to `packages/eval/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest packages/eval/test_runner.py::TestRunScenario -v
```

Expected: FAIL — `ImportError: cannot import name 'ScenarioError' from 'packages.eval.runner'`

- [ ] **Step 3: Write the run_scenario function and ScenarioError**

Add to `packages/eval/runner.py`, after the existing code:

```python
from datetime import date, time

from packages.voice_agent.config.models import CustomField
from packages.voice_agent.data.mock_adapter import MockDataAdapter
from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
from packages.voice_agent.dialogue.manager import DialogueManager
from packages.voice_agent.dialogue.models import CallEvent, EventType
from packages.voice_agent.dialogue.nlu.stub import StubNLUService
from packages.voice_agent.resolver.date_resolver import DateResolver
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


class ScenarioError(AssertionError):
    pass


def _fixed_date_resolver(config):
    bm = config.booking_model
    return DateResolver(
        business_hours=bm.business_hours,
        booking_window_days=bm.booking_window_days,
        min_notice_min=bm.min_notice_min,
        reference_date=date(2026, 6, 1),  # Monday
        reference_time=time(9, 0),
    )


async def run_scenario(scenario: Scenario) -> None:
    config = make_tenant_config()
    data = MockDataAdapter()
    checkpoint = InMemoryCheckpointStore()
    budget = InMemoryBudgetTracker()

    # --- Apply setup overrides ---
    setup = scenario.setup

    if setup.action_policy is not None:
        config.auth.action_policy = setup.action_policy

    if setup.force_no_bookings:
        data._bookings = {}

    if setup.force_no_availability:
        data.check_availability = lambda *a, **kw: []

    if setup.guardrails_max_turns is not None:
        config.guardrails.max_turns = setup.guardrails_max_turns

    if setup.add_custom_fields:
        svc = config.booking_model.services[0]
        svc.custom_fields = [
            CustomField(**cf) for cf in setup.add_custom_fields
        ]

    if setup.budget_used_inr > 0:
        tenant_id = config.meta.tenant_id
        cost_per_min = config.guardrails.cost_per_minute_inr
        fake_seconds = (setup.budget_used_inr / cost_per_min) * 60
        await budget.record_usage(tenant_id, fake_seconds)

    if setup.caller_is_returning:
        data.resolve_or_create_caller(
            config.meta.tenant_id, scenario.caller_phone
        )

    # --- Select NLU ---
    nlu = StubNLUService()

    # --- Create DialogueManager ---
    dm = DialogueManager(
        config=config,
        data_adapter=data,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone=scenario.caller_phone,
        call_id=f"eval-{scenario.name}",
        budget_tracker=budget,
    )
    dm._make_date_resolver = _fixed_date_resolver

    # --- Run ---
    await dm.start()

    for i, turn in enumerate(scenario.turns):
        turn_label = f"Scenario '{scenario.name}' turn {i + 1}"

        if turn.user == "__SILENCE__":
            event = CallEvent(type=EventType.SILENCE)
        else:
            event = CallEvent(type=EventType.TRANSCRIPTION, text=turn.user)

        action = await dm.handle_event(event)

        actual_state = dm.current_state.name
        if actual_state != turn.expect_state:
            raise ScenarioError(
                f"{turn_label}:\n"
                f"  User said: \"{turn.user}\"\n"
                f"  Expected state: {turn.expect_state}\n"
                f"  Actual state:   {actual_state}"
            )

        if turn.expect_action is not None and action.type != turn.expect_action:
            raise ScenarioError(
                f"{turn_label}:\n"
                f"  User said: \"{turn.user}\"\n"
                f"  Expected action: {turn.expect_action}\n"
                f"  Actual action:   {action.type}"
            )

        if turn.expect_intent is not None and dm.context.intent != turn.expect_intent:
            raise ScenarioError(
                f"{turn_label}:\n"
                f"  User said: \"{turn.user}\"\n"
                f"  Expected intent: {turn.expect_intent}\n"
                f"  Actual intent:   {dm.context.intent}"
            )

        if turn.expect_slot_service is not None:
            actual_svc = dm.context.slots.service_name
            if actual_svc != turn.expect_slot_service:
                raise ScenarioError(
                    f"{turn_label}:\n"
                    f"  User said: \"{turn.user}\"\n"
                    f"  Expected service: {turn.expect_slot_service}\n"
                    f"  Actual service:   {actual_svc}"
                )

        if turn.expect_text_contains is not None:
            if action.text is None or turn.expect_text_contains not in action.text:
                raise ScenarioError(
                    f"{turn_label}:\n"
                    f"  User said: \"{turn.user}\"\n"
                    f"  Expected text containing: \"{turn.expect_text_contains}\"\n"
                    f"  Actual text: \"{action.text}\""
                )

        if turn.expect_fallback_reason is not None:
            actual_reason = dm.context.fallback_reason
            if actual_reason != turn.expect_fallback_reason:
                raise ScenarioError(
                    f"{turn_label}:\n"
                    f"  User said: \"{turn.user}\"\n"
                    f"  Expected fallback_reason: {turn.expect_fallback_reason}\n"
                    f"  Actual fallback_reason:   {actual_reason}"
                )

    # --- End-of-scenario assertions ---
    if scenario.expect_fallback_reason is not None:
        actual = dm.context.fallback_reason
        if actual != scenario.expect_fallback_reason:
            raise ScenarioError(
                f"Scenario '{scenario.name}' end:\n"
                f"  Expected fallback_reason: {scenario.expect_fallback_reason}\n"
                f"  Actual fallback_reason:   {actual}"
            )

    if scenario.expect_min_turns is not None:
        if dm.context.turn_count < scenario.expect_min_turns:
            raise ScenarioError(
                f"Scenario '{scenario.name}' end:\n"
                f"  Expected min turns: {scenario.expect_min_turns}\n"
                f"  Actual turn count:  {dm.context.turn_count}"
            )

    if scenario.expect_max_turns is not None:
        if dm.context.turn_count > scenario.expect_max_turns:
            raise ScenarioError(
                f"Scenario '{scenario.name}' end:\n"
                f"  Expected max turns: {scenario.expect_max_turns}\n"
                f"  Actual turn count:  {dm.context.turn_count}"
            )
```

Note: the imports at the top of the new section (`from datetime import date, time`, etc.) should be placed at the top of the file with the other imports. The `_fixed_date_resolver`, `ScenarioError`, and `run_scenario` go after the existing `_parse_scenario` function.

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest packages/eval/test_runner.py -v
```

Expected: All tests PASS (both TestLoadScenarios and TestRunScenario).

- [ ] **Step 5: Commit**

```bash
git add packages/eval/runner.py packages/eval/test_runner.py
git commit -m "feat(eval): add run_scenario function and ScenarioError"
```

---

### Task 3: Pytest Wiring and pyproject.toml

**Files:**
- Create: `packages/eval/conftest.py`
- Create: `packages/eval/test_scenarios.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create a minimal test scenario YAML for wiring verification**

Create `packages/eval/scenarios/booking_happy.yaml` with just one scenario for now (full scenarios in Task 4):

```yaml
name: happy_path_booking
description: Caller books a haircut with Priya on Tuesday at 10am
turns:
  - user: "I want to book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "haircut"
    expect_state: collect_datetime
    expect_slot_service: Haircut
  - user: "tomorrow at 10am"
    expect_state: offer_slots
  - user: "Priya"
    expect_state: read_back
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: null
expect_min_turns: 5
expect_max_turns: 5
```

- [ ] **Step 2: Write conftest.py**

Create `packages/eval/conftest.py`:

```python
import os
from pathlib import Path

import pytest

from packages.eval.runner import load_scenarios

_SCENARIO_DIR = Path(__file__).parent / "scenarios"


def pytest_collect_file(parent, file_path):
    if file_path.name == "test_scenarios.py":
        return None
    return None


def pytest_generate_tests(metafunc):
    if "scenario" not in metafunc.fixturenames:
        return

    scenarios = load_scenarios(_SCENARIO_DIR)

    use_claude = os.environ.get("EVAL_NLU") == "claude"
    filtered = []
    for s in scenarios:
        if s.nlu == "claude" and not use_claude:
            continue
        filtered.append(s)

    metafunc.parametrize(
        "scenario",
        filtered,
        ids=[s.name for s in filtered],
    )
```

- [ ] **Step 3: Write test_scenarios.py**

Create `packages/eval/test_scenarios.py`:

```python
import pytest

from packages.eval.runner import Scenario, run_scenario


@pytest.mark.asyncio
async def test_scenario(scenario: Scenario):
    await run_scenario(scenario)
```

- [ ] **Step 4: Update pyproject.toml testpaths**

In `pyproject.toml`, change:

```toml
testpaths = ["packages/voice-agent/tests"]
```

to:

```toml
testpaths = ["packages/voice-agent/tests", "packages/eval"]
```

- [ ] **Step 5: Run the scenario test to verify wiring works**

```bash
.venv/bin/python -m pytest packages/eval/test_scenarios.py -v
```

Expected: 1 test (`test_scenario[happy_path_booking]`) PASS.

- [ ] **Step 6: Verify existing tests still pass**

```bash
.venv/bin/python -m pytest packages/voice-agent/tests/ -v --tb=short
```

Expected: All existing tests still PASS (no regressions from pyproject.toml change).

- [ ] **Step 7: Commit**

```bash
git add packages/eval/conftest.py packages/eval/test_scenarios.py packages/eval/scenarios/booking_happy.yaml pyproject.toml
git commit -m "feat(eval): add pytest wiring and first scenario"
```

---

### Task 4: Booking Scenarios (5 scenarios)

**Files:**
- Modify: `packages/eval/scenarios/booking_happy.yaml`
- Create: `packages/eval/scenarios/booking_fallbacks.yaml`

The `happy_path_booking` scenario was created in Task 3. Now add the remaining 4 booking scenarios.

- [ ] **Step 1: Add custom_fields and edit_at_readback to booking_happy.yaml**

Replace `packages/eval/scenarios/booking_happy.yaml` with the full file containing 3 scenarios (multi-document YAML):

```yaml
name: happy_path_booking
description: Caller books a haircut with Priya on Tuesday at 10am
turns:
  - user: "I want to book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "haircut"
    expect_state: collect_datetime
    expect_slot_service: Haircut
  - user: "tomorrow at 10am"
    expect_state: offer_slots
  - user: "Priya"
    expect_state: read_back
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: null
expect_min_turns: 5
expect_max_turns: 5
---
name: booking_with_custom_fields
description: Booking flow visits collect_custom state when service has custom fields
setup:
  add_custom_fields:
    - key: "notes"
      type: "str"
      required: true
      prompt: "Any special requests for your haircut?"
turns:
  - user: "book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "haircut"
    expect_state: collect_datetime
    expect_slot_service: Haircut
  - user: "tomorrow at 10am"
    expect_state: offer_slots
  - user: "Priya"
    expect_state: collect_custom
  - user: "just a trim please"
    expect_state: read_back
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_min_turns: 6
expect_max_turns: 6
---
name: booking_edit_at_readback
description: Caller says no at readback, changes datetime, then confirms
turns:
  - user: "book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "haircut"
    expect_state: collect_datetime
    expect_slot_service: Haircut
  - user: "tomorrow at 10am"
    expect_state: offer_slots
  - user: "Priya"
    expect_state: read_back
  - user: "no"
    expect_state: read_back
  - user: "time"
    expect_state: collect_datetime
  - user: "tomorrow at 2pm"
    expect_state: offer_slots
  - user: "Priya"
    expect_state: read_back
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_min_turns: 9
expect_max_turns: 9
```

- [ ] **Step 2: Create booking_fallbacks.yaml**

Create `packages/eval/scenarios/booking_fallbacks.yaml`:

```yaml
name: booking_unknown_service
description: Two unrecognized services trigger callback
turns:
  - user: "book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "massage"
    expect_state: collect_service
  - user: "facial"
    expect_state: callback_capture
    expect_fallback_reason: repeated_failure
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: repeated_failure
---
name: booking_no_availability
description: No available slots at requested time triggers callback
setup:
  force_no_availability: true
turns:
  - user: "book an appointment"
    expect_state: collect_service
    expect_intent: new_booking
  - user: "haircut"
    expect_state: collect_datetime
    expect_slot_service: Haircut
  - user: "tomorrow at 10am"
    expect_state: callback_capture
    expect_fallback_reason: no_availability
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: no_availability
```

- [ ] **Step 3: Run all booking scenarios**

```bash
.venv/bin/python -m pytest packages/eval/test_scenarios.py -v
```

Expected: 5 scenarios PASS:
- `test_scenario[happy_path_booking]`
- `test_scenario[booking_with_custom_fields]`
- `test_scenario[booking_edit_at_readback]`
- `test_scenario[booking_unknown_service]`
- `test_scenario[booking_no_availability]`

- [ ] **Step 4: Commit**

```bash
git add packages/eval/scenarios/booking_happy.yaml packages/eval/scenarios/booking_fallbacks.yaml
git commit -m "feat(eval): add 5 booking flow scenarios"
```

---

### Task 5: Cancel and Status Scenarios (3 scenarios)

**Files:**
- Create: `packages/eval/scenarios/cancel_flows.yaml`

- [ ] **Step 1: Create cancel_flows.yaml**

Create `packages/eval/scenarios/cancel_flows.yaml`:

```yaml
name: cancel_happy_path
description: Returning caller cancels their existing booking
setup:
  caller_is_returning: true
turns:
  - user: "cancel my appointment"
    expect_state: confirm_cancel
    expect_intent: cancel
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_min_turns: 2
expect_max_turns: 2
---
name: cancel_no_bookings
description: Caller tries to cancel but has no bookings
setup:
  force_no_bookings: true
turns:
  - user: "cancel my booking"
    expect_state: close
    expect_action: end_call
    expect_intent: cancel
expect_min_turns: 1
expect_max_turns: 1
---
name: status_check
description: Returning caller checks booking status then exits
setup:
  caller_is_returning: true
turns:
  - user: "check my booking"
    expect_state: read_status
    expect_intent: status
  - user: "no"
    expect_state: close
    expect_action: end_call
expect_min_turns: 2
expect_max_turns: 2
```

- [ ] **Step 2: Run cancel/status scenarios**

```bash
.venv/bin/python -m pytest packages/eval/test_scenarios.py -v -k "cancel or status"
```

Expected: 3 scenarios PASS:
- `test_scenario[cancel_happy_path]`
- `test_scenario[cancel_no_bookings]`
- `test_scenario[status_check]`

- [ ] **Step 3: Run all scenarios to check for regressions**

```bash
.venv/bin/python -m pytest packages/eval/test_scenarios.py -v
```

Expected: 8 scenarios PASS.

- [ ] **Step 4: Commit**

```bash
git add packages/eval/scenarios/cancel_flows.yaml
git commit -m "feat(eval): add cancel and status check scenarios"
```

---

### Task 6: Auth, Budget, and Guardrail Scenarios (6 scenarios)

**Files:**
- Create: `packages/eval/scenarios/auth_gating.yaml`
- Create: `packages/eval/scenarios/budget_guard.yaml`
- Create: `packages/eval/scenarios/guardrails.yaml`

- [ ] **Step 1: Create auth_gating.yaml**

Create `packages/eval/scenarios/auth_gating.yaml`:

```yaml
name: auth_blocks_new_caller
description: New caller blocked from cancelling with soft auth after 2 attempts
setup:
  action_policy:
    cancel: "soft"
turns:
  - user: "cancel my appointment"
    expect_state: intent
  - user: "cancel my appointment"
    expect_state: callback_capture
    expect_fallback_reason: auth_required
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: auth_required
---
name: auth_allows_existing_caller
description: Returning caller with soft auth can cancel
setup:
  caller_is_returning: true
  action_policy:
    cancel: "soft"
turns:
  - user: "cancel my appointment"
    expect_state: confirm_cancel
    expect_intent: cancel
  - user: "yes"
    expect_state: close
    expect_action: end_call
```

- [ ] **Step 2: Create budget_guard.yaml**

Create `packages/eval/scenarios/budget_guard.yaml`:

```yaml
name: budget_exceeded
description: Budget pre-loaded above limit skips greeting and goes to callback
setup:
  budget_used_inr: 6000
turns:
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: budget_exceeded
```

- [ ] **Step 3: Create guardrails.yaml**

Create `packages/eval/scenarios/guardrails.yaml`:

```yaml
name: max_turns_exceeded
description: Guardrail fires when turn count exceeds max_turns
setup:
  guardrails_max_turns: 2
turns:
  - user: "book"
    expect_state: collect_service
  - user: "hmm"
    expect_state: collect_service
  - user: "uh"
    expect_state: callback_capture
    expect_fallback_reason: max_turns_exceeded
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: max_turns_exceeded
---
name: repeated_silence
description: Two consecutive silence events trigger callback
turns:
  - user: "__SILENCE__"
    expect_state: intent
  - user: "__SILENCE__"
    expect_state: callback_capture
    expect_fallback_reason: repeated_silence
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: repeated_silence
---
name: unrecognized_intents
description: Three gibberish inputs at intent trigger callback
turns:
  - user: "blah blah"
    expect_state: intent
  - user: "asdfg"
    expect_state: intent
  - user: "xyz123"
    expect_state: callback_capture
    expect_fallback_reason: repeated_failure
  - user: "yes"
    expect_state: close
    expect_action: end_call
expect_fallback_reason: repeated_failure
```

- [ ] **Step 4: Run all 14 scenarios**

```bash
.venv/bin/python -m pytest packages/eval/test_scenarios.py -v
```

Expected: 14 scenarios PASS:
- `test_scenario[happy_path_booking]`
- `test_scenario[booking_with_custom_fields]`
- `test_scenario[booking_edit_at_readback]`
- `test_scenario[booking_unknown_service]`
- `test_scenario[booking_no_availability]`
- `test_scenario[cancel_happy_path]`
- `test_scenario[cancel_no_bookings]`
- `test_scenario[status_check]`
- `test_scenario[auth_blocks_new_caller]`
- `test_scenario[auth_allows_existing_caller]`
- `test_scenario[budget_exceeded]`
- `test_scenario[max_turns_exceeded]`
- `test_scenario[repeated_silence]`
- `test_scenario[unrecognized_intents]`

- [ ] **Step 5: Run full test suite (existing + eval)**

```bash
.venv/bin/python -m pytest -v --tb=short
```

Expected: All tests pass — existing voice-agent tests + 14 scenario tests + runner unit tests.

- [ ] **Step 6: Commit**

```bash
git add packages/eval/scenarios/auth_gating.yaml packages/eval/scenarios/budget_guard.yaml packages/eval/scenarios/guardrails.yaml
git commit -m "feat(eval): add auth, budget, and guardrail scenarios (14 total)"
```

- [ ] **Step 7: Update BUILD.md — mark Step 7 complete**

In `BUILD.md`, change Step 7 from `☐` to `☑`.

```bash
git add BUILD.md
git commit -m "docs: mark Step 7 (regression harness) complete in BUILD.md"
```
