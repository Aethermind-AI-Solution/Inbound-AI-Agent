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
