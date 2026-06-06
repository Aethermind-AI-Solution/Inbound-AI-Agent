import os
from pathlib import Path

from packages.eval.runner import load_scenarios

_SCENARIO_DIR = Path(__file__).parent / "scenarios"


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
