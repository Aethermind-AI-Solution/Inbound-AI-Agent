import pytest

from packages.eval.runner import Scenario, run_scenario


@pytest.mark.asyncio
async def test_scenario(scenario: Scenario):
    await run_scenario(scenario)
