from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


@pytest.fixture
def flow_state():
    config = make_tenant_config()
    return {
        "config": config,
        "data_adapter": MagicMock(),
        "checkpoint_store": InMemoryCheckpointStore(),
        "caller_phone": "+919876543210",
        "caller_id": "caller-1",
        "tenant_id": "t1",
        "turn_count": 0,
        "call_start": time.monotonic(),
        "intent": None,
    }


@pytest.fixture
def flow_manager(flow_state):
    fm = MagicMock()
    fm.state = flow_state
    return fm
