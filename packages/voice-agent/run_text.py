"""
Text-based voice agent — type into the terminal, see agent responses.

Bypasses all audio (STT/TTS/VAD) and exercises the full dialogue state machine
+ NLU directly. Uses Claude Haiku NLU if ANTHROPIC_API_KEY is set.

Usage:
  PYTHONPATH=. .venv/bin/python packages/voice-agent/run_text.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


async def main() -> None:
    from packages.voice_agent.data.mock_adapter import MockDataAdapter
    from packages.voice_agent.dialogue.budget.memory_tracker import InMemoryBudgetTracker
    from packages.voice_agent.dialogue.checkpoint.memory import InMemoryCheckpointStore
    from packages.voice_agent.dialogue.manager import DialogueManager
    from packages.voice_agent.dialogue.models import (
        ActionType,
        CallEvent,
        CallState,
        EventType,
    )
    from packages.voice_agent.tests.test_states.conftest import make_tenant_config

    config = make_tenant_config()
    data = MockDataAdapter()
    checkpoint = InMemoryCheckpointStore()
    budget = InMemoryBudgetTracker()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from packages.voice_agent.dialogue.nlu.claude_nlu import ClaudeNLUService
        from packages.voice_agent.dialogue.nlu.llm_client import AnthropicLLMClient
        llm_client = AnthropicLLMClient(api_key=anthropic_key)
        nlu = ClaudeNLUService(llm_client, config)
        print("NLU: Claude Haiku (natural language)")
    else:
        from packages.voice_agent.dialogue.nlu.stub import StubNLUService
        nlu = StubNLUService()
        print("NLU: StubNLU (keywords only — set ANTHROPIC_API_KEY for natural language)")

    dm = DialogueManager(
        config=config,
        data_adapter=data,
        nlu=nlu,
        checkpoint_store=checkpoint,
        caller_phone="+919876543210",
        call_id="text-test-001",
        budget_tracker=budget,
    )

    print()
    print("=" * 55)
    print("  VOICE AGENT — text mode")
    print("  Type what you'd say. Ctrl+C or 'quit' to exit.")
    print("=" * 55)
    print()

    action = await dm.start()
    print(f"  Agent [{dm.current_state.name}]: {action.text}")
    print()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        event = CallEvent(type=EventType.TRANSCRIPTION, text=user_input)
        action = await dm.handle_event(event)

        print(f"  Agent [{dm.current_state.name}]: {action.text}")
        print()

        if action.type == ActionType.END_CALL:
            break

    elapsed = time.monotonic() - dm.context.call_start
    await dm.record_call_usage(elapsed)
    print(f"  [Session ended — {dm.context.turn_count} turns, {elapsed:.1f}s]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
