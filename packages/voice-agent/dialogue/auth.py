from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.voice_agent.dialogue.models import CallContext


def check_auth(context: CallContext, intent: str) -> bool:
    policy = context.tenant_config.auth.action_policy
    required_level = policy.get(intent, "none")
    if required_level == "none":
        return True
    if required_level == "soft":
        if context.caller is None:
            return False
        return not context.caller.is_new
    return True
