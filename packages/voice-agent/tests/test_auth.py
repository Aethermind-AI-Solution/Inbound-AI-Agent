from __future__ import annotations

import pytest

from packages.voice_agent.data.adapter import CallerInfo
from packages.voice_agent.dialogue.models import CallContext
from packages.voice_agent.tests.test_states.conftest import make_tenant_config


def make_context(is_new: bool, action_policy: dict[str, str] | None = None) -> CallContext:
    overrides = {}
    if action_policy is not None:
        from packages.voice_agent.config.models import AuthConfig
        overrides["auth"] = AuthConfig(
            levels=["soft"],
            action_policy=action_policy,
            soft_match_fields=["caller_number"],
        )
    ctx = CallContext(
        tenant_config=make_tenant_config(**overrides),
        caller_phone="+919876543210",
        call_id="call-001",
    )
    ctx.caller = CallerInfo(
        id="c1", phone="+919876543210", tenant_id="t1",
        verified_at=None, is_new=is_new,
    )
    return ctx


class TestCheckAuth:
    def test_action_policy_none_always_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"new_booking": "none"})
        assert check_auth(ctx, "new_booking") is True

    def test_soft_with_existing_caller_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=False, action_policy={"cancel": "soft"})
        assert check_auth(ctx, "cancel") is True

    def test_soft_with_new_caller_not_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"cancel": "soft"})
        assert check_auth(ctx, "cancel") is False

    def test_intent_not_in_policy_defaults_to_authorized(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"new_booking": "none"})
        assert check_auth(ctx, "status") is True

    def test_no_caller_defaults_to_not_authorized_for_soft(self):
        from packages.voice_agent.dialogue.auth import check_auth
        ctx = make_context(is_new=True, action_policy={"cancel": "soft"})
        ctx.caller = None
        assert check_auth(ctx, "cancel") is False

    def test_new_booking_none_cancel_soft_combo(self):
        from packages.voice_agent.dialogue.auth import check_auth
        policy = {"new_booking": "none", "cancel": "soft", "status": "soft"}
        ctx_new = make_context(is_new=True, action_policy=policy)
        ctx_existing = make_context(is_new=False, action_policy=policy)
        assert check_auth(ctx_new, "new_booking") is True
        assert check_auth(ctx_new, "cancel") is False
        assert check_auth(ctx_existing, "cancel") is True
        assert check_auth(ctx_existing, "status") is True
