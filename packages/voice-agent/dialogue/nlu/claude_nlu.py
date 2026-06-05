from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from packages.voice_agent.dialogue.nlu.base import (
    IntentResult,
    NLUService,
    ServiceResult,
)

if TYPE_CHECKING:
    from packages.voice_agent.config.models import Service, TenantConfig
    from packages.voice_agent.dialogue.nlu.llm_client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_INTENTS = "new_booking, cancel, reschedule, status, unknown"

_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "sure", "confirm",
    "go ahead", "ok", "okay",
}
_NEGATIVE_WORDS = {"no", "nope", "nah", "don't", "stop"}


class ClaudeNLUService(NLUService):
    def __init__(self, client: LLMClient, tenant_config: TenantConfig) -> None:
        self._client = client
        self._tpl_intent = Template((_PROMPTS_DIR / "classify_intent.txt").read_text())
        self._tpl_service = Template((_PROMPTS_DIR / "extract_service.txt").read_text())
        self._tpl_yes_no = Template((_PROMPTS_DIR / "yes_no.txt").read_text())

        services_list = ", ".join(
            f"{s.id}:{s.name}"
            for s in tenant_config.booking_model.services
        )
        system_tpl = Template((_PROMPTS_DIR / "system.txt").read_text())
        self._system_prompt = system_tpl.safe_substitute(
            business_name=tenant_config.persona.business_name,
            sector=tenant_config.meta.sector,
            services_list=services_list,
        )
        self._last_yes_no: tuple[str, str] | None = None

    async def classify_intent(
        self, text: str, available_intents: list[str]
    ) -> IntentResult:
        intents_str = ", ".join(available_intents) if available_intents else _DEFAULT_INTENTS
        valid_intents = set(available_intents) if available_intents else {
            "new_booking", "cancel", "reschedule", "status", "unknown"
        }
        user_prompt = self._tpl_intent.safe_substitute(
            text=text, available_intents=intents_str
        )
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            intent = response.strip().lower()
            if intent in valid_intents and intent != "unknown":
                return IntentResult(intent=intent, confidence=0.9)
            return IntentResult(intent="unknown", confidence=0.0)
        except Exception:
            logger.exception("LLM error in classify_intent")
            return IntentResult(intent="unknown", confidence=0.0)

    async def extract_service(
        self, text: str, services: list[Service]
    ) -> ServiceResult:
        services_list = ", ".join(f"{s.id}: {s.name}" for s in services)
        valid_ids = {s.id for s in services}
        user_prompt = self._tpl_service.safe_substitute(
            text=text, services_list=services_list
        )
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            service_id = response.strip()
            if service_id in valid_ids:
                return ServiceResult(service_id=service_id, confidence=0.9)
            return ServiceResult(
                service_id=None, confidence=0.0,
                alternatives=[s.id for s in services[:3]],
            )
        except Exception:
            logger.exception("LLM error in extract_service")
            return ServiceResult(
                service_id=None, confidence=0.0,
                alternatives=[s.id for s in services[:3]],
            )

    async def is_affirmative(self, text: str) -> bool:
        lower = text.lower().strip()
        if lower in _AFFIRMATIVE_WORDS:
            return True
        if lower in _NEGATIVE_WORDS:
            return False
        return await self._classify_yes_no(text) == "yes"

    async def is_negative(self, text: str) -> bool:
        lower = text.lower().strip()
        if lower in _NEGATIVE_WORDS:
            return True
        if lower in _AFFIRMATIVE_WORDS:
            return False
        return await self._classify_yes_no(text) == "no"

    async def _classify_yes_no(self, text: str) -> str:
        if self._last_yes_no and self._last_yes_no[0] == text:
            return self._last_yes_no[1]
        user_prompt = self._tpl_yes_no.safe_substitute(text=text)
        try:
            response = await self._client.complete(self._system_prompt, user_prompt)
            result = response.strip().lower()
            if result not in ("yes", "no", "unclear"):
                result = "unclear"
            self._last_yes_no = (text, result)
            return result
        except Exception:
            logger.exception("LLM error in _classify_yes_no")
            self._last_yes_no = (text, "unclear")
            return "unclear"
