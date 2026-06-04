import pytest

from packages.voice_agent.config.models import CustomField, Service
from packages.voice_agent.dialogue.nlu.base import IntentResult, ServiceResult
from packages.voice_agent.dialogue.nlu.stub import StubNLUService


@pytest.fixture
def nlu():
    return StubNLUService()


@pytest.fixture
def services():
    return [
        Service(
            id="s1", name="Haircut", resource_type="stylist",
            duration_min=30, booking_mode="exclusive", custom_fields=[],
        ),
        Service(
            id="s2", name="Hair Color", resource_type="stylist",
            duration_min=60, booking_mode="exclusive", custom_fields=[],
        ),
        Service(
            id="s3", name="Facial", resource_type="beautician",
            duration_min=45, booking_mode="exclusive", custom_fields=[],
        ),
    ]


class TestClassifyIntent:
    @pytest.mark.asyncio
    async def test_book_keyword(self, nlu):
        result = await nlu.classify_intent("I want to book an appointment", [])
        assert result.intent == "new_booking"

    @pytest.mark.asyncio
    async def test_cancel_keyword(self, nlu):
        result = await nlu.classify_intent("cancel my appointment", [])
        assert result.intent == "cancel"

    @pytest.mark.asyncio
    async def test_reschedule_keyword(self, nlu):
        result = await nlu.classify_intent("I need to reschedule", [])
        assert result.intent == "reschedule"

    @pytest.mark.asyncio
    async def test_status_keyword(self, nlu):
        result = await nlu.classify_intent("check my booking status", [])
        assert result.intent == "status"

    @pytest.mark.asyncio
    async def test_unknown_input(self, nlu):
        result = await nlu.classify_intent("the weather is nice", [])
        assert result.intent == "unknown"

    @pytest.mark.asyncio
    async def test_explicit_mapping(self, nlu):
        nlu.intent_map["custom phrase"] = "new_booking"
        result = await nlu.classify_intent("custom phrase", [])
        assert result.intent == "new_booking"
        assert result.confidence == 1.0


class TestExtractService:
    @pytest.mark.asyncio
    async def test_exact_match(self, nlu, services):
        result = await nlu.extract_service("I want a haircut", services)
        assert result.service_id == "s1"

    @pytest.mark.asyncio
    async def test_no_match(self, nlu, services):
        result = await nlu.extract_service("something random", services)
        assert result.service_id is None
        assert len(result.alternatives) > 0

    @pytest.mark.asyncio
    async def test_explicit_mapping(self, nlu, services):
        nlu.service_map["my service"] = "s2"
        result = await nlu.extract_service("my service", services)
        assert result.service_id == "s2"


class TestAffirmativeNegative:
    @pytest.mark.asyncio
    async def test_yes(self, nlu):
        assert await nlu.is_affirmative("yes") is True
        assert await nlu.is_affirmative("Yeah") is True
        assert await nlu.is_affirmative("sure") is True

    @pytest.mark.asyncio
    async def test_no(self, nlu):
        assert await nlu.is_negative("no") is True
        assert await nlu.is_negative("Nope") is True

    @pytest.mark.asyncio
    async def test_not_affirmative(self, nlu):
        assert await nlu.is_affirmative("I think so") is False

    @pytest.mark.asyncio
    async def test_not_negative(self, nlu):
        assert await nlu.is_negative("maybe") is False
