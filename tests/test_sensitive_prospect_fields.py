import json
import os
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import (
    build_prospect_profile,
    get_prospect,
    score_prospect,
)


PROSPECT_ID = "LEAD-12853"
SENSITIVE_PROFILE = {
    "prospect_id": PROSPECT_ID,
    "name": "Test Prospect",
    "billing_qualification": {"tax_id": "123-45-6789"},
    "annual_revenue": 1_000_000,
    "tech_stack": ["Python"],
}


def test_get_prospect_excludes_sensitive_fields():
    result = get_prospect.invoke({"prospect_id": PROSPECT_ID})

    assert "billing_qualification" not in json.dumps(result)


def test_build_prospect_profile_excludes_sensitive_fields_and_caches_sanitized_profile():
    data_service._PROFILES.pop(PROSPECT_ID, None)

    result = build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

    assert "billing_qualification" not in json.dumps(result)
    assert "billing_qualification" not in json.dumps(data_service._PROFILES[PROSPECT_ID])


def test_build_prospect_profile_repairs_cached_sensitive_fields():
    data_service._PROFILES[PROSPECT_ID] = dict(SENSITIVE_PROFILE)

    result = build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

    assert "billing_qualification" not in json.dumps(result)
    assert "billing_qualification" not in json.dumps(data_service._PROFILES[PROSPECT_ID])


def test_score_prospect_excludes_sensitive_fields_from_user_message():
    captured = {}

    class FakeScoringLlm:
        def invoke(self, messages):
            captured["user"] = messages[1]["content"]
            return type("Result", (), {"model_dump": lambda self: {"score": 1}})()

    with patch("gtm_agent.gtm_agent._scoring_llm", FakeScoringLlm()):
        score_prospect.invoke({
            "prospect_profile": SENSITIVE_PROFILE,
            "offering": {
                "required_tech_stack": ["Python"],
                "min_annual_revenue": 1,
                "description": "Test offering",
            },
        })

    assert "billing_qualification" not in captured["user"]
