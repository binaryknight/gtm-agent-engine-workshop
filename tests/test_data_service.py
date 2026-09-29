from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile


def test_update_prospect_info_persists_and_refreshes_profile():
    prospect_id = "LEAD-90001"
    original_stack = list(data_service.PROSPECTS[prospect_id]["tech_stack"])
    original_profile = data_service._PROFILES.pop(prospect_id, None)
    try:
        build_prospect_profile.invoke({"prospect_id": prospect_id})
        result = data_service.update_prospect_info(prospect_id, "Okta")

        assert result["already_present"] is False
        assert "Okta" in data_service.fetch_tech_stack(prospect_id)
        assert "Okta" in build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]["tech_stack"]
    finally:
        data_service.PROSPECTS[prospect_id]["tech_stack"] = original_stack
        data_service._PROFILES.pop(prospect_id, None)
        if original_profile is not None:
            data_service._PROFILES[prospect_id] = original_profile
