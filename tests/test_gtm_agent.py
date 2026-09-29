import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_blocks_disqualified_prospect_from_authoritative_record(self):
        result = send_prospect_email.func(
            {
                "prospect_id": "LEAD-50001",
                "name": "Priya Nair",
                "email": "priya.nair@brightwaveapps.com",
                "disqualified": False,
            },
            "Follow-up",
            "Hello",
            None,
            {"name": "Rep", "email": "rep@example.com"},
        )

        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("message_id", result)

    def test_sends_qualified_prospect(self):
        result = send_prospect_email.func(
            {
                "prospect_id": "LEAD-70001",
                "name": "Marcus Chen",
                "email": "marcus.chen@northstarhealth.com",
            },
            "Follow-up",
            "Hello",
            None,
            {"name": "Rep", "email": "rep@example.com"},
        )

        self.assertEqual(result["status"], "sent")


if __name__ == "__main__":
    unittest.main()
