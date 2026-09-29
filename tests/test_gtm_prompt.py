import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "gtm_agent" / "gtm_agent.py"


def system_prompt():
    tree = ast.parse(SOURCE.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
        for target in node.targets
    ))
    return ast.literal_eval(assignment.value)


def test_scoring_does_not_require_rep_lookup_but_email_does():
    prompt = system_prompt()

    assert "As a first step in any request" not in prompt
    assert "Before any action attributed to or sent on behalf of the rep" in prompt
    assert "especially before calling send_prospect_email" in prompt
    assert "score prospects" in prompt
    assert "send emails to prospects" in prompt
