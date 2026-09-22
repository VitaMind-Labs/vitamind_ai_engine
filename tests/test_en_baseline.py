"""T2 regression: clinically rich EN session must not score 0.000 everywhere."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

RICH_MESSAGES = [
    "I have been distracted since school and I forget appointments all the time at work.",
    "My mind races from one idea to the next and I cannot finish what I start each day.",
    "I sleep only three hours some nights yet I wake up full of energy and start big projects.",
    "In the evening I crash and feel empty and worthless and nothing feels worth doing.",
    "When alone I sometimes feel people are watching me and I do not trust what is real.",
    "I spend money impulsively and take on too much, then feel exhausted for days after.",
    "My attention drifts constantly and I lose track of time when I finally focus on tasks.",
    "Overall this pattern repeats for months and it affects my job and my relationships.",
    "This has been happening on and off for years and impacts my family and daily functioning.",
    "I would like to understand what clinical follow-up is most appropriate for this pattern.",
]


def test_rich_session_produces_nonzero_report():
    sid = client.post("/api/v1/mira/session", json={"language": "en"}).json()["session_id"]
    payload = None
    for text in RICH_MESSAGES:
        payload = client.post(f"/api/v1/mira/session/{sid}/message", json={"text": text, "language": "en"}).json()
    assert payload["assessment_complete"] is True
    result = payload["result"]
    assert max(result["condition_scores"].values()) > 0, "0.000 false negative on rich content"
    assert result["requires_clinician_review"] is True
    assert len(result["supporting_features"]) > 0
