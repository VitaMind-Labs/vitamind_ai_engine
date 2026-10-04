"""Safety continuity across conversations.

A patient can start a fresh chat thread at any time. The backend hands the agent
the most serious safety moment of the last day (`recent_safety`, structure only)
so a new thread follows up a crisis instead of starting as if it never happened.
Like a journal context, it escalates a turn but never lowers one, never reaches
CRISIS on its own, and stops repeating once the follow-up happened.
"""
from __future__ import annotations

import pytest

from lumina.orchestrator import Lumina
from lumina.safety import recent_safety_level

from test_service import client, post  # noqa: F401  (fixture re-export)

CRISIS_EARLIER = {"level": "CRISIS", "hours_ago": 2, "follow_up_done": False}
NEUTRAL = "How do I get through this evening?"


@pytest.fixture(scope="module")
def lumina():
    return Lumina.load()


def test_an_unanswered_crisis_opens_the_next_turn_with_a_safety_check():
    assert recent_safety_level(CRISIS_EARLIER) == "HIGH"


def test_after_the_follow_up_or_a_high_turn_it_only_holds_elevated():
    assert recent_safety_level({**CRISIS_EARLIER, "follow_up_done": True}) == "ELEVATED"
    assert recent_safety_level({**CRISIS_EARLIER, "hours_ago": 13}) == "ELEVATED"
    assert recent_safety_level({"level": "HIGH", "hours_ago": 1, "follow_up_done": True}) == "ELEVATED"


def test_it_fades_after_the_window_and_never_reaches_crisis():
    assert recent_safety_level({**CRISIS_EARLIER, "hours_ago": 25}) == "NORMAL"
    assert recent_safety_level(None) == "NORMAL"
    assert recent_safety_level({"level": "CRISIS", "hours_ago": 0}) != "CRISIS"


def test_a_new_thread_after_a_crisis_opens_with_a_safety_check(lumina):
    without = lumina.turn(text=NEUTRAL, track="ADHD")
    after = lumina.turn(text=NEUTRAL, track="ADHD", recent_safety=CRISIS_EARLIER)
    assert without["decision"]["type"] != "ELEVATED_SAFETY_WORKFLOW"
    assert after["decision"]["type"] == "ELEVATED_SAFETY_WORKFLOW"
    assert after["safety"]["decided_by"] == "recent_safety"
    # The backend uses this code to tell an echo from new evidence.
    assert "RECENT_SAFETY" in after["decision"]["reason_codes"]
    assert after["decision"]["requires_human_review"] is True
    assert set(after["safety"]["recent_safety"]) == {"level", "hours_ago"}


def test_once_followed_up_the_same_crisis_stops_repeating_the_check(lumina):
    out = lumina.turn(text=NEUTRAL, track="ADHD",
                      recent_safety={**CRISIS_EARLIER, "follow_up_done": True})
    assert out["decision"]["type"] != "ELEVATED_SAFETY_WORKFLOW"
    assert out["safety"]["level"] == "ELEVATED"


def test_recent_safety_never_lowers_what_the_message_raised(lumina):
    out = lumina.turn(text="I want to kill myself", track="ADHD",
                      recent_safety={"level": "HIGH", "hours_ago": 1, "follow_up_done": True})
    assert out["safety"]["level"] == "CRISIS"
    assert out["decision"]["type"] == "CRISIS_WORKFLOW"


def test_the_service_accepts_recent_safety_and_rejects_unknown_keys(client):  # noqa: F811
    body = {"request_id": "rs1", "patient_id": "p1", "track": "ADHD", "text": NEUTRAL,
            "recent_safety": dict(CRISIS_EARLIER)}
    out = post(client, "/api/v1/lumina/chat", body).json()
    assert out["decision"]["type"] == "ELEVATED_SAFETY_WORKFLOW"
    body["recent_safety"]["message"] = "raw text"
    assert post(client, "/api/v1/lumina/chat", body).status_code == 422
    body["recent_safety"] = {"level": "NORMAL", "hours_ago": 1}
    assert post(client, "/api/v1/lumina/chat", body).status_code == 422


# --- known defect: the head over-reads executive-function overload -----------
# Recorded rather than silenced. The safety head scores ADHD task-overload language
# as HIGH ("I have so much to do and I cannot start any of it" -> 0.864), because its
# corpus is Reddit depression and SuicideWatch text that contains nothing phrased
# that way. Two mitigations were tried and rejected - see lumina/safety.py for the
# measurements and for why a confidence floor and a lexicon-gated pattern are both
# unsafe here. The fix is labelled supervision for this language, with clinician
# review of every HIGH row.
#
# xfail(strict=True) so that whoever fixes the corpus is told to delete this block:
# the test starts failing as soon as the behaviour is correct.
OVERLOAD_MISREAD_AS_DISTRESS = [
    "I have ten things to do and I am doing none of them",
    "I have so much to do and I cannot start any of it",
    "everything is piling up and I keep putting it off",
]


@pytest.mark.xfail(strict=True, reason="known defect: safety head reads ADHD "
                                       "task-overload as HIGH distress; needs "
                                       "labelled supervision, not a threshold")
@pytest.mark.parametrize("text", OVERLOAD_MISREAD_AS_DISTRESS)
def test_ordinary_overload_should_not_reach_high(lumina, text):
    from lumina.safety import SAFETY_ORDER
    verdict = lumina.safety.assess(text)
    assert SAFETY_ORDER[verdict["level"]] < SAFETY_ORDER["HIGH"], (
        text, verdict["level"], verdict["model"]["confidence"])


def test_real_distress_still_reaches_high_through_the_head_alone(lumina):
    """The counterpart the defect must not be 'fixed' at the cost of."""
    verdict = lumina.safety.assess("I feel hopeless and like a burden")
    assert verdict["level"] == "HIGH"


def test_the_head_can_never_reach_crisis_by_itself(lumina):
    """Whatever the head says, CRISIS stays with the deterministic rules."""
    verdict = lumina.safety.assess("I feel hopeless and like a burden")
    assert verdict["model_can_reach_crisis"] is False
    assert verdict["level"] != "CRISIS"
