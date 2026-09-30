"""An intent-routed reply must not claim data the patient never provided.

Found by tracing the pipeline: "I slept badly last night", with no check-ins on
file, was answered "Your sleep has been shorter than your own usual pattern" -
a comparison against a baseline that does not exist. The strategy templates were
written for a *measured* concern and were reused verbatim on the intent route
(`NO_STATE_CONCERN`), where the only evidence is what the patient just said.
"""
from __future__ import annotations

import re

import pytest

from lumina.orchestrator import Lumina

# Phrases that only make sense when a baseline comparison or a prior session exists.
UNMEASURED_CLAIMS = [
    r"shorter than your own usual",
    r"less settled than usual",
    r"less contact with people than usual",
    r"last time we tried",
    r"things look steady",
    r"compared with your own recent pattern, things",
    r"نومك كان أقصر",
    r"أقل استقرارًا من المعتاد",
    r"تواصلك مع الناس كان أقل",
    r"في المرة الماضية جرّبنا",
]

INTENT_ONLY_TURNS = [
    ("I slept badly last night", "en"),
    ("my routine is a mess", "en"),
    ("I feel lonely and I have no friends", "en"),
    ("How have I been doing lately?", "en"),
    ("I have no energy", "en"),
    ("نمت بشكل سيء", "ar"),
    ("روتيني فوضى", "ar"),
    ("Set a goal for me", "en"),
]


@pytest.fixture(scope="module")
def lumina():
    return Lumina.load()


@pytest.mark.parametrize("text,language", INTENT_ONLY_TURNS)
@pytest.mark.parametrize("track", ["ADHD", "BIPOLAR", "SCHIZOPHRENIA", "UNSPECIFIED"])
def test_intent_only_reply_makes_no_claim_about_unmeasured_state(lumina, text, language, track):
    out = lumina.turn(text=text, track=track, language=language)
    if "NO_STATE_CONCERN" not in out["decision"]["reason_codes"]:
        pytest.skip("not an intent-only route for this input")
    reply = out["response"]["text"]
    for claim in UNMEASURED_CLAIMS:
        assert not re.search(claim, reply, re.I), (claim, reply)


def test_measured_sleep_uses_absolute_wording_without_claiming_a_baseline(lumina):
    """An absolute sleep reading must not be phrased as a comparison to history."""
    from lumina.decision import Decision
    from lumina.response import render

    measured = Decision(type="SUPPORT", strategy="SLEEP_SUPPORT",
                        reason_codes=["CONCERN_LOW_SLEEP", "CAPACITY_NORMAL"],
                        capacity="NORMAL", track="BIPOLAR")
    assert "Your sleep was short" in render(measured, "en").text

    intent_only = Decision(type="SUPPORT", strategy="SLEEP_SUPPORT",
                           reason_codes=["INTENT_SLEEP", "NO_STATE_CONCERN"],
                           capacity="NORMAL", track="BIPOLAR")
    assert "shorter than your own usual pattern" not in render(intent_only, "en").text


def test_measured_history_still_reaches_the_reply(lumina):
    """With a real baseline and a real drop, the comparison is legitimate."""
    history = [{"sleep_hours": 7.5, "energy": 5, "stress": 4, "mood": 5,
                "routine_stability": 6, "focus": 5, "task_completion": 5,
                "social_connection": 5} for _ in range(12)]
    from lumina.state import build_state
    out = lumina.turn(checkin={"sleep_hours": 3.0}, history=[build_state(h) for h in history],
                      track="BIPOLAR", language="en", request_class="CHECKIN")
    assert any(c["dimension"] == "sleep" for c in out["changes"]), out["changes"]


def test_understanding_head_is_reported_as_advisory(lumina):
    """It is computed for chat, but nothing may read it as a decision input."""
    out = lumina.turn(text="I feel so sad today", track="ADHD", language="en")
    assert out["understanding"] is not None
    assert out["understanding"]["advisory_only"] is True


def test_decision_does_not_depend_on_the_understanding_head(lumina):
    """Same routed intent and state -> same decision, whatever the emotion head says."""
    class ForcedEmotion:
        version = "forced"

        def __init__(self, label):
            self.label = label

        def predict(self, text):
            return {"emotion": {"label": self.label, "confidence": 0.99, "abstain": False},
                    "act": {"label": "INFORM", "confidence": 0.9, "abstain": False}}

    text = "I need some help with my tasks"
    base = Lumina.load()
    decisions = set()
    for label in ("ANGRY", "CALM", "SAD", "NEUTRAL"):
        base.understanding = ForcedEmotion(label)
        out = base.turn(text=text, track="ADHD", language="en")
        decisions.add((out["decision"]["type"], out["decision"]["strategy"],
                       out["response"]["text"]))
    assert len(decisions) == 1, decisions


# --- the state Lumina hands back carries the capacity it decided ------------
def test_envelope_state_carries_the_estimated_capacity(lumina):
    """`state.capacity` used to stay at its default (UNKNOWN) on every turn, so the
    snapshot the backend persists never held what the capacity engine computed."""
    out = lumina.turn(checkin={"energy": 2, "stress": 8, "focus": 2, "mood": 3, "sleep_hours": 4},
                      track="ADHD", request_class="CHECKIN")
    assert out["capacity"]["level"] in ("REDUCED", "VERY_LOW")
    assert out["state"]["capacity"] == out["capacity"]["level"]

    calm = lumina.turn(checkin={"energy": 8, "stress": 2, "focus": 7, "mood": 8, "sleep_hours": 8,
                                "routine_stability": 7, "social_connection": 6, "task_completion": 7},
                       track="ADHD", request_class="CHECKIN")
    assert calm["state"]["capacity"] == calm["capacity"]["level"] != "UNKNOWN"


def test_no_signals_is_reported_as_unknown_not_invented(lumina):
    out = lumina.turn(text="hello there", track="ADHD")
    assert out["state"]["capacity"] == out["capacity"]["level"] == "UNKNOWN"
