from vitamind.clinical.patient_state import (
    Chapter,
    FeatureStatus,
    ObservationPolarity,
    PatientState,
)


def test_new_state_has_unknown_features():
    state = PatientState.new(age_group="adult", language="en")
    assert state.get_status("attention", "distractibility") == FeatureStatus.UNKNOWN


def test_present_observation_updates_status():
    state = PatientState.new(age_group="adult")
    state.add_observation(
        domain="attention",
        feature="distractibility",
        polarity=ObservationPolarity.PRESENT,
        confidence=0.90,
        evidence_excerpt="I lose focus constantly.",
        chapter=Chapter.MIDDAY,
    )
    assert state.get_status("attention", "distractibility") == FeatureStatus.PRESENT


def test_absent_is_not_unknown():
    state = PatientState.new(age_group="adult")
    state.add_observation(
        domain="psychosis",
        feature="auditory_perceptual_experience",
        polarity="absent",
        confidence=0.92,
        evidence_excerpt="I have never heard voices or sounds others could not hear.",
        chapter="INNER_VOICE",
    )
    assert (
        state.get_status("psychosis", "auditory_perceptual_experience")
        == FeatureStatus.ABSENT
    )


def test_conflicting_evidence_is_preserved():
    state = PatientState.new(age_group="adult")

    state.add_observation(
        domain="sleep",
        feature="decreased_need_for_sleep",
        polarity="present",
        confidence=0.85,
        evidence_excerpt="I sleep three hours and still feel full of energy.",
        chapter="MORNING",
    )

    state.add_observation(
        domain="sleep",
        feature="decreased_need_for_sleep",
        polarity="absent",
        confidence=0.80,
        evidence_excerpt="Actually I usually feel exhausted after sleeping that little.",
        chapter="EVENING",
    )

    assert (
        state.get_status("sleep", "decreased_need_for_sleep")
        == FeatureStatus.CONFLICTING
    )


def test_unknown_features_support_followup_planning():
    state = PatientState.new(age_group="adult")

    required = [
        ("sleep", "decreased_need_for_sleep"),
        ("episode_history", "episodic_pattern"),
        ("mood", "elevated_mood"),
    ]

    missing = state.unknown_features(required)
    assert len(missing) == 3

    state.add_observation(
        domain="sleep",
        feature="decreased_need_for_sleep",
        polarity="present",
        confidence=0.9,
        evidence_excerpt="I can sleep very little and still have energy.",
    )

    missing = state.unknown_features(required)
    assert ("sleep", "decreased_need_for_sleep") not in missing
    assert len(missing) == 2


def test_serialization_round_trip():
    state = PatientState.new(
        age_group="adult",
        age_years=31,
        language="en",
        country_context="UAE",
    )

    state.add_observation(
        domain="developmental_history",
        feature="childhood_onset",
        polarity="present",
        confidence=0.88,
        evidence_excerpt="I had the same attention issues at school.",
        chapter="MIDDAY",
        message_id="msg-12",
    )

    rebuilt = PatientState.from_json(state.to_json())

    assert rebuilt.context.age_years == 31
    assert (
        rebuilt.get_status("developmental_history", "childhood_onset")
        == FeatureStatus.PRESENT
    )


def test_safety_snapshot():
    state = PatientState.new(age_group="adult")

    state.add_observation(
        domain="safety",
        feature="suicidal_intent",
        polarity="present",
        confidence=0.95,
        evidence_excerpt="Safety example.",
        chapter="SAFETY",
    )

    snapshot = state.safety_snapshot()
    assert snapshot["urgent"] is True
    assert snapshot["active_features"][0]["feature"] == "suicidal_intent"
