from vitamind.clinical.patient_state import PatientState
from vitamind.clinical.feature_extractor import FeatureExtractor
from vitamind.clinical.assessment.assessment_engine import AssessmentEngine
from vitamind.clinical.differential import DifferentialEngine
from vitamind.clinical.uncertainty import UncertaintyEngine
from vitamind.safety.detector import SafetyDetector
from vitamind.mira.interview import InterviewPlanner


def make_state():
    return PatientState.new(age_group="adult", age_years=29, language="en")


def test_adhd_style_case_extracts():
    state = make_state()
    extractor = FeatureExtractor()
    extractor.extract_into_state(
        state,
        "Since school I have always been distracted, forget things and procrastinate.",
        chapter="MIDDAY",
    )
    assert state.get_status("developmental_history", "childhood_onset").value == "present"
    assert state.get_status("attention", "distractibility").value == "present"


def test_reduced_sleep_with_exhaustion_not_decreased_need():
    state = make_state()
    extractor = FeatureExtractor()
    extractor.extract_into_state(
        state,
        "I sleep 3 hours but I am exhausted the next day.",
        chapter="MORNING",
    )
    assert state.get_status("sleep", "reduced_sleep").value == "present"
    assert state.get_status("sleep", "decreased_need_for_sleep").value in {"absent", "conflicting"}


def test_bipolar_assessment_scores_episode_features():
    state = make_state()
    extractor = FeatureExtractor()
    text = (
        "For a week I slept 3 hours and still felt full of energy. "
        "I had racing thoughts, started many projects and spent too much."
    )
    extractor.extract_into_state(state, text, chapter="EVENING")
    engine = AssessmentEngine()
    result = engine.assess(state)
    assert result.assessments["BIPOLAR_SPECTRUM"].raw_score > 0


def test_psychosis_pathway():
    state = make_state()
    extractor = FeatureExtractor()
    extractor.extract_into_state(
        state,
        "I hear voices when alone and sometimes feel people are watching me.",
        chapter="INNER_VOICE",
    )
    result = AssessmentEngine().assess(state)
    assert result.assessments["PSYCHOSIS_SPECTRUM"].raw_score > 0


def test_safety_detector():
    state = make_state()
    result = SafetyDetector().detect(
        state,
        "I am going to kill myself."
    )
    assert result.urgent is True
    assert state.safety_snapshot()["urgent"] is True


def test_uncertainty_engine_can_abstain():
    state = make_state()
    assessment = AssessmentEngine().assess(state)
    diff = DifferentialEngine().analyze(state, assessment)
    u = UncertaintyEngine().decide(state, assessment, diff)
    assert u.abstain is True


def test_interview_planner_returns_question():
    state = make_state()
    assessment = AssessmentEngine().assess(state)
    q = InterviewPlanner().next_question(
        state,
        assessment,
        chapter="MORNING",
    )
    assert isinstance(q.question, str)
    assert len(q.question) > 10
