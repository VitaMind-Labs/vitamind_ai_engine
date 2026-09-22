"""Application service for one Mira conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from ..clinical.assessment.assessment_engine import AssessmentEngine
from ..clinical.differential import DifferentialEngine
from ..clinical.feature_extractor import FeatureExtractor
from ..clinical.patient_state import PatientState
from ..clinical.uncertainty import UncertaintyEngine
from ..safety.detector import SafetyDetector
from .interview import DEFAULT_SESSION_QUESTIONS, MAX_SESSION_QUESTIONS, InterviewPlanner, is_extension_requested


@dataclass
class MiraReply:
    text: str
    chapter: str
    chapter_progress: float
    assessment_complete: bool = False
    safety: dict = field(default_factory=dict)
    result: dict | None = None
    language: str = "en"


@dataclass
class MiraSession:
    session_id: str
    state: PatientState
    language: str
    chapter: str = "MORNING"
    messages: int = 0
    complete: bool = False
    result: dict | None = None
    assistant_message: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MiraAgent:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.extractor = FeatureExtractor()
        self.assessments = AssessmentEngine()
        self.safety = SafetyDetector()
        self.differential = DifferentialEngine()
        self.uncertainty = UncertaintyEngine()
        self.interview = InterviewPlanner()

    def create_session(self, age_group: str = "adult", language: str = "en", country_context: str | None = None) -> tuple[MiraSession, MiraReply]:
        if language not in {"en", "ar"}:
            raise ValueError("language must be 'en' or 'ar'")
        session = MiraSession(uuid4().hex, PatientState.new(age_group, language=language, country_context=country_context), language)
        assessment = self.assessments.assess(session.state)
        opening = self.interview.next_question(session.state, assessment, session.chapter)
        session.assistant_message = opening.question
        return session, MiraReply(opening.question, session.chapter, 0.0, language=language)

    @staticmethod
    def _report(assessment, differential, uncertainty, language="en", safety_level="routine", safety_flags=None):
        """Genere le rapport d'orientation (pas un diagnostic).
        - recommended_pathway = orientation vers quel type de professionnel
        - match_strength = niveau de confiance (HIGH/MODERATE/LOW)
        - condition_scores avec supporting/contradictory pour tracabilite
        """
        safety_flags = safety_flags or []
        scores = {
            "adhd": min(1.0, assessment.assessments["ADHD"].raw_score / 4),
            "bipolar": min(1.0, assessment.assessments["BIPOLAR_SPECTRUM"].raw_score / 10),
            "psychosis": min(1.0, assessment.assessments["PSYCHOSIS_SPECTRUM"].raw_score / 3),
        }
        leading = assessment.leading_condition
        pathway = {
            "ADHD": "ADHD_focused_clinical_assessment",
            "BIPOLAR_SPECTRUM": "mood_disorder_clinical_assessment",
            "PSYCHOSIS_SPECTRUM": "psychosis_spectrum_clinical_assessment",
        }.get(leading, "NO_STRONG_TARGET_SIGNAL") if not uncertainty.abstain else "NO_STRONG_TARGET_SIGNAL"
        # Orientation explicite (pas diagnostic): mapping vers professionnel
        orientation_labels = {
            "ADHD_focused_clinical_assessment": "orientation: attention/neurodevelopmental assessment with a qualified clinician",
            "mood_disorder_clinical_assessment": "orientation: mood disorder clinical assessment with a psychiatrist/psychologist",
            "psychosis_spectrum_clinical_assessment": "orientation: psychosis spectrum clinical assessment with a psychiatrist",
            "NO_STRONG_TARGET_SIGNAL": "orientation: no strong target signal — routine clinical review recommended",
        }
        orientation_text = orientation_labels.get(pathway, pathway)
        top = max(scores.values())
        strength = "HIGH" if top >= 0.75 else "MODERATE" if top >= 0.4 else "LOW"
        supporting = []
        contradictory = []
        missing = []
        for item in assessment.assessments.values():
            supporting.extend(item.supporting_evidence)
            contradictory.extend(item.contradictory_evidence)
            missing.extend(f"{item.condition}.{feature}" for feature in item.missing_information)
        return {
            "assessment_complete": True,
            "recommended_pathway": pathway,
            "orientation": orientation_text,  # Tache 4: wording orientation, pas diagnostic
            "match_strength": strength,  # niveau de confiance
            "condition_scores": scores,  # scores par condition + confiance
            "supporting_features": supporting,  # tracabilite: soutient
            "contradictory_features": contradictory,  # tracabilite: nuance
            "other_signals": [signal.upper() for signal in differential.alternatives],
            "missing_information": missing,
            "safety": {"level": safety_level, "flags": safety_flags},
            "recommended_test": "clinical consultation",
            "requires_clinician_review": True,
            "disclaimer": "This is an orientation for clinical follow-up, not a diagnosis.",
            "language": language,
        }

    def respond(self, session: MiraSession, text: str) -> MiraReply:
        if session.complete:
            raise ValueError("assessment already complete")
        if not text.strip():
            raise ValueError("message text cannot be empty")
        safety = self.safety.detect(session.state, text)
        if safety.urgent:
            report = self._report(
                self.assessments.assess(session.state),
                self.differential.analyze(session.state, self.assessments.assess(session.state)),
                self.uncertainty.decide(
                    session.state,
                    self.assessments.assess(session.state),
                    self.differential.analyze(session.state, self.assessments.assess(session.state)),
                ),
                session.language,
                "urgent",
                [safety.reason],
            )
            session.complete = True
            session.result = report
            session.assistant_message = reply_text = (
                "I am concerned about your immediate safety. Please contact local emergency services or a trusted person now."
                if session.language == "en"
                else "أنا قلق بشأن سلامتك الفورية. يرجى الاتصال بخدمات الطوارئ المحلية أو بشخص تثق به الآن."
            )
            return MiraReply(
                reply_text,
                "SAFETY",
                1.0,
                True,
                report["safety"],
                report,
                session.language,
            )
        self.extractor.extract_into_state(session.state, text, session.chapter, str(session.messages))
        # Extension a la demande: si l'utilisateur demande plus de questions
        # et qu'on est entre DEFAULT et MAX, on ne clot pas encore.
        wants_more = is_extension_requested(text)
        session.messages += 1
        assessment = self.assessments.assess(session.state)
        differential = self.differential.analyze(session.state, assessment)
        uncertainty = self.uncertainty.decide(session.state, assessment, differential)
        # Tache 2 — Logique de fin de session documentee:
        # Par defaut: DEFAULT_SESSION_QUESTIONS (10), determine par T1 (19 free_text -> 10 ≈ moitie).
        # Extension: si l'utilisateur demande explicitement plus de questions (wants_more)
        # et que messages < MAX (19, plafond = taille du pool free_text), on prolonge.
        # Plafond dur: MAX_SESSION_QUESTIONS (19) — meme si extension demandee, on clot.
        if session.messages >= MAX_SESSION_QUESTIONS:
            session.complete = True
        elif session.messages >= DEFAULT_SESSION_QUESTIONS:
            if wants_more and session.messages < MAX_SESSION_QUESTIONS:
                session.complete = False
            else:
                session.complete = True
        else:
            session.complete = False
        if session.complete:
            result = self._report(assessment, differential, uncertainty, session.language)
            session.result = result
            # Tache 4: wording orientation, pas diagnostic
            orientation_msg = (
                "Thank you for sharing this. This is an orientation (not a diagnosis) "
                f"toward {result['orientation']} with confidence {result['match_strength']}. "
                "A qualified clinician should review this orientation with you."
            )
            session.assistant_message = orientation_msg
            return MiraReply(orientation_msg, session.chapter, 1.0, True, result["safety"], result, session.language)
        question = self.interview.next_question(session.state, assessment, session.chapter)
        progress = min(0.99, session.messages / DEFAULT_SESSION_QUESTIONS)
        session.assistant_message = question.question
        return MiraReply(question.question, session.chapter, progress, False, {"level": "routine", "flags": []}, language=session.language)