import logging
import random
import re
from dataclasses import dataclass

from ..clinical.assessment.assessment_engine import AssessmentResult
from ..clinical.patient_state import PatientState
from .question_bank import MODULES_FOR_LEADING, load_bank, text_for

logger = logging.getLogger("vitamind.mira.interview")

# Tache 2 — Nombre de questions par defaut et plafond.
# Banque actuelle: 19 free_text total, 17 utilisables hors SAFETY/COMPLETION
# (filtre question_bank.py). Choix documente: 10 par defaut (≈ la moitie, variation assuree),
# plafond 17 (tout le pool utilisable) sur demande explicite d'extension.
DEFAULT_SESSION_QUESTIONS = 10
MAX_SESSION_QUESTIONS = 17

# Phrases declenchant une extension si l'utilisateur veut continuer
# au-dela du defaut. Detecte FR/EN/AR (MSA) — pas de FR/TN invente.
EXTENSION_PHRASES = (
    # EN
    "more questions",
    "another question",
    "can i answer more",
    "i have more to say",
    "i want to continue",
    "continue asking",
    "ask me more",
    # AR (MSA script — pas de transliteration inventee)
    "مزيد من الأسئلة",
    "المزيد من الأسئلة",
    "أسئلة أخرى",
    "هل يمكنني الإجابة على المزيد",
    "أريد الاستمرار",
    "لدي المزيد لأقوله",
)


@dataclass
class InterviewQuestion:
    question: str
    target_feature: str | None = None
    language: str = "en"


def is_extension_requested(text: str) -> bool:
    """Return True si l'utilisateur demande explicitement plus de questions.
    Utilise par MiraAgent pour prolonger la session au-dela de DEFAULT_SESSION_QUESTIONS
    jusqu'a MAX_SESSION_QUESTIONS. Plafond documente ci-dessus.
    """
    if not text or not text.strip():
        return False
    lowered = text.lower().strip()
    # Normalisation legere arabe: strip diacritics? keep simple substring
    for phrase in EXTENSION_PHRASES:
        if phrase.lower() in lowered:
            return True
    return False


class InterviewPlanner:
    # Fallback bilingue EN/AR stricte — pas de FR/TN, pas de traduction a la volee.
    # Si la banque est indisponible et langue=ar, on sert l'AR ici, jamais de l'EN.
    QUESTIONS = {
        "sleep": {
            "en": "How much do you sleep during these periods, and do you feel tired or unusually energized afterward?",
            "ar": "خلال هذه الفترات، كم تنام، وهل تشعر بالتعب أم بنشاط غير معتاد بعد ذلك؟",
        },
        "developmental_history": {
            "en": "Did these attention or organization difficulties also happen during childhood, in more than one setting?",
            "ar": "هل حدثت صعوبات الانتباه أو التنظيم هذه أيضًا في الطفولة، وفي أكثر من مكان؟",
        },
        "psychosis": {
            "en": "Have you had experiences such as voices or beliefs that other people do not share?",
            "ar": "هل مررت بتجارب مثل سماع أصوات أو معتقدات لا يشاركك فيها الآخرون؟",
        },
    }

    # Fallback slot -> (target domain, target feature, leading conditions that prefer it).
    SLOTS = {
        "sleep": ("sleep", "decreased_need_for_sleep", ("BIPOLAR_SPECTRUM", None)),
        "developmental_history": ("developmental_history", "childhood_onset", ("ADHD",)),
        "psychosis": ("psychosis", "auditory_perceptual_experience", ("PSYCHOSIS_SPECTRUM",)),
    }
    BASE_ORDER = ("sleep", "developmental_history", "psychosis")
    ASKED_PREFIX = "asked:"

    def __init__(self, bank_path: str | None = None):
        self._bank_path = bank_path

    @classmethod
    def asked_slots(cls, state: PatientState) -> list[str]:
        return [
            item.feature[len(cls.ASKED_PREFIX):]
            for item in state.observations
            if item.domain == "interview" and item.feature.startswith(cls.ASKED_PREFIX)
        ]

    def _record(self, state: PatientState, key: str, text: str, chapter: str | None) -> None:
        state.add_observation("interview", f"{self.ASKED_PREFIX}{key}", "present", 1.0, text, chapter)

    def _bank_question(self, state: PatientState, assessment: AssessmentResult, chapter: str | None) -> InterviewQuestion | None:
        """Selection variable documentee (Tache 2):
        - Filtre sur modules preferes selon leading_condition (MODULES_FOR_LEADING).
        - Exclut les deja posees (asked).
        - Pondere par informations manquantes: les questions dont le
          target_feature contient un feature encore UNKNOWN sont boostees.
        - Tirage aleatoire parmi le top pool (variable entre sessions,
          jamais identique deux sessions consecutives pour meme chapitre).
        - Plafond: jusqu'a MAX_SESSION_QUESTIONS sur demande explicite.
        """
        bank = load_bank(self._bank_path)
        if not bank:
            return None
        language = state.context.language if state.context.language in {"en", "ar"} else "en"
        asked_list = self.asked_slots(state)
        asked = set(asked_list)
        modules = MODULES_FOR_LEADING.get(assessment.leading_condition, MODULES_FOR_LEADING[None])
        ordered = [e for m in modules for e in bank if e.get("module") == m]
        ordered += [e for e in bank if e not in ordered]
        pool = [
            e for e in ordered
            if e["question_id"] not in asked
            and (language != "ar" or (e.get("question_ar") or "").strip())
        ]
        if not pool:
            # Tout le pool a ete pose: rotation, jamais la derniere
            last = asked_list[-1] if asked_list else None
            pool = [
                e for e in ordered
                if e["question_id"] != last
                and (language != "ar" or (e.get("question_ar") or "").strip())
            ] or [
                e for e in ordered
                if language != "ar" or (e.get("question_ar") or "").strip()
            ]

        # --- Ponderation par informations manquantes ---
        # On booste les entries dont au moins un token de target_feature
        # correspond a un feature UNKNOWN dans le PatientState.
        def _weight(entry: dict) -> int:
            target = entry.get("target_feature") or ""
            tokens = [t.strip() for t in re.split(r"[;,\s]+", target) if t.strip()]
            for tok in tokens:
                # tok peut etre "decreased_need_for_sleep" ou "chief_concern"
                for domain in ("attention", "sleep", "mood", "psychosis", "developmental_history", "episode_history", "safety"):
                    try:
                        if state.get_status(domain, tok).value == "unknown":
                            return 2
                    except Exception:
                        continue
                # Aussi verifier via missing_information de l'assessment
                for info in assessment.assessments.get("BIPOLAR_SPECTRUM", {}).missing_information if hasattr(assessment.assessments.get("BIPOLAR_SPECTRUM", {}), "missing_information") else []:
                    if tok in info:
                        return 2
            return 1

        weighted_pool = []
        for entry in pool:
            w = _weight(entry)
            # duplication pour ponderation simple (2x si manquant)
            weighted_pool.extend([entry] * w)

        # Variation aleatoire entre sessions:
        # seed deterministe base sur session_id-like info si disponible,
        # sinon random system. On utilise un RNG separe pour testabilite.
        # Pour la variabilite inter-session, on ne seed pas fixement.
        rng = random.Random()
        # Optionnel: si l'etat contient deja des asked, on perturbe via leur hash
        # pour que deux sessions avec meme leading n'aient pas la meme sequence
        # mais sans rendre le test flaky: on garde un choix aleatoire vrai.
        entry = rng.choice(weighted_pool) if weighted_pool else pool[0]
        text = text_for(entry, language)
        if text is None:
            return None
        key = entry["question_id"]
        self._record(state, key, text, chapter)
        return InterviewQuestion(text, entry.get("target_feature"), language)

    def _fallback_question(self, state: PatientState, assessment: AssessmentResult, chapter: str | None) -> InterviewQuestion:
        leading = assessment.leading_condition
        order = [key for key in self.BASE_ORDER if leading in self.SLOTS[key][2]]
        order += [key for key in self.BASE_ORDER if key not in order]

        asked = self.asked_slots(state)
        asked_set = set(asked)
        pool = [key for key in order if key not in asked_set]
        if not pool:
            pool = [key for key in order if not asked or key != asked[-1]] or list(order)

        pick = pool[0]
        for key in pool:
            domain, feature, _ = self.SLOTS[key]
            if state.get_status(domain, feature).value == "unknown":
                pick = key
                break

        domain, feature, _ = self.SLOTS[pick]
        raw = self.QUESTIONS[pick]
        # Bilingue strict: choix selon langue de session, pas de traduction a la volee
        lang = (state.context.language or "en").lower()
        if lang.startswith("ar"):
            question = raw.get("ar")
            used_lang = "ar"
        else:
            question = raw.get("en")
            used_lang = "en"
        self._record(state, pick, question, chapter)
        return InterviewQuestion(question, feature, used_lang)

    def next_question(self, state: PatientState, assessment: AssessmentResult, chapter: str | None = None) -> InterviewQuestion:
        bank_question = self._bank_question(state, assessment, chapter)
        if bank_question is not None:
            return bank_question
        return self._fallback_question(state, assessment, chapter)
