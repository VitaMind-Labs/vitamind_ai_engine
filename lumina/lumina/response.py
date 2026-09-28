"""Controlled response generation: templates, not free generation.

The decision engine has already chosen the strategy and the intervention. This
layer only renders them into language, in the patient's language, within the
limits capacity set (spec s13, s14, s55).

Templates rather than a generative model is a deliberate choice, and the spec
argues it directly (s14): a small model trained from scratch on a few thousand
conversations will not produce reliable clinical-adjacent language, while a
template filled from structured state is predictable, testable and translatable.
English and Arabic are authored side by side from the same strategy so the two
languages cannot drift apart (s84).

Crisis language is not templated here at all. It comes from a fixed approved set,
and the emergency resources inside it are supplied by the backend - the system
must never invent a phone number, a hospital or a service (s53).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .capacity import CONSTRAINTS
from .taxonomy import LANGUAGES, RESPONSE_STRATEGIES

TEMPLATE_VERSION = "lumina-templates-v1"

# Wording that must never survive into a patient-facing reply, whatever produced
# it. Checked after rendering, so a template edit cannot bypass the rule.
PROHIBITED_PATTERNS = {
    "diagnosis": re.compile(
        r"\byou (?:have|are)\s+(?:bipolar|schizophreni\w*|adhd|depressed|manic|psychotic)\b"
        r"|\byou are (?:entering|in) (?:a |an )?(?:manic|depressive|psychotic) (?:episode|phase)\b",
        re.I),
    "episode_naming": re.compile(r"\b(?:manic|hypomanic|depressive|psychotic) episode\b", re.I),
    "medication_advice": re.compile(
        r"\b(?:take|stop|increase|decrease|reduce|double|skip)\b.{0,24}\b"
        r"(?:medication|meds|dose|dosage|pills?|tablets?)\b", re.I),
    "mania_claim": re.compile(r"\byou (?:are|seem|appear|might be) (?:manic|hypomanic)\b", re.I),
    "validate_unverified_interpretation": re.compile(
        r"\b(?:yes|indeed),?\s+(?:they|he|she|people)\s+(?:are|is)\s+"
        r"(?:really\s+)?(?:watching|following|sending|controlling|spying)\b", re.I),
    "confront_or_dispute_belief": re.compile(
        r"\bthat (?:is|isn't|is not) (?:not )?real\b|\bthat did not happen\b"
        r"|\byou(?:'re| are) imagining\b", re.I),
    "treatment_replacement": re.compile(
        r"\b(?:instead of|rather than|replaces?)\b.{0,20}\b(?:therapy|treatment|medication)\b", re.I),
}

# One entry per strategy, per language. `{action}` is filled from the chosen
# intervention's first step; templates without it never mention an exercise.
TEMPLATES = {
    # Generic on purpose: this strategy is reached from a medication mention, a
    # journal entry and a steady check-in alike, so it must not assume any of them.
    "ACKNOWLEDGE": {
        "en": "Thank you for telling me, I have noted it.",
        "ar": "شكرًا لإخباري، لقد سجّلت ذلك.",
    },
    "ACKNOWLEDGE_CHECKIN": {
        "en": "Thanks for telling me how today went. Nothing looks out of the ordinary compared with your usual pattern.",
        "ar": "شكرًا لإخباري كيف مرّ يومك. لا يبدو أن هناك ما يختلف عن نمطك المعتاد.",
    },
    "CLARIFY": {
        "en": "I do not have enough from you yet to say anything useful about today. How has it been so far?",
        "ar": "ليس لديّ ما يكفي بعد لأقول شيئًا مفيدًا عن يومك. كيف كان حتى الآن؟",
    },
    "REFLECT": {
        "en": "What you are describing sounds heavy to carry.",
        "ar": "ما تصفه يبدو ثقيلًا.",
    },
    "NORMALIZE_WITHOUT_MINIMIZING": {
        "en": "This comes up for a lot of people, and that does not make it any lighter for you.",
        "ar": "هذا يحدث لكثيرين، وهذا لا يجعله أخف عليك.",
    },
    "MICRO_ACTION": {
        "en": "Getting started seems to be the hard part today, so let's make the first step smaller. {action}",
        "ar": "يبدو أن البداية هي الجزء الصعب اليوم، فلنجعل الخطوة الأولى أصغر. {action}",
    },
    "TASK_BREAKDOWN": {
        "en": "There seems to be more on you than one day holds. Let's narrow it. {action}",
        "ar": "يبدو أن ما عليك أكثر مما يتّسع له يوم واحد. فلنضيّق النطاق. {action}",
    },
    "GROUNDING": {
        "en": "Let's slow this down before anything else. {action}",
        "ar": "لنهدّئ الإيقاع قبل أي شيء آخر. {action}",
    },
    "ROUTINE_SUPPORT": {
        "en": "Your routine looks less settled than usual. One fixed point is enough to start. {action}",
        "ar": "يبدو روتينك أقل استقرارًا من المعتاد. نقطة ثابتة واحدة تكفي للبداية. {action}",
    },
    "SLEEP_SUPPORT": {
        "en": "Your sleep has been shorter than your own usual pattern. {action}",
        "ar": "نومك كان أقصر من نمطك المعتاد. {action}",
    },
    "STATE_MONITORING": {
        "en": "Compared with your own recent pattern, {observation}. I am not drawing a conclusion from that - let's keep watching it together. {action}",
        "ar": "مقارنةً بنمطك الأخير، {observation}. لا أستنتج شيئًا من ذلك، ولنواصل ملاحظته معًا. {action}",
    },
    "ENCOURAGE_SUPPORT_CONNECTION": {
        "en": "You have had less contact with people than usual. {action}",
        "ar": "تواصلك مع الناس كان أقل من المعتاد. {action}",
    },
    "FOLLOW_UP": {
        "en": "Last time we tried something together. How did it go?",
        "ar": "في المرة الماضية جرّبنا شيئًا معًا. كيف سار الأمر؟",
    },
    "SAFETY_CHECK": {
        "en": "I want to check in on how you are doing right now.",
        "ar": "أريد أن أطمئن على حالك الآن.",
    },
    "DECLINE_UNSAFE_REQUEST": {
        "en": "That is not something I can help with. What I can do is stay with what you are dealing with today.",
        "ar": "هذا ليس شيئًا أستطيع المساعدة فيه. ما أستطيعه هو البقاء معك فيما تواجهه اليوم.",
    },
    "UNKNOWN_SUPPORT": {
        "en": "I am not sure I follow yet. Can you say a little more?",
        "ar": "لست متأكدًا أنني فهمت بعد. هل يمكنك أن تقول المزيد؟",
    },
}

# Approved crisis text. Fixed wording, never generated, never paraphrased.
# `{resources}` is filled from backend configuration; with none supplied the
# message still stands and simply does not name a service.
CRISIS_MESSAGE = {
    "en": ("I am staying with you, and you can keep writing. If you might act on "
           "these thoughts, or you are in immediate danger, please contact local "
           "emergency services now. Is there someone you trust who could be with "
           "you? I cannot contact anyone on your behalf.{resources}"),
    "ar": ("أنا معك، ويمكنك مواصلة الكتابة. إذا كنت قد تتصرف وفق هذه الأفكار أو "
           "كنت في خطر مباشر، فاتصل بخدمات الطوارئ المحلية الآن. هل هناك شخص تثق "
           "به يمكن أن يكون معك؟ لا أستطيع الاتصال بأحد نيابة عنك.{resources}"),
}

DISCLAIMER = {
    "en": "This is support, not a diagnosis.",
    "ar": "هذا دعم وليس تشخيصًا.",
}

OBSERVATION = {
    "sleep": {"DECREASE": ("your sleep has been shorter", "نومك كان أقصر"),
              "INCREASE": ("your sleep has been longer", "نومك كان أطول")},
    "energy": {"DECREASE": ("your energy has been lower", "طاقتك كانت أقل"),
               "INCREASE": ("your energy has been higher", "طاقتك كانت أعلى")},
    "stress": {"DECREASE": ("your stress has been lower", "توترك كان أقل"),
               "INCREASE": ("your stress has been higher", "توترك كان أعلى")},
    "focus": {"DECREASE": ("your focus has been lower", "تركيزك كان أقل"),
              "INCREASE": ("your focus has been higher", "تركيزك كان أعلى")},
    "routine_stability": {"DECREASE": ("your routine has been less settled", "روتينك كان أقل استقرارًا"),
                          "INCREASE": ("your routine has been steadier", "روتينك كان أكثر ثباتًا")},
}


class ProhibitedContent(Exception):
    """Raised when rendered text violates a rule. It is never sent."""


@dataclass
class Reply:
    text: str
    language: str
    strategy: str
    sentences: int
    questions: int
    intervention_id: str | None = None
    template_version: str = TEMPLATE_VERSION
    truncated: bool = False

    def to_dict(self):
        return asdict(self)


def _count_sentences(text):
    return len([s for s in re.split(r"[.!?؟]+", text) if s.strip()])


def _count_questions(text):
    return text.count("?") + text.count("؟")


def _observation_phrase(changes, language):
    """Describe the most significant change in plain, non-clinical words."""
    if not changes:
        return ("things look steady" if language == "en" else "الأمور تبدو مستقرة")
    change = max(changes, key=lambda c: abs(c.delta or 0))
    phrases = OBSERVATION.get(change.dimension)
    if not phrases or change.direction not in phrases:
        generic = ("something has shifted" if language == "en" else "هناك تغيّر ما")
        return generic
    return phrases[change.direction][0 if language == "en" else 1]


def check_prohibited(text, prohibited):
    """Reject rendered text that breaks a track or global rule."""
    violations = []
    for rule in tuple(prohibited) + ("diagnosis", "medication_advice",
                                     "treatment_replacement"):
        pattern = PROHIBITED_PATTERNS.get(rule)
        if pattern and pattern.search(text):
            violations.append(rule)
    return sorted(set(violations))


def _enforce_capacity(text, capacity):
    """Trim to the sentence budget capacity allows, keeping whole sentences."""
    limits = CONSTRAINTS.get(capacity, CONSTRAINTS["UNKNOWN"])
    parts = re.findall(r"[^.!?؟]+[.!?؟]+|[^.!?؟]+$", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= limits["max_sentences"]:
        return text, False
    return " ".join(parts[:limits["max_sentences"]]), True


def render(decision, language="en", changes=(), resources=(), disclaimer=True):
    """Turn a Decision into approved patient-facing text.

    Raises ProhibitedContent if the rendered result breaks a rule, rather than
    sending something that merely looks plausible.
    """
    if language not in LANGUAGES:
        raise ValueError(f"unsupported language {language!r}; supported: {LANGUAGES}")
    if decision.strategy not in RESPONSE_STRATEGIES:
        raise ValueError(f"unknown strategy {decision.strategy!r}")

    # Crisis text is fixed and bypasses the templates entirely.
    if decision.type == "CRISIS_WORKFLOW":
        listed = ""
        if resources:
            label = "Emergency resources: " if language == "en" else "موارد الطوارئ: "
            listed = " " + label + "، ".join(resources) if language == "ar" \
                else " " + label + ", ".join(resources)
        text = CRISIS_MESSAGE[language].format(resources=listed)
        return Reply(text=text, language=language, strategy=decision.strategy,
                     sentences=_count_sentences(text), questions=_count_questions(text))

    key = decision.strategy
    if key == "ACKNOWLEDGE" and "NO_ACTIVE_CONCERN" in decision.reason_codes:
        key = "ACKNOWLEDGE_CHECKIN"
    template = TEMPLATES.get(key, TEMPLATES["UNKNOWN_SUPPORT"])[language]

    action = ""
    if decision.intervention:
        steps = decision.intervention.get(f"steps_{language}", [])
        action = steps[0] if steps else ""
    elif "{action}" in template:
        # No approved intervention survived the filters. Say that plainly rather
        # than rendering an empty instruction or inventing one.
        action = ("I do not have an approved exercise that fits right now, so let's "
                  "just talk it through." if language == "en"
                  else "ليس لديّ تمرين معتمد يناسب الآن، فلنتحدث عن الأمر فقط.")

    text = template.format(action=action,
                           observation=_observation_phrase(changes, language)).strip()
    text = re.sub(r"\s{2,}", " ", text)

    if disclaimer and decision.type in ("SUPPORT", "ELEVATED_SAFETY_WORKFLOW"):
        text = f"{text} {DISCLAIMER[language]}"

    text, truncated = _enforce_capacity(text, decision.capacity)

    violations = check_prohibited(text, decision.prohibited)
    if violations:
        raise ProhibitedContent(
            f"rendered reply violates {violations} for track {decision.track}")

    return Reply(text=text, language=language, strategy=decision.strategy,
                 sentences=_count_sentences(text), questions=_count_questions(text),
                 intervention_id=decision.intervention_id, truncated=truncated)
