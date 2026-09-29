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

TEMPLATE_VERSION = "lumina-templates-v2"

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

# One entry per strategy, per language. A value is one approved sentence or a
# list of approved variants; a turn picks one variant by a stable seed, so a
# replay says the same thing while a conversation does not repeat itself.
# `{action}` is filled from the chosen intervention's first step; templates
# without it never mention an exercise.
TEMPLATES = {
    # Generic on purpose: this strategy is reached from a medication mention, a
    # journal entry and a steady check-in alike, so it must not assume any of them.
    "ACKNOWLEDGE": {
        "en": ["Thank you for sharing that with me.",
               "I'm glad you told me.",
               "Thanks for letting me know."],
        "ar": ["شكرًا لأنك شاركتني ذلك.",
               "يسعدني أنك أخبرتني.",
               "شكرًا لإعلامي."],
    },
    "ACKNOWLEDGE_CHECKIN": {
        "en": ["Thanks for telling me how today went. Nothing looks out of the ordinary compared with your usual pattern.",
               "Thank you for checking in. Today looks close to your usual rhythm, which is good to see."],
        "ar": ["شكرًا لإخباري كيف مرّ يومك. لا يبدو أن هناك ما يختلف عن نمطك المعتاد.",
               "شكرًا على تسجيل حالتك. يومك قريب من إيقاعك المعتاد، وهذا جميل."],
    },
    "CLARIFY": {
        "en": ["I'd like to understand a little better. What's been on your mind today?",
               "Tell me a bit more, so I can follow you properly. How has today been so far?"],
        "ar": ["أودّ أن أفهم أكثر. ما الذي يشغل بالك اليوم؟",
               "أخبرني المزيد لأفهمك جيدًا. كيف كان يومك حتى الآن؟"],
    },
    "REFLECT": {
        "en": ["What you are describing sounds heavy to carry. I'm here, and we can take it slowly. What feels hardest right now?",
               "That sounds like a lot to hold. You don't have to sort it all out at once. Which part is weighing on you most?"],
        "ar": ["ما تصفه يبدو ثقيلًا. أنا هنا، ويمكننا أن نأخذ الأمر بهدوء. ما الأصعب الآن؟",
               "يبدو هذا كثيرًا عليك. لستَ مضطرًا لترتيب كل شيء دفعة واحدة. أيّ جزء يثقل عليك أكثر؟"],
    },
    "NORMALIZE_WITHOUT_MINIMIZING": {
        "en": "This comes up for a lot of people, and that does not make it any lighter for you.",
        "ar": "هذا يحدث لكثيرين، وهذا لا يجعله أخف عليك.",
    },
    "MICRO_ACTION": {
        "en": ["Getting started seems to be the hard part today, so let's make the first step smaller. {action}",
               "Starting is often the steepest part. Let's shrink the first step until it feels doable. {action}"],
        "ar": ["يبدو أن البداية هي الجزء الصعب اليوم، فلنجعل الخطوة الأولى أصغر. {action}",
               "البداية غالبًا هي الأصعب. لنصغّر الخطوة الأولى حتى تصبح ممكنة. {action}"],
    },
    "TASK_BREAKDOWN": {
        "en": ["There seems to be more on you than one day holds. Let's narrow it. {action}",
               "That's a lot on one plate. Let's pick just one piece for now. {action}"],
        "ar": ["يبدو أن ما عليك أكثر مما يتّسع له يوم واحد. فلنضيّق النطاق. {action}",
               "هذا كثير في وقت واحد. لنختر جزءًا واحدًا فقط الآن. {action}"],
    },
    "GROUNDING": {
        "en": ["Let's slow this down before anything else. {action}",
               "Let's pause together for a moment and let your body settle first. {action}"],
        "ar": ["لنهدّئ الإيقاع قبل أي شيء آخر. {action}",
               "لنتوقف لحظة معًا وندع جسدك يهدأ أولًا. {action}"],
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

# What the patient raised in their own words, when no check-in data backs a
# claim about their pattern. These never say "compared with your usual" -
# Lumina may not assert a change it has not measured.
HEARD = {
    "SLEEP_SUPPORT": {
        "en": ["Sleep trouble makes everything else feel heavier. {action}",
               "Nights like that wear you down. Let's look at it gently. {action}"],
        "ar": ["اضطراب النوم يجعل كل شيء آخر أثقل. {action}",
               "ليالٍ كهذه مُتعِبة. لننظر إلى الأمر بهدوء. {action}"],
    },
    "ENCOURAGE_SUPPORT_CONNECTION": {
        "en": ["Feeling on your own is hard, and I'm glad you said it out loud. {action}",
               "Connection matters, even in small doses. {action}"],
        "ar": ["الشعور بالوحدة صعب، ويسعدني أنك قلته. {action}",
               "التواصل مهم، ولو بقدر بسيط. {action}"],
    },
    # LOW_SLEEP is an absolute reading, not a comparison with a baseline.
    "SLEEP_SUPPORT_LOW": {
        "en": ["Your sleep was short, and that can make everything feel harder today. {action}"],
        "ar": ["نومك كان قصيرًا، وهذا قد يجعل كل شيء أصعب اليوم. {action}"],
    },
    "ROUTINE_SUPPORT": {
        "en": ["A steadier day can start from one fixed point. {action}"],
        "ar": ["يمكن أن يبدأ يوم أكثر ثباتًا من نقطة ثابتة واحدة. {action}"],
    },
    "STATE_MONITORING": {
        "en": ["Energy is worth paying attention to. How has it moved across your day - morning, afternoon, evening?"],
        "ar": ["الطاقة تستحق الانتباه. كيف تغيّرت خلال يومك - صباحًا وظهرًا ومساءً؟"],
    },
    "FOLLOW_UP_PROGRESS": {
        "en": ["Each check-in adds to your picture, and your weekly report in Reports shows what is shifting. Which part would you like to look at together - sleep, mood or energy?"],
        "ar": ["كل تسجيل يضيف إلى صورتك، وتقريرك الأسبوعي في التقارير يُظهر ما يتغيّر. أيّ جانب تودّ أن ننظر إليه معًا - النوم أم المزاج أم الطاقة؟"],
    },
    "FOLLOW_UP_GOAL": {
        "en": ["Let's make it small enough to start this week. What is one thing you would like to be different?"],
        "ar": ["لنجعله صغيرًا بما يكفي لنبدأ هذا الأسبوع. ما الشيء الواحد الذي تودّ أن يتغيّر؟"],
    },
    "ACKNOWLEDGE_MEDICATION_MENTION": {
        "en": ["Thanks for telling me. Your prescriber is the right person for anything about your treatment - if you like, note your questions in your journal so they are ready for your next appointment."],
        "ar": ["شكرًا لإخباري. الطبيب المعالج هو الشخص المناسب لكل ما يخص علاجك - إن أردت، دوّن أسئلتك في يومياتك لتكون جاهزة لموعدك القادم."],
    },
    "ACKNOWLEDGE_CLINICIAN_MENTION": {
        "en": ["Thanks for sharing that. If it helps, we can think together about what you would like to bring up with your care team."],
        "ar": ["شكرًا لمشاركتي ذلك. إن كان مفيدًا، يمكننا أن نفكّر معًا فيما تودّ أن تطرحه على فريق رعايتك."],
    },
    "ACKNOWLEDGE_EXERCISE": {
        "en": ["Moving your body counts, and it is good that you made room for it. How did you feel afterwards?"],
        "ar": ["تحريك جسدك مهم، وجميل أنك خصصت له وقتًا. كيف شعرت بعدها؟"],
    },
    "ACKNOWLEDGE_JOURNAL": {
        "en": ["Putting things into words can make them lighter. You can write here with me, or in your Smart Journal - whichever feels easier."],
        "ar": ["وضع الأشياء في كلمات قد يجعلها أخفّ. يمكنك أن تكتب هنا معي أو في يومياتك الذكية - أيّهما أسهل عليك."],
    },
    "ACKNOWLEDGE_CHECK_IN": {
        "en": ["Thanks for telling me how today is going. What stood out the most?"],
        "ar": ["شكرًا لإخباري كيف يمضي يومك. ما الذي لفت انتباهك أكثر؟"],
    },
    "CLARIFY_QUESTION": {
        "en": ["That's a fair question. I may not have the full answer, but I can think it through with you. What's behind it for you?"],
        "ar": ["سؤال في محلّه. قد لا أملك الإجابة كاملة، لكن يمكنني التفكير فيه معك. ما الذي يدفعك إليه؟"],
    },
    "OPEN_LISTENING": {
        "en": ["I'm listening. Tell me a little more about what's on your mind.",
               "I'm here with you. Take your time - what would you like to talk about?",
               "Go on, I'm with you. What's been happening?"],
        "ar": ["أنا أستمع. أخبرني قليلًا عمّا يدور في ذهنك.",
               "أنا هنا معك. خذ وقتك - عمّ تودّ أن نتحدث؟",
               "تابع، أنا معك. ما الذي حدث؟"],
    },
}

# The conversational move sets how a reply opens (or, for a few acts, the whole
# reply). `{name}` is ", <preferred name>" or empty.
ACT_OPENERS = {
    "GREETING": {
        "en": ["Hi{name}, it's good to hear from you.", "Hello{name}, I'm glad you're here.",
               "Hey{name}, it's nice to see you."],
        "ar": ["أهلًا{name}، سعيدة بسماع صوتك.", "مرحبًا{name}، يسعدني وجودك هنا.",
               "أهلًا بك{name}، سعيدة برؤيتك."],
    },
    "SADNESS": {
        "en": ["I'm sorry you're feeling this way{name}.", "That sounds really hard{name}, and I'm glad you told me."],
        "ar": ["آسفة أنك تشعر بهذا{name}.", "يبدو هذا صعبًا حقًا{name}، ويسعدني أنك أخبرتني."],
    },
    "ANXIETY": {
        "en": ["Worry can make everything feel urgent at once.", "That kind of unease is exhausting to sit with."],
        "ar": ["القلق يجعل كل شيء يبدو ملحًّا في وقت واحد.", "هذا النوع من القلق مُتعِب."],
    },
    "ANGER": {
        "en": ["It sounds like something really got to you.", "Feeling angry makes sense when something matters to you."],
        "ar": ["يبدو أن شيئًا ما أزعجك حقًا.", "الغضب مفهوم حين يكون الأمر مهمًا لك."],
    },
}

# Acts that are a complete reply on their own when nothing more pressing applies.
ACT_REPLIES = {
    "GREETING": {
        "en": ["How are you feeling right now?", "How is your day going so far?",
               "What's on your mind today?"],
        "ar": ["كيف تشعر الآن؟", "كيف يمضي يومك حتى الآن؟", "ما الذي يشغل بالك اليوم؟"],
    },
    "THANKS": {
        "en": ["You're welcome{name}. I'm here whenever you want to talk.",
               "Anytime{name}. Take care of yourself today."],
        "ar": ["على الرحب والسعة{name}. أنا هنا متى أردت الحديث.",
               "في أي وقت{name}. اعتنِ بنفسك اليوم."],
    },
    "CLOSING": {
        "en": ["Take care{name}. I'll be here when you come back.",
               "Rest well{name}. You can pick this up with me anytime."],
        "ar": ["اعتنِ بنفسك{name}. سأكون هنا حين تعود.",
               "ارتح جيدًا{name}. يمكنك أن تكمل معي في أي وقت."],
    },
    "ABOUT_LUMINA": {
        "en": ["I'm Lumina, your companion between appointments - I can help you notice patterns in your sleep, energy and mood, break a hard moment into small steps, and guide short calming exercises. I don't diagnose or replace your care team. What would help most today?"],
        "ar": ["أنا لومينا، رفيقتك بين المواعيد، أساعدك على ملاحظة أنماط نومك وطاقتك ومزاجك، وتقسيم اللحظات الصعبة إلى خطوات صغيرة، وأرافقك في تمارين تهدئة قصيرة. لا أشخّص ولا أحلّ محل فريق رعايتك. ما الذي يفيدك أكثر اليوم؟"],
    },
    "POSITIVE": {
        "en": ["That's really good to hear{name}. What do you think helped today go well?",
               "I love hearing that. What made the difference today?"],
        "ar": ["يسعدني سماع ذلك حقًا{name}. ما الذي ساعد يومك أن يمضي جيدًا برأيك؟",
               "جميل أن أسمع هذا. ما الذي صنع الفرق اليوم؟"],
    },
    "UNSURE": {
        "en": ["That's okay - you don't need the words yet. We can start small: does your body feel tired, tense, or okay right now?"],
        "ar": ["لا بأس - لست بحاجة للكلمات بعد. لنبدأ ببساطة: هل يشعر جسدك الآن بالتعب أم التوتر أم أنه بخير؟"],
    },
    "SADNESS": {
        "en": ["You don't have to carry it alone here. Would you like to tell me what's been weighing on you?"],
        "ar": ["لست مضطرًا لحمله وحدك هنا. هل تودّ أن تخبرني بما يثقل عليك؟"],
    },
    "ANGER": {
        "en": ["Do you want to tell me what happened? I'm listening, no judgement."],
        "ar": ["هل تودّ أن تخبرني بما حدث؟ أنا أستمع دون أحكام."],
    },
}

# When no approved exercise fits, ask something useful instead of announcing it.
NO_ACTION_FOLLOW_UP = {
    "SLEEP_SUPPORT": ("What usually keeps you up - your thoughts, your body, or your routine?",
                      "ما الذي يُبقيك مستيقظًا عادة - أفكارك أم جسدك أم روتينك؟"),
    "ENCOURAGE_SUPPORT_CONNECTION": ("Is there someone it would feel easy to hear from, even briefly?",
                                     "هل هناك شخص يسهل عليك أن تسمع منه، ولو لوقت قصير؟"),
    "GROUNDING": ("Try one slow breath with me, out longer than in. What do you notice?",
                  "جرّب معي نفسًا بطيئًا واحدًا، زفيره أطول من شهيقه. بماذا تشعر؟"),
    "_default": ("Would you like to tell me a bit more about it?",
                 "هل تودّ أن تخبرني المزيد عن ذلك؟"),
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


def _pick(entry, language, seed):
    """One approved variant, chosen by a stable per-turn seed."""
    value = entry[language]
    if isinstance(value, str):
        return value
    return value[seed % len(value)]


def _codes(decision, prefix):
    return [c[len(prefix):] for c in decision.reason_codes if c.startswith(prefix)]


def _body_template(decision, language, seed):
    """The main sentence(s) for this strategy, worded for what we actually know."""
    key = decision.strategy
    codes = decision.reason_codes
    topics = _codes(decision, "INTENT_") + _codes(decision, "TOPIC_")
    topic = topics[0] if topics else None
    heard = "NO_STATE_CONCERN" in codes

    if key == "ACKNOWLEDGE" and "NO_ACTIVE_CONCERN" in codes and not topic:
        return _pick(TEMPLATES["ACKNOWLEDGE_CHECKIN"], language, seed)
    if key == "CLARIFY" and ("OPEN_LISTENING" in codes or topic == "UNKNOWN"):
        return _pick(HEARD["OPEN_LISTENING"], language, seed)
    if key == "CLARIFY" and topic == "QUESTION":
        return _pick(HEARD["CLARIFY_QUESTION"], language, seed)
    if key == "SLEEP_SUPPORT" and "CONCERN_LOW_SLEEP" in codes:
        return _pick(HEARD["SLEEP_SUPPORT_LOW"], language, seed)
    if heard:
        if key == "FOLLOW_UP" and topic in ("PROGRESS", "GOAL"):
            return _pick(HEARD[f"FOLLOW_UP_{topic}"], language, seed)
        if key == "ACKNOWLEDGE" and f"ACKNOWLEDGE_{topic}" in HEARD:
            return _pick(HEARD[f"ACKNOWLEDGE_{topic}"], language, seed)
        if key in HEARD:
            return _pick(HEARD[key], language, seed)
    return _pick(TEMPLATES.get(key, TEMPLATES["UNKNOWN_SUPPORT"]), language, seed)


def render(decision, language="en", changes=(), resources=(), disclaimer=True,
           name=None, seed=0):
    """Turn a Decision into approved patient-facing text.

    `name` is the patient's own preferred name (never inferred); `seed` picks
    among approved variants. Raises ProhibitedContent if the rendered result
    breaks a rule, rather than sending something that merely looks plausible.
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

    safety_path = decision.type in ("ELEVATED_SAFETY_WORKFLOW", "SAFETY_CLARIFICATION")
    acts = [] if safety_path else _codes(decision, "ACT_")
    act = acts[0] if acts else None
    name_part = (f", {name}" if language == "en" else f" {name}") if name else ""
    has_concern = any(c.startswith("CONCERN_") for c in decision.reason_codes)
    conversational = decision.type == "GENERAL_SUPPORT" or \
        (decision.strategy in ("ACKNOWLEDGE", "CLARIFY", "REFLECT") and not has_concern)

    opener, body = "", None
    if "CONVERSATION_CLOSE" in decision.reason_codes and act in ACT_REPLIES:
        body = _pick(ACT_REPLIES[act], language, seed)
    elif act == "ABOUT_LUMINA":
        body = _pick(ACT_REPLIES[act], language, seed)
    elif conversational and act in ("GREETING", "POSITIVE", "UNSURE", "SADNESS", "ANGER"):
        # Nothing in the data asks for more: answer the person, not the pipeline.
        if act in ACT_OPENERS:
            opener = _pick(ACT_OPENERS[act], language, seed)
        body = _pick(ACT_REPLIES[act], language, seed)
    elif act in ACT_OPENERS:
        opener = _pick(ACT_OPENERS[act], language, seed)
        # A greeting in front of a check-in concern keeps the concern as the body.

    if body is None:
        template = _body_template(decision, language, seed)
        action = ""
        if decision.intervention:
            steps = decision.intervention.get(f"steps_{language}", [])
            action = steps[0] if steps else ""
        elif "{action}" in template:
            # No approved intervention survived the filters. Ask something useful
            # rather than rendering an empty instruction or inventing one.
            follow = NO_ACTION_FOLLOW_UP.get(decision.strategy, NO_ACTION_FOLLOW_UP["_default"])
            action = follow[0 if language == "en" else 1]
        body = template.format(action=action,
                               observation=_observation_phrase(changes, language))

    body = body.replace("{name}", name_part)
    opener = opener.replace("{name}", name_part)
    limit = CONSTRAINTS.get(decision.capacity, CONSTRAINTS["UNKNOWN"])["max_sentences"]
    if opener and _count_sentences(f"{opener} {body}") > limit:
        # Short on room: the substance matters more than the greeting.
        opener = ""
    text = re.sub(r"\s{2,}", " ", f"{opener} {body}").strip()

    # Said where it matters - when Lumina comments on a pattern, or a safety
    # check - rather than stamped on every warm sentence.
    if disclaimer and (decision.type == "ELEVATED_SAFETY_WORKFLOW"
                       or decision.strategy == "STATE_MONITORING"):
        text = f"{text} {DISCLAIMER[language]}"

    text, truncated = _enforce_capacity(text, decision.capacity)

    violations = check_prohibited(text, decision.prohibited)
    if violations:
        raise ProhibitedContent(
            f"rendered reply violates {violations} for track {decision.track}")

    return Reply(text=text, language=language, strategy=decision.strategy,
                 sentences=_count_sentences(text), questions=_count_questions(text),
                 intervention_id=decision.intervention_id, truncated=truncated)
