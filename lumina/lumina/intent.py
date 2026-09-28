"""Intent routing by inspectable rules, with the learned head as a fallback.

The learned intent model trained on the authored seed set abstains on every input
and scores macro-F1 0.17 on held-out meanings - correct behaviour for a model
with under a hundred training sentences across seventeen classes, and useless as
a router. The spec anticipates exactly this trade (s70, s101): prefer
deterministic code, and do not run a model where a rule does the job better.

So routing is done by bilingual cue patterns that a reviewer can read and correct.
When no rule fires, the router returns UNKNOWN and the decision engine falls back
to a generic supportive strategy - the same conservative default the learned head
would have produced, reached honestly.

This is not a permanent answer. Once reviewed real conversations exist, the
learned head is retrained and takes over where it measurably beats these rules.
Intent never influences safety: that path is owned by lumina/safety.py.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .taxonomy import INTENTS
from .text import normalize

RULE_VERSION = "intent-rules-v1"

# Ordered most specific first: the first rule that matches wins, so narrow
# patterns are not swallowed by broad ones.
PATTERNS = [
    ("MEDICATION_MENTION", r"\b(?:medication|meds|prescription|dose|dosage|pills?|tablets?|side effects?)\b"
                           r"|(?:دوا(?:ء|ئي)|الدواء|وصفة|جرعة|حبوب|اثار جانبية|آثار جانبية)"),
    ("CLINICIAN_MENTION", r"\b(?:psychiatrist|therapist|clinician|my doctor|appointment|session|clinic)\b"
                          r"|(?:طبيبي|معالجي|الطبيب النفسي|موعد|جلسة|العيادة)"),
    ("CRISIS", None),   # never matched here; safety owns it
    ("SAFETY", None),
    ("PROGRESS", r"\b(?:how have i been|summary of my|my patterns?|looking back|last month|"
                 r"better or worse|since i started|has anything changed)\b"
                 r"|(?:ملخص|كيف كان ادائي|انماطي|الشهر الماضي|افضل ام اسوا|تغير شيء)"),
    ("GOAL", r"\b(?:set a goal|my goal|a goal for|build a habit|target|drop the goal)\b"
             r"|(?:هدف|اهداف|عادة جديدة)"),
    ("SLEEP", r"\b(?:sleep|slept|sleeping|insomnia|wake up|woke up|bed(?:time)?|nap)\b"
              r"|(?:نوم|نمت|انام|ارق|استيقظ|اصحى|السرير)"),
    ("FOCUS", r"\b(?:focus|concentrat\w+|attention|distract\w+|other tabs|clear head)\b"
              r"|(?:تركيز|اركز|انتباه|شرود|مشتت)"),
    ("ENERGY", r"\b(?:energy|energetic|exhaust\w+|drained|tired|fatigue|running on empty|"
               r"sit still|drive)\b|(?:طاقة|نشاط|مرهق|ارهاق|تعب|منهك|ساكن)"),
    ("EXERCISE", r"\b(?:walk|walking|gym|exercise|stretch\w*|workout|run(?:ning)?|active)\b"
                 r"|(?:مشي|النادي|تمارين|رياضة|اطالة|نشاط بدني)"),
    ("ROUTINE", r"\b(?:routine|schedule|regular|rhythm|same time)\b"
                r"|(?:روتين|جدول|منتظم|ايقاع|الوقت نفسه)"),
    ("SOCIAL", r"\b(?:friends?|people|lonely|alone|social|messages|cancelled on|"
               r"spoken to anyone|replied to)\b|(?:اصدقاء|صديق|الناس|وحدة|بوحدي|رسائل|اجتماعي)"),
    ("TASK_SUPPORT", r"\b(?:task|tasks|start(?:ing|ed)?|finish\w*|to ?do|list|deadline|"
                     r"report|avoid\w* starting|piling up|putting it off|procrastinat\w+)\b"
                     r"|(?:مهمة|مهام|ابدا|البدء|انهي|قائمة|تاجيل|اؤجل|تتراكم|التقرير)"),
    ("STRESS", r"\b(?:stress\w*|pressure|overwhelm\w*|too much|calmed down)\b"
               r"|(?:توتر|ضغط|مرهق|اكثر مما احتمل|هدات)"),
    ("EMOTIONAL_SUPPORT", r"\b(?:just need someone|need to be heard|listen|sit with me|"
                          r"hard time|feel (?:flat|sad|low|empty)|pointless|nobody understands|"
                          r"lonely)\b|(?:استمع|يسمع لي|وقت صعب|حزين|فراغ|بلا معنى|لا احد يفهم)"),
    ("JOURNAL", r"\b(?:write about|writing this|put down what|journal|entry|"
                r"thoughts i wanted|looking back on the week)\b"
                r"|(?:اكتب عن|ادون|يوميات|افكار اردت)"),
    ("CHECK_IN", r"\b(?:checking in|check.?in|here is how today|today was|reporting in|"
                 r"mood is|answering your question|you asked how)\b"
                 r"|(?:اسجل حالتي|هذا ما مر به يومي|كان اليوم|مزاجي|سالتني عن)"),
    ("QUESTION", r"^\s*(?:what|how|who|can you|are you|do you|should i|is this|am i)\b"
                 r"|\?\s*$|(?:^|\s)(?:ماذا|كيف|من|هل|لماذا)\b|؟\s*$"),
    ("GENERAL_CONVERSATION", r"\b(?:good morning|good evening|hello|hi|thanks|thank you|"
                             r"that is all|talk to you|okay|understood|makes sense)\b"
                             r"|(?:صباح الخير|مساء الخير|مرحبا|اهلا|شكرا|هذا كل شيء|اراك|حسنا|فهمت)"),
]

COMPILED = [(intent, re.compile(pattern, re.I | re.UNICODE))
            for intent, pattern in PATTERNS if pattern]


@dataclass
class IntentReading:
    intent: str
    matched_rule: str | None = None
    all_matches: tuple = ()
    source: str = "rules"
    abstained: bool = False
    rule_version: str = RULE_VERSION

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent {self.intent!r}")

    def to_dict(self):
        return {**asdict(self), "all_matches": list(self.all_matches),
                "influences_safety": False}


def classify(text, model=None):
    """Route a message to an intent. Rules first; the learned head only fills gaps.

    `model` is an optional MultiHeadLinear with an `intent` head. It is consulted
    only when no rule matched, and only when it does not abstain - so a model
    that knows nothing changes nothing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    normalized = normalize(text)
    matches = [intent for intent, pattern in COMPILED if pattern.search(normalized)]
    if matches:
        return IntentReading(intent=matches[0], matched_rule=matches[0],
                             all_matches=tuple(dict.fromkeys(matches)))

    if model is not None:
        try:
            prediction = model.predict(text)["intent"]
        except Exception:
            prediction = None
        if prediction and not prediction["abstain"]:
            return IntentReading(intent=prediction["label"], source="model",
                                 matched_rule=None)

    return IntentReading(intent="UNKNOWN", source="rules", abstained=True)
