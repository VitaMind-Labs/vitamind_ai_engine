"""High-priority safety detector, intentionally independent of diagnosis."""

import re
from dataclasses import dataclass

from ..clinical.patient_state import PatientState


@dataclass
class SafetyResult:
    urgent: bool
    reason: str | None = None


class SafetyDetector:
    # Every English motif has its MSA Arabic equivalent (verified phrasing,
    # no invented translations: standard clinical/formal wording).
    PATTERNS = (
        ("suicidal_intent", (
            "kill myself", "suicide", "end my life", "hurt myself",
            "want to die",
            "أقتل نفسي", "أن أقتل نفسي", "انتحار", "أنهي حياتي",
            "إنهاء حياتي", "أريد أن أموت", "أؤذي نفسي",
        )),
        ("homicidal_intent", (
            "kill someone", "hurt someone", "harm others",
            "أقتل شخصا", "أقتل أحدا", "أن أقتل شخصا",
            "أؤذي شخصا", "أؤذي أحدا", "ألحق الأذى بالآخرين",
        )),
        ("immediate_danger", (
            "in immediate danger", "not safe right now",
            "في خطر فوري", "في خطر مباشر",
            "لست آمنا الآن", "لست بأمان", "لست في أمان",
        )),
    )

    # Denial tokens (EN + AR, matched as whole words, never substrings):
    # a match preceded by one of these is a denial ("I don't want to ...",
    # "لا أريد أن ..."), not intent.
    NEGATION_WORDS = frozenset({
        # NOTE: bare "no" deliberately excluded ("No. I want to kill myself"
        # must stay urgent).
        "not", "never",
        "لا", "لن", "لم", "ليس", "ليست", "لست",
        "أبدا", "أبدًا", "دون", "بدون",
    })

    def _negated(self, lowered: str, index: int) -> bool:
        window = lowered[max(0, index - 40):index]
        words = re.findall(r"[\w']+", window, flags=re.UNICODE)[-4:]
        return any(w in self.NEGATION_WORDS or w.endswith("n't") for w in words)

    def detect(self, state: PatientState, text: str) -> SafetyResult:
        lowered = text.lower()
        for feature, phrases in self.PATTERNS:
            for phrase in phrases:
                index = lowered.find(phrase.lower())
                if index >= 0 and not self._negated(lowered, index):
                    state.add_observation("safety", feature, "present", 0.98, text, "SAFETY")
                    return SafetyResult(True, feature)
        return SafetyResult(False)
