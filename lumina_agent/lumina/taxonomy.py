"""Closed label vocabularies. Every learned head predicts into one of these sets.

A finite taxonomy is what makes Lumina testable: a decision can be compared to an
expected value, and anything outside the set is a contract error rather than a
plausible-sounding sentence. UNKNOWN is a first-class member of every set so a
model is always allowed to abstain (spec s54, s124).
"""

# --- Understanding -------------------------------------------------------
INTENTS = ("CHECK_IN", "JOURNAL", "EMOTIONAL_SUPPORT", "TASK_SUPPORT", "SLEEP",
           "ENERGY", "FOCUS", "ROUTINE", "STRESS", "SOCIAL", "EXERCISE", "GOAL",
           "PROGRESS", "MEDICATION_MENTION", "CLINICIAN_MENTION", "SAFETY",
           "CRISIS", "QUESTION", "GENERAL_CONVERSATION", "UNKNOWN")

# NEUTRAL is an addition to the spec list: the annotated corpora mark "no
# emotion expressed", which is a different statement from CALM (an assessed
# absence of distress) and from UNKNOWN (the model declining to answer).
# Collapsing the three would make the head untestable.
EMOTIONS = ("NEUTRAL", "CALM", "CONTENT", "MOTIVATED", "HOPEFUL", "CONFUSED",
            "FRUSTRATED", "ANXIOUS", "SAD", "ANGRY", "OVERWHELMED", "LONELY",
            "FEARFUL", "IRRITABLE", "LOW_ENERGY", "DISTRESSED", "UNKNOWN")

# DailyDialog communication acts. Real human labels, kept as their own head
# rather than being forced into the Lumina intent set.
ACTS = ("INFORM", "QUESTION", "DIRECTIVE", "COMMISSIVE")

# --- Safety --------------------------------------------------------------
# Ordered least to most severe; ordering is relied upon when fusing signals.
SAFETY_LEVELS = ("NORMAL", "ELEVATED", "HIGH", "CRISIS", "UNKNOWN")
SAFETY_ORDER = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRISIS": 3, "UNKNOWN": 1}

# --- Tracks --------------------------------------------------------------
TRACKS = ("ADHD", "BIPOLAR", "SCHIZOPHRENIA", "UNSPECIFIED")

# Condition-associated *language* classes. These describe how text reads, not
# what a person has. Never used to set or change a track (spec s79, s80).
CONDITION_LANGUAGE = ("NORMAL", "DEPRESSION", "SUICIDAL", "ANXIETY", "BIPOLAR",
                      "STRESS", "PERSONALITY_DISORDER", "UNKNOWN")

# --- Capacity & state ----------------------------------------------------
CAPACITY = ("HIGH", "NORMAL", "REDUCED", "VERY_LOW", "UNKNOWN")

STATE_DIMENSIONS = ("sleep", "energy", "stress", "mood", "focus",
                    "routine_stability", "social_connection", "task_completion")

OBSERVATION_QUALITY = ("observed", "estimated", "missing", "unknown")

TRENDS = ("STABLE", "IMPROVING", "DECLINING", "VOLATILE",
          "SIGNIFICANT_CHANGE", "INSUFFICIENT_DATA")

# --- Response ------------------------------------------------------------
RESPONSE_STRATEGIES = ("ACKNOWLEDGE", "CLARIFY", "REFLECT",
                       "NORMALIZE_WITHOUT_MINIMIZING", "MICRO_ACTION",
                       "TASK_BREAKDOWN", "GROUNDING", "ROUTINE_SUPPORT",
                       "SLEEP_SUPPORT", "STATE_MONITORING",
                       "ENCOURAGE_SUPPORT_CONNECTION", "FOLLOW_UP",
                       "SAFETY_CHECK", "DECLINE_UNSAFE_REQUEST",
                       "UNKNOWN_SUPPORT")

MEMORY_TYPES = ("PREFERENCE", "ORIENTATION", "BASELINE", "PATTERN", "GOAL",
                "INTERVENTION_RESPONSE", "SAFETY", "CONTEXT",
                "CLINICIAN_CONSTRAINT", "PATIENT_CORRECTION")

# Higher wins when two memories disagree (spec s41).
MEMORY_PRIORITY = {"PATIENT_CORRECTION": 5, "PATIENT_EXPLICIT_STATEMENT": 4,
                   "STRUCTURED_CHECK_IN": 3, "OBSERVED_PATTERN": 2,
                   "MODEL_INFERENCE": 1}

LANGUAGES = ("en", "ar")


def validate(value, vocabulary, field):
    if value not in vocabulary:
        raise ValueError(f"{field}={value!r} is outside the closed vocabulary {vocabulary}")
    return value
