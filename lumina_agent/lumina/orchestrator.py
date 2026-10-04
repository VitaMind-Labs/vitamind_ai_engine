"""The Lumina turn: observe, understand, check safety, decide, speak, measure.

This is the intelligence loop from spec s132, wired in a fixed order. Safety runs
before anything is generated; the rule engine decides what is allowed; the
response layer only renders it. No step may be skipped or reordered by input.

Model routing (s101, s102) matters here: not every model runs on every request. A
check-in needs state, baseline and safety. A chat turn needs safety and
understanding. Running everything on everything would waste most of the work.

The output is a structured envelope with reason codes and persistence hints. The
backend owns the database; this returns what it may write, never writes it.
"""
from __future__ import annotations

import datetime as dt

from . import CONTRACT_VERSION
from .baseline import compute_baseline, detect_changes, CONFIG as BASELINE_CONFIG
from .capacity import estimate
from .decision import decide
from .intent import classify as classify_intent
from .journal import JournalAnalyzer, JournalAnalysisResult, fuse_journal_safety, journal_context_level
from .interventions import InterventionCatalog, OutcomeHistory
from .memory import MemoryStore
from .planning import extract_items
from .response import render
from .safety import SafetyEngine, recent_safety_level
from .state import build_state
from .taxonomy import SAFETY_ORDER, TRACKS
from .tracks import read_tracks

AGENT = "LUMINA"
AGENT_VERSION = "0.1.0"

REQUEST_CLASSES = ("CHAT_SHORT", "CHAT_DEEP", "CHECKIN", "JOURNAL_ANALYSIS",
                   "SAFETY_CLASSIFICATION")


def _preferred_name(store):
    """The name the patient asked to be called, from their own onboarding answer.

    Stored by the backend as "<label>: <value>" under key preferred_name. Only a
    short run of letters is used, so a stored sentence can never be echoed back.
    """
    import re
    for memory in store.find(key="preferred_name"):
        if not memory.active:
            continue
        value = memory.content.split(":", 1)[-1].strip()
        match = re.match(r"[^\W\d_][^\W\d_' -]{0,23}", value)
        if match:
            return match.group(0).strip()
    return None


def _variant_seed(request_id, text):
    """Stable per turn, different across turns: a replay gets the same wording."""
    import zlib
    return zlib.crc32(f"{request_id or ''}|{text or ''}".encode("utf-8"))


class Lumina:
    """One assembled engine. Stateless with respect to patients.

    Every patient-specific input arrives in the call; nothing is cached between
    patients, so one patient's history can never leak into another's turn.
    """

    def __init__(self, safety=None, catalog=None, allow_unreviewed=True,
                 understanding=None, intent_model=None, journal=None):
        self.safety = safety or SafetyEngine.load()
        self.catalog = catalog or InterventionCatalog(allow_unreviewed=allow_unreviewed)
        self.understanding = understanding
        # Journal AI runs inside the agent, as an analyzer module rather than a
        # separate service (spec s44). It shares the emotion head with the chat
        # path so the two cannot disagree about the same sentence.
        self.journal = journal or JournalAnalyzer(emotion_model=understanding)
        # The learned intent head abstains on everything it was trained for, so
        # it is off by default. Pass one explicitly once it is retrained on real
        # reviewed data and beats the rules on a held-out split.
        self.intent_model = intent_model

    @classmethod
    def load(cls, allow_unreviewed=True):
        from pathlib import Path

        from .linear import MultiHeadLinear
        root = Path(__file__).resolve().parents[1] / "artifacts" / "models"
        understanding = None
        try:
            understanding = MultiHeadLinear.load(root / "understanding")
        except (OSError, ValueError, KeyError):
            understanding = None
        # The learned intent head runs as a fallback only: rules route first, and the head
        # answers just the messages no rule matched, and only above the confidence floor
        # fitted on validation (0.6; it abstains otherwise), so it can add coverage but
        # never override a rule. On the held-out split the rules abstain on 73 of 171 rows
        # and the head answers 27 of those with 70% precision.
        intent_model = None
        try:
            intent_model = MultiHeadLinear.load(root / "intent")
        except (OSError, ValueError, KeyError):
            intent_model = None
        return cls(allow_unreviewed=allow_unreviewed, understanding=understanding,
                   intent_model=intent_model)

    # -- journal ----------------------------------------------------------
    def analyze_journal(self, text, *, entry_id=None, language=None,
                        content_version=1, is_private=False,
                        analysis_consent=True, request_id=None):
        """Analyze one journal entry inside the agent and return an envelope.

        Only structured signals come back. The entry text stays with the caller,
        so the chat path can never be handed journal prose (spec s46).
        """
        started = dt.datetime.now(dt.timezone.utc)
        analysis = self.journal.analyze(
            text, entry_id=entry_id, language=language,
            content_version=content_version, is_private=is_private,
            analysis_consent=analysis_consent)

        persistence = ["PERSIST_JOURNAL_ANALYSIS"]
        if analysis.memory_candidates:
            persistence.append("PERSIST_MEMORY_CANDIDATE")
        if SAFETY_ORDER[analysis.safety["level"]] >= SAFETY_ORDER["ELEVATED"]:
            persistence.append("PERSIST_SAFETY_EVENT")

        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds() * 1000
        return {
            "requestId": request_id,
            "agent": AGENT,
            "agentVersion": AGENT_VERSION,
            "contractVersion": CONTRACT_VERSION,
            "requestClass": "JOURNAL_ANALYSIS",
            "language": analysis.language,
            "journalAnalysis": analysis.to_dict(),
            "safety": analysis.safety,
            "persistence": persistence,
            "idempotencyKey": analysis.idempotency_key,
            "performance": {"processingTimeMs": round(elapsed, 2)},
            "storesChainOfThought": False,
            "clinicallyValidated": False,
        }

    # -- the turn ---------------------------------------------------------
    def turn(self, *, text=None, checkin=None, history=(), track="UNSPECIFIED",
             secondary_track=None, language="en", memories=None, outcomes=None,
             journal_signals=None, journal_context=None, recent_safety=None,
             resources=(),
             request_class="CHAT_SHORT", request_id=None):
        if track not in TRACKS:
            raise ValueError(f"unknown track {track!r}")
        if request_class not in REQUEST_CLASSES:
            raise ValueError(f"unknown request class {request_class!r}")
        if text is None and checkin is None:
            raise ValueError("a turn needs either text or a check-in")

        started = dt.datetime.now(dt.timezone.utc)
        store = memories if isinstance(memories, MemoryStore) else MemoryStore(memories)
        outcome_history = outcomes or OutcomeHistory()
        steps = []

        # 1-2. Observe and validate into a structured snapshot. A journal
        # analysis may be passed in place of a bare signal dict; only its
        # numeric signals are read, never any text.
        journal_analysis = None
        if isinstance(journal_signals, JournalAnalysisResult):
            journal_analysis = journal_signals
            journal_signals = dict(journal_analysis.signals)
        state = build_state(checkin or {}, journal_signals or {})
        steps.append("state_built")

        # 3. Intent routing, then understanding - both only where useful.
        intent = None
        if text:
            intent = classify_intent(text, self.intent_model)
            steps.append("intent")

        understanding = None
        if text and self.understanding is not None and \
                request_class in ("CHAT_SHORT", "CHAT_DEEP"):
            prediction = self.understanding.predict(text)
            understanding = {
                "emotion": prediction["emotion"]["label"],
                "emotion_confidence": prediction["emotion"]["confidence"],
                "emotion_abstained": prediction["emotion"]["abstain"],
                "act": prediction.get("act", {}).get("label"),
                "model_version": self.understanding.version,
                "not_a_diagnosis": True,
                # Its emotion head scores macro-F1 0.36 (MODELS.md), so nothing
                # downstream may steer a decision from it. Say so in the envelope
                # rather than leaving a caller to assume it is an input.
                "advisory_only": True,
            }
            steps.append("understanding")

        # 4. Safety, before any response is produced.
        if text:
            verdict = self.safety.assess(text)
        else:
            verdict = {"level": "NORMAL", "decided_by": "no_free_text",
                       "rules": {"available": False}, "model": {"available": False},
                       "requires_human_review": False, "clinically_validated": False}

        if journal_analysis is not None:
            # A journal entry's own safety level escalates the turn, never lowers it.
            fused = fuse_journal_safety(journal_analysis, verdict)
            if fused != verdict["level"]:
                verdict = {**verdict, "level": fused,
                           "decided_by": "journal_escalation",
                           "journal": journal_analysis.safety,
                           "requires_human_review": True}
            else:
                verdict = {**verdict, "journal": journal_analysis.safety}
        # A recent journal entry lends its follow-up to this turn: when the journal
        # itself recommended a safety check, the conversation that follows opens with
        # one. Escalation only - never lowers a level, never reaches CRISIS.
        journal_escalated = False
        lent = journal_context_level(journal_context)
        if SAFETY_ORDER[lent] > SAFETY_ORDER[verdict["level"]]                 and verdict["level"] != "UNKNOWN":
            journal_escalated = True
            verdict = {**verdict, "level": lent, "decided_by": "journal_context",
                       "journal_context": {"tier": journal_context.get("tier"),
                                           "cues": list(journal_context.get("cues") or ()),
                                           "hours_ago": journal_context.get("hours_ago")},
                       "requires_human_review": (verdict.get("requires_human_review")
                                                 or lent == "HIGH")}
        # An earlier turn's crisis or high-risk moment carries into this one, in
        # whatever conversation it happens, so a fresh thread cannot skip the
        # follow-up. Same rules: escalation only, never CRISIS.
        recent_escalated = False
        lent = recent_safety_level(recent_safety)
        if SAFETY_ORDER[lent] > SAFETY_ORDER[verdict["level"]] \
                and verdict["level"] != "UNKNOWN":
            recent_escalated = True
            journal_escalated = False
            verdict = {**verdict, "level": lent, "decided_by": "recent_safety",
                       "recent_safety": {"level": recent_safety.get("level"),
                                         "hours_ago": recent_safety.get("hours_ago")},
                       "requires_human_review": (verdict.get("requires_human_review")
                                                 or lent == "HIGH")}
        steps.append("safety")

        # 5-6. Baseline and change detection against the patient's own history.
        full_history = list(history) + [state]
        baseline = compute_baseline(full_history,
                                    exclude_recent=BASELINE_CONFIG["recent_window"])
        changes = detect_changes(full_history, baseline)
        steps.append("baseline_and_changes")

        # 7. Capacity, bounded by safety.
        capacity = estimate(state, verdict["level"])
        # The snapshot travels in the envelope and is what the backend stores as the
        # day's state. Its `capacity` field was a default nothing ever set, so every
        # persisted snapshot said UNKNOWN whatever the engine had just decided.
        state.capacity = capacity.level
        steps.append("capacity")

        # 8. Track logic, kept separate per track.
        readings = read_tracks(track, state, changes, baseline, secondary_track)
        steps.append("tracks")

        # 9. Relevant memory only - never the whole history.
        relevant = store.retrieve(
            keys=[c.dimension for reading in readings for c in reading.concerns],
            categories=("PREFERENCE", "INTERVENTION_RESPONSE", "CLINICIAN_CONSTRAINT"))
        steps.append("memory_retrieval")

        # 10-11. The rule engine decides what is allowed.
        decision = decide(safety=verdict, state=state, changes=changes,
                          capacity=capacity, track_readings=readings,
                          catalog=self.catalog, history=outcome_history,
                          memories=relevant,
                          intent=intent.intent if intent else None,
                          act=intent.act if intent else None)
        if journal_escalated and decision.type == "ELEVATED_SAFETY_WORKFLOW":
            decision.reason_codes.append("JOURNAL_CONTEXT")
        if recent_escalated and decision.type == "ELEVATED_SAFETY_WORKFLOW":
            decision.reason_codes.append("RECENT_SAFETY")
        steps.append("decision")

        # 12. Render, within capacity limits and the track's language rules.
        # A request to plan the day is answered from the list the patient just wrote.
        plan_items = extract_items(text) if intent is not None and intent.act == "PLAN_REQUEST" else None
        reply = render(decision, language=language, changes=changes,
                       resources=resources, name=_preferred_name(store),
                       seed=_variant_seed(request_id, text), plan_items=plan_items)
        steps.append("response")

        # 13-16. Persistence hints. The backend decides what actually gets stored.
        persistence = ["PERSIST_INTERACTION"]
        if decision.intervention_id:
            persistence.append("PERSIST_OUTCOME")
        if verdict["level"] in ("HIGH", "CRISIS", "UNKNOWN"):
            persistence.append("PERSIST_SAFETY_EVENT")
        memory_updates = outcome_history.to_memory_updates()
        if memory_updates:
            persistence.append("PERSIST_MEMORY_UPDATE")

        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds() * 1000
        return {
            "requestId": request_id,
            "agent": AGENT,
            "agentVersion": AGENT_VERSION,
            "contractVersion": CONTRACT_VERSION,
            "language": language,
            "requestClass": request_class,
            "response": {"text": reply.text,
                         "style": {"strategy": reply.strategy,
                                   "sentences": reply.sentences,
                                   "questions": reply.questions,
                                   "truncated": reply.truncated}},
            "intent": intent.to_dict() if intent else None,
            "journalAnalysis": journal_analysis.to_dict() if journal_analysis else None,
            "understanding": understanding,
            "state": state.to_dict(),
            "baseline": baseline.to_dict(),
            "changes": [c.to_dict() for c in changes],
            "capacity": capacity.to_dict(),
            "tracks": [r.to_dict() for r in readings],
            "safety": verdict,
            "decision": decision.to_dict(),
            "memoryUpdates": memory_updates,
            "relevantMemory": [m.to_dict() for m in relevant],
            "reviewFlag": decision.review_flag,
            "persistence": persistence,
            "pipeline": steps,
            "performance": {"processingTimeMs": round(elapsed, 2)},
            "storesChainOfThought": False,
            "clinicallyValidated": False,
        }
