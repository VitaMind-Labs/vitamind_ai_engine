"""Clause-scoped context checks are authoritative over isolated cue words."""
from .schema import Context, RANK

def rule_context(hits):
    candidates=[]
    for h in hits:
        if h.kind=="idiom":
            tier="none"
        elif h.negated or h.quoted:
            tier="low"
        elif h.subject=="other":
            # Concern about another person still merits support, without patient attribution.
            tier="moderate_flagged" if h.kind in ("explicit","urgent_wish","serious") else "low"
        elif h.temporal=="past":
            tier="moderate" if h.kind in ("explicit","urgent_wish","serious") else "low"
        else:
            tier={"explicit":"high","urgent_wish":"high","serious":"moderate_flagged","moderate":"moderate","marker":"low"}[h.kind]
        candidates.append((tier,h))
    if not candidates:
        return Context(rationale="No configured cue matched; this does not establish safety.")
    tier,h=max(candidates,key=lambda pair:RANK[pair[0]])
    return Context(tier=tier,categories=sorted({hit.category for _,hit in candidates}),subject=h.subject,temporal=h.temporal,negated=h.negated,is_idiom=all(hit.kind=="idiom" for _,hit in candidates),confidence=0.0,rationale="Clause-scoped cues; heuristic confidence is not a probability.",source="rules")
