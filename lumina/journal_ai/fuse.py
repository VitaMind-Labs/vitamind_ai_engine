"""Deterministic, inspectable policy. Never infer a diagnosis from a journal."""
from .context import rule_context
from .schema import RANK

def fuse(hits, model_context):
    rules=rule_context(hits)
    active=[h for h in hits if h.kind not in ("idiom",) and not (h.negated or h.quoted or h.subject=="other" or h.temporal=="past")]
    explicit=[h for h in active if h.kind=="explicit"]
    if explicit:
        return "high", "affirmed_current_self_explicit", rules
    if any(h.kind=="urgent_wish" for h in active):
        return "high", "current_self_death_wish_support", rules
    if model_context is None:
        # Model absence is visible and uncertain, not a silent 'none'.
        tier=rules.tier if RANK[rules.tier]>=RANK["moderate"] else "moderate_flagged"
        return tier, "model_unavailable_review_required", rules
    # Context must scope to the risk-bearing span, not an unrelated sentence.
    serious=[h for h in hits if h.kind in ("explicit","urgent_wish","serious","moderate")]
    softened=serious and all(h.negated or h.quoted or h.subject=="other" or h.temporal=="past" for h in serious)
    if softened:
        return rules.tier, "risk_spans_contextualized", rules
    if rules.is_idiom and not active:
        # Do not let an idiom in one clause erase model-detected risk in another.
        if model_context.tier=="high" and not model_context.is_idiom and model_context.confidence>=0.65:
            return "moderate_flagged", "idiom_model_disagreement", rules
        return "none", "idiom_only", rules
    if model_context.tier=="high":
        if model_context.negated or model_context.temporal=="past" or model_context.subject=="other" or model_context.is_idiom or model_context.confidence<0.65:
            return max((rules.tier,"moderate_flagged"),key=lambda t:RANK[t]), "model_high_needs_context_review", rules
        return "high", "model_high_current_self", rules
    if RANK[rules.tier]>=RANK["moderate"]:
        return rules.tier, "serious_cue_minimum_support", rules
    if model_context.tier=="moderate":
        return "moderate" if model_context.confidence>=0.5 else "moderate_flagged", "model_moderate", rules
    return max((rules.tier,model_context.tier),key=lambda t:RANK[t]), "combined_nonurgent", rules
