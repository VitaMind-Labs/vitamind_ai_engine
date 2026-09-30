import time
from datetime import datetime, timedelta, timezone
from .fuse import fuse
from .lexicon import scan, VERSION as LEXICON_VERSION
from .normalize import language, clauses
from .schema import RANK, validate_text
from .signature import signature_summary, timestamp
from .templates import response

class JournalSentinel:
    """Load once, analyze many times. No writes, network, UI or message sending."""
    def __init__(self, model_dir=None, model=None):
        self.model=model
        self.load_error=None
        if model is None:
            try:
                from .model import LocalModel, DEFAULT_MODEL
                self.model=LocalModel(model_dir or DEFAULT_MODEL)
            except (ImportError, OSError, ValueError, KeyError) as exc:
                self.load_error=type(exc).__name__

    def analyze(self,text,*,lang=None,entry_id=None,now=None,signature_phrases=(),signature_history=(),signature_enabled=False,last_support_at=None,adaptive_enabled=False,include_audit=False):
        text=validate_text(text)
        if lang not in (None,"en","ar"):
            raise ValueError("language must be en or ar")
        if entry_id is not None and (not isinstance(entry_id,str) or not 1<=len(entry_id)<=128):
            raise ValueError("entry_id must be a string of 1-128 characters")
        start=time.perf_counter()
        now=now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must include a timezone")
        lang=lang or language(text)
        hits=scan(text)
        prediction=None
        failure=self.load_error
        if self.model is not None:
            try:
                prediction=self.model.predict(text)
            except Exception as exc:
                # A runtime failure is explicit in output; do not log journal text.
                failure=type(exc).__name__
        tier,reason,rules=fuse(hits,prediction)
        scoped_audit=[]
        # Negation, attribution or an idiom in one clause must not suppress another.
        units=clauses(text)
        if prediction is not None and len(units)>1 and any(h.negated or h.quoted or h.temporal=="past" or h.subject=="other" or h.kind=="idiom" for h in hits):
            for index,unit in enumerate(units):
                try:
                    local_prediction=self.model.predict(unit)
                except Exception:
                    local_prediction=None
                local_tier,local_reason,local_rules=fuse([h for h in hits if h.clause==index],local_prediction)
                scoped_audit.append({"clause":index,"tier":local_tier,"reason":local_reason})
                if RANK[local_tier]>RANK[tier]:
                    tier,reason,rules=local_tier,"clause_"+local_reason,local_rules
                    if not local_rules.categories and local_prediction is not None:
                        rules.subject=local_prediction.subject
                        rules.temporal=local_prediction.temporal
        signature=signature_summary(text,signature_phrases,signature_history,entry_id,now,signature_enabled)
        if signature["threshold_reached"] and RANK[tier]<RANK["moderate_flagged"]:
            tier,reason="moderate_flagged","personal_phrase_recurrence"
        categories=sorted(set(rules.categories+(prediction.categories if prediction else [])))
        subject=rules.subject if hits else prediction.subject if prediction else "self"
        temporal=rules.temporal if hits else prediction.temporal if prediction else "current"
        cooldown=False
        if tier in ("moderate","moderate_flagged") and last_support_at:
            elapsed=now-timestamp(last_support_at)
            cooldown=timedelta(0)<=elapsed<timedelta(hours=3)
        action="crisis_support" if tier=="high" else "support_panel" if tier in ("moderate","moderate_flagged") and not cooldown else "optional_reflection" if tier!="none" else "none"
        theme="grounding_earth" if "paranoia" in categories else "calm_deep" if RANK[tier]>=RANK["moderate"] else "calm_gentle" if tier=="low" else None
        result={
            "schema_version":"1.0","entry_id":entry_id,"analyzed_at":now.isoformat(),"tier":tier,
            "categories":categories,"subject":subject,"temporal":temporal,
            "negated":rules.negated if hits else prediction.negated if prediction else False,
            "is_idiom":rules.is_idiom if hits else prediction.is_idiom if prediction else False,
            "response":response(tier,lang,categories,subject,prediction is None and tier!="high"),
            "action":{"type":action,"keep_input_editable":True,"preserve_text":True,"dismissible":True,"cooldown_applied":cooldown,"suggested_theme":theme if adaptive_enabled else None,"adaptive_enabled":adaptive_enabled,"disclosure_required":bool(adaptive_enabled and theme),"reason_code":reason,"exercises":["sensory_grounding","comfortable_breathing"] if tier!="none" else [],"motion":"none"},
            "review":{"recommended":RANK[tier]>=RANK["moderate"],"priority":"urgent" if tier=="high" else "review" if RANK[tier]>=RANK["moderate"] else "routine","delivery_status":"not_sent","trusted_contact_status":"not_contacted","requires_human_review":True},
            "signature":signature,
            "model":{"version":getattr(self.model,"version",None),"source":prediction.source if prediction else "rules_fallback","available":prediction is not None,"confidence":round(prediction.confidence,5) if prediction else None,"confidence_meaning":"classifier score, not probability of harm","clinically_validated":False},
        }
        if include_audit:
            result["audit"]={"lexicon_version":LEXICON_VERSION,"hits":[h.to_dict() for h in hits],"context":prediction.to_dict() if prediction else None,"rules":rules.to_dict(),"scoped_checks":scoped_audit,"fusion_reason":reason,"model_error_type":failure,"latency_ms":round((time.perf_counter()-start)*1000,3)}
        return result
