"""Opt-in clinician-defined phrases; distinct saved entries, never keystrokes."""
from datetime import datetime, timedelta, timezone
from .normalize import contains_phrase

def timestamp(value):
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)

def signature_summary(text, phrases=(), history=(), entry_id=None, now=None, enabled=False, threshold=3, window_days=5):
    now=now or datetime.now(timezone.utc)
    if not enabled:
        return {"enabled":False,"matched_phrases":[],"matching_entries":0,"threshold_reached":False}
    if not entry_id or not 1<=threshold<=20 or not 1<=window_days<=90:
        raise ValueError("signature tracking requires entry_id and valid window/threshold")
    if len(phrases)>20 or any(not isinstance(p,str) or len(p.strip())<3 or len(p)>120 for p in phrases):
        raise ValueError("use up to 20 phrases, each 3-120 characters")
    matched=[p for p in dict.fromkeys(phrases) if contains_phrase(text,p)]
    recent={}
    for row in history:
        when=timestamp(row["timestamp"])
        if now-timedelta(days=window_days)<=when<=now and row["entry_id"]!=entry_id:
            # Caller supplies prior match metadata, not raw journal history.
            if set(row.get("matched_phrases",[])) & set(phrases):
                recent[row["entry_id"]]=True
    if matched:
        recent[entry_id]=True
    return {"enabled":True,"matched_phrases":matched,"matching_entries":len(recent),"threshold":threshold,"window_days":window_days,"threshold_reached":len(recent)>=threshold,"method":"normalized_phrase_match","clinical_interpretation":"Personal phrase recurrence only; not a relapse diagnosis."}
