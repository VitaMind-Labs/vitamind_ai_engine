from .schemas import FocusSession

def choose_focus(request,capacity,available,action,patterns):
    if action is None or (available is not None and available<5): return None
    minutes=5 if capacity in ('VERY_LOW','REDUCED') else 10
    basis='capacity-aware starting block, not a duration estimate'
    pref=request.patient.preferences.preferredFocusMinutes
    if pref is not None:
        minutes=pref; basis='explicit patient preference'
    else:
        relevant=[m for m in request.relevantMemory if m.key=='preferred_focus_duration' and m.value in (5,10,15,25) and m.confidence>=.7 and m.evidenceCount>=4]
        learned=[p for p in patterns if p.key=='preferred_focus_duration']
        if learned: minutes=learned[0].value; basis='repeated distinct outcome history'
        elif relevant: minutes=relevant[0].value; basis='provided repeated-evidence memory'
    if capacity in ('REDUCED','VERY_LOW'): minutes=min(minutes,5)
    if available is not None:
        allowed=[m for m in (5,10,15,25) if m<=available and m<=minutes]
        if not allowed: return None
        minutes=max(allowed)
    return FocusSession(minutes=minutes,successCondition=action.text,basis=basis)
