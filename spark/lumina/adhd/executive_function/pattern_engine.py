from collections import defaultdict
from datetime import timedelta
from .schemas import Candidate

def learn_patterns(outcomes,patient_id,now):
    latest={}
    for o in outcomes:
        if o.patientId!=patient_id or not now-timedelta(days=60)<=o.timestamp<=now: continue
        if o.attemptId not in latest or latest[o.attemptId].timestamp<o.timestamp: latest[o.attemptId]=o
    groups=defaultdict(list)
    for o in latest.values():
        if o.focusMinutes and o.outcome in ('DONE','PARTIAL','NOT_STARTED','TOO_HARD','TOO_LONG','INTERRUPTED'): groups[o.focusMinutes].append(o)
    candidates=[]
    eligible=[]
    for minutes,rs in groups.items():
        days=len({r.timestamp.date() for r in rs})
        if len(rs)<4 or days<2: continue
        done=sum(r.outcome=='DONE' for r in rs)
        rate=(done+1)/(len(rs)+2)
        eligible.append((rate,minutes,len(rs),days,done))
    if eligible:
        rate,minutes,count,days,done=max(eligible,key=lambda row:(row[0],-row[1]))
        if rate>=.6:
            candidates.append(Candidate(key='preferred_focus_duration',value=minutes,category='PREFERENCE',confidence=round(rate,3),evidenceCount=count,evidenceDays=days,explanation=f'{done}/{count} distinct attempts completed; smoothed completion fraction, not clinical confidence.'))
    hard=[r for r in latest.values() if r.outcome in ('TOO_HARD','TOO_LONG','NOT_STARTED')]
    if len(hard)>=4 and len({r.timestamp.date() for r in hard})>=2:
        candidates.append(Candidate(key='smaller_start',value='MINIMUM_STEP',category='PATTERN',confidence=round(len(hard)/max(len(latest),1),3),evidenceCount=len(hard),evidenceDays=len({r.timestamp.date() for r in hard}),explanation='Repeated difficulty reports; try a smaller starting action.'))
    evening=[r for r in latest.values() if r.complexity=='HIGH' and r.localHour is not None and r.localHour>=21 and r.outcome in ('DONE','PARTIAL','NOT_STARTED','TOO_HARD','TOO_LONG','INTERRUPTED')]
    if len(evening)>=4 and len({r.timestamp.date() for r in evening})>=2:
        incomplete=sum(r.outcome!='DONE' for r in evening)
        if incomplete/len(evening)>=.75:
            candidates.append(Candidate(key='evening_complex_tasks',value='LOW_COMPLETION',category='PATTERN',confidence=round(incomplete/len(evening),3),evidenceCount=len(evening),evidenceDays=len({r.timestamp.date() for r in evening}),explanation='Repeated evening complex-task outcomes; no causal or diagnostic claim.'))
    return candidates
