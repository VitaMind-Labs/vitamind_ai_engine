from datetime import timedelta

ORDER={'VERY_LOW':0,'REDUCED':1,'NORMAL':2,'HIGH':3}
def capacity(request,friction,day):
    state=request.currentState
    value=state.capacity or 'NORMAL'; evidence=[]
    if state.capacity: evidence.append('EXPLICIT_CAPACITY')
    if state.energy=='LOW' or 'LOW_ENERGY' in friction:
        value=min((value,'REDUCED'),key=ORDER.get); evidence.append('LOW_ENERGY')
    if state.stress=='HIGH' or 'OVERWHELM' in friction:
        value=min((value,'REDUCED'),key=ORDER.get); evidence.append('STRESS_OR_OVERWHELM')
    # Old/future journals must not silently set current capacity.
    journals=[j for j in request.journalContext if day and day-timedelta(days=2)<=j.date<=day]
    if journals:
        j=max(journals,key=lambda j:j.date).signals
        if j.energy=='LOW' or j.overwhelm or (j.sleep=='LOW' and j.stress=='HIGH'):
            value=min((value,'REDUCED'),key=ORDER.get); evidence.append('RECENT_JOURNAL')
        if j.sleep=='LOW' and j.energy=='LOW' and j.stress=='HIGH':
            value='VERY_LOW'; evidence.append('RECENT_SLEEP_ENERGY_STRESS')
    if not evidence: evidence.append('CONSERVATIVE_DEFAULT_NOT_MEASURED')
    return value,evidence
