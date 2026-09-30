from .schemas import RankedTask, PriorityFactor

DEFAULT_WEIGHTS={'overdue':12,'due_today':10,'due_soon':6,'importance':5,'consequence':5,'quick_win':2,'time_mismatch':-5,'energy_mismatch':-4,'cognitive_mismatch':-3,'startup_mismatch':-2,'postponement':2,'location_match':2,'location_mismatch':-3,'evening_complex':-4}

def rank_tasks(tasks,request,capacity,available,day,patterns,weights=None,completed_ids=None):
    weights={**DEFAULT_WEIGHTS,**(weights or {})}
    done={t.temporaryId for t in tasks if t.status=='DONE'} | set(completed_ids or ())
    ranked=[]
    for task in tasks:
        if task.status!='TODO': continue
        factors=[]
        def factor(key,evidence): factors.append(PriorityFactor(name=key,points=weights[key],evidence=evidence))
        blocked=any(d not in done for d in task.dependencies)
        if task.deadline and day:
            delta=(task.deadline-day).days
            if delta<0: factor('overdue','provided deadline before planning day')
            elif delta==0: factor('due_today','provided deadline equals planning day')
            elif delta<=2: factor('due_soon','provided deadline within two days')
        if task.importance=='HIGH': factor('importance','explicit patient importance HIGH')
        if task.consequence=='HIGH': factor('consequence','provided consequence HIGH')
        if task.durationMinutes is not None:
            if task.durationMinutes<=5: factor('quick_win','provided duration <= 5 minutes')
            if available is not None and task.durationMinutes>available: factor('time_mismatch','whole task exceeds available time; only a starting step can be proposed')
        if capacity in ('REDUCED','VERY_LOW'):
            if task.energyRequirement=='HIGH': factor('energy_mismatch','provided high energy need and reduced capacity')
            if task.cognitiveLoad=='HIGH': factor('cognitive_mismatch','provided high cognitive load and reduced capacity')
            if task.startupDifficulty=='HIGH': factor('startup_mismatch','provided high startup difficulty and reduced capacity')
        if task.postponedCount>=3: factor('postponement','provided count >= 3')
        if task.location and request.currentState.location:
            factor('location_match' if task.location.casefold()==request.currentState.location.casefold() else 'location_mismatch','provided location comparison')
        if request.timeContext.localTime and request.timeContext.localTime.hour>=21 and task.cognitiveLoad=='HIGH' and any(p.key=='evening_complex_tasks' for p in patterns):
            factor('evening_complex','repeated personal outcome pattern and current local hour')
        ranked.append(RankedTask(task=task,score=sum(f.points for f in factors),factors=factors,blocked=blocked,bucket='DEFER' if blocked else 'OPTIONAL'))
    # Stable input order resolves ties; no unsupported deadline or importance is inferred.
    return sorted(ranked,key=lambda r:(r.blocked,-r.score))
