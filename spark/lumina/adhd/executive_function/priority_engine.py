import re
from .language import normalize
from .schemas import RankedTask, PriorityFactor

# What a task's own name says about how movable it is. An appointment or a prayer has a
# time that will not wait for the patient; a coffee with a friend can be moved. Neither
# says anything about importance to the patient, so the weights stay small: stated
# deadlines and importance always outrank them, and they only order tasks that nothing
# else separates. The reason is reported in the factor, so the order is never a guess.
FIXED_TIME=re.compile(r"\b(?:appointment|doctor|dentist|clinic|hospital|therapist|psychiatrist|physio|checkup|check-up|meeting|interview|exam|class|lecture|school|flight|train|shift|pray|praying|prayer|salah|salat)\b|موعد|طبيب|دكتور|عياد[هة]|مستشفى|اجتماع|مقابل[هة]|امتحان|محاضر[هة]|حص[هة]|صلا[هة]|اصلي|صلي",re.I)
FLEXIBLE_SOCIAL=re.compile(r"\b(?:friends?|coffee|hang out|catch up|party|movie|game|gaming|shopping)\b|صديق|اصدقاء|قهو[هة]|سهر[هة]|فيلم",re.I)

DEFAULT_WEIGHTS={'overdue':12,'due_today':10,'due_soon':6,'importance':5,'consequence':5,'quick_win':2,'time_mismatch':-5,'energy_mismatch':-4,'cognitive_mismatch':-3,'startup_mismatch':-2,'postponement':2,'location_match':2,'location_mismatch':-3,'evening_complex':-4,'fixed_time':3,'flexible_social':-1,'stated_time':1}

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
        name=normalize(task.title)
        # Studying for an exam is preparation, not the exam: only the event is fixed.
        preparing=re.match(r'^(?:study|revise|prepare|read|review|اذاكر|ادرس|اراجع)\b',name)
        if FIXED_TIME.search(name) and not preparing: factor('fixed_time','named appointment or prayer: its time cannot move for you')
        elif FLEXIBLE_SOCIAL.search(name) and not task.startTime: factor('flexible_social','named social plan: it can be moved if the day is full')
        if task.startTime: factor('stated_time','a clock time was stated for this task')
        if task.location and request.currentState.location:
            factor('location_match' if task.location.casefold()==request.currentState.location.casefold() else 'location_mismatch','provided location comparison')
        if request.timeContext.localTime and request.timeContext.localTime.hour>=21 and task.cognitiveLoad=='HIGH' and any(p.key=='evening_complex_tasks' for p in patterns):
            factor('evening_complex','repeated personal outcome pattern and current local hour')
        ranked.append(RankedTask(task=task,score=sum(f.points for f in factors),factors=factors,blocked=blocked,bucket='DEFER' if blocked else 'OPTIONAL'))
    # Stable input order resolves ties; no unsupported deadline or importance is inferred.
    return sorted(ranked,key=lambda r:(r.blocked,-r.score))
