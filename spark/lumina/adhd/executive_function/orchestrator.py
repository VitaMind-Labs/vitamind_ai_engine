from datetime import datetime,timezone,timedelta
from time import perf_counter
from uuid import uuid5
from .schemas import *
from .language import choose,languages,normalize,mentions,referenced_task
from .intent_classifier import IntentClassifier
from .friction_classifier import FrictionClassifier
from .time_engine import available_minutes,parse_temporal
from .task_extractor import extract_tasks
from .capacity_engine import capacity
from .priority_engine import rank_tasks
from .task_breakdown import first_action
from .focus_engine import choose_focus
from .pattern_engine import learn_patterns
from .response_engine import compose
from .errors import AssistantError
from ...safety.adapter import SafetyAdapter

# Recorded for transparency, but they answer themselves; they never gate a save.
# The deadline codes belong here because the tasks are certain even when the due
# date is not: losing "call the bank" and "buy groceries" because a trailing
# "they're due Friday" was ambiguous would be far worse than saving them undated.
INFORMATIONAL_NOTES={'RECURRING_TASK_SAVED_ONCE','DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS',
                     'DEADLINE_CLAUSE_CONFLICTS_WITH_TASK_DATE','DEADLINE_CLAUSE_NOT_RESOLVED',
                     'DEADLINE_CLAUSE_WITHOUT_TASK'}

class LuminaADHD:
    def __init__(self,calendar=None,weights=None):
        self.calendar=calendar
        self.intent_classifier=IntentClassifier()
        self.friction_classifier=FrictionClassifier()
        self.safety=SafetyAdapter()
        self.weights=weights
    @property
    def ready(self): return self.safety.ready

    def organize(self,request,now=None):
        start=perf_counter()
        if not isinstance(request,OrganizeRequest): request=OrganizeRequest.model_validate(request)
        now=now or datetime.now(timezone.utc)
        if any(o.timestamp>now+timedelta(minutes=5) for o in request.outcomes):
            raise AssistantError('INVALID_REQUEST','outcomes cannot be dated in the future')
        if request.calendar.commit and not self.calendar:
            raise AssistantError('INVALID_REQUEST','calendar.commit requires an explicitly configured local calendar')
        if request.calendar.commit:
            cached=self.calendar.cached(request)
            if cached: return cached
        safety,safety_text=self.safety.check(request)
        intent,confidence,source=self.intent_classifier.predict(request.message.text)
        friction,friction_source=self.friction_classifier.predict(request.message.text)
        day=request.timeContext.day or request.timeContext.referenceDate
        cap,cap_evidence=capacity(request,friction,request.timeContext.referenceDate)
        uncertainty=[]
        analysis=Analysis(taskCount=0,needsClarification=False,dataQuality='LIMITED',uncertainty=uncertainty,capacityEvidence=cap_evidence,intentSource=source,intentConfidence=confidence,frictionSource=friction_source,languagesObserved=languages(request.message.text))
        if safety.level!='NORMAL':
            return OrganizeResponse(requestId=request.requestId,language=request.patient.language,intent=intent,friction=friction,capacity=cap,analysis=analysis,plan=Plan(strategy='SAFETY_HANDOFF',day=day),response=Response(text=safety_text,templateId='safety.handoff'),followUp=FollowUp(type='NONE'),safety=safety,performance=Performance(processingTimeMs=round((perf_counter()-start)*1000,3)))
        supplied=[t.model_copy(deep=True) for t in request.tasks]
        stored=self.calendar.tasks(request.patient.id) if self.calendar else []
        existing={t.temporaryId:t for t in stored}
        for i,t in enumerate(supplied):
            if not t.temporaryId: t.temporaryId='task_'+uuid5(request.requestId,'provided:'+str(i)).hex[:16]
            existing[t.temporaryId]=t
        extracted,issues=extract_tasks(request.message.text,request.requestId,request.timeContext.referenceDate)
        uncertainty.extend(issues)
        # Notes that describe how a task was recorded are reported, but they are not
        # unresolved questions, so they must not stop the task from being saved.
        issues=[i for i in issues if i not in INFORMATIONAL_NOTES]
        # Naming a task is not a report of being interrupted. "I forgot to pay rent"
        # carries the task itself, so it is treated the same way as an unknown label.
        if extracted and intent in ('UNKNOWN','INTERRUPTION_RECOVERY'): intent='ADD_TASK'
        # A date in a planning query chooses the calendar day; a task date is parsed per task.
        query_time=parse_temporal(request.message.text,request.timeContext.referenceDate)
        if not extracted and intent in ('ORGANIZE_DAY','PRIORITIZE'):
            day=query_time['date'] or day
            uncertainty.extend(query_time['uncertainty'])
            issues.extend(query_time['uncertainty'])
        operations=[]
        resolved=[]
        for t in extracted:
            duplicate=next((x for x in existing.values() if normalize(x.title)==normalize(t.title) and x.scheduledDate==t.scheduledDate and x.status=='TODO'),None)
            # "I keep putting off the email" points at the open "email James"; it is
            # not a second commitment. The turn is then about the existing task.
            if not duplicate: duplicate=referenced_task(t.title,[x for x in existing.values() if x.status=='TODO'])
            if duplicate:
                resolved.append(duplicate); continue
            existing[t.temporaryId]=t
            resolved.append(t)
            operations.append(TaskOperation(operation='ADD',task=t,taskId=t.temporaryId))
        extracted=resolved
        outcomes=(self.calendar.outcomes(request.patient.id) if self.calendar else [])+request.outcomes
        patterns=learn_patterns(outcomes,request.patient.id,now)
        for o in request.outcomes:
            if o.outcome=='DONE' and o.taskId in existing:
                existing[o.taskId].status='DONE'
                operations.append(TaskOperation(operation='COMPLETE',taskId=o.taskId))
        # A completion stated in the message closes the named task. Only an existing
        # task can be closed, and only one named plainly enough to match; a vague
        # "I'm done" never guesses which commitment the user meant.
        completed_now=[]
        if intent=='TASK_COMPLETED':
            said=normalize(request.message.text)
            completed_now=[t for t in existing.values() if t.status=='TODO' and mentions(t.title,said)]
            for t in completed_now:
                t.status='DONE'
                operations.append(TaskOperation(operation='COMPLETE',taskId=t.temporaryId))
        all_tasks=list(existing.values())
        # Daily plans exclude other days but can include unscheduled tasks explicitly supplied now.
        tasks=[t for t in all_tasks if t.scheduledDate is None or day is None or t.scheduledDate==day]
        # A task named in this message is what the turn is about, whichever label the
        # classifier produced. Keying this on ADD_TASK alone meant "I have to buy
        # groceries tomorrow", read as ORGANIZE_DAY, was filtered out by today's day
        # window and answered with a clarifying question instead of being saved.
        if extracted:
            tasks=extracted
            if len({t.scheduledDate for t in extracted})==1: day=extracted[0].scheduledDate or day
        available=request.timeContext.availableMinutes
        if available is None: available=available_minutes(request.message.text)
        if 'UNKNOWN' in friction and len(tasks)>=10:
            uncertainty.append('MANY_TASKS_WITHOUT_CAPACITY_SIGNAL')
        completed_ids={t.temporaryId for t in all_tasks if t.status=='DONE'}
        ranks=rank_tasks(tasks,request,cap,available,day,patterns,self.weights,completed_ids)
        eligible=[r for r in ranks if not r.blocked]
        clarification=None
        primary=eligible[0].task if eligible else None
        action=first_action(primary,request.patient.language,cap) if primary else None
        strategy='ONE_NEXT_ACTION'
        if intent=='INTERRUPTION_RECOVERY':
            context=sorted((c for c in request.recentContext if now-timedelta(days=7)<=c.timestamp<=now),key=lambda c:c.timestamp,reverse=True)
            context=next((c for c in context if c.taskId in existing and existing[c.taskId].status=='TODO' and c.nextStep),None)
            if context:
                primary=existing[context.taskId]
                action=NextAction(text=context.nextStep,source='PROVIDED_CONTEXT',templateId='recovery.provided')
            else:
                primary=action=None
                clarification=choose(request.patient.language,'Which task were you working on before the interruption?','ما المهمة التي كنت تعمل عليها قبل المقاطعة؟')
        elif intent=='RESCHEDULE':
            # Never guess which event a pronoun refers to.
            target=next((t for t in all_tasks if t.status=='TODO' and (normalize(t.title) in normalize(request.message.text) or mentions(t.title,request.message.text))),None)
            if target and query_time['date'] and not query_time['uncertainty']:
                target=target.model_copy(update={'scheduledDate':query_time['date']})
                primary=target; action=first_action(target,request.patient.language,cap)
                operations.append(TaskOperation(operation='RESCHEDULE',task=target,taskId=target.temporaryId))
                day=query_time['date']
                ranks=rank_tasks([target],request,cap,available,day,patterns,self.weights,completed_ids)
            else:
                primary=action=None
                clarification=choose(request.patient.language,'Name the task and the new date you want.','اذكر المهمة والتاريخ الجديد الذي تريده.')
        elif intent=='TASK_COMPLETED' and not completed_now and not extracted:
            # Saying something is finished is not a request to start work. If no open
            # task is named plainly enough to close, ask which one instead of
            # answering with the next action for a different task.
            primary=action=None
            clarification=choose(request.patient.language,'Which task did you finish?','ما المهمة التي أنهيتها؟')
        elif completed_now and not primary:
            pass
        elif not primary:
            clarification=choose(request.patient.language,'What is one task you want help with?','ما المهمة الواحدة التي تريد المساعدة فيها؟') if not tasks else choose(request.patient.language,'Which dependency needs to be completed before these tasks can start?','ما المهمة السابقة التي يجب ان تنتهي قبل بدء هذه المهام؟')
        if issues:
            clarification=choose(request.patient.language,'Please clarify the task date or time; I have not assumed one.','وضح تاريخ المهمة او وقتها؛ لم افترض موعدا من عندي.')
        if available==0:
            primary=action=None; strategy='NO_TIME'; clarification=None
        elif clarification: strategy='CLARIFY'
        elif completed_now and not primary: strategy='TASK_COMPLETED_ACK'
        elif not primary: strategy='NO_TASKS'
        secondary=[]
        if primary and cap in ('NORMAL','HIGH') and intent in ('ORGANIZE_DAY','PRIORITIZE'):
            strategy='SHORT_DAY_PLAN'
            # Unknown durations never count as fitting a time budget.
            remaining=available-primary.durationMinutes-2 if available is not None and primary.durationMinutes is not None else None
            for r in eligible:
                if r.task.temporaryId==primary.temporaryId: continue
                if len(secondary)>=2: break
                if remaining is not None and r.task.durationMinutes is not None and r.task.durationMinutes+2<=remaining:
                    secondary.append(r.task); remaining-=r.task.durationMinutes+2
                elif available is None:
                    # No time budget was stated, so this is a question about what the
                    # day holds, not a request to fit work into a window. List the
                    # other commitments; the uncertainty flags below record that no
                    # schedule was computed for them.
                    secondary.append(r.task)
            if available is None: uncertainty.append('AVAILABLE_TIME_UNKNOWN_NO_TIMED_SCHEDULE')
            elif primary.durationMinutes is None: uncertainty.append('TASK_DURATION_UNKNOWN_STARTER_ONLY')
        for r in ranks:
            r.bucket='PRIMARY' if primary and r.task.temporaryId==primary.temporaryId else 'SECONDARY' if r.task in secondary else r.bucket
        if primary and primary.durationMinutes is None: uncertainty.append('DURATION_NOT_PROVIDED')
        if intent=='TIME_ESTIMATION' and primary and primary.durationMinutes is None:
            uncertainty.append('NO_EVIDENCE_FOR_DURATION_ESTIMATE')
            clarification=choose(request.patient.language,'How long did a similar task take you before?','كم استغرقت مهمة مشابهة منك من قبل؟')
            strategy='CLARIFY'
        if primary:
            guarded=self.safety.check_action_text(primary.title,request.patient.language)
            if guarded is None and action and action.source=='PROVIDED_CONTEXT':
                guarded=self.safety.check_action_text(action.text,request.patient.language)
            if guarded:
                guarded_safety,guarded_text=guarded
                return OrganizeResponse(requestId=request.requestId,language=request.patient.language,intent=intent,friction=friction,capacity=cap,analysis=analysis,plan=Plan(strategy='SAFETY_HANDOFF',day=day),response=Response(text=guarded_text,templateId='safety.action_boundary'),followUp=FollowUp(type='NONE'),safety=guarded_safety,performance=Performance(processingTimeMs=round((perf_counter()-start)*1000,3)))
        if clarification:
            primary=action=None
            secondary=[]
            strategy='CLARIFY'
            for rank in ranks:
                if rank.bucket in ('PRIMARY','SECONDARY'): rank.bucket='OPTIONAL'
        plan=Plan(strategy=strategy,primaryTask=primary,nextAction=action,secondaryTasks=secondary,rankedTasks=ranks,day=day)
        future_plan=bool(day and request.timeContext.referenceDate and day>request.timeContext.referenceDate)
        focus=None if clarification or future_plan else choose_focus(request,cap,available,action,patterns)
        analysis=Analysis(taskCount=len(tasks),needsClarification=bool(clarification),dataQuality='MODERATE' if tasks and not issues else 'LIMITED',uncertainty=sorted(set(uncertainty)),clarification=clarification,capacityEvidence=cap_evidence,intentSource=source,intentConfidence=confidence,frictionSource=friction_source,languagesObserved=languages(request.message.text))
        result=OrganizeResponse(requestId=request.requestId,language=request.patient.language,intent=intent,friction=friction,capacity=cap,analysis=analysis,plan=plan,focusSession=focus,response=compose(request,plan,cap,intent,clarification,completed_now,friction,uncertainty),taskOperations=operations,memoryCandidates=[p for p in patterns if p.category=='PREFERENCE'],patternCandidates=[p for p in patterns if p.category=='PATTERN'],followUp=FollowUp(type='CLARIFICATION' if clarification else 'AFTER_ACTION' if action and not future_plan else 'NONE',recommendedAfterMinutes=focus.minutes if focus else None),safety=safety,performance=Performance(processingTimeMs=round((perf_counter()-start)*1000,3)))
        if request.calendar.commit and not clarification:
            return self.calendar.commit(request,result)
        return result
