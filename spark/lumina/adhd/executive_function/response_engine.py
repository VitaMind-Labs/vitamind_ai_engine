from .schemas import Response
from .language import choose
from .friction_support import ask_for_task,adapt_step,leading

# Reasons that come from the patient's own deadlines or ratings. When none of them
# separated the tasks, the order rests on the names alone and the reply says so.
NL=chr(10)
STATED_FACTORS={'overdue','due_today','due_soon','importance','consequence','stated_time'}

def _shown(title):
    return title[:1].upper()+title[1:]

def _order_is_from_names(plan):
    ranked=[r for r in plan.rankedTasks if not r.blocked]
    return len(ranked)>1 and not any(f.name in STATED_FACTORS for r in ranked for f in r.factors)

def compose(request,plan,capacity,intent,clarification=None,completed=(),friction=(),notes=(),added=()):
    lang=request.patient.language
    variant=request.requestId.int%2
    if clarification:
        # The friction classifier has already named why this is hard. Ask the question
        # that fits that block instead of the same generic prompt every time.
        # "What should I do first today?" is a planning question, not a report of being
        # unable to start, so those two labels do not override the request for a list.
        asked_to_plan=intent in ('ORGANIZE_DAY','PRIORITIZE') and leading(friction,intent) in ('NO_CLEAR_START','TOO_MANY_CHOICES')
        targeted=ask_for_task(friction,lang,intent) if plan.strategy=='CLARIFY' and not plan.primaryTask and not asked_to_plan else None
        text=targeted or clarification; key='clarify_friction' if targeted else 'clarify'
    elif completed:
        # Name what was closed, then offer at most one next thing. Nothing is invented:
        # the follow-on is only mentioned when a task is actually ranked next.
        done='; '.join(t.title for t in completed)
        text=choose(lang,f'Marked done: {done}.',f'تم تسجيل الإنجاز: {done}.')
        if plan.nextAction: text+=choose(lang,f' Next: {plan.nextAction.text}',f' التالي: {plan.nextAction.text}')
        key='completed'
    elif plan.primaryTask and plan.nextAction and plan.day and request.timeContext.referenceDate and plan.day>request.timeContext.referenceDate:
        # "Proposed" describes something just captured. A question about a day that is
        # already planned gets told what is on it, not that it is being proposed.
        if intent in ('ORGANIZE_DAY','PRIORITIZE'):
            listed='; '.join([plan.primaryTask.title]+[t.title for t in plan.secondaryTasks])
            text=choose(lang,f'On {plan.day.isoformat()}: {listed}. When you start: {plan.nextAction.text}',
                             f'يوم {plan.day.isoformat()}: {listed}. عندما تبدا: {plan.nextAction.text}'); key='future_day'
        else:
            text=choose(lang,f'Proposed for {plan.day.isoformat()}. When you start: {plan.nextAction.text}',f'الموعد المقترح {plan.day.isoformat()}. عندما تبدا: {plan.nextAction.text}'); key='future_task'
    elif plan.strategy=='NO_TIME':
        text=choose(lang,'There is no available time in this plan. Choose a later time when you are ready.','لا يوجد وقت متاح في هذه الخطة. اختر وقتا لاحقا عندما تكون مستعدا.'); key='no_time'
    elif not plan.nextAction:
        text=choose(lang,'Tell me one task you would like help starting.','اذكر مهمة واحدة تحب ان اساعدك في بدئها.'); key='ask_task'
    elif intent=='INTERRUPTION_RECOVERY' and plan.nextAction.source=='PROVIDED_CONTEXT':
        text=choose(lang,f'You were working on {plan.primaryTask.title}. Continue with: {plan.nextAction.text}',f'كنت تعمل على {plan.primaryTask.title}. اكمل من هنا: {plan.nextAction.text}'); key='resume'
    elif capacity in ('REDUCED','VERY_LOW') and not (plan.strategy=='SHORT_DAY_PLAN' and plan.secondaryTasks and intent in ('ORGANIZE_DAY','PRIORITIZE')):
        lead=choose(lang,'Just one small start. ' if variant else 'One thing for now. ','بداية صغيرة فقط. ' if variant else 'شيء واحد الآن. ')
        text=lead+(adapt_step(friction,lang,plan.nextAction.text,intent) or plan.nextAction.text); key='one_start'
    elif plan.strategy=='SHORT_DAY_PLAN' and plan.secondaryTasks and intent in ('ORGANIZE_DAY','PRIORITIZE'):
        # A "what does my day hold" question is answered with the whole day in order,
        # then the first step. Every open task is listed: the plan's secondary list is
        # capped for the UI, but a person who asked to prioritise four things must not
        # be shown three of them.
        order=[r.task for r in plan.rankedTasks if not r.blocked and r.task.status=='TODO']
        if plan.primaryTask.temporaryId not in {t.temporaryId for t in order}: order.insert(0,plan.primaryTask)
        trimmed=capacity in ('REDUCED','VERY_LOW') and len(order)>3
        if capacity in ('REDUCED','VERY_LOW'): order=order[:3]
        items=NL.join(f'{i}. {_shown(t.title)}' for i,t in enumerate(order,1))
        text=choose(lang,f'Here is the order I would take them in:{NL}{items}{NL}Start here: {plan.nextAction.text}',
                         f'هذا الترتيب الذي اقترحه:{NL}{items}{NL}ابدا من هنا: {plan.nextAction.text}')
        if _order_is_from_names(plan):
            text+=NL+choose(lang,'I ordered by what has a fixed time first. Tell me a deadline or what matters most and I will reorder.',
                                 'رتبتها بحيث تأتي المواعيد الثابتة اولا. اخبرني بموعد نهائي او بالاهم لديك وسأعيد الترتيب.')
        if trimmed: text+=NL+choose(lang,'The rest can wait until these are done.','والباقي يمكنه الانتظار حتى تنتهي من هذه.')
        key='day_overview'
    else:
        adapted=adapt_step(friction,lang,plan.nextAction.text,intent)
        text=adapted or choose(lang,'Start here: ','ابدا من هنا: ')+plan.nextAction.text
        key='start_friction' if adapted else 'start'
        if plan.secondaryTasks and request.patient.preferences.responseLength=='NORMAL':
            text+=choose(lang,' After that: ',' بعد ذلك: ')+'; '.join(t.title for t in plan.secondaryTasks)
    # Saying "I need to X and Y" saves X and Y; the reply confirms what was written so the
    # patient can see the list they were just given is the list that was kept.
    if added and key in ('start','start_friction','one_start','future_task','day_overview'):
        names='; '.join(_shown(t.title) for t in added)
        text=choose(lang,f'Added: {names}. ',f'تمت الاضافة: {names}. ')+text
    # The tasks were saved; only the due date could not be placed. Say so in one line
    # rather than dropping it silently or refusing the whole message.
    if 'DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS' in notes:
        text+=choose(lang,' Saved without a due date - tell me which one is due, and when.',
                          ' حُفظت دون موعد نهائي - اخبرني اي واحدة لها موعد، ومتى.')
    return Response(text=text,length='SHORT' if capacity in ('REDUCED','VERY_LOW') else request.patient.preferences.responseLength,templateId=f'en-ar-v1.{lang}.{key}.{variant}')
