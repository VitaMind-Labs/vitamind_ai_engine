from .schemas import Response
from .language import choose
from .friction_support import ask_for_task,adapt_step

def compose(request,plan,capacity,intent,clarification=None,completed=(),friction=(),notes=()):
    lang=request.patient.language
    variant=request.requestId.int%2
    if clarification:
        # The friction classifier has already named why this is hard. Ask the question
        # that fits that block instead of the same generic prompt every time.
        targeted=ask_for_task(friction,lang,intent) if plan.strategy=='CLARIFY' and not plan.primaryTask else None
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
    elif capacity in ('REDUCED','VERY_LOW'):
        lead=choose(lang,'Just one small start. ' if variant else 'One thing for now. ','بداية صغيرة فقط. ' if variant else 'شيء واحد الآن. ')
        text=lead+(adapt_step(friction,lang,plan.nextAction.text,intent) or plan.nextAction.text); key='one_start'
    elif plan.strategy=='SHORT_DAY_PLAN' and plan.secondaryTasks and intent in ('ORGANIZE_DAY','PRIORITIZE'):
        # A "what does my day hold" question is answered with the day, then the step.
        rest='; '.join(t.title for t in plan.secondaryTasks)
        text=choose(lang,f'{plan.primaryTask.title} first, then: {rest}. Start here: {plan.nextAction.text}',
                         f'ابدا بـ {plan.primaryTask.title}، ثم: {rest}. ابدا من هنا: {plan.nextAction.text}')
        key='day_overview'
    else:
        adapted=adapt_step(friction,lang,plan.nextAction.text,intent)
        text=adapted or choose(lang,'Start here: ','ابدا من هنا: ')+plan.nextAction.text
        key='start_friction' if adapted else 'start'
        if plan.secondaryTasks and request.patient.preferences.responseLength=='NORMAL':
            text+=choose(lang,' After that: ',' بعد ذلك: ')+'; '.join(t.title for t in plan.secondaryTasks)
    # The tasks were saved; only the due date could not be placed. Say so in one line
    # rather than dropping it silently or refusing the whole message.
    if 'DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS' in notes:
        text+=choose(lang,' Saved without a due date - tell me which one is due, and when.',
                          ' حُفظت دون موعد نهائي - اخبرني اي واحدة لها موعد، ومتى.')
    return Response(text=text,length='SHORT' if capacity in ('REDUCED','VERY_LOW') else request.patient.preferences.responseLength,templateId=f'en-ar-v1.{lang}.{key}.{variant}')
