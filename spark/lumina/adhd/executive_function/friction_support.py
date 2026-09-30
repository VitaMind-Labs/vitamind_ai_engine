"""Turn a detected friction into a concrete move.

The friction classifier already names why the user is stuck. Without this module
that finding was computed and discarded: every struggle message, whatever it
described, was answered with the same "What is one task you want help with?".

Two shapes are offered per friction. ASK is used when no task is known yet: it
reflects the specific block back and asks the one question that unblocks it.
STEP prefixes a known next action, adapting how the task is approached.

These are authored coaching templates, not clinical instruments and not generated
text. They state nothing about the user that the message did not say, and they
never claim an outcome.
"""
from .language import choose

# Ordered by how much the block decides what to do next; the first match wins, so a
# message flagged both OVERWHELM and BORING is answered as overwhelm.
# UNCLEAR is deliberately absent: it is the classifier's catch-all and fires on plain
# queries such as "what do I have today?", where reflecting a block back is noise.
# Its templates stay defined so a future, better-separated label can use them.
PRIORITY=['INTERRUPTION','OVERWHELM','NO_CLEAR_START','TOO_BIG','AVOIDANCE',
          'PERFECTIONISM','ANXIETY','LOW_ENERGY','DISTRACTION','BORING','TOO_MANY_CHOICES','TIME_BLINDNESS']

ASK={
 'OVERWHELM':('When everything is loud at once, the way out is one item. Name the task that worries you most.',
              'عندما يتزاحم كل شيء، المخرج مهمة واحدة. اذكر المهمة التي تقلقك اكثر.'),
 'NO_CLEAR_START':('Not knowing where to start is a starting problem, not a willpower problem. Name one task and I will give you its first physical action.',
                   'عدم معرفة نقطة البداية مشكلة بداية، وليست مشكلة ارادة. اذكر مهمة واحدة واعطيك اول خطوة ملموسة فيها.'),
 'TOO_BIG':('A task feels too big when its first step is still hidden. Name it and I will cut it down to one step.',
            'تبدو المهمة كبيرة عندما تكون خطوتها الاولى غير واضحة. اذكرها واقسمها لك الى خطوة واحدة.'),
 'UNCLEAR':('Name the task, and what "done" would look like for it.',
            'اذكر المهمة، وما الذي يعنيه "انتهيت" بالنسبة لها.'),
 'AVOIDANCE':('Putting something off usually means it feels unpleasant, not that you are lazy. Which task is it?',
              'تأجيل شيء ما يعني غالبا انه مزعج، لا انك كسول. ما هي المهمة؟'),
 'PERFECTIONISM':('A rough version finished beats a perfect one never started. Which task is it?',
                  'نسخة غير مكتملة انتهت افضل من نسخة مثالية لم تبدأ. ما هي المهمة؟'),
 'ANXIETY':('Name the task. A smaller step usually lowers the pressure.',
            'اذكر المهمة. الخطوة الاصغر تخفف الضغط عادة.'),
 'LOW_ENERGY':('Low energy is a real constraint, not an excuse. Name one task and I will shrink it to fit today.',
               'انخفاض الطاقة قيد حقيقي، وليس عذرا. اذكر مهمة واحدة واصغّرها لتناسب يومك.'),
 'DISTRACTION':('Name the task you meant to do, and I will start with removing what pulls you away.',
                'اذكر المهمة التي كنت تنوي عملها، وسنبدأ بإبعاد ما يشتتك.'),
 'BORING':('Boring tasks stall on the first minute, not the whole job. Name it and I will make the start tiny.',
           'المهام المملة تتعثر في الدقيقة الاولى، لا في العمل كله. اذكرها واجعل البداية صغيرة جدا.'),
 'TOO_MANY_CHOICES':('Too many options is its own block. Name any two tasks and I will pick one.',
                     'كثرة الخيارات عائق بحد ذاته. اذكر اي مهمتين واختار لك واحدة.'),
 'TIME_BLINDNESS':('Name the task and I will attach a length to it before you start.',
                   'اذكر المهمة واحدد لها مدة قبل ان تبدأ.'),
 'INTERRUPTION':('Which task were you working on before the interruption?',
                 'ما المهمة التي كنت تعمل عليها قبل المقاطعة؟'),
}

STEP={
 'OVERWHELM':('Everything else can wait. Only this: ','كل ما عداها ينتظر. هذه فقط: '),
 'NO_CLEAR_START':('The first physical action: ','اول خطوة ملموسة: '),
 'TOO_BIG':('Only the first slice, not the whole task: ','الجزء الاول فقط، لا المهمة كلها: '),
 'UNCLEAR':('Decide what "done" means, then: ','حدد معنى "انتهيت"، ثم: '),
 'AVOIDANCE':('Five minutes, and you may stop when they are up: ','خمس دقائق، ولك ان تتوقف بعدها: '),
 'PERFECTIONISM':('A rough first pass is allowed: ','يُسمح بنسخة اولى غير متقنة: '),
 'ANXIETY':('One small step, nothing beyond it: ','خطوة صغيرة واحدة، لا اكثر: '),
 'LOW_ENERGY':('The smallest useful version: ','اصغر نسخة مفيدة: '),
 'DISTRACTION':('Put one distraction out of reach, then: ','ابعد مصدر تشتيت واحد، ثم: '),
 'BORING':('Ten minutes only, then decide whether to go on: ','عشر دقائق فقط، ثم قرر الاستمرار: '),
 'TOO_MANY_CHOICES':('Take this one and drop the comparison: ','خذ هذه واترك المقارنة: '),
 'TIME_BLINDNESS':('Set a visible timer first: ','اضبط مؤقتا ظاهرا اولا: '),
 'INTERRUPTION':('Pick up where you stopped: ','اكمل من حيث توقفت: '),
}

# Some intents name the block outright. When the friction classifier found nothing,
# that label is the evidence available, so it stands in rather than being discarded.
FROM_INTENT={'BREAK_DOWN_TASK':'TOO_BIG','PROCRASTINATION':'AVOIDANCE',
             'OVERWHELMED_WITH_TASKS':'OVERWHELM','FOCUS_HELP':'DISTRACTION',
             'INTERRUPTION_RECOVERY':'INTERRUPTION','START_TASK':'NO_CLEAR_START'}

def leading(friction,intent=None):
    """The one friction that should shape the reply, or None if nothing informative."""
    return next((f for f in PRIORITY if f in (friction or [])),None) or FROM_INTENT.get(intent)

def ask_for_task(friction,lang,intent=None):
    """Reflect the block back and ask the question that unblocks it."""
    key=leading(friction,intent)
    if not key: return None
    return choose(lang,*ASK[key])

def adapt_step(friction,lang,action_text,intent=None):
    """Prefix a known next action with how to approach it under this block."""
    key=leading(friction,intent)
    if not key: return None
    return choose(lang,*STEP[key])+action_text
