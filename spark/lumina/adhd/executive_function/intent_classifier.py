import re
from .language import normalize

RULES_VERSION='intent-rules-1.0.0'
INTENT_RULES=[
('INTERRUPTION_RECOVERY',r"forgot what i was doing|where was i|interrupted|نسيت (?:شو|ايش|ماذا) كنت|انقطعت|قاطعني"),
('TASK_FAILED',r"(?:didn't|did not|couldn't|could not) (?:do|finish|complete)|ما قدرت (?:اخلص|اكمل)|لم (?:اكمل|استطع)|ما خلصت"),
('TASK_COMPLETED',r"\b(?:finished|completed|done)\b|\bخلصت\b|انهيت"),
('RESCHEDULE',r"\b(?:reschedule|postpone|move .* to)\b|اجل|تاجيل|غير الموعد"),
('OVERWHELMED_WITH_TASKS',r"too many|overwhelm|\b(?:15|20) things|don't know where to start|وايد شغلات|مهام كثير|مب عارف من وين ابدا|كثير.*(?:مهام|شغل)"),
('PRIORITIZE',r"prioriti[sz]|what.*first|which.*first|الاولو|اولويه|وش.*اول|ماذا.*اولا"),
('BREAK_DOWN_TASK',r"break.*down|smaller steps|too big|خطوات صغير|قسم|الخطو[هة] كبير[هة]|المهم[هة] كبير[هة]"),
('TIME_ESTIMATION',r"how long|estimate.*time|كم (?:وقت|تاخذ|يستغرق)|تقدير.*وقت"),
('FOCUS_HELP',r"focus|distract|pomodoro|concentrat|تركيز|اركز|مشتت"),
('PROCRASTINATION',r"procrastinat|avoiding|putting.*off|اسوف|اتجنب|تسويف"),
('START_TASK',r"can't start|cannot start|help.*start|ما قدرت ابدا|مش قادر ابدا|ابدا"),
('CONTINUE_TASK',r"\b(?:continue|resume)\b|اكمل|كمل"),
('ORGANIZE_DAY',r"plan|schedule for|organize.*day|tasks.*today|رتب.*يوم|خطط|جدول.*اليوم|مهام.*اليوم"),
('ADD_TASK',r"\b(?:i need to|i have to|add|schedule|remind me to)\b|لازم|احتاج|اضف|ذكرني"),
]

def rule_intent(text):
    text=normalize(text)
    return next((label for label,pattern in INTENT_RULES if re.search(pattern,text)), 'UNKNOWN')

def protected_intent(text):
    """Small explicit-command contract; learned labels never authorize negated writes."""
    value=normalize(text)
    if re.search(r"\b(?:don't|do not|never)\s+(?:add|schedule|reschedule|postpone|move)\b|لا (?:تضف|تجدول|توجل|تنقل)",value): return 'UNKNOWN'
    if re.search(r'what.*(?:word|mean)|معني كلمة|ما معنى',value): return 'UNKNOWN'
    if re.search(r'\bprioriti[sz]e\b|ترتيب اولويات|رتب.*اولوي',value): return 'PRIORITIZE'
    if re.search(r'\b(?:order|organi[sz]e|sort|rank)\s+(?:them|these|those|it|my tasks)\b|\bprioriti[sz]e\b',value): return 'PRIORITIZE'
    # "Order my day" is a request to plan, whichever label a learned model would give it
    # (it read "رتب لي يومي" as ADD_TASK and asked for a task instead).
    if re.search(r'^رتب لي(?: يومي| يوم)?$|رتب.{0,12}(?:يومي|اليوم|مهامي|اعمالي|شغلي)',value): return 'ORGANIZE_DAY'
    # An unambiguous move/postpone command outranks a learned label, the same way
    # a negated write does; otherwise the classifier reads it as CONTINUE_TASK.
    if re.search(r'\b(?:reschedule|postpone|push back)\b|\bmove\b.{0,40}\bto\b|اجل |تاجيل|غير الموعد|انقل.{0,30}الى',value): return 'RESCHEDULE'
    # A stated completion is a report about the past, not a request to plan.
    if re.search(r"\b(?:i(?:'ve)?\s+(?:just\s+)?(?:finished|completed|did)|i am done with|i'm done with|done with)\b|^خلصت|انهيت ",value): return 'TASK_COMPLETED'
    return None

class IntentClassifier:
    """Baseline until independently labeled intent data validates an ML replacement."""
    version=RULES_VERSION
    def __init__(self):
        from .classical_model import ClassicalModel
        self.model=ClassicalModel('intent')
    def predict(self,text):
        protected=protected_intent(text)
        if protected is not None: return protected,None,'EXPLICIT_COMMAND_GUARD'
        if self.model.supports(text): return self.model.predict(text)[0],None,'LOCAL_TRAINED_CLASSIFIER'
        return rule_intent(text),None,'RULES_NO_VALIDATED_ARTIFACT_FOR_LANGUAGE'
