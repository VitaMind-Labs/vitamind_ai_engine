import re
from datetime import timedelta
from .journal_ai import JournalSentinel
from .journal_ai.lexicon import scan
from .journal_ai.context import rule_context
from ..adhd.executive_function.schemas import Safety
from ..adhd.executive_function.language import normalize,choose

# fuse() trusts an unaided model 'high' only at >=0.65; the same bar applies here.
MODEL_ONLY_TRUST=0.65
# Cue kinds that carry risk on their own. 'marker' is deliberately excluded: the
# lexicon tiers a bare marker 'low', and its vocabulary -- overwhelmed, deadline,
# piling up, exhausted, can't focus -- is the ordinary way an ADHD user describes a
# hard day, so a marker must not license an escalation the model alone is driving.
CORROBORATING_KINDS={'explicit','urgent_wish','serious','moderate'}

class SafetyAdapter:
    def __init__(self): self.agent=JournalSentinel()
    @property
    def ready(self): return self.agent.model is not None
    def check_action_text(self,text,lang):
        # Task/recovery fields are data too. Never turn a harmful stored step into coaching.
        s=normalize(text)
        if re.search(r'\b(?:double|increase|decrease|stop|start|skip)\b.{0,35}\b(?:dose|dosage|medication|medicine|pills?|mg)\b|(?:ضاعف|ازيد|اوقف|زود|قلل).{0,25}(?:جرع|دوا|علاج)',s):
            return Safety(level='ELEVATED',flags=['MEDICAL_ACTION_BOUNDARY'],source='action_rules'),choose(lang,'Medication changes need your clinician. I cannot turn this into a productivity step.','تغيير الدواء يحتاج الى طبيبك. لا يمكنني تحويله الى خطوة تنظيمية.')
        context=rule_context(scan(text))
        if context.tier in ('moderate','moderate_flagged','high'):
            return Safety(level='CRISIS' if context.tier=='high' else 'ELEVATED',flags=['UNSAFE_TASK_OR_RECOVERY_CONTEXT'],source='action_rules'),choose(lang,'Let us pause this task and focus on support. If you cannot stay safe, contact local emergency services or someone you trust.','لنوقف هذه المهمة ونركز على الدعم. اذا لم تستطع البقاء بامان، تواصل مع الطوارئ المحلية او شخص تثق به.')
        return None
    def check(self,request):
        s=normalize(request.message.text)
        medical=bool(re.search(r'what dose|dosage|should i (?:stop|start|take|increase|decrease).{0,35}(?:medication|medicine|mg|pill)|do i have adhd|diagnose me|كم.*جرع|هل.*(?:اوقف|ازيد|انقص).*(?:دوا|علاج)|هل عندي.*(?:adhd|فرط)|شخصني',s))
        result=self.agent.analyze(request.message.text,lang=request.patient.language.lower(),include_audit=True)
        level={'none':'NORMAL','low':'NORMAL','moderate':'ELEVATED','moderate_flagged':'ELEVATED','high':'CRISIS'}[result['tier']]
        flags=[] if level=='NORMAL' else ['JOURNAL_SAFETY_'+result['tier'].upper()]
        # The journal classifier was trained on reflective journal entries, not on
        # productivity requests, so on planning text it fires with no interpretable
        # cue behind it. Every configured crisis cue lives in the lexicon, and the
        # lexicon alone catches the crisis battery in tests/test_assistant.py, while
        # the model's score on benign planning text overlaps its score on real risk.
        # So an ELEVATED level that rests on the model alone -- no lexicon hit at all,
        # and below the same 0.65 confidence bar fuse() already requires before it
        # trusts the model unaided -- is recorded but does not interrupt planning.
        # Lexicon hits, CRISIS, provided journal context and the medical boundary are
        # all untouched; this only removes model-only noise.
        corroborated=any(h['kind'] in CORROBORATING_KINDS for h in result['audit']['hits'])
        if level=='ELEVATED' and not corroborated and result['model']['available'] and (result['model']['confidence'] or 0)<MODEL_ONLY_TRUST:
            level='NORMAL'; flags=['UNCORROBORATED_MODEL_SIGNAL_ON_PLANNING_TEXT']
        text=result['response']['text']
        # Current structured safety context can increase urgency, never lower it.
        day=request.timeContext.referenceDate
        journal=[j for j in request.journalContext if day and day-timedelta(days=2)<=j.date<=day]
        if any(j.signals.safetyLevel=='CRISIS' for j in journal):
            level='CRISIS'; flags.append('PROVIDED_CURRENT_CRISIS_CONTEXT')
            text=choose(request.patient.language,'Your safety comes first. If you cannot stay safe, contact local emergency services or someone you trust now.','سلامتك اولا. اذا لم تستطع البقاء بامان، تواصل الآن مع خدمات الطوارئ المحلية او شخص تثق به.')
        elif level=='NORMAL' and any(j.signals.safetyLevel=='ELEVATED' for j in journal):
            level='ELEVATED'; flags.append('PROVIDED_CURRENT_SAFETY_CONTEXT')
            text=choose(request.patient.language,'Let us pause planning and check what support you need right now.','لنوقف التخطيط قليلا ونتاكد من الدعم الذي تحتاجه الآن.')
        if medical and level!='CRISIS':
            level='ELEVATED'; flags.append('MEDICAL_BOUNDARY')
            text=choose(request.patient.language,'I can help organize tasks, but assessment and medication decisions need a qualified clinician.','يمكنني مساعدتك في تنظيم المهام، لكن التقييم وقرارات الدواء تحتاج الى مختص مؤهل.')
        if not result['model']['available']: flags.append('SAFETY_MODEL_UNAVAILABLE')
        return Safety(level=level,flags=flags,source='local_journal_model_and_rules'),text
