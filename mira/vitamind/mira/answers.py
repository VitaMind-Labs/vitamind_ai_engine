"""Interpret bounded answers to the exact question currently being asked."""
import re
from ..clinical.language import normalize

YES = {'yes','yes i do','yes it does','yes they do','sometimes','often','نعم','اجل','احيانا','نعم يحدث ذلك'}
NO = {'no','no i do not','no i don\'t','never','not at all','لا','كلا','ابدا','لا يحدث ذلك'}
SKIP = {'skip','prefer not to answer','تخطي','افضل عدم الاجابة'}
UNKNOWN = {'i don\'t know','i do not know','not sure','unsure','لا اعرف','لست متاكدا','غير متاكد','maybe','ربما'}

def interpret(question,text):
    value=normalize(text).strip(' .!؟?')
    if value in SKIP: return 'skip',None
    if value in UNKNOWN: return 'unknown',None
    if question.kind=='narrative':
        return ('answered',text) if len(value.split())>=3 else ('unknown',None)
    if question.kind=='duration':
        if re.search(r'\b(?:day|days|week|weeks|month|months|year|years|childhood)\b|(?:يوم|ايام|اسبوع|اسابيع|شهر|اشهر|سنة|سنوات|طفول)',value):
            return 'answered',text
        return 'unknown',None
    if question.kind=='context':
        if value in NO: return 'answered','denied'
        if re.search(r'\b(?:medication|medicine|drug|alcohol|cannabis|stimulant|caffeine|sleep|thyroid|illness|prescription)\b|(?:دواء|ادوية|كحول|مخدر|نوم|كافيين|مرض|الغدة)',value):
            return 'answered',text
        return 'unknown',None
    if question.kind=='sleep':
        if re.search(r'\b(?:exhausted|tired)\b|(?:متعب|مرهق)',value) and not re.search(r"(?:not|never|don't|without|لا|بدون)\s+(?:feel\s+)?(?:tired|exhausted|متعب|مرهق)",value):
            return 'answered',False
    if value in YES: return 'answered',True
    if value in NO: return 'answered',False
    if re.match(r'^(?:yes|yeah|yep|نعم|اجل)(?:\b|[،,])',value): return 'answered',True
    if re.match(r'^(?:no|nope|لا|كلا)(?:\b|[،,])',value): return 'answered',False
    # The extractor handles detailed symptom descriptions; don't guess from unrelated prose.
    return 'unknown',None
