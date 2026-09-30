from datetime import date, time, timedelta
import re, unicodedata
from .language import normalize

DAYS={'monday':0,'tuesday':1,'wednesday':2,'thursday':3,'friday':4,'saturday':5,'sunday':6,'الاثنين':0,'الثلاثاء':1,'الاربعاء':2,'الخميس':3,'الجمعه':4,'الجمعة':4,'السبت':5,'الاحد':6}
NUMBERS={'five':5,'ten':10,'fifteen':15,'twenty':20,'thirty':30,'خمس':5,'خمسه':5,'خمسة':5,'عشر':10,'عشره':10,'عشرة':10,'خمسة عشر':15,'خمسه عشر':15,'عشرين':20,'ثلاثين':30}

def surface(text):
    text=unicodedata.normalize('NFKC',text).replace('’',"'")
    text=re.sub(r'[\u064b-\u065f\u0670\u0640]','',text)
    return text.translate(str.maketrans('أإآٱى٠١٢٣٤٥٦٧٨٩','ااااي0123456789'))

def available_minutes(text):
    s=normalize(text)
    numbers=r'\d{1,4}|'+ '|'.join(sorted(map(re.escape,NUMBERS),key=len,reverse=True))
    match=re.search(r"(?:\b(?:have|got|available|only)\s+(?:about\s+)?|عندي\s+|عندي فقط\s+)("+numbers+r")\s*(?:minutes?|دقيقه|دقيقة|دقايق|دقائق)",s)
    if not match: return None
    value=int(match[1]) if match[1].isdigit() else NUMBERS[match[1]]
    return value if 0<=value<=1440 else None

def parse_temporal(text,reference_date=None):
    clean=surface(text)
    s=clean.lower()
    spans=[]; dates=[]; errors=[]; start=None; duration=None
    # Explicit dates are unambiguous. Slash dates intentionally need clarification.
    for m in re.finditer(r'\b\d{4}-\d{2}-\d{2}\b',s):
        spans.append(m.span())
        try: dates.append(date.fromisoformat(m[0]))
        except ValueError: errors.append('INVALID_DATE')
    if re.search(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',s): errors.append('AMBIGUOUS_NUMERIC_DATE')
    relative=[(r'\bday after tomorrow\b|بعد غد',2),(r'\btomorrow\b|\bغدا\b|\bباجر\b|\bبكر[هة]\b',1),(r'\btoday\b|\bاليوم\b',0)]
    for pattern,offset in relative:
        for m in re.finditer(pattern,s):
            if any(a<=m.start()<b for a,b in spans): continue
            spans.append(m.span())
            if reference_date: dates.append(reference_date+timedelta(days=offset))
            else: errors.append('REFERENCE_DATE_REQUIRED')
    for m in re.finditer(r'\bin (\d{1,3}) days?\b|بعد (\d{1,3}) ايام',s):
        spans.append(m.span())
        if reference_date: dates.append(reference_date+timedelta(days=int(m[1] or m[2])))
        else: errors.append('REFERENCE_DATE_REQUIRED')
    day_names='|'.join(DAYS)
    for m in re.finditer(r'(?<!\w)(?:(next)\s+)?('+day_names+r')(?:\s+(القادم|الجاي))?(?!\w)',s):
        spans.append(m.span())
        if reference_date:
            delta=(DAYS[m[2]]-reference_date.weekday())%7
            if (m[1] or m[3]) and delta==0: delta=7
            dates.append(reference_date+timedelta(days=delta))
        else: errors.append('REFERENCE_DATE_REQUIRED')
    if re.search(r'next week|الاسبوع (?:الجاي|القادم)|sometime|بعدين',s): errors.append('DATE_NEEDS_CLARIFICATION')
    if re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d|\d\s+(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b|يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|اغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر',s):
        errors.append('USE_ISO_DATE_FOR_MONTH_NAMES')
    # The caller receives a local clock time, not a fabricated UTC appointment.
    clock=re.search(r'\b(?:at\s+)(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b|الساع[هة]\s+(\d{1,2})(?::(\d{2}))?\s*(صباحا|مساء)?',s)
    if clock:
        spans.append(clock.span())
        hour=int(clock[1] or clock[4]); minute=int(clock[2] or clock[5] or 0); period=clock[3] or clock[6]
        if period:
            if not 1<=hour<=12 or minute>59: errors.append('INVALID_TIME')
            else: start=time(hour%12+(12 if period in ('pm','مساء') else 0),minute)
        elif (clock[2] or clock[5]) and hour<=23 and minute<=59: start=time(hour,minute)
        else: errors.append('AM_PM_REQUIRED')
    match=re.search(r'\b(?:takes?|for|duration)\s+(\d{1,4})\s*(?:minutes?|mins?)\b|(?:تستغرق|مدتها|لمد[هة])\s+(\d{1,4})\s*(?:دقيقه|دقيقة)',s)
    if match:
        value=int(match[1] or match[2]); spans.append(match.span())
        if 1<=value<=1440: duration=value
        else: errors.append('INVALID_DURATION')
    if len(set(dates))>1: errors.append('MULTIPLE_DATES_IN_TASK')
    for a,b in sorted(spans,reverse=True): clean=clean[:a]+clean[b:]
    clean=re.sub(r'\b(?:on|at|by|due)\s*$','',clean.strip(),flags=re.I)
    return {'date':dates[0] if dates and not errors else None,'start':start,'duration':duration,'text':re.sub(r'\s+',' ',clean).strip(' .,;،؛'), 'uncertainty':sorted(set(errors)),'is_deadline':bool(re.search(r'\b(?:due|deadline|by)\b|موعد نهائي|اخر موعد',s))}
