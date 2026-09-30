import re
from uuid import uuid5
from .language import normalize
from .time_engine import surface, parse_temporal
from .schemas import Task

EN_VERBS=r'email|send|call|phone|text|message|finish|write|prepare|buy|get|pick|drop|go|pay|clean|wash|study|review|read|book|submit|complete|open|edit|organize|organise|reply|exercise|take|meet|attend|visit|renew|refill|fix|print|fill|apply|order|file|return|deposit|collect|arrange|install|update|cancel|confirm|check|sort|tidy|pack|cook|walk|water|charge|back up|sign|renew|pray|see|talk to|speak to|work out|revise|practice|practise|rest|nap|shower|eat|meditate|stretch|tidy up|do'
AR_VERBS=r'اصلي|اشوف|اشوفه|اشوفها|اراجع|ارتاح|انام|اتمرن|اتكلم|اكلم|اتصل|اخلص|انهي|اشتري|اشتر|ارسل|اكتب|احضر|ادفع|انظف|اغسل|ادرس|اذاكر|اقرا|اراجع|احجز|افتح|اذهب|اروح|ارتب|اخذ|اقابل|ازور|اجدد|اصلح|اطبع|اعبي|اقدم|اطلب|ارجع|اودع|اجمع|انزل|احدث|اؤكد|اتفقد|اطبخ|امشي|اوقع'
VERBS=r'(?:'+EN_VERBS+'|'+AR_VERBS+')'
# Lead-ins that introduce an intention. "forgot/keep forgetting" are intentions too:
# the task is still outstanding, which is exactly when an ADHD user needs it captured.
INTRO=re.compile(r"^(?:(?:and|then|also|first|next|later|finally|after that)\s+)*(?:please\s+)?(?:(?:i(?:'m| am)\s+)?(?:going|gonna)\s+to|i(?:'ll| will)|i plan to|i(?:'d| would) like to|i (?:need|have|want|ought|got)\s+to|i(?:'ve)?\s+(?:got|have)\s+to|i (?:should|must|gotta)|i forgot to|i keep forgetting to|i (?:still )?haven'?t|don'?t let me forget to|remind me to|add(?: a task to)?|schedule(?: a task to)?|make sure (?:to|i)|لازم|احتاج(?: ان)?|ابغي|اريد(?: ان)?|ذكرني(?: ان)?|عندي|نسيت(?: ان)?|المفروض|يجب(?: ان)?|لا تنساني)\s+",re.I)
# Nouns that name a commitment rather than an action. "I have a dentist appointment"
# is a real obligation even though no verb introduces it.
EVENT_NOUNS=r'appointment|meeting|interview|exam|test|class|lecture|session|reservation|checkup|check-up|deadline|due date|shift|flight|موعد|اجتماع|مقابل[هة]|اختبار|امتحان|محاضر[هة]|حص[هة]|حجز|رحل[هة]|دوام'
EVENT=re.compile(r"^(?:i(?:'ve)?\s+(?:have|got)\s+|there(?:'s| is)\s+)?(?:an?|my|the)\s+(.{0,40}?(?:"+EVENT_NOUNS+r"))(?!\w)",re.I)
EVENT_AR=re.compile(r"^((?:"+EVENT_NOUNS+r").{0,40}?)$",re.I)
# "I am going to the doctor" names a place to be, not a verb. It is rewritten as the
# task it means ("go to the doctor") before the lead-in is stripped; "I'm going to
# study" has no determiner after "to", so it stays an ordinary intention.
DESTINATION=re.compile(r"^(?:(?:and|then|also|first|next|later|finally)\s+)*(?:i(?:'m| am)\s+)?(?:going|heading|off)\s+to\s+((?:the|my|a|an)\s+.+)$",re.I)
# The request half of a message ("... so help me organise them by priority") is an
# instruction to Spark, not a task. It is cut off before extraction so that its verbs
# ("organise", "help") can never become tasks of their own.
REQUEST_TAIL=re.compile(r"[,;.\s]*(?:\b(?:so|and|then|now|please)\s+)*(?:(?:can|could|would|will) you\s+)?(?:help(?:\s+me)?(?:\s+to)?\b|(?:i\s+)?(?:want|need|would like)\s+(?:you\s+)?to\s+(?:organi[sz]e|prioriti[sz]e|order|sort|rank)\b|(?:organi[sz]e|prioriti[sz]e|plan|sort|rank|order)\s+(?:them|these|those|it|all|my|the day|today|everything|things)\b|و?(?:ساعدني|رتبهم|رتبها|رتب لي|رتب يومي|حدد الاولويات)).*$",re.I)
REQUEST_ONLY=re.compile(r"^(?:please\s+)?(?:(?:can|could|would|will) you\s+)?(?:help(?:\s+me)?|(?:organi[sz]e|prioriti[sz]e|order|sort|rank)\s+(?:them|these|those|it|all|my\s+\w+|the day)|(?:i\s+)?(?:want|need)\s+(?:you\s+)?to\s+(?:organi[sz]e|prioriti[sz]e|order|sort|rank)\b.*|رتبهم|رتبها)\W*$",re.I)
# Everyday commitments people name without a verb: "I have a workout and school and praying".
ACTIVITY=r'workout|work out|gym|school|class|classes|work|homework|prayer|praying|shopping|laundry|dishes|cleaning|lunch|dinner|breakfast|meeting|training|revision|studying|errands|commute'
ACTIVITY_ONLY=re.compile(r"^(?:(?:an?|the|my)\s+)?(?:"+ACTIVITY+r")(?:\s+(?:session|practice|class))?$",re.I)
HAVE_A=re.compile(r"^i(?:'ve)?\s+(?:have|got|had)\s+(?:an?\s+|the\s+|my\s+)?",re.I)
REQUEST_HEAD=re.compile(r"^(?:no[,.]?\s*)?(?:i\s+)?(?:want|need)\s+you\s+to\s+(?:order|organi[sz]e|prioriti[sz]e|plan|sort|rank)\s+(?:my|the)\s+(?:day|tasks?|things|list)\s*[,.:]?\s*",re.I)
# Activities that are complete without an object: "pray" and "study" are tasks.
SELF_CONTAINED=re.compile(r"^(?:pray|study|exercise|rest|nap|shower|meditate|stretch|eat|cook|walk|revise|practi[sc]e|work out|اصلي|ادرس|اذاكر|ارتاح|انام|اتمرن|امشي|اطبخ)$",re.I)
NEGATIVE=re.compile(r"\b(?:don't|do not|no need to|cancel(?:led)?|already|finished|completed)\b|ما (?:ابغي|احتاج)|لا (?:اريد|احتاج)|لغيت|خلصت|انتهيت",re.I)
# Naming what you are stuck on names the task. "I keep putting off my taxes" is a
# commitment as much as "I need to do my taxes", and it is how the block is usually
# reported, so the object is captured instead of being lost with the complaint.
STUCK_ON=re.compile(r"\b(?:putting off|put off|avoiding|procrastinating (?:on|about)|dreading|stuck on|struggling with|delaying|postponing|been meaning to|keep forgetting)\s+(.{2,60}?)(?:\s+for\s+(?:weeks|days|months|ages|so long))?$|(?:اتجنب|اسوف في|مأجل|مؤجل|عالق في|متعثر في)\s+(.{2,60}?)$",re.I)
TOO_BIG_FOR_ME=re.compile(r"^(?:the\s+|my\s+)?(.{2,40}?)\s+(?:feels?|is|seems|looks)\s+(?:way\s+|really\s+|just\s+)?too\s+(?:big|much|hard)|^(.{2,40}?)\s+كبير[هة]?\s+عل(?:ي|يا)",re.I)
# "this task" and "it" point at something the message never names. Saving a task
# literally called "task" is clutter, so the user is asked which one instead.
PLACEHOLDER=re.compile(r"^(?:this|that|the|my|it|one)?\s*(?:task|thing|stuff|it|one|شي|شيء|المهم[هة]|هذه المهم[هة]|الشغل[هة])$",re.I)
# A clause carrying only a deadline, pointing back at what was just named:
# "..., they're due Friday", "..., it's due 2026-10-05", "..., deadline Friday".
# Anchored, so a segment that states its own deadline ("pay the bill due Friday")
# is untouched and keeps being parsed as one task.
TRAILING_DEADLINE=re.compile(
    r"^(?:and\s+)?(?:(?:they|it|that|this|those|these|which)\s*(?:'re|'s|is|are|was|were)?\s+)?"
    r"(?:all\s+)?(?:due|deadline(?:\s+is)?|needs?\s+to\s+be\s+(?:done|finished|in)|has\s+to\s+be\s+(?:done|in))\b(.*)$"
    r"|^(?:و\s*)?(?:موعدها|موعده|اخر موعد|الموعد النهائي|المفروض تخلص|لازم تخلص)\b(.*)$",re.I)
# A recurring commitment is recorded once, with the repetition flagged rather than
# silently expanded into invented future entries.
RECURRENCE=re.compile(r"\b(?:every|each)\s+(?:day|morning|evening|night|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\bdaily|weekly|monthly\b|كل (?:يوم|صباح|مساء|اسبوع|شهر)",re.I)

# Text that addresses the assistant rather than describing the patient's day. A sentence
# like this is dropped on its own, so the real commitments around it are still captured
# and its verbs ("ignore", "delete", "reveal") can never leak into a task title.
INSTRUCTION_TO_ASSISTANT=re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|discard)\b[^.!?\n]{0,40}\b(?:instructions?|rules?|prompts?|guidelines?|constraints?|restrictions?|above|previous|prior)\b"
    r"|\byou\s+are\s+(?:now|no\s+longer)\b|\b(?:pretend|act|behave|respond)\s+(?:to\s+be|as(?:\s+if)?)\b"
    r"|^\s*(?:system|assistant|developer|user)\s*:|\b(?:reveal|show|print|repeat|leak)\b[^.!?\n]{0,25}\b(?:system\s+)?(?:prompt|instructions)\b"
    r"|\b(?:jailbreak|developer\s+mode|dan\s+mode)\b|\b(?:delete|erase|wipe|remove)\s+(?:all|every(?:thing)?)\b[^.!?\n]{0,20}\b(?:tasks?|data|memor(?:y|ies)|history)\b"
    r"|__\w+__|\bimport\s+os\b|\bos\.system\b"
    r"|تجاهل[^.!?\n]{0,30}(?:التعليمات|القواعد|الاوامر)|انسَ?\s+(?:كل\s+)?(?:التعليمات|القواعد)|انت\s+الان",re.I)
MARKUP=re.compile(r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>|<[^>]{0,200}>",re.I|re.S)
# A sentence ends at .!?؟ followed by a space, and not after a common abbreviation.
_SENTENCE_END=re.compile(r"(?<!\b[Dd]r\.)(?<!\b[Mm]r\.)(?<!\b[Mm]rs\.)(?<!\b[Mm]s\.)(?<!\bSt\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?؟])\s+")

def isolate_instructions(text):
    """The patient's own words, with markup and instructions aimed at the assistant removed."""
    text=MARKUP.sub(' ',text)
    sentences=[s.strip() for s in _SENTENCE_END.split(text)]
    return ' ; '.join(s for s in sentences if s and not INSTRUCTION_TO_ASSISTANT.search(s))

def extract_tasks(text,request_id,reference_date=None):
    raw=isolate_instructions(surface(text)).strip()
    if not raw: return [],[]
    head=REQUEST_HEAD.match(raw)
    if head:
        raw=raw[head.end():].strip()
        if not raw: return [],[]
    tail=REQUEST_TAIL.search(raw)
    if tail and raw[:tail.start()].strip(): raw=raw[:tail.start()].strip()
    # "order them" on its own is an instruction about tasks already named, not a task.
    if REQUEST_ONLY.match(raw): return [],[]
    # Split only conjunctions introducing another action, not a name or object list.
    # "and I have to ..." and "then I'm going to ..." each start a new intention.
    pieces=re.split(r"[,;\n،؛]+|\s+and\s+(?="+EN_VERBS+r"|(?:i|we)\s+(?:need|have|want|should|must|got|gotta|am|will|plan)\b|(?:i|we)'(?:m|ll|d)\b|(?:going|heading)\s+to\b|(?:(?:an?|the|my)\s+)?(?:"+ACTIVITY+r")\b)|\s+(?:and\s+)?(?:then|also|after that)\s+|\s+و(?="+AR_VERBS+r")|(?<=\w)\s*و(?="+AR_VERBS+r")",raw,flags=re.I)
    tasks=[]; uncertainty=[]; trailing=[]
    for segment in pieces:
        segment=segment.strip().rstrip('.!?؟ ').strip()
        if not segment: continue
        # A clause that only carries a deadline is held aside; which task it belongs
        # to is decided after the whole message has been read.
        carried=TRAILING_DEADLINE.match(segment)
        if carried:
            when=parse_temporal(carried[1] or carried[2] or '',reference_date)
            trailing.append(when)
            continue
        recurring=bool(RECURRENCE.search(segment))
        named_event=False
        destination=DESTINATION.match(segment)
        candidate='go to '+destination[1].strip() if destination else INTRO.sub('',segment).strip()
        if NEGATIVE.search(candidate): continue
        if not re.match(r'^'+VERBS+r'(?:\s|$)',candidate,re.I):
            # Recognize an explicitly named report/presentation with a request to finish.
            m=re.match(r'^(?:عندي\s+)?(presentation|report|تقرير|عرض(?: تقديمي)?)\s+لازم\s+(?:اخلصه|اكمله)',candidate,re.I)
            if m: candidate='اخلص '+m[1]+candidate[m.end():]
            else:
                # Otherwise accept a named commitment, keeping any trailing date/time
                # text so parse_temporal can still resolve when it happens.
                stuck=STUCK_ON.search(candidate) or TOO_BIG_FOR_ME.match(candidate)
                event=None if stuck else (EVENT.match(candidate) or EVENT_AR.match(candidate))
                if stuck:
                    candidate=(stuck[1] or stuck[2]).strip()
                    named_event=True
                elif event:
                    candidate=event[1]+candidate[event.end():]
                    named_event=True
                elif ACTIVITY_ONLY.match(parse_temporal(HAVE_A.sub('',candidate).strip(),reference_date)['text']):
                    candidate=HAVE_A.sub('',candidate).strip(); named_event=True
                else: continue
        # Keep the task title separate from the following report of difficulty.
        candidate=re.split(r"\s+(?:but|بس|لكن)\s+(?=(?:i\b|ما\b|مش\b|مو\b|لا\b))",candidate,flags=re.I)[0]
        if recurring: candidate=RECURRENCE.sub('',candidate).strip()
        temporal=parse_temporal(candidate,reference_date)
        uncertainty.extend(temporal['uncertainty'])
        title=temporal['text'].strip(' .!?؟')
        title=re.sub(r'\s+(?:again|once more|مر[هة] ثاني[هة])$','',title,flags=re.I).strip()
        if PLACEHOLDER.match(title):
            uncertainty.append('TASK_NOT_NAMED'); continue
        # A bare verb needs an object to be actionable, but a named commitment
        # ("a meeting on Thursday") is already complete on its own.
        if len(title.split())<2 and not (named_event and title) and not SELF_CONTAINED.match(title):
            uncertainty.append('TASK_OBJECT_REQUIRED'); continue
        if recurring: uncertainty.append('RECURRING_TASK_SAVED_ONCE')
        ident='task_'+uuid5(request_id,str(len(tasks))+':'+normalize(title)).hex[:16]
        tasks.append(Task(temporaryId=ident,title=title,scheduledDate=temporal['date'],deadline=temporal['date'] if temporal['is_deadline'] else None,startTime=temporal['start'],durationMinutes=temporal['duration'],source='MESSAGE'))
    for when in trailing:
        uncertainty.extend(when['uncertainty'])
        if not when['date']:
            # "they're due next week" resolves to nothing definite; say so rather
            # than attach a guess.
            if not when['uncertainty']: uncertainty.append('DEADLINE_CLAUSE_NOT_RESOLVED')
        elif len(tasks)!=1:
            # "call the bank, buy groceries, they're due Friday" gives no honest way
            # to tell which commitment the clause meant. Never guess a due date.
            uncertainty.append('DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS' if tasks else 'DEADLINE_CLAUSE_WITHOUT_TASK')
        elif tasks[0].deadline or tasks[0].scheduledDate:
            # The task already stated its own date; a later clause does not override it.
            uncertainty.append('DEADLINE_CLAUSE_CONFLICTS_WITH_TASK_DATE')
        else:
            tasks[0]=tasks[0].model_copy(update={'deadline':when['date'],'scheduledDate':when['date']})
    return tasks,sorted(set(uncertainty))
