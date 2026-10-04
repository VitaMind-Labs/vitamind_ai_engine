"""Intent routing by inspectable rules, with the learned head as a fallback.

Routing is done by bilingual cue patterns that a reviewer can read and correct.
The spec asks for exactly this trade (s70, s101): prefer deterministic code, and
do not run a model where a rule does the job better.

The learned head is consulted only when no rule fires, and only when it does not
abstain. It used to score macro-F1 0.09 on held-out meanings and was useless even
as a fallback, because the authored seed set held three phrasings-of-one-meaning
per intent and the split gave training exactly one of them. With nine or ten
meaning-families per intent it scores macro-F1 0.46 with 0.84 precision on the
predictions it does accept - still not a router on its own, now worth consulting
when the rules have nothing.

When neither fires, the router returns UNKNOWN and the decision engine falls back
to a generic supportive strategy - the conservative default, reached honestly.

Retrain on reviewed real conversations before letting the head lead. Intent never
influences safety: that path is owned by lumina/safety.py.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .taxonomy import INTENTS
from .text import normalize

RULE_VERSION = "intent-rules-v2"

# Ordered most specific first: the first rule that matches wins, so narrow
# patterns are not swallowed by broad ones.
PATTERNS = [
    ("ABOUT_BOT", r"\b(?:who are you|what are you|what can you do|how can you help|"
                  r"what do you do|what is lumina|are you (?:a )?(?:bot|robot|ai|human|real))\b"
                  r"|(?:من انت|ماذا تستطيع|ما هي لومينا|كيف تساعدني|ماذا تفعلين|من تكونين)"),
    ("GOODBYE", r"^\s*(?:bye|goodbye|good night|talk to you later|see you|that is all|that's all|"
                r"وداعا|باي|تصبح على خير|الى اللقاء|هذا كل شيء|اراك)\s*[.!]*\s*$"),
    ("THANKS", r"^\s*(?:thanks|thank you|thx|شكرا|اشكرك|مشكور)\s*[.!]*\s*$"),
    ("GREETING", r"^\s*(?:hello|hi|hey|good morning|good evening|good afternoon|"
                 r"مرحبا|اهلا|السلام عليكم|صباح الخير|مساء الخير|هاي)\s*[.!]*\s*$"),
    ("DISENGAGE", r"\b(?:leave me alone|stop talking|do not contact|don't contact|"
                  r"لا تكلمني|اتركني وحدي|لا اريد الحديث)\b"),
    ("BOT_FEEDBACK", r"\b(?:you are helpful|you are not helpful|this is helpful|"
                      r"this is not helpful|that helped|لم تساعدني|ساعدتني)\b"),
    ("MEDICATION_MENTION", r"\b(?:medication|meds|prescription|dose|dosage|pills?|tablets?|side effects?)\b"
                           r"|(?:دوا(?:ء|ئي)|الدواء|وصفة|جرعة|حبوب|اثار جانبية|آثار جانبية)"),
    ("CLINICIAN_MENTION", r"\b(?:psychiatrist|therapist|clinician|my doctor|appointment|session|clinic)\b"
                          r"|(?:طبيبي|معالجي|الطبيب النفسي|موعد|جلسة|العيادة)"),
    ("CRISIS", None),   # never matched here; safety owns it
    ("SAFETY", None),
    ("PROGRESS", r"\b(?:how have i been|summary of my|my patterns?|looking back|last month|"
                 r"better or worse|since i started|has anything changed)\b"
                 r"|(?:ملخص|كيف كان ادائي|انماطي|الشهر الماضي|افضل ام اسوا|تغير شيء)"),
    ("GOAL", r"\b(?:set a goal|my goal|a goal for|build a habit|target|drop the goal)\b"
             r"|(?:هدف|اهداف|عادة جديدة)"),
    # A falling-out with someone, reported as an event. Filed as EMOTIONAL_SUPPORT and
    # not SOCIAL on purpose: SOCIAL answers with "connection with other people can
    # help", which lands badly on someone who has just argued with their brother. What
    # they want first is for it to be heard. SOCIAL keeps the *relationship* cases -
    # isolation, unanswered messages, ongoing tension.
    #
    # The learned head cannot be trusted with this: "I had a fight with my brother" has
    # the same shape as "I made bread today", and the head scored it
    # GENERAL_CONVERSATION at 0.82 confidence - which a stressed check-in then answered
    # with a productivity nudge. Kinship and conflict cues are named here instead,
    # where a reviewer can read and correct them.
    ("EMOTIONAL_SUPPORT",
     r"\b(?:fight|fought|argument|argu(?:ed|ing)|row|fell out|falling out|"
     r"shouted|yelled|screamed)\b(?:\s+\w+){0,3}\s+(?:with|at)\s+"
     r"(?:my|the|his|her|their|our)?\s*"
     r"(?:brother|sister|mother|father|mum|mom|dad|parents?|wife|husband|"
     r"partner|boyfriend|girlfriend|cousin|son|daughter|uncle|aunt|family|"
     r"friend|flat ?mate|room ?mate|colleague|boss|neighbou?r|him|her|them)\b"
     r"|\b(?:not|stopped)\s+speaking\b"
     r"|(?:تشاجرت|تخاصمنا|زعلت من|لا نتكلم|خلاف مع|اختلفت مع|صرخ في)"),
    # Named low moods outrank topic words in the same sentence ("depressed and I can't focus").
    ("EMOTIONAL_SUPPORT", r"\b(?:depress\w*|hopeless|worthless|numb)\b|(?:اكتئاب|مكتئب|يائس)"),
    ("SLEEP", r"\b(?:sleep|slept|sleeping|insomnia|wake up|woke up|bed(?:time)?|nap)\b"
              r"|(?:نوم|نمت|انام|ارق|استيقظ|اصحى|السرير)"),
    ("FOCUS", r"\b(?:focus|concentrat\w+|attention|distract\w+|other tabs|clear head)\b"
              r"|(?:تركيز|اركز|انتباه|شرود|مشتت)"),
    ("ENERGY", r"\b(?:energy|energetic|exhaust\w+|drained|tired|fatigue|running on empty|"
               r"sit still|drive)\b|(?:طاقة|نشاط|مرهق|ارهاق|تعب|منهك|ساكن)"),
    ("EXERCISE", r"\b(?:walk|walking|gym|exercise|stretch\w*|workout|run(?:ning)?|active)\b"
                 r"|(?:مشي|النادي|تمارين|رياضة|اطالة|نشاط بدني)"),
    ("ROUTINE", r"\b(?:routine|schedule|regular|rhythm|same time)\b"
                r"|(?:روتين|جدول|منتظم|ايقاع|الوقت نفسه)"),
    # Interpersonal conflict, plainly reported. A learned head reads "I had a fight
    # with my brother" as neutral small talk - it has the same shape as "I made
    # bread today" - and a stressed check-in then answers it with a productivity
    # nudge. Naming the conflict and kinship cues here keeps that in rules, where a
    # reviewer can see and correct it.
    ("SOCIAL", r"\b(?:friends?|people|lonely|alone|social|messages|cancelled on|"
               r"spoken to anyone|replied to)\b|(?:اصدقاء|صديق|الناس|وحدة|بوحدي|رسائل|اجتماعي)"),
    ("TASK_SUPPORT", r"\b(?:task|tasks|start(?:ing|ed)?|finish\w*|to ?do|list|deadline|"
                     r"report|avoid\w* starting|piling up|putting it off|procrastinat\w+)\b"
                     r"|(?:مهمة|مهام|ابدا|البدء|انهي|قائمة|تاجيل|اؤجل|تتراكم|التقرير)"),
    ("STRESS", r"\b(?:stress\w*|pressure|overwhelm\w*|too much|calmed down|anxious|anxiety|"
               r"worr(?:y|ied|ying)|nervous|scared|afraid|panic\w*|on edge|tense)\b"
               r"|(?:توتر|ضغط|مرهق|اكثر مما احتمل|هدات|قلق|خايف|خائف|هلع|متوتر)"),
    ("EMOTIONAL_SUPPORT", r"\b(?:just need someone|need to be heard|listen|sit with me|"
                          r"hard time|feel (?:flat|sad|low|empty)|pointless|nobody understands|"
                          r"lonely|sad|sadness|unhappy|miserable|crying|cried|heartbroken|"
                          r"(?:feel(?:ing)?|i'?m|i am) (?:down|low|bad|awful|terrible)|"
                          r"not (?:good|great|okay|ok|well|fine)|angry|mad at|furious|upset|"
                          r"annoyed|frustrated|irritated|hurt)\b"
                          r"|(?:استمع|يسمع لي|وقت صعب|حزين|فراغ|بلا معنى|لا احد يفهم|زعلان|"
                          r"مكتئب|ابكي|لست بخير|مش كويس|غاضب|معصب|منزعج|محبط)"),
    ("JOURNAL", r"\b(?:write about|writing this|put down what|journal|entry|"
                r"thoughts i wanted|looking back on the week)\b"
                r"|(?:اكتب عن|ادون|يوميات|افكار اردت)"),
    ("CHECK_IN", r"\b(?:checking in|check.?in|here is how today|today was|reporting in|"
                 r"mood is|answering your question|you asked how)\b"
                 r"|(?:اسجل حالتي|هذا ما مر به يومي|كان اليوم|مزاجي|سالتني عن)"),
    ("QUESTION", r"^\s*(?:what|how|who|can you|are you|do you|should i|is this|am i)\b"
                 r"|\?\s*$|(?:^|\s)(?:ماذا|كيف|من|هل|لماذا)\b|؟\s*$"),
    ("GENERAL_CONVERSATION", r"\b(?:good morning|good evening|hello|hi|hey|thanks|thank you|"
                             r"that is all|talk to you|okay|understood|makes sense|bye|goodbye|"
                             r"good night|good day|great day|happy|feel(?:ing)? (?:better|good|great|calm)|"
                             r"proud|went well)\b"
                             r"|(?:صباح الخير|مساء الخير|مرحبا|اهلا|السلام عليكم|شكرا|هذا كل شيء|اراك|"
                             r"حسنا|فهمت|يوم جيد|سعيد|افضل|بخير|مرتاح|تصبح على خير|الى اللقاء)"),
]

COMPILED = [(intent, re.compile(pattern, re.I | re.UNICODE))
            for intent, pattern in PATTERNS if pattern]

# The conversational move a message makes, separate from its topic. It only
# shapes wording (an opener, a closing line) - never safety, never the decision's
# priority order. Ordered most specific first; a stated negative feeling wins
# over a greeting ("hi, I feel awful" is about feeling awful).
ACTS = [
    ("ABOUT_LUMINA", r"\b(?:who are you|what are you|what can you do|how can you help|"
                     r"what do you do|what is lumina|are you (?:a )?(?:bot|robot|ai|human|real))\b"
                     r"|(?:من انت|ماذا تستطيع|ما هي لومينا|كيف تساعدني|ماذا تفعلين|من تكونين)"),
    # A request to plan or order the day. It is a concrete ask, so it outranks the mood
    # acts below and, in the decision engine, outranks stored state concerns: the patient
    # said what they want, and answering with an unrelated exercise is not listening.
    ("PLAN_REQUEST", r"\b(?:prioriti[sz]\w*|organi[sz]e\s+(?:my|the|our|them|these|those|it|things|everything|all)\b|"
                     r"organi[sz]e\b.{0,24}\b(?:day|list|tasks?|things|schedule)\b|"
                     r"plan\s+(?:my|the|our)\s+(?:day|week|morning|afternoon|evening|tomorrow|time|schedule)\b|"
                     r"(?:what|which)\s+(?:should|do|to)\s+(?:i\s+)?(?:do|start|tackle|focus on)\s+(?:first|next)\b|"
                     r"what to do first\b|sort\s+(?:out\s+)?my\s+(?:day|tasks|list|things)\b|"
                     r"put (?:them|these|things|my day) in order\b|order of my tasks\b|"
                     r"manage my (?:day|time)\b|help me (?:plan|organi[sz]e|sort)\b)"
                     r"|(?:(?<![ا-ي])و?رتب|(?<![ا-ي])(?:ال)?ترتيب|اولوي\w*|نظم (?:يومي|وقتي)|خطط (?:ليومي|ليوم)|"
                     r"شو اسوي اول|ايش اسوي اول|ماذا افعل اولا)"),
    ("CONFUSED", r"\b(?:i )?(?:don'?t|do not|dont) (?:und\w+|get (?:it|this))|what do you mean|not clear|confus\w+"
                 r"|(?:لا افهم|مش فاهم|ما فهمت|غير واضح)"),
    ("AGREE", r"^\s*(?:yes|yeah|yep|yup|ok|okay|sure|alright|نعم|ايوه|اوكي|حسنا)\s*[.!]*\s*$"),
    ("DECLINE", r"^\s*(?:no|nope|nah|not now|لا|كلا|مش الان)\s*[.!]*\s*$"),
    ("SADNESS", r"\b(?:sad|sadness|depress\w*|numb|unhappy|miserable|crying|cried|heartbroken|lonely|empty|"
                r"(?:feel(?:ing)?|i'?m|i am) (?:down|low|bad|awful|terrible)|"
                r"not (?:good|great|okay|ok|well|fine))\b"
                r"|(?:حزين|زعلان|مكتئب|ابكي|لست بخير|مش كويس|وحيد|فراغ)"),
    ("ANXIETY", r"\b(?:anxious|anxiety|worr(?:y|ied|ying)|nervous|scared|afraid|panic\w*|on edge)\b"
                r"|(?:قلق|خايف|خائف|هلع|متوتر)"),
    ("ANGER", r"\b(?:angry|mad at|furious|upset|annoyed|frustrated|irritated)\b"
              r"|(?:غاضب|معصب|منزعج|محبط)"),
    ("CLOSING", r"\b(?:bye|goodbye|good night|talk (?:to you )?later|see you|that is all|that's all)\b"
                r"|(?:الى اللقاء|تصبح على خير|وداعا|باي|هذا كل شيء|اراك)"),
    ("THANKS", r"\b(?:thanks|thank you|thx|appreciate it)\b|(?:شكرا|اشكرك|مشكور)"),
    ("POSITIVE", r"\b(?:good day|great day|nice day|happy|feel(?:ing)? (?:better|good|great|calm)|"
                 r"proud|went well|i'?m (?:good|great|fine|okay|ok)|i am (?:good|great|fine|okay|ok))\b"
                 r"|(?:يوم جيد|سعيد|افضل|بخير|مرتاح|فخور)"),
    ("UNSURE", r"\b(?:i don'?t know|idk|not sure|no idea|nothing really)\b"
               r"|(?:لا اعرف|مش عارف|لست متاكد|ما بعرف)"),
    ("GREETING", r"\b(?:hello|hi|hey|good morning|good evening|good afternoon)\b"
                 r"|(?:مرحبا|اهلا|السلام عليكم|صباح الخير|مساء الخير|هاي)"),
]
COMPILED_ACTS = [(act, re.compile(pattern, re.I | re.UNICODE)) for act, pattern in ACTS]


# A "thanks" that also asks something or reports a difficulty is not a closing: "thanks, but I
# still can't sleep" and "thank you - how do I start?" have to be answered, not waved off.
CARRIES_MORE = re.compile(
    r"\?|؟|\b(?:can|could|would|will|do|does|did|should) (?:you|i|we)\b|\bhelp me\b"
    r"|\b(?:but|still|however|though|although|what|how|why|when|where|which)\b"
    r"|\b(?:can'?t|cannot|couldn'?t|won'?t|don'?t|doesn'?t|didn'?t|isn'?t|not working)\b"
    r"|لكن|بس|ما زال|مازال|كيف|ليش|لماذا|متى|ساعدني|لا استطيع|ما اقدر", re.I | re.UNICODE)
CLOSING_ACT_NAMES = ("THANKS", "CLOSING")


def conversation_act(text):
    """Return the first matching conversational act, or None."""
    normalized = normalize(text)
    for act, pattern in COMPILED_ACTS:
        if pattern.search(normalized):
            if act in CLOSING_ACT_NAMES and CARRIES_MORE.search(normalized):
                continue
            return act
    return None


@dataclass
class IntentReading:
    intent: str
    matched_rule: str | None = None
    all_matches: tuple = ()
    source: str = "rules"
    abstained: bool = False
    act: str | None = None
    rule_version: str = RULE_VERSION

    def __post_init__(self):
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent {self.intent!r}")

    def to_dict(self):
        return {**asdict(self), "all_matches": list(self.all_matches),
                "influences_safety": False}


def classify(text, model=None):
    """Route a message to an intent. Rules first; the learned head only fills gaps.

    `model` is an optional MultiHeadLinear with an `intent` head. It is consulted
    only when no rule matched, and only when it does not abstain - so a model
    that knows nothing changes nothing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    normalized = normalize(text)
    act = conversation_act(text)
    if act == "PLAN_REQUEST":
        # Whatever else the sentence mentions (a friend, a doctor), what was asked for is
        # help with tasks; the topic rules would otherwise read the list as small talk.
        return IntentReading(intent="TASK_SUPPORT", matched_rule="PLAN_REQUEST",
                             all_matches=("TASK_SUPPORT",), act=act)
    matches = [intent for intent, pattern in COMPILED if pattern.search(normalized)]
    if matches:
        return IntentReading(intent=matches[0], matched_rule=matches[0],
                             all_matches=tuple(dict.fromkeys(matches)), act=act)

    if model is not None:
        try:
            prediction = model.predict(text)["intent"]
        except Exception:
            prediction = None
        if prediction and not prediction["abstain"]:
            return IntentReading(intent=prediction["label"], source="model",
                                 matched_rule=None, act=act)

    return IntentReading(intent="UNKNOWN", source="rules", abstained=True, act=act)
