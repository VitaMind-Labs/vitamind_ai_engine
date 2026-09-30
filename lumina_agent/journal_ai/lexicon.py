"""Interpretable safety cues. A match alone is not a diagnosis or crisis decision."""
import re
from dataclasses import dataclass, asdict
from .normalize import clauses, normalize

VERSION = "journal-cues-v1"

# Entries are authored development rules, not clinician-reviewed instruments.
PATTERNS = [
    ("self_harm", "explicit", "suicidal_ideation", r"\b(?:kill(?:ing)?|hurt(?:ing)?|harm(?:ing)?) (?:myself|himself|herself)\b|\b(?:end(?:ing)?|take|taking) (?:my|his|her) (?:own )?life\b|\b(?:commit|attempt) suicide\b|(?:انهي|انهاء) (?:حياتي|كل شي)|(?:اقتل|اؤذي|اوذي|اذي|اذيت|اذاء|ايذاء|تؤذي|تقتل) (?:نفسي|نفسها|نفسه)|\bانتحر\b"),
    ("death_wish", "explicit", "suicidal_ideation", r"\b(?:wants?|wish|plan|intend) to die\b|(?:ابغي|اريد|ودي|تريد|يريد) (?:ان )?(?:اموت|تموت|يموت)"),
    # These are support triggers for expressed death/absence wishes, not proof of intent.
    ("passive_death_wish", "urgent_wish", "suicidal_ideation", r"\b(?:don't|dont|do not) want to (?:be here anymore|wake up(?: tomorrow| again)?|live|be alive|exist)\b|\b(?:wish|rather).{0,16}not wake up\b|\bwish (?:i was|i were|i'd be|to be) dead\b|\bbetter off dead\b|\bwish i (?:had never been|was never|wasn't) born\b|\bno reason to (?:keep )?(?:living|going on)\b|\b(?:can't|cant|cannot) (?:go on|keep going) (?:anymore|living)\b|(?:ما ابغي اكون موجود بعد|ما ابغي اصحي|ما اريد ان استيقظ|(?:لا|ما) (?:اريد|ابغي|بدي) (?:ان )?اعيش|اتمني (?:الموت|لو كنت ميت|اني ميت)|الموت احسن لي)"),
    ("intent_ending", "explicit", "suicidal_ideation", r"\b(?:thinking|planning) (?:about|of) ending it\b|\b(?:going to|will) end it (?:tonight|today|now)\b|\bwrote goodbye (?:letters|messages)\b|(?:كتبت|اكتب) رسا(?:يل|ئل) وداع"),
    ("harmful_commands", "explicit", "command_hallucination", r"\b(?:voices?|someone)\b.{0,60}\b(?:tell(?:ing)?|order(?:ing)?|command(?:ing)?)\b.{0,45}\b(?:hurt|harm|kill|bad|punish|terrible)\b|(?:صوت|اصوات|احد).{0,45}(?:يامر|تامر|تقول|يقول).{0,40}(?:اذي|اوذي|اؤذي|اقتل|شين|فظيع|سيئ|سيء)|(?:يامرني|تامرني).{0,30}(?:اذي|اوذي|اؤذي|اقتل)"),
    ("commands", "serious", "command_hallucination", r"\bvoices?\b.{0,45}\b(?:commands?|orders?|telling me|tells me)\b|(?:اصوات|صوت).{0,35}(?:يامر|تامر|تقول لي|يقول لي)|التلفزيون.{0,30}(?:اوامر|يقولون لي)"),
    ("absence_wish", "serious", "suicidal_ideation", r"\b(?:disappear|gone) for (?:good|ever)\b|\b(?:don't|do not|not) want to (?:be here|wake up)\b|\b(?:better off|better) without me\b|\bno (?:point|reason) (?:in|to) (?:living|live)\b|\bnobody would (?:notice|care)\b|\bnot wake up\b|(?:اختفي للابد|ما ابغي اكون موجود|ما ابغي اصحي|ما في فايده من الحياه|احسن من دوني|ولا احد بيلاحظ|ما الفايده من الحياه|شو الفايده من العيشه)"),
    ("suicide_mention", "serious", "suicidal_ideation", r"\bsuicidal\b|\bsuicide\b|(?:الانتحار|افكار انتحاريه)"),
    ("hopelessness", "moderate", "hopelessness", r"\bhopeless\b|\bburden\b|\bnothing (?:will ever|ever|can) get better\b|\bno (?:point|hope)\b|\b(?:don't|do not) see the point\b|\bwhy (?:even )?try\b|\btrapped\b|\bnothing left to look forward to\b|(?:عبء|ميؤوس|بلا امل|ما اشوف فايده|ولا شي بيتحسن|ما في امل|مافي امل|ما في شي اقدر اسويه)"),
    ("persecutory_themes", "moderate", "paranoia", r"\b(?:watching|recording|spying on|following|whispering about) me\b|\bsecret messages\b|\b(?:camera|cameras) hidden\b|\b(?:plan|plot) against me\b|\breading my messages\b|\b(?:poison|poisoning)\b|(?:يراقبوني|تراقبني|يسجلوني|يتهامسون عني|يتجسسون|رسائل سريه|رسايل سريه|مخططين|مؤامره|يسمموني)"),
    ("anxiety", "marker", "anxiety", r"\b(?:anxious|anxiety|worried|worrying|worry|nervous|panic|scared|worst cases)\b|\bchest.{0,15}tight\b|(?:قلق|خايف|خائف|اخاف|هلع|اسوا الاحتمالات|صدري ضايق)"),
    ("sadness", "marker", "sadness", r"\b(?:sad|sadness|lonely|cry|cried|crying|empty|exhausted|feeling flat)\b|\beverything feels heavy\b|\b(?:out of|leave) bed\b|(?:حزين|زعلان|بوحده|بفراغ|ثقيل|ابكي|بكيت|من الفراش)"),
    ("overload", "marker", "overload", r"\b(?:overwhelmed|overload|deadline|deadlines|piling up)\b|\b(?:can't|couldn't) (?:focus|concentrate)\b|\bforgot.{0,20}appointment\b|\bfinished (?:nothing|none)\b|(?:متراكم|ديدلاين|ما اقدر اركز|ما قدرت اركز|ما خلصت ولا|نسيت موعد)"),
    ("elevated", "marker", "elevated", r"\b(?:unstoppable|invincible)\b|\b(?:new|three) projects\b|\b(?:don't|do not) need sleep\b|\bso many ideas\b|\b(?:spent|bought).{0,25}(?:money|laptop|things)\b|(?:ما احتاج نوم|مشاريع جديده|افكار وايد|صرفت وايد|حجزت سفره)"),
    ("idioms", "idiom", "idiom", r"\b(?:traffic|feet|work|homework|exam).{0,30}(?:killing|murdering|death of)\b|\bdying to (?:see|watch|meet|try)\b|\b(?:kill|die) for (?:a |some |good )?(?:coffee|karak|shawarma|pizza|chocolate)\b|\bdied laughing\b|(?:الزحمه ذبحتني|اموت علي|ميت تعب|ميت ابغي اشوف|مت من الضحك)"),
]
COMPILED = [(id_,kind,category,re.compile(pattern)) for id_,kind,category,pattern in PATTERNS]

NEGATION = re.compile(r"(?:\b(?:don't|do not|never|not|wouldn't|won't|will not)\s+(?:(?:want|wanting|plan|planning|intend|intending|think|thinking|going)\s+(?:to|about|of)\s+|to\s+)?|\bno (?:intention|plan|plans|desire) (?:of|to|for)\s+|(?:ما|لا|لن|لست)\s+(?:(?:ابغي|اريد|افكر|ناوي|انوي|راح|سوف)\s+){0,2}(?:في\s+)?)$")
UNCERTAIN_SAFETY = re.compile(r"\b(?:can't|cannot|can not|couldn't|not able to)\b.{0,30}\b(?:promise|guarantee|sure)\b|(?:لا استطيع|ما اقدر).{0,20}(?:اضمن|اوعد)")
VOICE_DENIAL = re.compile(r"\b(?:don't|do not|never|not) (?:hear|hearing)\b.{0,25}\bvoices?\b|\bno voices?\b|\bvoices?\b.{0,20}\b(?:isn't|is not|aren't|are not|not) (?:telling|ordering|commanding)\b|(?:ما|لا) اسمع.{0,20}(?:اصوات|صوت)|(?:اصوات|صوت).{0,20}(?:لا|ما) (?:تامر|يامر|تقول|يقول)")
PAST = re.compile(r"\b(?:years? ago|last year|last winter|used to|back then|there was a time|during my episode|in the past)\b|(?:السنه اللي طافت|الشتا اللي طاف|النوبه اللي طافت|كنت|سابقا|في الماضي|قبل سنوات)")
OTHER = re.compile(r"\bmy (?:friend|brother|sister|cousin|dad|father|mom|mother|son|daughter|colleague|neighbor)\b.{0,35}\b(?:said|says|told|thinks|thought|wants|wanted|feels|felt|sent|keeps|wishes)\b|\b(?:he|she) (?:wants|wanted|said|says)\b|(?:اخوي|اختي|صديقي|صديقتي|ابوي|امي|جارتنا).{0,25}(?:قال|تقول|يقول|قالت|اعترف|تظن|يحس|تشعر)")
QUOTED = re.compile(r"\b(?:a character|the character|the actor|the movie|the book|the film|the song|the article)\b.{0,65}\b(?:says|said|line|quote|reads)\b|\b(?:lyrics|quoted|quoting)\b|(?:الشخصيه|الممثل|الفيلم|الروايه).{0,40}(?:قال|تقول|يقول|عباره)|\bاقتباس\b")

@dataclass
class Hit:
    cue: str
    kind: str
    category: str
    clause: int
    start: int
    end: int
    negated: bool
    temporal: str
    subject: str
    quoted: bool

    def to_dict(self):
        return asdict(self)

def scan(text):
    hits=[]
    for index,clause in enumerate(clauses(text)):
        for cue,kind,category,pattern in COMPILED:
            for match in pattern.finditer(clause):
                if cue=="passive_death_wish" and re.search(r"\b(?:at work|at the office|at this party|in this room|in this city|in class)\b|(?:في الدوام|في الحفله|في المكتب|في هذه الغرفه)",clause):
                    continue
                prefix=clause[:match.start()]
                # A present-tense contrast or new first-person subject resets history/attribution.
                scope=re.split(r"\b(?:now|today|tonight|and i|and now i)\b|(?:والحين|والان|واليوم|وانا)",prefix)[-1]
                negated=bool(NEGATION.search(prefix[-90:]))
                if VOICE_DENIAL.search(clause[:match.end()]) and category in ("suicidal_ideation","command_hallucination"):
                    negated=True
                if UNCERTAIN_SAFETY.search(prefix):
                    negated=False
                temporal="past" if PAST.search(scope) else "current"
                subject="other" if OTHER.search(scope+match.group()) or re.search(r"\b(?:himself|herself|his life|her life)\b|(?:نفسها|نفسه)",match.group()) else "self"
                if re.search(r"\b(?:told|tells|telling|ordered|ordering) me to\s*$",prefix):
                    subject="self"
                quoted=bool(QUOTED.search(scope))
                hits.append(Hit(cue,kind,category,index,match.start(),match.end(),negated,temporal,subject,quoted))
    return hits
