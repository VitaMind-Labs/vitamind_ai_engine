"""Authored (synthetic) sentences, part A: meta intents, missing emotions, safety seeds.

Everything here is written for coverage, not collected from patients. Builders tag
it source_type=synthetic, verified=false, needs_human_review=true. Arabic dialect
wording (Gulf/Egyptian/Levantine/Arabizi) needs a native clinician reviewer before
use; it was written without one.
"""

# ---------------------------------------------------------------------------
# Meta intents. Keys: en | MSA | GLF | EGY | LEV | FRANCO (Arabizi)
# ---------------------------------------------------------------------------
INTENT_SEEDS = {
    "GREETING": {
        "en": ["hi", "hello", "hey there", "good morning", "good evening", "good afternoon",
               "hi lumina, are you there?", "hello again", "hey, it's me", "morning lumina",
               "hi, can we talk?"],
        "MSA": ["مرحبا", "أهلاً بك", "السلام عليكم", "صباح الخير", "مساء الخير",
                "مرحباً لومينا، هل أنت هنا؟", "أهلاً من جديد"],
        "GLF": ["هلا", "هلا والله", "يا هلا", "اهلين", "هلا فيك"],
        "EGY": ["ازيك", "إزيك عامل إيه", "صباح الفل", "أهلا يا لومينا"],
        "LEV": ["مرحبا كيفك", "هاي", "أهلين", "صباح الخير كيفك"],
        "FRANCO": ["salam", "marhaba", "hala", "ahlan", "hi lumina"],
    },
    "GOODBYE": {
        "en": ["bye", "goodbye", "see you tomorrow", "talk to you later", "good night",
               "i have to go now", "that's all for today", "i'm done for today",
               "catch you later", "gotta go, bye"],
        "MSA": ["مع السلامة", "إلى اللقاء", "تصبح على خير", "أراك غداً", "سأذهب الآن",
                "هذا كل شيء لليوم", "انتهيت لهذا اليوم", "أكلمك لاحقاً"],
        "GLF": ["مع السلامة", "لازم أروح الحين", "أشوفك بكرة", "خلصنا لليوم", "بكلمك بعدين"],
        "EGY": ["مع السلامة", "سلام", "هكلمك بعدين", "همشي دلوقتي", "خلصت كده النهاردة"],
        "LEV": ["مع السلامة", "بشوفك بكرا", "بحكيك بعدين", "لازم روح هلق", "خلصنا لليوم"],
        "FRANCO": ["bye", "yalla bye", "tesbah ala kher", "ashoofak bokra", "bkellmak ba3dein"],
    },
    "THANKS": {
        "en": ["thanks", "thank you", "thanks a lot", "that helped, thank you",
               "i appreciate it", "thank you lumina", "thanks, that was useful",
               "cheers for that"],
        "MSA": ["شكراً", "شكراً لك", "شكراً جزيلاً", "ساعدتني، شكراً", "أقدّر ذلك",
                "شكراً لومينا", "كان هذا مفيداً، شكراً"],
        "GLF": ["مشكور", "يعطيك العافية", "الله يعطيك العافية", "تسلم", "شكراً وايد"],
        "EGY": ["ميرسي", "تسلم", "شكراً جداً", "كتر خيرك"],
        "LEV": ["يعطيك العافية", "مرسي كتير", "شكراً كتير", "تسلم"],
        "FRANCO": ["shukran", "mashkoor", "yislamo", "merci", "thanks lumina"],
    },
    "ABOUT_BOT": {
        "en": ["who are you", "what are you", "are you a real person", "are you a robot",
               "are you a doctor", "what can you do", "who made you", "are you an ai",
               "do you remember me", "is this conversation private",
               "where do you get your information"],
        "MSA": ["من أنت؟", "ما أنت؟", "هل أنت إنسان حقيقي؟", "هل أنت روبوت؟",
                "هل أنت طبيب؟", "ماذا تستطيع أن تفعل؟", "من صنعك؟", "هل أنت ذكاء اصطناعي؟",
                "هل تتذكرني؟", "هل هذه المحادثة سرية؟"],
        "GLF": ["منو انتي؟", "انتي روبوت؟", "شنو تقدرين تسوين؟", "انتي بشر ولا برنامج؟",
                "هالمحادثة سرية؟"],
        "EGY": ["انتي مين؟", "انتي روبوت؟", "بتعملي إيه؟", "مين عملك؟", "الكلام ده سري؟"],
        "LEV": ["مين انتي؟", "انتي روبوت؟", "انتي بشر ولا برنامج؟", "شو بتعملي؟", "مين صنعك؟"],
        "FRANCO": ["inti min", "enti robot?", "inti bashar wala program", "shu beti3mali"],
    },
    "BOT_FEEDBACK": {
        "en": ["you keep repeating yourself", "you already said that",
               "why do you keep saying the same thing", "that doesn't make sense",
               "you're not listening to me", "you don't understand me",
               "that was a useless answer", "you sound like a robot",
               "stop repeating yourself", "you're not helping", "that's not what i asked",
               "you're not understanding me at all"],
        "MSA": ["أنت تكرر نفس الكلام", "قلت لي هذا من قبل", "لماذا تعيد نفس الكلام؟",
                "كلامك لا معنى له", "أنت لا تستمع إلي", "أنت لا تفهمني", "هذا رد غير مفيد",
                "تتكلم مثل الروبوت", "توقف عن التكرار", "أنت لا تساعدني",
                "ليس هذا ما سألتك عنه"],
        "GLF": ["ليش تعيدين نفس الكلام؟", "قلتيه لي قبل", "ما تفهمين علي",
                "مو هذا اللي سألتك عنه", "ردك ما يفيدني"],
        "EGY": ["انتي بتكرري نفس الكلام", "انتي مش فاهماني", "الرد ده مش مفيد",
                "ده مش اللي سألتك عليه", "انتي مش بتسمعيني"],
        "LEV": ["عم تعيدي نفس الحكي", "ما عم تفهمي علي", "هاد الرد ما بيفيد",
                "هاد مش يلي سألتك عنه", "ما عم تسمعيني"],
        "FRANCO": ["enta betkarrer nafs el kalam", "ma tefhamni", "el radd mesh mufeed"],
    },
    "DISENGAGE": {
        "en": ["i don't want to talk about it", "let's talk about something else",
               "leave me alone", "i'd rather not say", "not now", "drop it",
               "i don't feel like talking", "can we stop for now", "skip this question",
               "i don't want to answer that"],
        "MSA": ["لا أريد الحديث عن هذا", "دعنا نتحدث عن شيء آخر", "اتركني وحدي",
                "أفضّل ألا أقول", "ليس الآن", "لا أرغب في الكلام",
                "هل يمكننا التوقف الآن؟", "تخطَّ هذا السؤال", "لا أريد الإجابة عن هذا"],
        "GLF": ["ما أبغى أتكلم عن هذا", "خلينا نتكلم عن شي ثاني", "خلني وحدي", "مو الحين",
                "ما عندي مزاج أتكلم", "ما أبي أجاوب"],
        "EGY": ["مش عايز اتكلم في الموضوع ده", "خلينا نتكلم في حاجة تانية", "سيبني لوحدي",
                "مش دلوقتي", "مش عايز أجاوب"],
        "LEV": ["ما بدي احكي عن هالشي", "خلينا نحكي عن شي تاني", "خليني لحالي", "مش هلق",
                "ما بدي جاوب"],
        "FRANCO": ["ma abgha atkallam", "mesh 3ayez a2ol", "khalina nghayer el mawdoo3",
                   "ma badi ehki"],
    },
}

DIALECT_LANG = {"en": ("en", "EN"), "MSA": ("ar", "MSA"), "GLF": ("ar", "GLF"),
                "EGY": ("ar", "EGY"), "LEV": ("ar", "LEV"), "FRANCO": ("ar", "FRANCO")}

# ---------------------------------------------------------------------------
# Emotions the public corpora do not cover well (OVERWHELMED, LONELY, LOW_ENERGY),
# plus ANXIOUS / CALM / DISTRESSED which are thin in GoEmotions.
# ---------------------------------------------------------------------------
EN_LEADINS = ["", "honestly, ", "lately ", "today "]
AR_LEADINS = ["", "بصراحة ", "اليوم "]

EMOTION_SEEDS = {
    "OVERWHELMED": {
        "en": ["i have too much on my plate", "everything is piling up and i can't keep up",
               "it's all too much right now", "i don't know where to start, there's so much",
               "i feel buried under everything", "my head is so full i can't sort anything out",
               "there are too many things to do and i'm freezing",
               "i can't handle all of this at once"],
        "MSA": ["لدي أشياء كثيرة جداً ولا أستطيع مجاراتها", "كل شيء يتراكم عليّ",
                "هذا كثير عليّ الآن", "لا أعرف من أين أبدأ فالأمور كثيرة",
                "أشعر بأن كل شيء يثقل عليّ دفعة واحدة", "رأسي ممتلئ ولا أستطيع ترتيب أي شيء"],
        "GLF": ["الشغلات كثيرة وايد ومب قادر ألحق", "ما أعرف من وين أبدأ، كل شي فوق راسي"],
    },
    "LONELY": {
        "en": ["i feel completely alone", "nobody checks in on me anymore",
               "i haven't talked to anyone in days", "i miss having people around",
               "the evenings are the hardest, it's so quiet and empty",
               "i have no one to talk to", "i feel cut off from everyone",
               "everyone has someone except me"],
        "MSA": ["أشعر بوحدة تامة", "لا أحد يسأل عني", "لم أتحدث مع أحد منذ أيام",
                "أفتقد وجود الناس حولي", "المساء هو أصعب وقت، هادئ وفارغ",
                "ليس لدي أحد أتحدث إليه"],
        "GLF": ["أحس روحي وحيد وايد", "ما عندي أحد أسولف معاه"],
    },
    "LOW_ENERGY": {
        "en": ["i have no energy at all", "i'm exhausted even after sleeping",
               "getting out of bed takes everything i have", "i feel drained all the time",
               "my body feels so heavy today", "i can't find the energy to do anything",
               "i'm running on empty", "everything feels like such an effort"],
        "MSA": ["ليس لدي أي طاقة", "أشعر بالإرهاق حتى بعد النوم",
                "النهوض من السرير يستهلك كل ما لدي", "أشعر بالإنهاك طوال الوقت",
                "جسدي ثقيل جداً اليوم", "لا أجد الطاقة لفعل أي شيء"],
        "GLF": ["ما عندي طاقة أبداً", "تعبان وايد حتى بعد ما أنام"],
    },
    "ANXIOUS": {
        "en": ["i can't stop worrying about what might go wrong",
               "my chest feels tight and my mind is racing with worries",
               "i keep imagining the worst that could happen", "i'm on edge and i can't relax",
               "i'm so nervous about tomorrow", "my stomach is in knots and i can't settle",
               "i keep checking things again and again because i'm afraid",
               "i feel a constant sense of dread"],
        "MSA": ["لا أتوقف عن القلق مما قد يحدث", "صدري مشدود وأفكاري تدور حول المخاوف",
                "أتخيل الأسوأ طوال الوقت", "أنا متوتر ولا أستطيع الاسترخاء",
                "أنا قلق جداً بشأن الغد", "أشعر بعقدة في معدتي ولا أستطيع الهدوء"],
        "GLF": ["قلبي يدق وما أقدر أهدأ", "خايف من بكرة وايد"],
    },
    "CALM": {
        "en": ["i feel calm and settled today", "things feel peaceful right now",
               "i'm relaxed and my mind is quiet", "today was slow and steady and i feel okay",
               "i feel at ease", "my breathing is slow and i feel grounded",
               "i feel steady today", "everything feels quiet in a good way"],
        "MSA": ["أشعر بالهدوء والاستقرار اليوم", "الأمور هادئة الآن",
                "أنا مسترخٍ وذهني هادئ", "كان يومي بطيئاً ومتزناً وأشعر بخير",
                "أشعر بالطمأنينة", "أشعر بالثبات اليوم"],
        "GLF": ["مرتاح اليوم وبالي هادي", "الحمدلله أحس بهدوء"],
    },
    "DISTRESSED": {
        "en": ["i'm in a lot of pain and i don't know how to cope",
               "i feel terrible and can't think straight", "i'm really struggling right now",
               "everything is falling apart", "i can't take how bad this feels",
               "i feel awful and completely shaken",
               "i'm so upset that i can't stop shaking", "i don't know how to get through this"],
        "MSA": ["أنا في ألم شديد ولا أعرف كيف أتحمل", "أشعر بسوء شديد ولا أستطيع التفكير بوضوح",
                "أنا أعاني كثيراً الآن", "كل شيء ينهار", "لا أحتمل كيف أشعر الآن",
                "لا أعرف كيف سأتجاوز هذا"],
        "GLF": ["تعبان نفسياً وايد ومب عارف شسوي"],
    },
}

# ---------------------------------------------------------------------------
# Safety seeds. Levels follow the documented behaviour in MODELS.md:
#   attribution -> ELEVATED, past-resolved -> ELEVATED, idiom -> NORMAL.
# CRISIS is never a training label (rules own it); it appears only in SAFETY_EVAL_CRISIS.
# No methods, places or means appear anywhere in this file.
# ---------------------------------------------------------------------------
SAFETY_SEEDS = {
    "ELEVATED": {
        "en": ["i've been really down lately", "i can't stop worrying", "i feel so alone",
               "i haven't been sleeping and everything feels heavy",
               "i'm exhausted and nothing helps", "i cry most nights and i don't know why",
               "i'm so stressed i can't think", "i feel like a failure",
               "i keep isolating myself from everyone", "i've lost interest in everything",
               "my anxiety is getting worse",
               "my brother said he wanted to die", "my friend told me she feels hopeless",
               "a few years ago i wished i could disappear but i'm in a much better place now"],
        "MSA": ["أشعر بحزن شديد هذه الأيام", "لا أتوقف عن القلق", "أشعر بالوحدة",
                "لا أنام وكل شيء يبدو ثقيلاً", "أنا منهك ولا شيء يفيد",
                "أبكي أغلب الليالي ولا أعرف السبب", "أشعر أنني فاشل", "أبتعد عن الجميع",
                "فقدت الاهتمام بكل شيء", "قلقي يزداد سوءاً", "قال أخي إنه يريد أن يموت",
                "صديقتي تقول إنها يائسة من كل شيء"],
        "GLF": ["نفسيتي تعبانة وايد هالأيام", "ما أنام وكل شي ثقيل علي", "حاس إني فاشل"],
    },
    "HIGH": {
        "en": ["i feel like a burden to everyone", "i don't see a way out of this",
               "i can't keep doing this", "i feel hopeless about everything",
               "nothing will ever get better", "i wish i could just disappear",
               "i don't want to be here anymore", "everyone would be better off without me"],
        "MSA": ["أشعر أنني عبء على الجميع", "لا أرى مخرجاً من هذا", "لا أستطيع الاستمرار هكذا",
                "أشعر باليأس من كل شيء", "لن يتحسن شيء أبداً", "أتمنى لو أختفي",
                "لا أريد أن أكون هنا بعد الآن"],
        "GLF": ["حاس إني عبء على الكل", "ما أشوف مخرج من هالوضع", "ما أقدر أكمل كذا",
                "ما أبغى أكون هنا بعد"],
        "EGY": ["حاسس إني عبء على الكل", "مش شايف مخرج", "مش قادر أكمل كده"],
        "LEV": ["حاسس حالي عبء على الكل", "ما عاد فيني كمل هيك"],
    },
    "NORMAL": {
        "en": ["i had a nice walk and slept well", "work was busy but fine",
               "i made pasta for dinner", "this traffic is killing me",
               "i'm dying to see that new movie", "my phone died again",
               "i could murder a coffee right now", "that joke killed me",
               "we're going to the park this weekend", "i finished my report early",
               "i'm exhausted after the gym but happy",
               "the meeting ran long but it went okay"],
        "MSA": ["قضيت يوماً جميلاً ونمت جيداً", "كان العمل مزدحماً لكنه بخير",
                "حضّرت المعكرونة للعشاء", "هذا الزحام يقتلني",
                "أموت شوقاً لمشاهدة الفيلم الجديد", "مات هاتفي مرة أخرى", "أموت من الضحك",
                "سنذهب إلى الحديقة نهاية الأسبوع", "أنهيت التقرير مبكراً",
                "أنا متعب بعد النادي لكنني سعيد"],
        "GLF": ["الزحمة بتموتني", "كنت أموت ضحك", "خلصت التقرير بدري",
                "رحنا الكورنيش وانبسطنا"],
    },
}

# Regression probes for the deterministic rules (NOT training data).
SAFETY_EVAL_CRISIS = {
    "en": ["i want to kill myself", "i am going to end my life tonight",
           "i don't want to live anymore", "i'm thinking about suicide",
           "i've decided to kill myself", "i want to die"],
    "MSA": ["أريد أن أقتل نفسي", "سأنهي حياتي الليلة", "لا أريد أن أعيش بعد الآن",
            "أفكر في الانتحار", "قررت أن أنتحر", "أريد أن أموت"],
    "GLF": ["أبغى أموت", "ما أبغى أعيش", "أفكر أنتحر", "ما أبغى أصحى بكرة"],
    "EGY": ["عايز أموت", "مش عايز أعيش", "بفكر أنتحر"],
    "LEV": ["بدي موت", "ما بدي عيش", "عم فكر انتحر"],
    "FRANCO": ["abgha amoot", "badi moot", "3ayez amoot", "ma abgha a3eesh"],
}
