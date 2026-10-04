"""Authored seed phrasings for the Lumina intent vocabulary.

Why this file exists: none of the corpora in `data/` carry Lumina intent labels.
DailyDialog annotates four communication acts, which is a different question from
"what is this patient trying to do right now". Rather than derive intent labels
from keywords and present the result as supervision, the seeds are written out
explicitly here, where they can be read, argued with and replaced.

What this is NOT: a training corpus. It is a few hundred authored sentences. A
model trained on it will handle phrasings close to these and will not generalise
to real patient language. Every row is marked `authored_seed` / `verified: false`
/ `needs_human_review: true`, and the trained model stays a candidate until it is
retrained on reviewed data from actual use.

Grouping: each entry is (intent, family, [phrasings]). A family is one underlying
meaning; its phrasings never span a train/test split, so the reported score
measures generalisation to unseen *phrasings within an authored set*, not
memorisation - and still not generalisation to real patients.
"""
from __future__ import annotations

from .intent_seed_families import MORE_SEEDS

# (intent, family, english phrasings, arabic phrasings)
BASE_SEEDS = [
    ("CHECK_IN", "daily_report", [
        "Here is how today went for me",
        "Checking in for today",
        "Today was okay overall, nothing unusual",
        "Reporting in: slept fine, average day",
    ], [
        "هذا ما مرّ به يومي",
        "أسجّل حالتي لليوم",
        "كان اليوم عاديًا، لا شيء غير معتاد",
    ]),
    ("CHECK_IN", "state_summary", [
        "Mood is about a six, energy lower than usual",
        "Sleep was short but the day was fine",
        "Stress is high today, everything else is normal",
    ], [
        "مزاجي حوالي ستة وطاقتي أقل من المعتاد",
        "نومي كان قصيرًا لكن اليوم كان جيدًا",
        "توتري مرتفع اليوم وبقية الأمور عادية",
    ]),
    ("JOURNAL", "evening_entry", [
        "I want to write about my day",
        "Let me put down what happened today",
        "Writing this before bed, it was a long one",
    ], [
        "أريد أن أكتب عن يومي",
        "دعني أدوّن ما حدث اليوم",
        "أكتب هذا قبل النوم، كان يومًا طويلًا",
    ]),
    ("EMOTIONAL_SUPPORT", "need_to_be_heard", [
        "I just need someone to listen right now",
        "I do not want advice, I want to be heard",
        "Can you just sit with me for a moment",
        "I am having a really hard time today",
    ], [
        "أحتاج فقط لمن يستمع إليّ الآن",
        "لا أريد نصيحة، أريد أن يُسمع لي",
        "أمر بيوم صعب جدًا",
    ]),
    ("TASK_SUPPORT", "cannot_start", [
        "I know what I need to do but I keep avoiding starting it",
        "I cannot get myself to start this task",
        "I have been staring at this for an hour and done nothing",
        "I could not start anything today",
    ], [
        "أعرف ما عليّ فعله لكنني أتجنب البدء",
        "لا أستطيع أن أجعل نفسي أبدأ هذه المهمة",
        "لم أستطع البدء بأي شيء اليوم",
    ]),
    ("TASK_SUPPORT", "too_much_to_do", [
        "I have ten things to do and I am doing none of them",
        "There is too much on my list and I do not know where to begin",
        "Everything is piling up and I keep putting it off",
    ], [
        "لديّ عشرة أشياء ولا أنجز أيًا منها",
        "قائمتي طويلة جدًا ولا أعرف من أين أبدأ",
        "كل شيء يتراكم وأستمر في التأجيل",
    ]),
    ("SLEEP", "sleep_problem", [
        "I barely slept last night",
        "I keep waking up at three in the morning",
        "Falling asleep has been taking hours",
        "My sleep has been all over the place this week",
    ], [
        "بالكاد نمت الليلة الماضية",
        "أستيقظ في الثالثة فجرًا كل ليلة",
        "يستغرق نومي ساعات حتى يأتي",
    ]),
    ("ENERGY", "energy_change", [
        "I have no energy at all today",
        "I feel unusually energetic even though I barely slept",
        "I have been running on empty for days",
    ], [
        "ليس لديّ أي طاقة اليوم",
        "أشعر بنشاط غير معتاد رغم قلة نومي",
        "أشعر بالإنهاك منذ أيام",
    ]),
    ("FOCUS", "attention_problem", [
        "I cannot focus today",
        "I keep opening other tabs instead of working",
        "My attention slides off everything I try to read",
    ], [
        "لا أستطيع التركيز اليوم",
        "أفتح صفحات أخرى بدل أن أعمل",
        "ينزلق انتباهي عن كل ما أحاول قراءته",
    ]),
    ("ROUTINE", "routine_change", [
        "My routine has completely fallen apart",
        "I have not kept a regular schedule in weeks",
        "I want to get back into some kind of daily rhythm",
    ], [
        "انهار روتيني تمامًا",
        "لم ألتزم بجدول منتظم منذ أسابيع",
        "أريد العودة إلى إيقاع يومي ما",
    ]),
    ("STRESS", "under_pressure", [
        "Everything feels like too much right now",
        "I am really stressed about work",
        "The pressure has not let up all week",
    ], [
        "كل شيء يبدو أكثر مما أحتمل الآن",
        "أنا متوتر جدًا بسبب العمل",
        "لم يخفّ الضغط طوال الأسبوع",
    ]),
    ("SOCIAL", "social_context", [
        "I have not spoken to anyone in days",
        "I cancelled on my friends again",
        "Being around people has felt like too much lately",
    ], [
        "لم أتحدث مع أحد منذ أيام",
        "ألغيت موعدي مع أصدقائي مرة أخرى",
        "وجودي بين الناس صار ثقيلًا عليّ مؤخرًا",
    ]),
    ("EXERCISE", "activity", [
        "I went for a walk this morning",
        "I have not moved from my desk all day",
        "I want to start being more active again",
    ], [
        "خرجت للمشي هذا الصباح",
        "لم أتحرك من مكتبي طوال اليوم",
        "أريد أن أعود للنشاط من جديد",
    ]),
    ("GOAL", "goal_setting", [
        "I want to set a goal for this week",
        "My goal is to finish the report by Friday",
        "Can we work on something I am trying to build a habit around",
    ], [
        "أريد أن أضع هدفًا لهذا الأسبوع",
        "هدفي أن أنهي التقرير قبل الجمعة",
        "أريد أن أبني عادة جديدة",
    ]),
    ("PROGRESS", "reviewing_progress", [
        "How have I been doing over the last few weeks",
        "Has anything changed since I started",
        "I want to look back at my last month",
    ], [
        "كيف كان أدائي في الأسابيع الماضية",
        "هل تغيّر شيء منذ أن بدأت",
        "أريد مراجعة الشهر الماضي",
    ]),
    ("MEDICATION_MENTION", "medication_context", [
        "I started a new medication last week",
        "My doctor changed my prescription",
        "I have been taking my medication as prescribed",
    ], [
        "بدأت دواءً جديدًا الأسبوع الماضي",
        "غيّر طبيبي الوصفة",
        "ألتزم بدوائي كما وُصف لي",
    ]),
    ("CLINICIAN_MENTION", "clinician_context", [
        "I have an appointment with my psychiatrist on Tuesday",
        "My therapist suggested I track this",
        "I saw my doctor yesterday",
    ], [
        "لديّ موعد مع طبيبي النفسي يوم الثلاثاء",
        "اقترح معالجي أن أتابع هذا",
        "زرت طبيبي أمس",
    ]),
    ("QUESTION", "asking_about_system", [
        "What can you actually help me with",
        "How does this work",
        "Can you explain what you do with what I tell you",
    ], [
        "بماذا يمكنك مساعدتي فعلًا",
        "كيف يعمل هذا",
        "هل يمكنك شرح ما تفعله بما أخبرك به",
    ]),
    ("GENERAL_CONVERSATION", "small_talk", [
        "Good morning",
        "Thanks, that helped",
        "Nothing much to report today",
        "Talk to you tomorrow",
    ], [
        "صباح الخير",
        "شكرًا، هذا ساعدني",
        "لا جديد اليوم",
        "أراك غدًا",
    ]),
    # --- second and third meaning-families per intent -------------------
    # An intent with one family cannot be evaluated: a grouped split puts its
    # only meaning in one partition and leaves the others empty. Each intent
    # below is carried to at least three genuinely different meanings so a
    # held-out split has something real to score.
    ("CHECK_IN", "prompted_response", [
        "You asked how I slept, it was about five hours",
        "Answering your question from this morning",
        "Yes, today felt harder than yesterday",
    ], [
        "سألتني عن نومي، كان حوالي خمس ساعات",
        "أجيب عن سؤالك هذا الصباح",
        "نعم، اليوم كان أصعب من أمس",
    ]),
    ("JOURNAL", "reflective_entry", [
        "Looking back on the week, a few things stand out",
        "I keep thinking about what happened at work",
        "Some thoughts I wanted to get out of my head",
    ], [
        "حين أنظر إلى الأسبوع، تبرز أمور قليلة",
        "أظل أفكر فيما حدث في العمل",
        "أفكار أردت إخراجها من رأسي",
    ]),
    ("JOURNAL", "event_entry", [
        "Today my sister visited and it went better than expected",
        "Had coffee with a friend, first time in months",
    ], [
        "زارتني أختي اليوم وكان الأمر أفضل مما توقعت",
        "شربت القهوة مع صديق، لأول مرة منذ شهور",
    ]),
    ("EMOTIONAL_SUPPORT", "feeling_low", [
        "I feel completely flat and I do not know why",
        "Everything feels pointless today",
        "I am just sad and I cannot shake it",
    ], [
        "أشعر بفراغ تام ولا أعرف السبب",
        "كل شيء يبدو بلا معنى اليوم",
        "أنا حزين ولا أستطيع التخلص من ذلك",
    ]),
    ("EMOTIONAL_SUPPORT", "feeling_alone", [
        "I feel like nobody really understands what this is like",
        "I am lonely even when people are around",
    ], [
        "أشعر أن لا أحد يفهم حقًا ما أمر به",
        "أشعر بالوحدة حتى حين يكون الناس حولي",
    ]),
    ("TASK_SUPPORT", "task_abandoned", [
        "I started three things today and finished none of them",
        "I keep switching between tasks and getting nowhere",
    ], [
        "بدأت ثلاثة أشياء اليوم ولم أُنه أيًا منها",
        "أنتقل بين المهام ولا أصل إلى شيء",
    ]),
    ("SLEEP", "sleep_improvement", [
        "I actually slept well for once",
        "Last night was the best sleep I have had in weeks",
    ], [
        "نمت جيدًا لأول مرة منذ مدة",
        "الليلة الماضية كانت أفضل نوم لي منذ أسابيع",
    ]),
    ("SLEEP", "sleep_schedule", [
        "I have been going to bed at four in the morning",
        "My sleep schedule has shifted completely",
    ], [
        "أنام في الرابعة فجرًا",
        "تغيّر جدول نومي تمامًا",
    ]),
    ("ENERGY", "fatigue", [
        "I am exhausted before the day even starts",
        "Getting out of bed took everything I had",
    ], [
        "أشعر بالإرهاق قبل أن يبدأ اليوم",
        "النهوض من السرير استهلك كل ما لديّ",
    ]),
    ("ENERGY", "restlessness", [
        "I cannot sit still today",
        "I have more drive than I know what to do with",
    ], [
        "لا أستطيع الجلوس ساكنًا اليوم",
        "لديّ اندفاع أكثر مما أعرف كيف أستخدمه",
    ]),
    ("FOCUS", "distraction", [
        "My phone has eaten the entire afternoon",
        "I get pulled away every few minutes",
    ], [
        "التهم هاتفي فترة ما بعد الظهر كاملة",
        "أنجذب بعيدًا كل بضع دقائق",
    ]),
    ("FOCUS", "focus_good", [
        "I had a really clear head this morning",
        "I managed two solid hours of work",
    ], [
        "كان ذهني صافيًا هذا الصباح",
        "أنجزت ساعتين متواصلتين من العمل",
    ]),
    ("ROUTINE", "routine_holding", [
        "I kept my morning routine three days running",
        "Getting up at the same time has been working",
    ], [
        "حافظت على روتيني الصباحي ثلاثة أيام متتالية",
        "الاستيقاظ في الوقت نفسه ينجح معي",
    ]),
    ("ROUTINE", "routine_disrupted", [
        "Travel has thrown off everything I had set up",
        "Since the schedule changed nothing has been regular",
    ], [
        "أفسد السفر كل ما رتّبته",
        "منذ تغيّر الجدول لم يعد شيء منتظمًا",
    ]),
    ("STRESS", "specific_stressor", [
        "The deadline on Thursday is what is getting to me",
        "A conversation with my family has been sitting on me",
    ], [
        "الموعد النهائي يوم الخميس هو ما يرهقني",
        "حديث مع عائلتي ما زال يثقل عليّ",
    ]),
    ("STRESS", "stress_easing", [
        "Things have calmed down since last week",
        "The pressure finally lifted a bit",
    ], [
        "هدأت الأمور منذ الأسبوع الماضي",
        "خفّ الضغط أخيرًا قليلًا",
    ]),
    ("SOCIAL", "social_positive", [
        "I saw a friend today and it helped",
        "I finally replied to the messages I had been avoiding",
    ], [
        "قابلت صديقًا اليوم وقد ساعدني ذلك",
        "رددت أخيرًا على الرسائل التي كنت أتجنبها",
    ]),
    ("EXERCISE", "exercise_done", [
        "I did twenty minutes of stretching",
        "I got to the gym for the first time this month",
    ], [
        "مارست تمارين الإطالة عشرين دقيقة",
        "ذهبت إلى النادي لأول مرة هذا الشهر",
    ]),
    ("EXERCISE", "exercise_blocked", [
        "I meant to go out but never did",
        "Moving feels like too much effort lately",
    ], [
        "نويت الخروج لكنني لم أفعل",
        "الحركة تبدو مجهودًا كبيرًا مؤخرًا",
    ]),
    ("GOAL", "goal_progress", [
        "I am halfway through what I set out to do",
        "The habit has stuck for about a week now",
    ], [
        "أنجزت نصف ما خططت له",
        "استمرت العادة نحو أسبوع الآن",
    ]),
    ("GOAL", "goal_change", [
        "I want to drop the goal I set last week",
        "That target was too big, I want a smaller one",
    ], [
        "أريد التخلي عن الهدف الذي وضعته الأسبوع الماضي",
        "كان ذلك الهدف كبيرًا جدًا، أريد هدفًا أصغر",
    ]),
    ("PROGRESS", "asking_for_summary", [
        "Can you show me a summary of my week",
        "What have you noticed about my patterns",
    ], [
        "هل يمكنك أن تريني ملخص أسبوعي",
        "ما الذي لاحظته في أنماطي",
    ]),
    ("PROGRESS", "comparing", [
        "Is this better or worse than last month",
        "Am I sleeping more than I was",
    ], [
        "هل هذا أفضل أم أسوأ من الشهر الماضي",
        "هل أنام أكثر مما كنت",
    ]),
    ("MEDICATION_MENTION", "medication_difficulty", [
        "I keep forgetting to take it in the evening",
        "The side effects have been hard to sit with",
    ], [
        "أنسى تناوله في المساء باستمرار",
        "الآثار الجانبية كانت صعبة عليّ",
    ]),
    ("MEDICATION_MENTION", "medication_question", [
        "Should I be taking this at a different time",
        "I am wondering whether this is still the right one for me",
    ], [
        "هل يجب أن أتناوله في وقت مختلف",
        "أتساءل إن كان هذا ما زال مناسبًا لي",
    ]),
    ("CLINICIAN_MENTION", "clinician_plan", [
        "We agreed I would track my sleep before the next visit",
        "She wants me to bring notes to the next session",
    ], [
        "اتفقنا أن أتابع نومي قبل الزيارة القادمة",
        "تريد مني أن أحضر ملاحظات للجلسة القادمة",
    ]),
    ("CLINICIAN_MENTION", "clinician_access", [
        "I have not been able to get an appointment for months",
        "I am between clinicians at the moment",
    ], [
        "لم أتمكن من الحصول على موعد منذ شهور",
        "أنا بين طبيبين في الوقت الحالي",
    ]),
    ("QUESTION", "asking_about_data", [
        "What do you do with what I write here",
        "Who else can see this",
    ], [
        "ماذا تفعل بما أكتبه هنا",
        "من غيرك يستطيع رؤية هذا",
    ]),
    ("QUESTION", "asking_about_limits", [
        "Can you tell me what is wrong with me",
        "Are you able to talk to my doctor",
    ], [
        "هل يمكنك أن تخبرني ما بي",
        "هل تستطيع التحدث إلى طبيبي",
    ]),
    ("GENERAL_CONVERSATION", "closing", [
        "That is all for now",
        "I will come back to this later",
    ], [
        "هذا كل شيء الآن",
        "سأعود إلى هذا لاحقًا",
    ]),
    ("GENERAL_CONVERSATION", "acknowledgement", [
        "Okay, that makes sense",
        "Understood, I will try that",
    ], [
        "حسنًا، هذا منطقي",
        "فهمت، سأجرب ذلك",
    ]),
]

# Three families per intent - all this file originally held - forces a grouped
# split to put one family in train, one in val and one in test. The head then
# trains on one way of saying a thing and is tested on a different meaning, which
# scored F1 0.00 on twelve of the seventeen intents. `intent_seed_families.py`
# raises every intent to seventeen families so train keeps thirteen or fourteen.
SEEDS = BASE_SEEDS + MORE_SEEDS

# Safety-bearing intents are deliberately NOT seeded here. Routing a message to
# the safety workflow is the job of lumina/safety.py, which fuses a deterministic
# lexicon with a calibrated head. Teaching a small intent model on a handful of
# authored crisis sentences would create a second, weaker, uncalibrated path to
# the most consequential decision in the system.
EXCLUDED_INTENTS = ("SAFETY", "CRISIS", "UNKNOWN")
EXCLUSION_REASON = (
    "Safety routing is owned by lumina/safety.py (deterministic rules + a "
    "calibrated distress head). A second uncalibrated path to a crisis decision "
    "would be a liability, not a feature.")
