"""Additional authored meaning-families for the Lumina intent vocabulary.

Why this file exists: the original seed set in `intent_seed.py` carried three
meaning-families per intent. Splitting by family - which is the right policy, see
that file's header - then left exactly one family per intent in each of train,
val and test. A model that has seen one way of saying "I cannot focus" is asked
at test time about a different meaning entirely, so twelve of the seventeen
intents scored F1 0.00 and the head was useless as a router.

The fix is more meanings, not a looser split. These families raise every intent
to seventeen, so a grouped 80/10/10 split leaves thirteen or fourteen families in
train and still never lets a phrasing cross a split boundary.

Written in two passes. After the first (ten families per intent) a learning curve
over family subsets was still climbing steeply - 312 rows scored 0.250 on
validation, 687 scored 0.391, 1040 scored 0.442, 1356 scored 0.505 - while
swapping the optimiser for a converged solver, ensembling over seeds and widening
the feature space each moved it by nothing or made it worse. Only meanings moved
the number, so a second pass was written.

Same caveats as `intent_seed.py`, undiminished: this is authored text, not
patient language. Arabic is mostly MSA with some Gulf, Egyptian and Levantine
phrasings; none of it was reviewed by a native clinical reviewer.

Format matches `intent_seed.SEEDS`: (intent, family, english, arabic).
"""
from __future__ import annotations

MORE_SEEDS = [
    # --- CHECK_IN ------------------------------------------------------
    ("CHECK_IN", "morning_report", [
        "Morning check-in: woke up at seven, feeling steady",
        "Starting the day, I would say I am about average",
        "Logging my morning before I get going",
    ], [
        "تسجيل الصباح: استيقظت السابعة وأنا مستقر",
        "أبدأ يومي، أقول إنني في المعدل تقريبًا",
        "أسجّل صباحي قبل أن أنطلق",
    ]),
    ("CHECK_IN", "weekly_report", [
        "Here is my week in short: two bad days, the rest fine",
        "Wrapping up the week for you",
        "Weekly check-in, overall a bit better than last week",
    ], [
        "أسبوعي باختصار: يومان سيئان والبقية جيدة",
        "أختم الأسبوع معك",
        "تسجيل أسبوعي، عمومًا أفضل قليلًا من الأسبوع الماضي",
    ]),
    ("CHECK_IN", "numbers_only", [
        "Sleep six, energy four, stress seven",
        "Mood three out of ten today",
        "Focus two, everything else about a five",
    ], [
        "نوم ست ساعات، طاقة أربعة، توتر سبعة",
        "مزاجي ثلاثة من عشرة اليوم",
        "تركيزي اثنان وبقية الأمور حوالي خمسة",
    ]),
    ("CHECK_IN", "bad_day_report", [
        "Today was a rough one, putting it on the record",
        "Recording a bad day, nothing catastrophic",
        "Check-in: today did not go well at all",
    ], [
        "كان اليوم قاسيًا، أسجّله للعلم",
        "أدوّن يومًا سيئًا، لا شيء كارثي",
        "تسجيل: اليوم لم يمضِ جيدًا أبدًا",
    ]),
    ("CHECK_IN", "good_day_report", [
        "Check-in: a good day for once, wanted to log it",
        "Today actually went well, noting it down",
        "Reporting a solid day, first in a while",
    ], [
        "تسجيل: يوم جيد لمرة واحدة، أردت تدوينه",
        "اليوم مضى جيدًا فعلًا، أسجّله",
        "أسجّل يومًا متينًا، الأول منذ فترة",
    ]),
    ("CHECK_IN", "missed_days", [
        "I have not checked in for a few days, catching up now",
        "Sorry for the gap, here is where I am at",
        "Missed the last week of check-ins, restarting today",
    ], [
        "لم أسجّل حالتي منذ أيام، ألحق بالمتأخر الآن",
        "أعتذر عن الانقطاع، هذا وضعي الحالي",
        "فوّتت أسبوعًا من التسجيل، أبدأ من جديد اليوم",
    ]),
    ("CHECK_IN", "since_last_time", [
        "Since we last spoke a couple of things shifted",
        "Update since the last check-in: sleep improved, focus did not",
        "Following up on what I told you last time",
    ], [
        "منذ آخر حديث بيننا تغيّر شيئان",
        "تحديث منذ آخر تسجيل: النوم تحسّن والتركيز لا",
        "أتابع ما أخبرتك به آخر مرة",
    ]),

    # --- JOURNAL -------------------------------------------------------
    ("JOURNAL", "want_to_vent_in_writing", [
        "I need to get this out of my head and onto the page",
        "Let me just write, I do not want an answer yet",
        "I want to type it all out without being interrupted",
    ], [
        "أحتاج أن أخرج هذا من رأسي إلى الورق",
        "دعني أكتب فقط، لا أريد ردًا الآن",
        "أريد أن أكتب كل شيء دون أن يقاطعني أحد",
    ]),
    ("JOURNAL", "gratitude_entry", [
        "Writing down three things that went right today",
        "I want to note the good parts of today somewhere",
        "Journal entry: what I was grateful for this week",
    ], [
        "أكتب ثلاثة أشياء مضت بشكل صحيح اليوم",
        "أريد أن أدوّن الجوانب الجيدة من يومي",
        "تدوينة: ما كنت ممتنًا له هذا الأسبوع",
    ]),
    ("JOURNAL", "night_entry", [
        "I woke up at four and want to write while it is fresh",
        "Middle of the night entry, cannot sleep anyway",
        "Writing this down before I forget the night",
    ], [
        "استيقظت الرابعة وأريد الكتابة وهو طازج",
        "تدوينة في منتصف الليل، لا أنام على أي حال",
        "أكتب هذا قبل أن أنسى ليلتي",
    ]),
    ("JOURNAL", "long_entry_warning", [
        "This is going to be a long entry, bear with me",
        "Fair warning, I have a lot to write today",
        "Long one today, a lot happened",
    ], [
        "ستكون هذه تدوينة طويلة، تحمّلني",
        "تنبيه، لديّ الكثير لأكتبه اليوم",
        "تدوينة طويلة اليوم، حدث الكثير",
    ]),
    ("JOURNAL", "continue_previous_entry", [
        "I want to add to what I wrote yesterday",
        "Picking up the entry I left unfinished",
        "Continuing my last journal entry",
    ], [
        "أريد أن أضيف إلى ما كتبته أمس",
        "أكمل التدوينة التي تركتها ناقصة",
        "أتابع تدوينتي الأخيرة",
    ]),
    ("JOURNAL", "entry_about_a_person", [
        "I want to write about the argument with my brother",
        "Journalling about what my mother said to me",
        "Writing about a conversation that is still bothering me",
    ], [
        "أريد أن أكتب عن الخلاف مع أخي",
        "أدوّن ما قالته لي أمي",
        "أكتب عن محادثة ما زالت تزعجني",
    ]),
    ("JOURNAL", "reread_own_entries", [
        "I want to read back what I wrote last month",
        "Can I look through my old entries",
        "Show me what I journalled about last week",
    ], [
        "أريد أن أقرأ ما كتبته الشهر الماضي",
        "هل أستطيع تصفّح تدويناتي القديمة",
        "أرني ما دوّنته الأسبوع الماضي",
    ]),

    # --- EMOTIONAL_SUPPORT --------------------------------------------
    ("EMOTIONAL_SUPPORT", "numb_and_flat", [
        "I do not feel anything at all lately and it scares me",
        "Everything is flat, good and bad look the same",
        "I am just numb, I cannot describe it better than that",
    ], [
        "لا أشعر بأي شيء مؤخرًا وهذا يخيفني",
        "كل شيء مسطح، الجيد والسيئ متشابهان",
        "أنا متبلّد فقط، لا أستطيع وصفه أفضل من ذلك",
    ]),
    ("EMOTIONAL_SUPPORT", "ashamed_of_state", [
        "I feel pathetic for still struggling with this",
        "I am embarrassed that I cannot handle a normal life",
        "Everyone else manages and I cannot, it makes me feel small",
    ], [
        "أشعر أنني تافه لأنني ما زلت أعاني من هذا",
        "أخجل من أنني لا أستطيع تدبير حياة عادية",
        "الجميع يتدبّر وأنا لا، هذا يجعلني أشعر بالصغر",
    ]),
    ("EMOTIONAL_SUPPORT", "crying_without_reason", [
        "I have been crying on and off and I do not know why",
        "I keep tearing up over nothing",
        "I cried twice today for no clear reason",
    ], [
        "أبكي على فترات ولا أعرف السبب",
        "تدمع عيني على لا شيء",
        "بكيت مرتين اليوم بلا سبب واضح",
    ]),
    ("EMOTIONAL_SUPPORT", "grief", [
        "I lost someone and I still cannot talk about it",
        "It has been months since the funeral and it is not easier",
        "I miss them so much it is physical",
    ], [
        "فقدت شخصًا وما زلت لا أستطيع الحديث عنه",
        "مرّت شهور على العزاء ولم يصبح أسهل",
        "أفتقدهم لدرجة أنني أشعر بها في جسدي",
    ]),
    ("EMOTIONAL_SUPPORT", "nobody_understands", [
        "My family thinks I am just lazy and it hurts",
        "Nobody around me gets what this is like",
        "I stopped explaining myself because no one listens",
    ], [
        "عائلتي تظن أنني كسول فقط وهذا يؤلمني",
        "لا أحد حولي يفهم كيف يكون هذا",
        "توقفت عن الشرح لأن لا أحد يستمع",
    ]),
    ("EMOTIONAL_SUPPORT", "just_need_comfort", [
        "Please just say something kind, I have had enough today",
        "I do not need a plan, I need a bit of comfort",
        "Can you be gentle with me right now",
    ], [
        "قل لي شيئًا لطيفًا فقط، تعبت اليوم",
        "لا أحتاج خطة، أحتاج بعض المواساة",
        "كن لطيفًا معي الآن من فضلك",
    ]),
    ("EMOTIONAL_SUPPORT", "anger_at_self", [
        "I am so angry at myself for wasting another day",
        "I hate how I handled everything this week",
        "I keep beating myself up over small mistakes",
    ], [
        "أنا غاضب جدًا من نفسي لأنني أهدرت يومًا آخر",
        "أكره كيف تعاملت مع كل شيء هذا الأسبوع",
        "أستمر في لوم نفسي على أخطاء صغيرة",
    ]),

    # --- TASK_SUPPORT --------------------------------------------------
    ("TASK_SUPPORT", "break_it_down", [
        "Can you break this into smaller steps for me",
        "Split this job into pieces I can actually do",
        "Give me the first step only, the whole thing is too big",
    ], [
        "هل يمكنك تقسيم هذا إلى خطوات أصغر",
        "جزّئ هذه المهمة إلى أجزاء أستطيع فعلها",
        "أعطني الخطوة الأولى فقط، الأمر كله كبير جدًا",
    ]),
    ("TASK_SUPPORT", "deadline_pressure", [
        "The report is due tomorrow and I have not opened it",
        "My deadline is in two days and I am frozen",
        "I have to submit this tonight and I cannot move",
    ], [
        "التقرير مطلوب غدًا ولم أفتحه بعد",
        "موعدي النهائي بعد يومين وأنا متجمد",
        "عليّ تسليم هذا الليلة ولا أستطيع الحركة",
    ]),
    ("TASK_SUPPORT", "switching_between_tasks", [
        "I start one thing, jump to another, and finish neither",
        "I have five tabs of half-done work",
        "I keep swapping tasks every ten minutes",
    ], [
        "أبدأ شيئًا وأنتقل لآخر ولا أنهي أيًا منهما",
        "لديّ خمس نوافذ من عمل نصف منجز",
        "أستمر في تبديل المهام كل عشر دقائق",
    ]),
    ("TASK_SUPPORT", "boring_admin_task", [
        "I have to make one phone call and I have avoided it for a week",
        "There is paperwork I cannot bring myself to touch",
        "It is a fifteen minute errand and I keep skipping it",
    ], [
        "عليّ إجراء مكالمة واحدة وأتجنبها منذ أسبوع",
        "هناك أوراق لا أستطيع أن ألمسها",
        "مهمة خمس عشرة دقيقة وأستمر في تخطيها",
    ]),
    ("TASK_SUPPORT", "finished_a_task", [
        "I actually finished the thing I have been dreading",
        "Got the task done, wanted to tell someone",
        "I cleared two items off the list today",
    ], [
        "أنجزت فعلًا الشيء الذي كنت أخشاه",
        "أنهيت المهمة، أردت أن أخبر أحدًا",
        "أزلت بندين من القائمة اليوم",
    ]),
    ("TASK_SUPPORT", "what_first", [
        "Tell me which one to do first, I cannot choose",
        "Help me put these in order",
        "Which of these actually matters today",
    ], [
        "أخبرني بأي واحدة أبدأ، لا أستطيع الاختيار",
        "ساعدني في ترتيب هذه",
        "أي من هذه يهم فعلًا اليوم",
    ]),
    ("TASK_SUPPORT", "task_too_hard_today", [
        "I do not have it in me for the big task today, what else",
        "Give me something small, my capacity is gone",
        "Can we lower the bar for today",
    ], [
        "لا أملك طاقة المهمة الكبيرة اليوم، ما البديل",
        "أعطني شيئًا صغيرًا، طاقتي انتهت",
        "هل نخفّض السقف اليوم",
    ]),

    # --- SLEEP ---------------------------------------------------------
    ("SLEEP", "oversleeping", [
        "I slept eleven hours and still feel wrecked",
        "I cannot get out of bed before noon anymore",
        "I am sleeping far too much lately",
    ], [
        "نمت إحدى عشرة ساعة وما زلت محطمًا",
        "لم أعد أستطيع النهوض قبل الظهر",
        "أنام أكثر من اللازم بكثير مؤخرًا",
    ]),
    ("SLEEP", "racing_thoughts_at_night", [
        "My head will not switch off when I lie down",
        "The moment the lights go off my brain starts",
        "I lie there replaying the whole day",
    ], [
        "رأسي لا ينطفئ عندما أستلقي",
        "لحظة إطفاء الضوء يبدأ عقلي",
        "أستلقي وأعيد شريط اليوم كله",
    ]),
    ("SLEEP", "nightmares", [
        "I keep having bad dreams and waking up sweating",
        "Nightmares three nights running now",
        "I am scared to sleep because of the dreams",
    ], [
        "تتكرر أحلامي السيئة وأستيقظ متعرّقًا",
        "كوابيس ثلاث ليال متتالية الآن",
        "أخاف النوم بسبب الأحلام",
    ]),
    ("SLEEP", "sleep_and_medication", [
        "Since the dose changed my sleep is different",
        "I think the evening tablet is keeping me up",
        "My sleep shifted right after the new prescription",
    ], [
        "منذ تغيّرت الجرعة صار نومي مختلفًا",
        "أظن أن حبة المساء تبقيني مستيقظًا",
        "تغيّر نومي بعد الوصفة الجديدة تمامًا",
    ]),
    ("SLEEP", "sleep_advice_request", [
        "What can I actually do to sleep better tonight",
        "Give me something to try before bed",
        "Any suggestion for winding down at night",
    ], [
        "ما الذي أستطيع فعله فعلًا لأنام أفضل الليلة",
        "أعطني شيئًا أجربه قبل النوم",
        "أي اقتراح للتهدئة في الليل",
    ]),
    ("SLEEP", "napping", [
        "I keep falling asleep in the afternoon and then cannot sleep at night",
        "My naps are getting longer than my nights",
        "I passed out on the sofa again at six",
    ], [
        "أنام بعد الظهر ثم لا أنام في الليل",
        "غفواتي أطول من ليلي",
        "غبت عن الوعي على الأريكة مجددًا في السادسة",
    ]),
    ("SLEEP", "sleep_stable_now", [
        "Four nights in a row of decent sleep now",
        "My sleep has settled into something normal",
        "I have been going to bed at the same time and it is working",
    ], [
        "أربع ليال متتالية من نوم معقول الآن",
        "استقر نومي على شيء طبيعي",
        "أنام في الوقت نفسه وهذا ينجح",
    ]),

    # --- ENERGY --------------------------------------------------------
    ("ENERGY", "energy_crash_midday", [
        "I am fine until two in the afternoon and then I collapse",
        "My energy falls off a cliff after lunch",
        "I get about four good hours a day and that is it",
    ], [
        "أكون بخير حتى الثانية بعد الظهر ثم أنهار",
        "طاقتي تسقط من جرف بعد الغداء",
        "أحصل على أربع ساعات جيدة في اليوم وهذا كل شيء",
    ]),
    ("ENERGY", "physically_heavy", [
        "My body feels like it weighs twice as much",
        "Even a shower takes everything I have",
        "Walking to the kitchen feels like a chore",
    ], [
        "جسدي يبدو وكأن وزنه تضاعف",
        "حتى الاستحمام يأخذ كل ما لديّ",
        "المشي إلى المطبخ يبدو عملًا شاقًا",
    ]),
    ("ENERGY", "too_much_energy", [
        "I have not stopped moving since five this morning",
        "I feel wired, I cannot sit down for a minute",
        "There is too much energy in me and nowhere for it to go",
    ], [
        "لم أتوقف عن الحركة منذ الخامسة صباحًا",
        "أشعر بشدّ عصبي، لا أستطيع الجلوس دقيقة",
        "لديّ طاقة فائضة ولا مكان تذهب إليه",
    ]),
    ("ENERGY", "energy_returning", [
        "I had enough energy to cook today, that is new",
        "Something came back this week, I can do things again",
        "I managed a full day without needing to lie down",
    ], [
        "كانت لديّ طاقة للطبخ اليوم، وهذا جديد",
        "عاد شيء هذا الأسبوع، أستطيع فعل أشياء مجددًا",
        "تدبّرت يومًا كاملًا دون حاجة للاستلقاء",
    ]),
    ("ENERGY", "energy_unpredictable", [
        "One day I can do everything, the next I can do nothing",
        "My energy swings with no pattern I can see",
        "I never know which version of me wakes up",
    ], [
        "يومًا أستطيع كل شيء ويومًا لا أستطيع شيئًا",
        "طاقتي تتأرجح بلا نمط أراه",
        "لا أعرف أبدًا أي نسخة مني ستستيقظ",
    ]),
    ("ENERGY", "caffeine_and_energy", [
        "I am on four coffees a day just to function",
        "The coffee stopped working on me",
        "I need energy drinks to get through the morning",
    ], [
        "أشرب أربع قهوات يوميًا فقط لأعمل",
        "لم تعد القهوة تؤثر فيّ",
        "أحتاج مشروبات الطاقة لأجتاز الصباح",
    ]),
    ("ENERGY", "tired_but_not_sleepy", [
        "I am exhausted and wide awake at the same time",
        "Tired to the bone but my eyes will not close",
        "Drained all day, alert all night",
    ], [
        "أنا منهك ومستيقظ تمامًا في الوقت نفسه",
        "متعب حتى العظم لكن عينيّ لا تُغمضان",
        "منهك طول النهار ومتنبّه طول الليل",
    ]),

    # --- FOCUS ---------------------------------------------------------
    ("FOCUS", "cannot_read", [
        "I read the same paragraph four times and took in nothing",
        "My eyes go over the words but nothing sticks",
        "I cannot follow a page of text anymore",
    ], [
        "قرأت المقطع نفسه أربع مرات ولم أستوعب شيئًا",
        "عيناي تمرّان على الكلمات ولا يثبت شيء",
        "لم أعد أستطيع متابعة صفحة نص",
    ]),
    ("FOCUS", "losing_track_mid_sentence", [
        "I forget what I was saying halfway through a sentence",
        "I lose the thread in the middle of my own thought",
        "I walk into a room and cannot remember why",
    ], [
        "أنسى ما كنت أقوله في منتصف الجملة",
        "أفقد الخيط في وسط فكرتي",
        "أدخل غرفة ولا أتذكر لماذا",
    ]),
    ("FOCUS", "phone_distraction", [
        "I pick up my phone every two minutes without deciding to",
        "I lost an hour scrolling again",
        "The phone is eating my whole day",
    ], [
        "أرفع هاتفي كل دقيقتين دون أن أقرر",
        "أهدرت ساعة في التمرير مجددًا",
        "الهاتف يأكل يومي كله",
    ]),
    ("FOCUS", "noise_sensitivity", [
        "Any sound in the room and my concentration is gone",
        "I cannot work unless it is completely silent",
        "The neighbours' noise makes focusing impossible",
    ], [
        "أي صوت في الغرفة ويذهب تركيزي",
        "لا أستطيع العمل إلا في صمت تام",
        "ضجيج الجيران يجعل التركيز مستحيلًا",
    ]),
    ("FOCUS", "hyperfocus", [
        "I worked six hours straight and forgot to eat",
        "I got locked into one thing and ignored everything else",
        "I only looked up when it got dark",
    ], [
        "عملت ست ساعات متواصلة ونسيت أن آكل",
        "انغلقت على شيء واحد وتجاهلت كل شيء آخر",
        "لم أرفع رأسي إلا عندما حلّ الظلام",
    ]),
    ("FOCUS", "focus_help_request", [
        "What can I do to concentrate for even twenty minutes",
        "Give me a way to hold my attention on one thing",
        "Any trick for staying on task",
    ], [
        "ما الذي أفعله لأركز لعشرين دقيقة على الأقل",
        "أعطني طريقة لأثبّت انتباهي على شيء واحد",
        "أي حيلة للبقاء على المهمة",
    ]),
    ("FOCUS", "focus_worse_than_before", [
        "My concentration is much worse than it was a year ago",
        "I used to be able to study for hours, now I cannot",
        "This is a real decline, not a bad day",
    ], [
        "تركيزي أسوأ بكثير مما كان قبل سنة",
        "كنت أستطيع الدراسة ساعات، الآن لا أستطيع",
        "هذا تراجع حقيقي وليس يومًا سيئًا",
    ]),

    # --- ROUTINE -------------------------------------------------------
    ("ROUTINE", "no_structure_at_all", [
        "My days have no shape, I just drift through them",
        "I have stopped having any kind of schedule",
        "Every day is different and none of them work",
    ], [
        "أيامي بلا شكل، أنا أطفو فيها فقط",
        "توقفت عن أن يكون لي أي جدول",
        "كل يوم مختلف ولا واحد منها ينجح",
    ]),
    ("ROUTINE", "meals_irregular", [
        "I eat once a day at random times",
        "I forget meals entirely and then eat at midnight",
        "My eating has no rhythm at all",
    ], [
        "آكل مرة واحدة في اليوم في أوقات عشوائية",
        "أنسى الوجبات تمامًا ثم آكل في منتصف الليل",
        "طعامي بلا إيقاع أبدًا",
    ]),
    ("ROUTINE", "want_to_build_a_routine", [
        "Help me set up something I can follow every day",
        "I want a simple structure for my mornings",
        "Can we build a routine I will actually keep",
    ], [
        "ساعدني في وضع شيء أتبعه كل يوم",
        "أريد نظامًا بسيطًا لصباحي",
        "هل نبني روتينًا سألتزم به فعلًا",
    ]),
    ("ROUTINE", "weekend_breaks_routine", [
        "I hold it together on weekdays and it falls apart on the weekend",
        "Two days off and my whole schedule is gone",
        "Holidays undo everything I have built",
    ], [
        "أتماسك في أيام العمل وينهار كل شيء في نهاية الأسبوع",
        "يومان إجازة ويذهب جدولي كله",
        "الإجازات تنقض كل ما بنيته",
    ]),
    ("ROUTINE", "routine_too_rigid", [
        "If one thing moves in my day I cannot cope",
        "My routine is so tight that any change ruins it",
        "I get upset when the plan shifts even slightly",
    ], [
        "إذا تحرك شيء واحد في يومي لا أستطيع التعامل",
        "روتيني محكم لدرجة أن أي تغيير يفسده",
        "أنزعج عندما تتغير الخطة قليلًا",
    ]),
    ("ROUTINE", "day_night_flipped", [
        "I am awake all night and asleep all day now",
        "My clock has completely flipped around",
        "I live at night and sleep through the daylight",
    ], [
        "أنا مستيقظ كل الليل ونائم كل النهار الآن",
        "انقلبت ساعتي تمامًا",
        "أعيش في الليل وأنام في النهار",
    ]),
    ("ROUTINE", "routine_rebuilding", [
        "I got two days of the new schedule right this week",
        "Slowly getting the structure back in place",
        "The routine is starting to stick again",
    ], [
        "نجحت في يومين من الجدول الجديد هذا الأسبوع",
        "أعيد النظام إلى مكانه ببطء",
        "بدأ الروتين يثبت مجددًا",
    ]),

    # --- STRESS --------------------------------------------------------
    ("STRESS", "money_worry", [
        "I cannot stop thinking about the bills",
        "The money situation is keeping me up at night",
        "Every time I check my account my chest tightens",
    ], [
        "لا أستطيع التوقف عن التفكير في الفواتير",
        "وضع المال يبقيني مستيقظًا في الليل",
        "كل مرة أتحقق من حسابي يضيق صدري",
    ]),
    ("STRESS", "work_pressure", [
        "My manager keeps adding to my plate and I cannot say no",
        "Work has been relentless for three weeks",
        "The pressure at the office is more than I can hold",
    ], [
        "مديري يزيد على ما لديّ ولا أستطيع الرفض",
        "العمل لا يرحم منذ ثلاثة أسابيع",
        "الضغط في المكتب أكثر مما أحتمل",
    ]),
    ("STRESS", "physical_symptoms_of_stress", [
        "My jaw aches from clenching and my shoulders are solid",
        "I have had a headache for four days from the tension",
        "My stomach is in knots the whole time",
    ], [
        "فكي يؤلمني من الشد وكتفاي متحجران",
        "لديّ صداع منذ أربعة أيام من التوتر",
        "معدتي معقودة طول الوقت",
    ]),
    ("STRESS", "family_conflict", [
        "The situation at home is wearing me down",
        "The arguments with my family never stop",
        "I dread going home at the end of the day",
    ], [
        "الوضع في البيت يستنزفني",
        "الخلافات مع عائلتي لا تتوقف",
        "أخشى العودة إلى البيت في آخر اليوم",
    ]),
    ("STRESS", "anticipating_an_event", [
        "I have an exam on Sunday and I cannot think about anything else",
        "The interview is tomorrow and I feel sick",
        "There is a meeting next week that I am dreading",
    ], [
        "لديّ امتحان الأحد ولا أستطيع التفكير في غيره",
        "المقابلة غدًا وأشعر بالغثيان",
        "هناك اجتماع الأسبوع القادم أخشاه",
    ]),
    ("STRESS", "need_to_calm_down_now", [
        "I need to bring this down a notch right now",
        "Help me get out of this spiral",
        "Talk me down, I am wound too tight",
    ], [
        "أحتاج أن أخفّض هذا درجة الآن",
        "ساعدني في الخروج من هذه الدوامة",
        "هدّئني، أنا مشدود أكثر من اللازم",
    ]),
    ("STRESS", "stress_after_it_passed", [
        "The deadline passed and I still cannot relax",
        "It is over but my body has not caught up",
        "The pressure is gone and I am still tense",
    ], [
        "مرّ الموعد النهائي وما زلت لا أستطيع الاسترخاء",
        "انتهى الأمر لكن جسدي لم يلحق",
        "ذهب الضغط وما زلت متوترًا",
    ]),

    # --- SOCIAL --------------------------------------------------------
    ("SOCIAL", "withdrawing", [
        "I have not left the house or spoken to anyone in four days",
        "I keep turning down invitations",
        "I have gone quiet on everyone",
    ], [
        "لم أخرج من البيت ولم أتحدث مع أحد منذ أربعة أيام",
        "أستمر في رفض الدعوات",
        "صمتُّ عن الجميع",
    ]),
    ("SOCIAL", "unanswered_messages", [
        "I have thirty messages I have not replied to",
        "My phone is full of people waiting on me",
        "I read the messages and cannot bring myself to answer",
    ], [
        "لديّ ثلاثون رسالة لم أرد عليها",
        "هاتفي مليء بأناس ينتظرون ردي",
        "أقرأ الرسائل ولا أستطيع أن أرد",
    ]),
    ("SOCIAL", "friend_let_me_down", [
        "My friend cancelled on me again and it stung",
        "I feel like I am the only one making an effort",
        "They forgot about me and I noticed",
    ], [
        "ألغى صديقي موعدنا مجددًا وقد أوجعني ذلك",
        "أشعر أنني الوحيد الذي يبذل جهدًا",
        "نسوني ولاحظت ذلك",
    ]),
    ("SOCIAL", "good_social_contact", [
        "I met a friend for coffee and it actually helped",
        "I called my sister and we talked for an hour",
        "I said yes to something social and I am glad I did",
    ], [
        "قابلت صديقًا على قهوة وقد ساعدني ذلك فعلًا",
        "اتصلت بأختي وتحدثنا ساعة",
        "قلت نعم لشيء اجتماعي وأنا سعيد بذلك",
    ]),
    ("SOCIAL", "social_anxiety_at_events", [
        "I stood in the corner the whole evening and left early",
        "Being in a group of people exhausts me",
        "I want to see people but I panic when I get there",
    ], [
        "بقيت في الزاوية طول المسائية وخرجت مبكرًا",
        "التواجد في مجموعة من الناس يستنزفني",
        "أريد رؤية الناس لكنني أُصاب بالهلع عندما أصل",
    ]),
    ("SOCIAL", "new_connection", [
        "I actually made a new friend at the course",
        "Someone at work reached out and it felt good",
        "I joined a group and went twice now",
    ], [
        "كوّنت صديقًا جديدًا في الدورة فعلًا",
        "تواصل معي أحد في العمل وكان شعورًا جيدًا",
        "انضممت إلى مجموعة وذهبت مرتين الآن",
    ]),
    ("SOCIAL", "wants_to_reconnect", [
        "I want to reach out to someone but I do not know how to start",
        "How do I message a friend after months of silence",
        "I owe people an explanation for disappearing",
    ], [
        "أريد التواصل مع أحد لكنني لا أعرف كيف أبدأ",
        "كيف أراسل صديقًا بعد شهور من الصمت",
        "أنا مدين للناس بتفسير لاختفائي",
    ]),

    # --- EXERCISE ------------------------------------------------------
    ("EXERCISE", "walk_done", [
        "I got out for a twenty minute walk today",
        "Managed a lap around the block, first in weeks",
        "I walked to the shop instead of driving",
    ], [
        "خرجت لمشي عشرين دقيقة اليوم",
        "تدبّرت لفة حول الحي، الأولى منذ أسابيع",
        "مشيت إلى الدكان بدل أن أقود",
    ]),
    ("EXERCISE", "stopped_going_to_gym", [
        "I have not been to the gym in two months",
        "I cancelled the membership because I never went",
        "My training just stopped and I cannot restart",
    ], [
        "لم أذهب إلى النادي منذ شهرين",
        "ألغيت العضوية لأنني لم أذهب أبدًا",
        "توقف تدريبي فقط ولا أستطيع البدء مجددًا",
    ]),
    ("EXERCISE", "movement_helped_mood", [
        "I felt noticeably better after moving today",
        "The walk cleared my head more than I expected",
        "Exercise is the only thing that shifts my mood",
    ], [
        "شعرت بتحسن واضح بعد الحركة اليوم",
        "صفّى المشي رأسي أكثر مما توقعت",
        "الرياضة الشيء الوحيد الذي يحرّك مزاجي",
    ]),
    ("EXERCISE", "overdoing_it", [
        "I trained twice a day all week and now I am broken",
        "I pushed too hard and cannot move today",
        "I went from nothing to two hours a day",
    ], [
        "تدربت مرتين يوميًا طول الأسبوع والآن أنا محطم",
        "دفعت نفسي بقوة ولا أستطيع الحركة اليوم",
        "انتقلت من لا شيء إلى ساعتين يوميًا",
    ]),
    ("EXERCISE", "wants_easy_start", [
        "What is the smallest bit of movement that counts",
        "Give me something I can do without leaving the flat",
        "I need an exercise that does not need energy I do not have",
    ], [
        "ما أصغر قدر من الحركة يُحتسب",
        "أعطني شيئًا أفعله دون مغادرة البيت",
        "أحتاج تمرينًا لا يحتاج طاقة لا أملكها",
    ]),
    ("EXERCISE", "weather_or_place_blocks", [
        "It has been too hot to go outside all week",
        "There is nowhere near me that is safe to walk",
        "The weather has stopped me every day",
    ], [
        "الجو حار جدًا للخروج طول الأسبوع",
        "لا يوجد قريب مني مكان آمن للمشي",
        "الطقس أوقفني كل يوم",
    ]),
    ("EXERCISE", "exercise_routine_holding", [
        "Three walks a week for a month now",
        "I have kept the stretching going every morning",
        "The habit has actually held this time",
    ], [
        "ثلاث مرات مشي أسبوعيًا منذ شهر الآن",
        "حافظت على تمارين الإطالة كل صباح",
        "ثبتت العادة فعلًا هذه المرة",
    ]),

    # --- GOAL ----------------------------------------------------------
    ("GOAL", "goal_too_ambitious", [
        "I set the target too high again and gave up",
        "I keep promising myself the whole thing and doing none of it",
        "My goal was unrealistic from the start",
    ], [
        "وضعت الهدف عاليًا مجددًا واستسلمت",
        "أستمر في وعد نفسي بكل شيء ولا أفعل شيئًا",
        "كان هدفي غير واقعي من البداية",
    ]),
    ("GOAL", "goal_achieved", [
        "I hit the target I set two weeks ago",
        "I did the thing I said I would do",
        "The goal is done, what next",
    ], [
        "وصلت إلى الهدف الذي وضعته قبل أسبوعين",
        "فعلت الشيء الذي قلت إنني سأفعله",
        "انتهى الهدف، ما التالي",
    ]),
    ("GOAL", "drop_a_goal", [
        "I want to drop the reading goal, it is not working",
        "Take that target off the list",
        "I am letting go of the one about mornings",
    ], [
        "أريد إسقاط هدف القراءة، إنه لا ينجح",
        "أزل ذلك الهدف من القائمة",
        "أتخلى عن الهدف المتعلق بالصباح",
    ]),
    ("GOAL", "long_term_goal", [
        "I want to get back to studying within the year",
        "My aim is to be working again by spring",
        "Long term I want to move out on my own",
    ], [
        "أريد العودة إلى الدراسة خلال هذه السنة",
        "هدفي أن أعود للعمل بحلول الربيع",
        "على المدى الطويل أريد أن أستقل بمسكني",
    ]),
    ("GOAL", "smaller_goal_request", [
        "Make the goal smaller so I can actually reach it",
        "Can we scale this target down",
        "Give me a version of this I will not fail",
    ], [
        "اجعل الهدف أصغر لأستطيع الوصول إليه",
        "هل نخفّض هذا الهدف",
        "أعطني نسخة منه لا أفشل فيها",
    ]),
    ("GOAL", "goal_stalled", [
        "I am stuck halfway to the goal and not moving",
        "Two weeks of no progress on my target",
        "The goal has stalled and I do not know why",
    ], [
        "أنا متوقف في نصف الطريق إلى الهدف ولا أتحرك",
        "أسبوعان بلا تقدم على هدفي",
        "تعطل الهدف ولا أعرف السبب",
    ]),
    ("GOAL", "habit_streak", [
        "Nine days in a row on the new habit",
        "I have kept the streak going all week",
        "The habit is on day twelve now",
    ], [
        "تسعة أيام متتالية على العادة الجديدة",
        "حافظت على التتابع طول الأسبوع",
        "العادة في يومها الثاني عشر الآن",
    ]),

    # --- PROGRESS ------------------------------------------------------
    ("PROGRESS", "am_i_getting_better", [
        "Honestly, am I getting any better or not",
        "Tell me straight whether this is working",
        "Is there any improvement in what you have on me",
    ], [
        "بصراحة، هل أتحسن أم لا",
        "أخبرني صراحة إن كان هذا ينجح",
        "هل هناك أي تحسن في ما لديك عني",
    ]),
    ("PROGRESS", "trend_over_time", [
        "What does the last month look like overall",
        "Show me the direction things have been going",
        "Give me the trend, not today's number",
    ], [
        "كيف يبدو الشهر الماضي عمومًا",
        "أرني الاتجاه الذي تسير فيه الأمور",
        "أعطني الاتجاه لا رقم اليوم",
    ]),
    ("PROGRESS", "what_helped", [
        "Which of the things we tried actually helped",
        "What made the difference in the good weeks",
        "Tell me what was working when I was doing better",
    ], [
        "أي من الأشياء التي جربناها ساعدت فعلًا",
        "ما الذي أحدث الفرق في الأسابيع الجيدة",
        "أخبرني ما كان ينجح عندما كنت أفضل",
    ]),
    ("PROGRESS", "noticed_own_pattern", [
        "I notice my bad days always follow a short night",
        "It seems to get worse every time my routine slips",
        "I think there is a pattern around the weekends",
    ], [
        "ألاحظ أن أيامي السيئة تتبع دائمًا ليلة قصيرة",
        "يبدو أنه يسوء كلما انزلق روتيني",
        "أظن أن هناك نمطًا حول نهايات الأسبوع",
    ]),
    ("PROGRESS", "setback_after_progress", [
        "I was doing well for a month and it all slipped back",
        "I lost the ground I gained",
        "Two good weeks and then straight back down",
    ], [
        "كنت بحال جيدة لشهر ثم انزلق كل شيء",
        "فقدت الأرض التي كسبتها",
        "أسبوعان جيدان ثم هبوط مباشر",
    ]),
    ("PROGRESS", "summary_for_clinician", [
        "I need a summary of the last six weeks for my appointment",
        "Can you put together what I should tell my doctor",
        "Prepare something I can show at the clinic",
    ], [
        "أحتاج ملخصًا للأسابيع الستة الماضية لموعدي",
        "هل تجمع ما يجب أن أخبر به طبيبي",
        "جهّز شيئًا أعرضه في العيادة",
    ]),
    ("PROGRESS", "compare_to_before", [
        "How does this week compare to the same week last month",
        "Am I better off than when I started",
        "Compare now with where I was in the winter",
    ], [
        "كيف يقارن هذا الأسبوع بالأسبوع نفسه الشهر الماضي",
        "هل أنا أفضل من حين بدأت",
        "قارن الآن بما كنت عليه في الشتاء",
    ]),

    # --- MEDICATION_MENTION --------------------------------------------
    ("MEDICATION_MENTION", "missed_doses", [
        "I forgot my tablets three times this week",
        "I keep missing the evening dose",
        "I have not taken them since Tuesday",
    ], [
        "نسيت حبوبي ثلاث مرات هذا الأسبوع",
        "أستمر في تفويت جرعة المساء",
        "لم آخذها منذ الثلاثاء",
    ]),
    ("MEDICATION_MENTION", "side_effects", [
        "The new tablet is making my mouth dry all day",
        "I have put on weight since the dose went up",
        "It makes me groggy until midday",
    ], [
        "الحبة الجديدة تجعل فمي جافًا طول النهار",
        "زاد وزني منذ رُفعت الجرعة",
        "تجعلني مترنحًا حتى الظهر",
    ]),
    ("MEDICATION_MENTION", "wants_to_stop", [
        "I am thinking about coming off them",
        "I want to stop taking this, I feel worse on it",
        "Part of me wants to just quit the medication",
    ], [
        "أفكر في التوقف عنها",
        "أريد التوقف عن هذا، أشعر أسوأ عليه",
        "جزء مني يريد ترك الدواء فقط",
    ]),
    ("MEDICATION_MENTION", "dose_changed", [
        "The doctor raised my dose last week",
        "They switched me to a different tablet on Monday",
        "My prescription changed and I am still adjusting",
    ], [
        "رفع الطبيب جرعتي الأسبوع الماضي",
        "نقلوني إلى حبة مختلفة يوم الاثنين",
        "تغيّرت وصفتي وما زلت أتأقلم",
    ]),
    ("MEDICATION_MENTION", "medication_helping", [
        "I think the tablets are starting to help",
        "Things have been steadier since the change",
        "The medication seems to be doing something",
    ], [
        "أظن أن الحبوب بدأت تساعد",
        "الأمور أكثر استقرارًا منذ التغيير",
        "يبدو أن الدواء يفعل شيئًا",
    ]),
    ("MEDICATION_MENTION", "prescription_logistics", [
        "I ran out and the pharmacy has none in",
        "My repeat prescription has not come through",
        "I need to reorder before the weekend",
    ], [
        "انتهت عندي والصيدلية ليس لديها",
        "لم تصل وصفتي المتكررة",
        "أحتاج أن أعيد الطلب قبل نهاية الأسبوع",
    ]),
    ("MEDICATION_MENTION", "medication_timing", [
        "Is it better to take it in the morning or at night",
        "I have been taking it with food, is that right",
        "I moved it to bedtime and I am not sure that was wise",
    ], [
        "هل الأفضل أخذها صباحًا أم ليلًا",
        "آخذها مع الطعام، هل هذا صحيح",
        "نقلتها إلى وقت النوم ولست متأكدًا أن ذلك حكيم",
    ]),

    # --- CLINICIAN_MENTION ---------------------------------------------
    ("CLINICIAN_MENTION", "appointment_upcoming", [
        "I see my psychiatrist on Thursday",
        "My next session is in two weeks",
        "I have a clinic appointment tomorrow morning",
    ], [
        "أرى طبيبي النفسي الخميس",
        "جلستي القادمة بعد أسبوعين",
        "لديّ موعد في العيادة صباح الغد",
    ]),
    ("CLINICIAN_MENTION", "appointment_went_badly", [
        "The session yesterday left me worse than I went in",
        "I do not think my therapist listened to me",
        "The appointment was ten minutes and felt pointless",
    ], [
        "تركتني جلسة الأمس أسوأ مما دخلت",
        "لا أظن أن معالجي استمع إليّ",
        "كان الموعد عشر دقائق وبدا بلا معنى",
    ]),
    ("CLINICIAN_MENTION", "what_to_say_to_clinician", [
        "I never know what to say in the session",
        "How do I tell my doctor it is not working",
        "Help me put into words what I want to raise",
    ], [
        "لا أعرف أبدًا ماذا أقول في الجلسة",
        "كيف أخبر طبيبي أن هذا لا ينجح",
        "ساعدني في صياغة ما أريد طرحه",
    ]),
    ("CLINICIAN_MENTION", "cannot_get_appointment", [
        "The waiting list is eight months long",
        "I have been trying to get seen since March",
        "There is nobody taking new patients near me",
    ], [
        "قائمة الانتظار ثمانية أشهر",
        "أحاول الحصول على موعد منذ مارس",
        "لا أحد يقبل مرضى جددًا قريبًا مني",
    ]),
    ("CLINICIAN_MENTION", "clinician_advice_recall", [
        "My therapist told me to track my sleep, so I am",
        "The doctor said to keep the routine steady",
        "They asked me to note when it gets worse",
    ], [
        "قال لي معالجي أن أتابع نومي، وأنا أفعل",
        "قال الطبيب أن أحافظ على ثبات الروتين",
        "طلبوا مني أن أدوّن متى يسوء",
    ]),
    ("CLINICIAN_MENTION", "changing_clinician", [
        "I am thinking of asking for a different therapist",
        "I want to switch to someone else",
        "My psychiatrist is leaving and I am being reassigned",
    ], [
        "أفكر في طلب معالج مختلف",
        "أريد الانتقال إلى شخص آخر",
        "طبيبي النفسي يترك العمل وسيُحوّلني إلى غيره",
    ]),
    ("CLINICIAN_MENTION", "trusts_clinician", [
        "My therapist is the one person who gets it",
        "The sessions are the most useful hour of my month",
        "I actually trust my doctor with this",
    ], [
        "معالجي هو الشخص الوحيد الذي يفهم",
        "الجلسات أنفع ساعة في شهري",
        "أنا أثق بطبيبي في هذا فعلًا",
    ]),

    # --- QUESTION ------------------------------------------------------
    ("QUESTION", "asking_how_it_works", [
        "How do you decide what to say to me",
        "Where do your answers come from",
        "Are you following a script or working it out",
    ], [
        "كيف تقرر ما تقوله لي",
        "من أين تأتي أجوبتك",
        "هل تتبع نصًا أم تحسبه",
    ]),
    ("QUESTION", "asking_about_memory", [
        "Do you remember what I told you yesterday",
        "How long do you keep what I say",
        "Will you still know this next week",
    ], [
        "هل تتذكر ما أخبرتك به أمس",
        "إلى متى تحفظ ما أقوله",
        "هل ستعرف هذا الأسبوع القادم",
    ]),
    ("QUESTION", "asking_about_sharing", [
        "Does my doctor see what I write here",
        "Is any of this sent to my family",
        "Who gets a copy of these conversations",
    ], [
        "هل يرى طبيبي ما أكتبه هنا",
        "هل يُرسل أي من هذا إلى عائلتي",
        "من يحصل على نسخة من هذه المحادثات",
    ]),
    ("QUESTION", "asking_for_a_fact", [
        "How many hours of sleep is normal for an adult",
        "What does a mood episode usually look like",
        "Is it common for focus to get worse with stress",
    ], [
        "كم ساعة نوم تُعد طبيعية للبالغ",
        "كيف تبدو نوبة المزاج عادة",
        "هل من الشائع أن يسوء التركيز مع التوتر",
    ]),
    ("QUESTION", "asking_whether_to_worry", [
        "Should I be worried about this or is it normal",
        "Is what I just described something to act on",
        "Do I need to tell someone about this",
    ], [
        "هل يجب أن أقلق من هذا أم أنه طبيعي",
        "هل ما وصفته للتو شيء يستدعي التحرك",
        "هل أحتاج أن أخبر أحدًا بهذا",
    ]),
    ("QUESTION", "asking_about_features", [
        "Can you remind me to take my tablets",
        "Is there a way to see my week in one place",
        "Can I turn the check-in prompts off",
    ], [
        "هل تستطيع تذكيري بأخذ حبوبي",
        "هل هناك طريقة لرؤية أسبوعي في مكان واحد",
        "هل أستطيع إيقاف تنبيهات التسجيل",
    ]),

    # --- GENERAL_CONVERSATION ------------------------------------------
    ("GENERAL_CONVERSATION", "weather_and_day", [
        "It has been raining here all day",
        "Nice weather for once this morning",
        "The heat is unbearable today",
    ], [
        "تمطر هنا طول اليوم",
        "طقس جميل لمرة هذا الصباح",
        "الحرارة لا تُحتمل اليوم",
    ]),
    ("GENERAL_CONVERSATION", "small_update_no_ask", [
        "I made bread today, nothing special",
        "Watched a film last night, it was fine",
        "Just tidied the kitchen, that is all",
    ], [
        "خبزت اليوم، لا شيء خاص",
        "شاهدت فيلمًا الليلة الماضية، كان جيدًا",
        "رتبت المطبخ فقط، هذا كل شيء",
    ]),
    ("GENERAL_CONVERSATION", "polite_reply", [
        "Yes, that sounds reasonable",
        "Alright, noted",
        "Fair enough, I will think about it",
    ], [
        "نعم، هذا يبدو معقولًا",
        "حسنًا، سجلت ذلك",
        "لا بأس، سأفكر في الأمر",
    ]),
    ("GENERAL_CONVERSATION", "feeling_fine", [
        "I am doing okay today, nothing to report",
        "All normal here",
        "Nothing much going on, I am fine",
    ], [
        "أنا بخير اليوم، لا شيء أبلغ عنه",
        "كل شيء عادي هنا",
        "لا يجري شيء كثير، أنا بخير",
    ]),
    ("GENERAL_CONVERSATION", "positive_news", [
        "My sister had her baby this week",
        "I got the job I applied for",
        "We are moving to a bigger flat next month",
    ], [
        "أنجبت أختي طفلها هذا الأسبوع",
        "حصلت على العمل الذي تقدمت له",
        "سننتقل إلى شقة أكبر الشهر القادم",
    ]),
    ("GENERAL_CONVERSATION", "hobby_talk", [
        "I have been getting back into drawing",
        "I started a new book last night",
        "I planted some things on the balcony",
    ], [
        "عدت إلى الرسم مؤخرًا",
        "بدأت كتابًا جديدًا الليلة الماضية",
        "زرعت بعض الأشياء في الشرفة",
    ]),
    ("GENERAL_CONVERSATION", "pause_the_conversation", [
        "Give me a minute, someone is at the door",
        "Hold on, I need to step away",
        "One moment, I will be back",
    ], [
        "أعطني دقيقة، أحد على الباب",
        "انتظر، أحتاج أن أبتعد قليلًا",
        "لحظة واحدة، سأعود",
    ]),

    # ==================================================================
    # Second expansion. The learning curve after the first one was still
    # climbing - 312 rows scored 0.250, 687 scored 0.391, 1040 scored 0.442,
    # 1356 scored 0.505 - so the head was still short of meanings, not of
    # tuning. Optimiser swaps, seed ensembling and wider feature spaces were
    # all measured first and none of them moved the number. These families
    # take each intent to roughly twenty.
    # ==================================================================

    # --- CHECK_IN ------------------------------------------------------
    ("CHECK_IN", "evening_wrapup", [
        "Closing out the day: nothing dramatic, just tired",
        "End of day report, I got through it",
        "Signing off for today, it was manageable",
    ], [
        "أختم يومي: لا شيء درامي، متعب فقط",
        "تقرير آخر اليوم، لقد تجاوزته",
        "أنهي تسجيل اليوم، كان محتملًا",
    ]),
    ("CHECK_IN", "comparison_to_yesterday", [
        "Better than yesterday, not by much",
        "About the same as yesterday, maybe a bit worse",
        "Today was easier than the last two days",
    ], [
        "أفضل من أمس، لكن ليس كثيرًا",
        "مثل أمس تقريبًا، ربما أسوأ قليلًا",
        "اليوم كان أسهل من اليومين الماضيين",
    ]),
    ("CHECK_IN", "partial_checkin", [
        "I can only tell you about my sleep today, skip the rest",
        "I do not want to answer all of it, just the mood part",
        "Only the sleep question today please",
    ], [
        "أستطيع أن أخبرك عن نومي فقط اليوم، تجاوز الباقي",
        "لا أريد الإجابة على كل شيء، جزء المزاج فقط",
        "سؤال النوم فقط اليوم لو سمحت",
    ]),
    ("CHECK_IN", "checkin_with_question", [
        "Logging today, and I wanted to ask you something after",
        "Here is my update, then I have a question",
        "Check-in first, then I need your help with something",
    ], [
        "أسجل اليوم، وأردت أن أسألك شيئًا بعدها",
        "هذا تحديثي، ثم لديّ سؤال",
        "التسجيل أولًا، ثم أحتاج مساعدتك في شيء",
    ]),
    ("CHECK_IN", "mixed_day", [
        "Good morning, terrible afternoon, so call it even",
        "Half the day was fine and half was not",
        "It swung a lot today, hard to give one number",
    ], [
        "صباح جيد وبعد ظهر فظيع، فلنعتبره متوازنًا",
        "نصف اليوم كان جيدًا ونصفه لا",
        "تأرجح كثيرًا اليوم، يصعب إعطاء رقم واحد",
    ]),
    ("CHECK_IN", "physical_symptoms_checkin", [
        "Reporting in: headache all day and no appetite",
        "Logging that my stomach has been bad since morning",
        "Check-in, mostly physical today, everything aches",
    ], [
        "أسجل: صداع طول اليوم ولا شهية",
        "أدوّن أن معدتي سيئة منذ الصباح",
        "تسجيل، جسدي أكثر من نفسي اليوم، كل شيء يؤلم",
    ]),
    ("CHECK_IN", "checkin_after_event", [
        "Logging how I am after the interview yesterday",
        "Check-in following the family visit",
        "Reporting in after the appointment I told you about",
    ], [
        "أسجل حالتي بعد مقابلة الأمس",
        "تسجيل بعد زيارة العائلة",
        "أسجل بعد الموعد الذي أخبرتك عنه",
    ]),

    # --- JOURNAL -------------------------------------------------------
    ("JOURNAL", "entry_about_work", [
        "I want to write about what happened at work today",
        "Journalling about the meeting that went wrong",
        "Putting down what my colleague said to me",
    ], [
        "أريد أن أكتب عما حدث في العمل اليوم",
        "أدوّن عن الاجتماع الذي فشل",
        "أسجل ما قاله لي زميلي",
    ]),
    ("JOURNAL", "entry_about_body", [
        "I want to write about how my body has felt this week",
        "Journalling about the pain and what it stops me doing",
        "Writing down what is going on physically",
    ], [
        "أريد أن أكتب عن حال جسدي هذا الأسبوع",
        "أدوّن عن الألم وما يمنعني من فعله",
        "أكتب ما يحدث لي جسديًا",
    ]),
    ("JOURNAL", "entry_unsent_letter", [
        "I want to write a letter I will never send",
        "Let me write as if I were talking to them directly",
        "I am going to write what I wish I had said",
    ], [
        "أريد أن أكتب رسالة لن أرسلها أبدًا",
        "دعني أكتب كأنني أحدثهم مباشرة",
        "سأكتب ما كنت أتمنى أن أقوله",
    ]),
    ("JOURNAL", "entry_list_form", [
        "I want to just list what happened, not write properly",
        "Bullet points today, I have no energy for sentences",
        "Let me put it down as a list",
    ], [
        "أريد أن أسرد ما حدث فقط، لا أن أكتب بشكل كامل",
        "نقاط فقط اليوم، لا طاقة لي للجمل",
        "دعني أدوّنها كقائمة",
    ]),
    ("JOURNAL", "entry_about_future", [
        "I want to write about what I am afraid is coming",
        "Journalling about where I want to be next year",
        "Writing down what I hope happens next",
    ], [
        "أريد أن أكتب عما أخشى أنه قادم",
        "أدوّن عن المكان الذي أريد أن أكون فيه العام القادم",
        "أكتب ما أتمنى أن يحدث بعد ذلك",
    ]),
    ("JOURNAL", "entry_about_memory", [
        "Something from years ago came back today and I want to write it",
        "Journalling about a memory that will not leave me",
        "I want to write about when I was younger",
    ], [
        "عاد إليّ شيء من سنوات مضت وأريد كتابته",
        "أدوّن عن ذكرى لا تفارقني",
        "أريد أن أكتب عن أيام صغري",
    ]),
    ("JOURNAL", "entry_short", [
        "Just one line today: I survived it",
        "Short entry, nothing more to say",
        "Two words for today and that is all",
    ], [
        "سطر واحد اليوم: لقد نجوت",
        "تدوينة قصيرة، لا شيء آخر لأقوله",
        "كلمتان لهذا اليوم وهذا كل شيء",
    ]),

    # --- EMOTIONAL_SUPPORT --------------------------------------------
    ("EMOTIONAL_SUPPORT", "comparing_to_others", [
        "Everyone my age is settled and I am nowhere",
        "I look at my friends and feel so far behind",
        "My cousin has everything figured out and I have nothing",
    ], [
        "كل من في عمري مستقر وأنا في لا مكان",
        "أنظر إلى أصدقائي وأشعر أنني متأخر جدًا",
        "ابن عمي يعرف كل شيء عن حياته وأنا لا شيء",
    ]),
    ("EMOTIONAL_SUPPORT", "fear_of_relapse", [
        "I am scared this is the start of it happening again",
        "It feels like last time and that terrifies me",
        "I do not want to go back to where I was",
    ], [
        "أخاف أن تكون هذه بداية تكراره",
        "الأمر يشبه المرة الماضية وهذا يرعبني",
        "لا أريد أن أعود إلى ما كنت عليه",
    ]),
    ("EMOTIONAL_SUPPORT", "lonely_in_a_crowd", [
        "I was surrounded by people and still felt completely alone",
        "I can be in a full room and feel invisible",
        "Being with them somehow made it worse",
    ], [
        "كنت محاطًا بالناس وما زلت أشعر بوحدة تامة",
        "أستطيع أن أكون في غرفة ممتلئة وأشعر أنني غير مرئي",
        "وجودي معهم زاد الأمر سوءًا بطريقة ما",
    ]),
    ("EMOTIONAL_SUPPORT", "needs_reassurance", [
        "Tell me this is going to pass",
        "I need to hear that I am not doing this wrong",
        "Is it normal to feel this way, I need to know",
    ], [
        "قل لي إن هذا سيمر",
        "أحتاج أن أسمع أنني لا أفعل هذا بشكل خاطئ",
        "هل من الطبيعي أن أشعر هكذا، أحتاج أن أعرف",
    ]),
    ("EMOTIONAL_SUPPORT", "missing_old_self", [
        "I do not recognise the person I have become",
        "I used to be funny and warm and now I am not",
        "I want to feel like myself again",
    ], [
        "لا أعرف الشخص الذي أصبحت عليه",
        "كنت مرحًا ودافئًا والآن لست كذلك",
        "أريد أن أشعر أنني أنا مرة أخرى",
    ]),
    ("EMOTIONAL_SUPPORT", "guilt_about_others", [
        "I feel awful for what I am putting my family through",
        "My partner deserves better than this version of me",
        "I keep letting the people around me down",
    ], [
        "أشعر بالسوء لما أُعرّض عائلتي له",
        "شريكي يستحق أفضل من هذه النسخة مني",
        "أستمر في إخلاف وعودي لمن حولي",
    ]),
    ("EMOTIONAL_SUPPORT", "emotional_exhaustion", [
        "I am so tired of managing my own head",
        "Holding it together every day is wearing me out",
        "I have nothing left to give anyone, including myself",
    ], [
        "تعبت جدًا من إدارة رأسي",
        "التماسك كل يوم يستنزفني",
        "لم يبق لديّ ما أعطيه لأحد، ولا لنفسي",
    ]),

    # --- TASK_SUPPORT --------------------------------------------------
    ("TASK_SUPPORT", "forgot_a_task", [
        "I completely forgot about it until the reminder went off",
        "It fell off my list and I only noticed today",
        "I forgot I had agreed to do it at all",
    ], [
        "نسيته تمامًا حتى رن التنبيه",
        "سقط من قائمتي ولم ألاحظ إلا اليوم",
        "نسيت أنني وافقت على فعله أصلًا",
    ]),
    ("TASK_SUPPORT", "task_took_too_long", [
        "Something that should take an hour took me all day",
        "I spent six hours on a small job",
        "Everything takes me three times longer than it should",
    ], [
        "شيء كان يجب أن يأخذ ساعة أخذ يومي كله",
        "أمضيت ست ساعات على مهمة صغيرة",
        "كل شيء يأخذ مني ثلاثة أضعاف ما ينبغي",
    ]),
    ("TASK_SUPPORT", "perfectionism_blocks", [
        "I will not start until I know I can do it properly",
        "I keep redoing the first part instead of moving on",
        "If it is not going to be right I would rather not begin",
    ], [
        "لن أبدأ حتى أعرف أنني أستطيع فعله كما يجب",
        "أستمر في إعادة الجزء الأول بدل أن أتقدم",
        "إن لم يكن سيخرج صحيحًا أفضّل ألا أبدأ",
    ]),
    ("TASK_SUPPORT", "interrupted_constantly", [
        "Every time I start, someone needs something from me",
        "I cannot get twenty uninterrupted minutes in this house",
        "The interruptions mean I never actually get going",
    ], [
        "كل مرة أبدأ، يحتاج أحدهم شيئًا مني",
        "لا أستطيع الحصول على عشرين دقيقة دون مقاطعة في هذا البيت",
        "المقاطعات تعني أنني لا أبدأ فعلًا أبدًا",
    ]),
    ("TASK_SUPPORT", "needs_accountability", [
        "Can you check whether I did it later",
        "I do better if someone is expecting it from me",
        "Ask me about this tomorrow so I actually do it",
    ], [
        "هل تتحقق لاحقًا إن كنت فعلته",
        "أنجز أفضل إن كان أحد ينتظره مني",
        "اسألني عن هذا غدًا لأفعله فعلًا",
    ]),
    ("TASK_SUPPORT", "delegating_or_asking_help", [
        "I think I need to ask someone to take part of this",
        "How do I tell them I cannot do all of it",
        "I should hand some of this over but I feel bad",
    ], [
        "أظن أنني أحتاج أن أطلب من أحد أن يأخذ جزءًا من هذا",
        "كيف أخبرهم أنني لا أستطيع فعله كله",
        "يجب أن أسلّم بعض هذا لكنني أشعر بالسوء",
    ]),
    ("TASK_SUPPORT", "housework", [
        "The washing up has been there for four days",
        "The flat is a mess and I cannot face it",
        "There is laundry everywhere and I keep walking past it",
    ], [
        "الجلي موجود منذ أربعة أيام",
        "البيت فوضى ولا أستطيع مواجهته",
        "الملابس في كل مكان وأستمر في المرور بجانبها",
    ]),

    # --- SLEEP ---------------------------------------------------------
    ("SLEEP", "waking_too_early", [
        "I am awake at half four every morning and cannot get back",
        "I wake before the alarm and just lie there",
        "My eyes open at five whatever time I go to bed",
    ], [
        "أستيقظ الرابعة والنصف كل صباح ولا أعود للنوم",
        "أصحو قبل المنبه وأبقى مستلقيًا",
        "تفتح عيناي الخامسة مهما كان وقت نومي",
    ]),
    ("SLEEP", "sleep_environment", [
        "The room is too hot and I cannot settle",
        "The street noise wakes me up two or three times",
        "I sleep much better when it is properly dark",
    ], [
        "الغرفة حارة جدًا ولا أستطيع الاستقرار",
        "ضجيج الشارع يوقظني مرتين أو ثلاثًا",
        "أنام أفضل بكثير عندما يكون الظلام تامًا",
    ]),
    ("SLEEP", "screen_before_bed", [
        "I am on my phone until two and then wonder why I am awake",
        "I watch things in bed and lose three hours",
        "I know the screen is the problem but I keep doing it",
    ], [
        "أبقى على هاتفي حتى الثانية ثم أتساءل لماذا أنا مستيقظ",
        "أشاهد أشياء في السرير وأخسر ثلاث ساعات",
        "أعرف أن الشاشة هي المشكلة لكنني أستمر",
    ]),
    ("SLEEP", "sleep_debt_weekend", [
        "I run on five hours all week and crash on Saturday",
        "I sleep till two at the weekend to catch up",
        "I am always paying back sleep I did not get",
    ], [
        "أعيش على خمس ساعات طول الأسبوع وأنهار السبت",
        "أنام حتى الثانية في نهاية الأسبوع لأعوّض",
        "أنا دائمًا أسدد نومًا لم أحصل عليه",
    ]),
    ("SLEEP", "shift_work_sleep", [
        "My shifts change every week and my sleep never settles",
        "I finish at midnight and cannot wind down until three",
        "Nights one week, mornings the next, it is wrecking me",
    ], [
        "دوامي يتغير كل أسبوع ونومي لا يستقر أبدًا",
        "أنتهي منتصف الليل ولا أهدأ حتى الثالثة",
        "ليالي أسبوعًا وصباحات الأسبوع التالي، هذا يدمرني",
    ]),
    ("SLEEP", "partner_disturbs_sleep", [
        "My husband snores and I am up half the night",
        "The baby is still waking twice and so am I",
        "I sleep better in the spare room and that upsets them",
    ], [
        "زوجي يشخر وأبقى مستيقظة نصف الليل",
        "الطفل ما زال يصحو مرتين وأنا معه",
        "أنام أفضل في الغرفة الأخرى وهذا يزعجهم",
    ]),
    ("SLEEP", "dreading_bedtime", [
        "I put off going to bed because I know what is coming",
        "Bedtime has become something I avoid",
        "I stay up late just to delay lying awake",
    ], [
        "أؤجل الذهاب إلى السرير لأنني أعرف ما ينتظرني",
        "صار وقت النوم شيئًا أتجنبه",
        "أسهر فقط لأؤخر الاستلقاء مستيقظًا",
    ]),

    # --- ENERGY --------------------------------------------------------
    ("ENERGY", "energy_after_eating", [
        "I need to lie down after every meal",
        "Lunch wipes me out completely",
        "I get so sleepy after I eat that I cannot work",
    ], [
        "أحتاج أن أستلقي بعد كل وجبة",
        "الغداء ينهكني تمامًا",
        "أشعر بنعاس شديد بعد الأكل لدرجة أنني لا أستطيع العمل",
    ]),
    ("ENERGY", "morning_energy_low", [
        "It takes me three hours to feel human in the morning",
        "Mornings are the worst, I am useless before eleven",
        "I cannot do anything until the middle of the day",
    ], [
        "أحتاج ثلاث ساعات لأشعر أنني إنسان في الصباح",
        "الصباح هو الأسوأ، لا نفع لي قبل الحادية عشرة",
        "لا أستطيع فعل شيء حتى منتصف النهار",
    ]),
    ("ENERGY", "energy_and_weather", [
        "The dark mornings have flattened me completely",
        "Since the clocks changed I have no energy at all",
        "Grey days take everything out of me",
    ], [
        "الصباحات المظلمة سطّحتني تمامًا",
        "منذ تغير الوقت لا طاقة لي أبدًا",
        "الأيام الرمادية تأخذ كل ما فيّ",
    ]),
    ("ENERGY", "needs_rest_permission", [
        "Is it alright if I just rest today and do nothing",
        "I feel guilty for needing to lie down",
        "Tell me it is okay to stop for a day",
    ], [
        "هل لا بأس أن أرتاح اليوم ولا أفعل شيئًا",
        "أشعر بالذنب لأنني أحتاج أن أستلقي",
        "قل لي إنه لا بأس أن أتوقف يومًا",
    ]),
    ("ENERGY", "energy_for_one_thing_only", [
        "I have enough for work or for the house, not both",
        "If I go out I will not manage anything else today",
        "One thing a day is my limit at the moment",
    ], [
        "لديّ طاقة للعمل أو للبيت، لا لكليهما",
        "إن خرجت فلن أتدبر أي شيء آخر اليوم",
        "شيء واحد في اليوم هو حدي حاليًا",
    ]),
    ("ENERGY", "energy_better_with_movement", [
        "Oddly I have more energy on the days I walk",
        "Sitting all day makes the tiredness worse, not better",
        "Moving actually gives me energy back",
    ], [
        "غريب أن طاقتي أعلى في الأيام التي أمشي فيها",
        "الجلوس طول اليوم يزيد التعب لا ينقصه",
        "الحركة تعيد لي طاقتي فعلًا",
    ]),
    ("ENERGY", "sudden_energy_drop", [
        "It hit me out of nowhere this afternoon",
        "I was fine and then suddenly could not stand up",
        "The tiredness came down like a shutter",
    ], [
        "أصابني من لا مكان بعد ظهر اليوم",
        "كنت بخير ثم فجأة لم أستطع الوقوف",
        "هبط التعب عليّ كالستارة",
    ]),

    # --- FOCUS ---------------------------------------------------------
    ("FOCUS", "cannot_finish_conversation", [
        "I drift off while someone is talking to me",
        "People tell me I have stopped listening mid-sentence",
        "I lose track of what my friend is saying",
    ], [
        "أسرح بينما يتحدث إليّ أحدهم",
        "يقول لي الناس إنني توقفت عن الإنصات في منتصف الكلام",
        "أفقد خيط ما يقوله صديقي",
    ]),
    ("FOCUS", "forgetting_appointments", [
        "I missed a booking again because it left my head",
        "I have to write everything down or it is gone",
        "I double-booked myself twice this week",
    ], [
        "فوّتت موعدًا مجددًا لأنه غاب عن رأسي",
        "عليّ أن أكتب كل شيء وإلا اختفى",
        "حجزت موعدين متعارضين مرتين هذا الأسبوع",
    ]),
    ("FOCUS", "misplacing_things", [
        "I have looked for my keys three times today",
        "I put things down and cannot find them minutes later",
        "I lost my wallet again, it is constant",
    ], [
        "بحثت عن مفاتيحي ثلاث مرات اليوم",
        "أضع الأشياء ولا أجدها بعد دقائق",
        "فقدت محفظتي مجددًا، الأمر مستمر",
    ]),
    ("FOCUS", "focus_better_on_medication", [
        "On the days I take it I can actually sit and work",
        "The difference in my concentration is obvious now",
        "I got through a full task for the first time in months",
    ], [
        "في الأيام التي آخذه فيها أستطيع الجلوس والعمل فعلًا",
        "الفرق في تركيزي واضح الآن",
        "أنجزت مهمة كاملة لأول مرة منذ شهور",
    ]),
    ("FOCUS", "focus_in_meetings", [
        "I cannot follow what is being said in a meeting",
        "By the time it is my turn I have lost the thread",
        "Long calls and my mind is completely elsewhere",
    ], [
        "لا أستطيع متابعة ما يُقال في الاجتماع",
        "حين يأتي دوري أكون قد فقدت الخيط",
        "المكالمات الطويلة وعقلي في مكان آخر تمامًا",
    ]),
    ("FOCUS", "mind_wandering_while_driving", [
        "I drove home and do not remember the journey",
        "I missed my turning because I was somewhere else",
        "I zone out completely behind the wheel",
    ], [
        "قدت إلى البيت ولا أتذكر الطريق",
        "فوّتت مخرجي لأنني كنت في مكان آخر",
        "أغيب تمامًا وأنا أقود",
    ]),
    ("FOCUS", "focus_time_of_day", [
        "I can only think clearly between nine and eleven",
        "After lunch there is no point trying to concentrate",
        "My head is sharpest very late at night",
    ], [
        "أستطيع التفكير بصفاء بين التاسعة والحادية عشرة فقط",
        "بعد الغداء لا جدوى من محاولة التركيز",
        "رأسي أصفى في وقت متأخر من الليل",
    ]),

    # --- ROUTINE -------------------------------------------------------
    ("ROUTINE", "travel_disrupts_routine", [
        "I was away for a week and everything fell apart",
        "Being in a hotel undid all my habits",
        "The time difference has thrown me completely",
    ], [
        "كنت مسافرًا أسبوعًا وانهار كل شيء",
        "الإقامة في فندق نقضت كل عاداتي",
        "فرق التوقيت أخرجني تمامًا عن مساري",
    ]),
    ("ROUTINE", "morning_routine_specific", [
        "I want the first hour of my day to look the same every day",
        "My mornings are chaos and it sets the tone",
        "If I get the morning right the rest usually follows",
    ], [
        "أريد أن تكون الساعة الأولى من يومي متشابهة كل يوم",
        "صباحاتي فوضى وهذا يحدد نبرة اليوم",
        "إن أصلحت صباحي يتبعه الباقي عادة",
    ]),
    ("ROUTINE", "evening_routine_specific", [
        "I have no wind-down at all, I just stop and collapse",
        "I want a proper end-of-day sequence",
        "My evenings have no shape and I drift until late",
    ], [
        "ليس لديّ أي تهدئة، أتوقف وأنهار فقط",
        "أريد تسلسلًا حقيقيًا لآخر اليوم",
        "أمسياتي بلا شكل وأطفو حتى وقت متأخر",
    ]),
    ("ROUTINE", "routine_after_illness", [
        "I was ill for two weeks and have not got back into anything",
        "Since the flu my whole structure has gone",
        "I am trying to restart everything after being unwell",
    ], [
        "كنت مريضًا أسبوعين ولم أعد إلى أي شيء",
        "منذ الإنفلونزا ذهب كل نظامي",
        "أحاول إعادة تشغيل كل شيء بعد المرض",
    ]),
    ("ROUTINE", "hygiene_routine", [
        "I have not showered in three days and I know I should",
        "Brushing my teeth has become something I forget",
        "Basic self-care is the first thing to go",
    ], [
        "لم أستحم منذ ثلاثة أيام وأعرف أنني يجب أن أفعل",
        "صار تنظيف أسناني شيئًا أنساه",
        "العناية الأساسية بنفسي هي أول ما يسقط",
    ]),
    ("ROUTINE", "routine_with_children", [
        "The school run decides my whole day and nothing else fits",
        "I cannot keep a routine with two small children",
        "My schedule is really their schedule",
    ], [
        "توصيل المدرسة يحدد يومي كله ولا يتسع لشيء آخر",
        "لا أستطيع الحفاظ على روتين مع طفلين صغيرين",
        "جدولي هو في الحقيقة جدولهم",
    ]),
    ("ROUTINE", "reminders_for_routine", [
        "Can you prompt me at the same time each evening",
        "I need a nudge or I will not remember",
        "Set something up so I do not have to hold it in my head",
    ], [
        "هل تنبهني في الوقت نفسه كل مساء",
        "أحتاج تنبيهًا وإلا لن أتذكر",
        "رتّب شيئًا حتى لا أضطر لحمله في رأسي",
    ]),

    # --- STRESS --------------------------------------------------------
    ("STRESS", "too_many_demands_from_others", [
        "Everyone wants something from me and I have nothing left",
        "I am being pulled in four directions at once",
        "I cannot be what all these people need",
    ], [
        "الجميع يريد شيئًا مني ولم يبق لديّ شيء",
        "أنا مسحوب في أربع جهات في الوقت نفسه",
        "لا أستطيع أن أكون ما يحتاجه كل هؤلاء",
    ]),
    ("STRESS", "news_and_world_events", [
        "I cannot stop reading the news and it is making me ill",
        "What is happening in the world is sitting on my chest",
        "I doom-scroll for an hour and feel worse every time",
    ], [
        "لا أستطيع التوقف عن قراءة الأخبار وهذا يمرضني",
        "ما يحدث في العالم جالس على صدري",
        "أتابع الأخبار السيئة ساعة وأشعر أسوأ كل مرة",
    ]),
    ("STRESS", "health_anxiety_stress", [
        "I have been searching my symptoms for days",
        "I am waiting on test results and cannot think straight",
        "Every ache convinces me something is seriously wrong",
    ], [
        "أبحث عن أعراضي منذ أيام",
        "أنتظر نتائج التحاليل ولا أستطيع التفكير بوضوح",
        "كل وجع يقنعني أن هناك خطأ جسيمًا",
    ]),
    ("STRESS", "housing_stress", [
        "The landlord wants us out and I do not know where we will go",
        "We are moving in three weeks and nothing is arranged",
        "The damp in the flat is making everything harder",
    ], [
        "المالك يريد أن نخرج ولا أعرف إلى أين سنذهب",
        "سننتقل بعد ثلاثة أسابيع ولم يُرتب شيء",
        "الرطوبة في الشقة تجعل كل شيء أصعب",
    ]),
    ("STRESS", "study_pressure", [
        "I have four assignments due and I have started none",
        "Revision is not going in and the exam is close",
        "I am behind on the whole term and it is crushing me",
    ], [
        "لديّ أربعة واجبات مستحقة ولم أبدأ أيًا منها",
        "المراجعة لا تثبت والامتحان قريب",
        "أنا متأخر عن الفصل كله وهذا يسحقني",
    ]),
    ("STRESS", "commute_or_traffic", [
        "Two hours on the bus each way is breaking me",
        "I arrive at work already depleted from the journey",
        "The commute is the worst part of my day by far",
    ], [
        "ساعتان في الباص في كل اتجاه تكسرني",
        "أصل إلى العمل منهكًا من الطريق أصلًا",
        "الطريق إلى العمل أسوأ جزء من يومي بلا منافس",
    ]),
    ("STRESS", "stress_from_uncertainty", [
        "Not knowing what happens next is worse than bad news",
        "I am waiting on a decision and it is eating me",
        "I could cope if someone would just tell me either way",
    ], [
        "عدم معرفة ما سيحدث أسوأ من الخبر السيئ",
        "أنتظر قرارًا وهذا يأكلني",
        "أستطيع التحمل لو أخبرني أحد بأي من الاتجاهين",
    ]),

    # --- SOCIAL --------------------------------------------------------
    ("SOCIAL", "family_obligations", [
        "There is a family gathering and I really do not want to go",
        "I am expected at my aunt's and I cannot face it",
        "Saying no to family is not an option in my house",
    ], [
        "هناك اجتماع عائلي ولا أريد الذهاب فعلًا",
        "يتوقعون حضوري عند عمتي ولا أستطيع مواجهة ذلك",
        "رفض العائلة ليس خيارًا في بيتنا",
    ]),
    ("SOCIAL", "conflict_with_partner", [
        "My wife and I have barely spoken for three days",
        "We keep having the same argument and getting nowhere",
        "Things at home with my partner are very tense",
    ], [
        "زوجتي وأنا بالكاد تحدثنا منذ ثلاثة أيام",
        "نكرر الخلاف نفسه ولا نصل إلى شيء",
        "الأمور في البيت مع شريكي متوترة جدًا",
    ]),
    ("SOCIAL", "feeling_left_out", [
        "They all went out and nobody thought to ask me",
        "I saw the photos and realised I was not invited",
        "I am always the one who finds out afterwards",
    ], [
        "خرجوا جميعًا ولم يفكر أحد في دعوتي",
        "رأيت الصور وأدركت أنني لم أُدعَ",
        "أنا دائمًا من يعرف بعد فوات الأمر",
    ]),
    ("SOCIAL", "online_vs_real_contact", [
        "All my contact with people is through a screen now",
        "I talk to strangers online more than anyone I know",
        "I have not seen a friend in person for months",
    ], [
        "كل تواصلي مع الناس عبر شاشة الآن",
        "أتحدث مع غرباء على الإنترنت أكثر من أي شخص أعرفه",
        "لم أرَ صديقًا وجهًا لوجه منذ شهور",
    ]),
    ("SOCIAL", "saying_no_to_people", [
        "I agreed to something I do not want to do again",
        "I cannot turn anyone down and it costs me",
        "How do I decline without upsetting them",
    ], [
        "وافقت على شيء لا أريد فعله مجددًا",
        "لا أستطيع رفض أحد وهذا يكلفني",
        "كيف أرفض دون أن أزعجهم",
    ]),
    ("SOCIAL", "workplace_relations", [
        "There is someone at work who makes every day harder",
        "I eat lunch alone every day and it has started to bother me",
        "My team talks around me rather than to me",
    ], [
        "هناك شخص في العمل يجعل كل يوم أصعب",
        "أتغدى وحدي كل يوم وقد بدأ هذا يزعجني",
        "فريقي يتحدث حولي لا إليّ",
    ]),
    ("SOCIAL", "missing_someone_far", [
        "My brother moved abroad and I miss him constantly",
        "The time difference means we never really talk",
        "My closest friend lives too far to see",
    ], [
        "انتقل أخي إلى الخارج وأفتقده باستمرار",
        "فرق التوقيت يعني أننا لا نتحدث فعلًا أبدًا",
        "أقرب صديق لي يسكن بعيدًا جدًا لأراه",
    ]),
    ("SOCIAL", "hosting_or_visiting", [
        "People are coming over on Friday and I am already anxious",
        "I said they could stay and now I regret it",
        "Having guests in the house is a lot for me",
    ], [
        "سيأتي ناس الجمعة وأنا قلق أصلًا",
        "قلت لهم أن يبقوا والآن أنا نادم",
        "وجود ضيوف في البيت كثير عليّ",
    ]),
    # A plainly reported falling-out, with no request attached. Filed under
    # EMOTIONAL_SUPPORT rather than SOCIAL to match lumina/intent.py: a discrete
    # upsetting event wants acknowledging, while SOCIAL's reply ("connection with
    # other people can help") suits isolation and ongoing relational tension.
    ("EMOTIONAL_SUPPORT", "argument_with_family", [
        "I had an argument with my sister last night",
        "Me and my dad had a big row this morning",
        "I fell out with my cousin and we have not spoken since",
    ], [
        "تشاجرت مع أختي الليلة الماضية",
        "أنا وأبي تخاصمنا بشدة هذا الصباح",
        "اختلفت مع ابن عمي ولم نتحدث منذ ذلك",
    ]),

    # --- EXERCISE ------------------------------------------------------
    ("EXERCISE", "pain_limits_exercise", [
        "My knee gives out after ten minutes of walking",
        "My back stops me doing anything much",
        "I want to move but my body will not let me",
    ], [
        "رُكبتي تخذلني بعد عشر دقائق من المشي",
        "ظهري يمنعني من فعل أي شيء يُذكر",
        "أريد الحركة لكن جسدي لا يسمح لي",
    ]),
    ("EXERCISE", "exercise_with_others", [
        "I only manage it if I go with my sister",
        "I joined a class so I would have to turn up",
        "Exercising alone never lasts more than a week",
    ], [
        "لا أتدبره إلا إن ذهبت مع أختي",
        "انضممت إلى حصة حتى أضطر للحضور",
        "الرياضة وحدي لا تستمر أكثر من أسبوع",
    ]),
    ("EXERCISE", "home_workout", [
        "I did twenty minutes in the living room today",
        "I found a video I can actually follow at home",
        "I cannot face a gym so I move in my room instead",
    ], [
        "تمرنت عشرين دقيقة في الصالة اليوم",
        "وجدت مقطعًا أستطيع متابعته في البيت فعلًا",
        "لا أستطيع مواجهة النادي فأتحرك في غرفتي بدلًا منه",
    ]),
    ("EXERCISE", "sport_specific", [
        "I played football for the first time in two years",
        "I have started swimming twice a week",
        "I went back to cycling and it felt good",
    ], [
        "لعبت كرة القدم لأول مرة منذ سنتين",
        "بدأت السباحة مرتين في الأسبوع",
        "عدت إلى الدراجة وكان شعورًا جيدًا",
    ]),
    ("EXERCISE", "exercise_and_sleep", [
        "I sleep much better on the days I have moved",
        "If I train late I cannot get to sleep at all",
        "Morning exercise seems to fix my nights",
    ], [
        "أنام أفضل بكثير في الأيام التي أتحرك فيها",
        "إن تدربت متأخرًا لا أستطيع النوم أبدًا",
        "يبدو أن رياضة الصباح تصلح ليلي",
    ]),
    ("EXERCISE", "exercise_motivation_gone", [
        "I know it helps and I still cannot make myself do it",
        "I have not wanted to move in weeks",
        "The kit is by the door and I walk past it",
    ], [
        "أعرف أنها تساعد وما زلت لا أستطيع إجبار نفسي",
        "لم أرغب في الحركة منذ أسابيع",
        "الحقيبة عند الباب وأمر بجانبها",
    ]),
    ("EXERCISE", "steps_target", [
        "I hit eight thousand steps for the third day running",
        "I am trying to get a few thousand steps in daily",
        "My watch says I moved more this week than last",
    ], [
        "وصلت إلى ثمانية آلاف خطوة لليوم الثالث تواليًا",
        "أحاول تحقيق بضعة آلاف خطوة يوميًا",
        "ساعتي تقول إنني تحركت هذا الأسبوع أكثر من الماضي",
    ]),

    # --- GOAL ----------------------------------------------------------
    ("GOAL", "streak_broken", [
        "I missed a day and now I have stopped altogether",
        "The chain broke on day eleven and I gave up",
        "One slip and the whole habit went",
    ], [
        "فوّتت يومًا والآن توقفت تمامًا",
        "انكسرت السلسلة في اليوم الحادي عشر واستسلمت",
        "زلة واحدة وذهبت العادة كلها",
    ]),
    ("GOAL", "goal_review_request", [
        "Can we go over the targets I set and see what is left",
        "Read me back what I said I would do",
        "I want to look at my goals again",
    ], [
        "هل نراجع الأهداف التي وضعتها ونرى ما تبقى",
        "اقرأ لي ما قلت إنني سأفعله",
        "أريد أن أنظر في أهدافي مرة أخرى",
    ]),
    ("GOAL", "competing_goals", [
        "I want to sleep more and train more and they clash",
        "I cannot work on my studies and my fitness at once",
        "Two of my goals are pulling against each other",
    ], [
        "أريد أن أنام أكثر وأتدرب أكثر وهما يتعارضان",
        "لا أستطيع العمل على دراستي ولياقتي في الوقت نفسه",
        "هدفان من أهدافي يتجاذبان",
    ]),
    ("GOAL", "goal_for_this_week", [
        "Give me one thing to aim for between now and Sunday",
        "I want a target just for the next few days",
        "What should I try to do this week",
    ], [
        "أعطني شيئًا واحدًا أسعى إليه من الآن حتى الأحد",
        "أريد هدفًا للأيام القادمة فقط",
        "ما الذي يجب أن أحاول فعله هذا الأسبوع",
    ]),
    ("GOAL", "external_deadline_goal", [
        "I need to be ready for the wedding in six weeks",
        "The course starts in a month and I want to be prepared",
        "I have to be back at work by October",
    ], [
        "أحتاج أن أكون مستعدًا للعرس بعد ستة أسابيع",
        "الدورة تبدأ بعد شهر وأريد أن أكون مستعدًا",
        "عليّ أن أعود إلى العمل بحلول أكتوبر",
    ]),
    ("GOAL", "goal_about_people", [
        "I want to call one friend a week, that is my aim",
        "My goal is to eat dinner with my family properly",
        "I want to reply to messages within a day",
    ], [
        "أريد أن أتصل بصديق كل أسبوع، هذا هدفي",
        "هدفي أن أتعشى مع عائلتي كما يجب",
        "أريد أن أرد على الرسائل خلال يوم",
    ]),
    ("GOAL", "reward_for_goal", [
        "I want something to look forward to if I manage it",
        "Should I promise myself a treat for finishing",
        "I do better with something at the end of it",
    ], [
        "أريد شيئًا أتطلع إليه إن نجحت",
        "هل أعد نفسي بمكافأة عند الإنجاز",
        "أنجز أفضل عندما يكون هناك شيء في النهاية",
    ]),

    # --- PROGRESS ------------------------------------------------------
    ("PROGRESS", "progress_in_one_area_only", [
        "My sleep is better but everything else is the same",
        "I have improved at work and gone backwards socially",
        "One thing is fixed and the rest has not moved",
    ], [
        "نومي أفضل لكن كل شيء آخر كما هو",
        "تحسنت في العمل وتراجعت اجتماعيًا",
        "شيء واحد أُصلح والباقي لم يتحرك",
    ]),
    ("PROGRESS", "doubting_the_data", [
        "I do not think what you have reflects how bad it has been",
        "The numbers look better than I actually feel",
        "I was not honest in some of those check-ins",
    ], [
        "لا أظن أن ما لديك يعكس مدى سوء الأمر",
        "الأرقام تبدو أفضل من شعوري الحقيقي",
        "لم أكن صادقًا في بعض تلك التسجيلات",
    ]),
    ("PROGRESS", "milestone_noticed", [
        "That is a month of check-ins without missing one",
        "First full week where I did everything I planned",
        "I have not had a really bad day in three weeks",
    ], [
        "هذا شهر من التسجيل دون أن أفوّت يومًا",
        "أول أسبوع كامل أفعل فيه كل ما خططت له",
        "لم يمر بي يوم سيئ فعلًا منذ ثلاثة أسابيع",
    ]),
    ("PROGRESS", "others_noticed_change", [
        "My mother said I seem more like myself lately",
        "A colleague told me I look better than I did",
        "People have started saying I seem lighter",
    ], [
        "قالت أمي إنني أبدو أشبه بنفسي مؤخرًا",
        "قال لي زميل إنني أبدو أفضل من قبل",
        "بدأ الناس يقولون إنني أبدو أخف",
    ]),
    ("PROGRESS", "slow_progress_frustration", [
        "It has been six months and I expected more by now",
        "The pace of this is so slow it barely counts",
        "I am tired of tiny improvements",
    ], [
        "مرت ستة أشهر وكنت أتوقع أكثر الآن",
        "وتيرة هذا بطيئة جدًا لدرجة أنها لا تُحتسب",
        "تعبت من التحسنات الضئيلة",
    ]),
    ("PROGRESS", "what_to_change_next", [
        "Based on what you have, what should I work on next",
        "Where is the biggest gap I could close",
        "What would make the most difference now",
    ], [
        "بناءً على ما لديك، على ماذا أعمل بعد ذلك",
        "ما أكبر فجوة أستطيع إغلاقها",
        "ما الذي سيحدث أكبر فرق الآن",
    ]),
    ("PROGRESS", "progress_since_medication", [
        "Has anything changed since the dose went up",
        "Compare me before and after the new tablet",
        "I want to see whether the change actually helped",
    ], [
        "هل تغيّر شيء منذ رُفعت الجرعة",
        "قارن حالي قبل الحبة الجديدة وبعدها",
        "أريد أن أرى إن كان التغيير ساعد فعلًا",
    ]),

    # --- MEDICATION_MENTION --------------------------------------------
    ("MEDICATION_MENTION", "medication_and_alcohol", [
        "Is it a problem if I have a drink on these",
        "I had two beers last night and now I am worried",
        "Nobody told me whether I can drink on this",
    ], [
        "هل هناك مشكلة إن شربت وأنا على هذه",
        "شربت اثنتين الليلة الماضية والآن أنا قلق",
        "لم يخبرني أحد إن كان بإمكاني الشرب على هذا",
    ]),
    ("MEDICATION_MENTION", "cost_of_medication", [
        "I cannot really afford the prescription this month",
        "The price went up and I have been stretching them out",
        "I have been taking half doses to make it last",
    ], [
        "لا أستطيع فعلًا تحمل ثمن الوصفة هذا الشهر",
        "زاد السعر وأنا أمددها",
        "آخذ نصف جرعات لتكفي أكثر",
    ]),
    ("MEDICATION_MENTION", "multiple_medications", [
        "I am on four different things now and I lose track",
        "One is for sleep, one for mood, I get confused",
        "Keeping the schedule of all of them straight is hard",
    ], [
        "أنا على أربعة أشياء مختلفة الآن وأفقد المتابعة",
        "واحد للنوم وواحد للمزاج، أختلط",
        "الحفاظ على جدول كلها بشكل صحيح صعب",
    ]),
    ("MEDICATION_MENTION", "medication_stigma", [
        "My family think I should not need tablets at all",
        "I feel weak for being on medication",
        "I do not tell people that I take anything",
    ], [
        "عائلتي تظن أنني لا يجب أن أحتاج حبوبًا أبدًا",
        "أشعر بالضعف لأنني على دواء",
        "لا أخبر الناس أنني آخذ شيئًا",
    ]),
    ("MEDICATION_MENTION", "restarting_medication", [
        "I stopped two months ago and I think I need to go back on",
        "I am starting them again tomorrow",
        "I came off it and things have got worse",
    ], [
        "توقفت منذ شهرين وأظن أنني أحتاج العودة إليها",
        "سأبدأها من جديد غدًا",
        "توقفت عنه وساءت الأمور",
    ]),
    ("MEDICATION_MENTION", "medication_not_working", [
        "Six weeks on this and I feel no different",
        "I do not think this one is doing anything for me",
        "The first one worked and this one does not",
    ], [
        "ستة أسابيع على هذا ولا أشعر بأي فرق",
        "لا أظن أن هذا يفعل شيئًا لي",
        "الأول كان ينجح وهذا لا",
    ]),
    ("MEDICATION_MENTION", "taking_as_needed", [
        "I only take the second one when it gets bad",
        "I am supposed to use it as required but I never know when",
        "How do I decide when I actually need the extra one",
    ], [
        "آخذ الثانية فقط عندما يسوء الأمر",
        "المفروض أن أستخدمه عند الحاجة لكنني لا أعرف متى",
        "كيف أقرر متى أحتاج الإضافية فعلًا",
    ]),

    # --- CLINICIAN_MENTION ---------------------------------------------
    ("CLINICIAN_MENTION", "appointment_missed", [
        "I did not turn up to my appointment and now I feel awful",
        "I slept through the session yesterday",
        "I cancelled at the last minute again",
    ], [
        "لم أحضر موعدي والآن أشعر بالسوء",
        "نمت عن الجلسة أمس",
        "ألغيت في آخر لحظة مجددًا",
    ]),
    ("CLINICIAN_MENTION", "bringing_someone_to_appointment", [
        "Should I take my sister in with me on Thursday",
        "I want someone there because I forget what is said",
        "My mother wants to come to the appointment",
    ], [
        "هل آخذ أختي معي الخميس",
        "أريد أحدًا هناك لأنني أنسى ما يُقال",
        "أمي تريد الحضور إلى الموعد",
    ]),
    ("CLINICIAN_MENTION", "clinician_disagreement", [
        "My doctor and I see this completely differently",
        "He says I am fine and I do not feel fine",
        "They want to change something I do not want changed",
    ], [
        "طبيبي وأنا نرى هذا بشكل مختلف تمامًا",
        "يقول إنني بخير وأنا لا أشعر أنني بخير",
        "يريدون تغيير شيء لا أريد تغييره",
    ]),
    ("CLINICIAN_MENTION", "referral_pending", [
        "My GP referred me and I have heard nothing since",
        "I am waiting to be assessed by the specialist team",
        "The referral went in six weeks ago",
    ], [
        "أحالني طبيب العائلة ولم أسمع شيئًا منذ ذلك",
        "أنتظر التقييم من فريق الأخصائيين",
        "قُدمت الإحالة قبل ستة أسابيع",
    ]),
    ("CLINICIAN_MENTION", "gp_vs_specialist", [
        "Should I go to my GP about this or wait for the psychiatrist",
        "I do not know which of them to ask",
        "My family doctor says it is not his area",
    ], [
        "هل أذهب إلى طبيب العائلة بهذا أم أنتظر الطبيب النفسي",
        "لا أعرف من منهما أسأل",
        "طبيب عائلتي يقول إن هذا ليس مجاله",
    ]),
    ("CLINICIAN_MENTION", "telehealth_appointment", [
        "My session is a video call and I find that harder",
        "We do it over the phone now and it is not the same",
        "The connection dropped halfway through my appointment",
    ], [
        "جلستي مكالمة فيديو وأجد ذلك أصعب",
        "نفعلها عبر الهاتف الآن وليست كما كانت",
        "انقطع الاتصال في منتصف موعدي",
    ]),
    ("CLINICIAN_MENTION", "notes_or_records", [
        "I want to see what is written in my file",
        "Can I get a copy of the letter they sent",
        "I asked for my notes and they have not sent them",
    ], [
        "أريد أن أرى ما هو مكتوب في ملفي",
        "هل أحصل على نسخة من الرسالة التي أرسلوها",
        "طلبت ملاحظاتي ولم يرسلوها",
    ]),

    # --- QUESTION ------------------------------------------------------
    ("QUESTION", "asking_about_language", [
        "Can I write to you in Arabic instead",
        "Do you understand me if I mix the two languages",
        "Which language should I use with you",
    ], [
        "هل أستطيع أن أكتب لك بالعربية بدلًا من ذلك",
        "هل تفهمني إن خلطت اللغتين",
        "بأي لغة أتحدث معك",
    ]),
    ("QUESTION", "asking_about_cost", [
        "Is there a charge for using this",
        "Will I have to pay for it later",
        "Is this free or is there a subscription",
    ], [
        "هل هناك رسوم لاستخدام هذا",
        "هل سأدفع مقابله لاحقًا",
        "هل هذا مجاني أم هناك اشتراك",
    ]),
    ("QUESTION", "asking_about_availability", [
        "Are you there at night if I need you",
        "Can I message you at any hour",
        "What happens if I write and nobody answers",
    ], [
        "هل تكون موجودًا في الليل إن احتجتك",
        "هل أستطيع مراسلتك في أي ساعة",
        "ماذا يحدث إن كتبت ولم يرد أحد",
    ]),
    ("QUESTION", "asking_for_definition", [
        "What does capacity mean in what you just said",
        "You used a word I do not know, what is it",
        "Explain what you mean by baseline",
    ], [
        "ماذا تعني الطاقة في ما قلته للتو",
        "استخدمت كلمة لا أعرفها، ما هي",
        "اشرح ما تعنيه بخط الأساس",
    ]),
    ("QUESTION", "asking_to_compare_options", [
        "Which of those two would you say is better for me",
        "Is it best to start with sleep or with the tasks",
        "Should I do the short one or the long one",
    ], [
        "أي من هذين تقول إنه أفضل لي",
        "هل الأفضل أن أبدأ بالنوم أم بالمهام",
        "هل أفعل القصير أم الطويل",
    ]),
    ("QUESTION", "asking_about_evidence", [
        "Is there any actual research behind that suggestion",
        "How do you know that works",
        "Who says this is the right thing to do",
    ], [
        "هل هناك بحث فعلي وراء هذا الاقتراح",
        "كيف تعرف أن هذا ينجح",
        "من يقول إن هذا هو الصواب",
    ]),
    ("QUESTION", "asking_who_made_it", [
        "Who built you and who do they answer to",
        "Is a doctor involved in what you tell me",
        "Who is responsible if you get something wrong",
    ], [
        "من بناك ولمن يرجعون",
        "هل هناك طبيب مشارك في ما تقوله لي",
        "من المسؤول إن أخطأت في شيء",
    ]),
    ("QUESTION", "asking_about_deleting_data", [
        "Can I delete everything I have written here",
        "How do I remove my history",
        "What happens to my entries if I stop using this",
    ], [
        "هل أستطيع حذف كل ما كتبته هنا",
        "كيف أزيل سجلي",
        "ماذا يحدث لتدويناتي إن توقفت عن استخدام هذا",
    ]),

    # --- GENERAL_CONVERSATION ------------------------------------------
    ("GENERAL_CONVERSATION", "food_and_cooking", [
        "I made a proper dinner for the first time in ages",
        "I tried a new recipe and it went badly",
        "I have been living on toast this week",
    ], [
        "طبخت عشاءً حقيقيًا لأول مرة منذ زمن",
        "جربت وصفة جديدة ولم تنجح",
        "أعيش على الخبز المحمص هذا الأسبوع",
    ]),
    ("GENERAL_CONVERSATION", "pets", [
        "The cat has been sitting on me all morning",
        "I took the dog out twice today",
        "My rabbit is the only one who puts up with me",
    ], [
        "القطة جالسة عليّ طول الصباح",
        "أخرجت الكلب مرتين اليوم",
        "أرنبي هو الوحيد الذي يتحملني",
    ]),
    ("GENERAL_CONVERSATION", "travel_plans", [
        "We are going to the coast next month",
        "I booked a few days away in the summer",
        "I might visit my brother over the holiday",
    ], [
        "سنذهب إلى الساحل الشهر القادم",
        "حجزت بضعة أيام بعيدًا في الصيف",
        "قد أزور أخي في العطلة",
    ]),
    ("GENERAL_CONVERSATION", "music_or_film", [
        "I finished a series last night and I miss it already",
        "I have had the same album on repeat all week",
        "I saw a film that stayed with me",
    ], [
        "أنهيت مسلسلًا الليلة الماضية وأفتقده أصلًا",
        "أعدت الألبوم نفسه طول الأسبوع",
        "شاهدت فيلمًا بقي معي",
    ]),
    ("GENERAL_CONVERSATION", "sports_talk", [
        "Did you see the match last night",
        "My team lost again, as usual",
        "The season starts next week and I am looking forward to it",
    ], [
        "هل رأيت المباراة الليلة الماضية",
        "خسر فريقي مجددًا، كالعادة",
        "الموسم يبدأ الأسبوع القادم وأنا أتطلع إليه",
    ]),
    ("GENERAL_CONVERSATION", "correcting_a_detail", [
        "Actually it was Tuesday, not Monday",
        "I said seven hours, not five",
        "You have the wrong week there",
    ], [
        "في الحقيقة كان الثلاثاء لا الاثنين",
        "قلت سبع ساعات لا خمسًا",
        "الأسبوع الذي لديك خطأ",
    ]),
    ("GENERAL_CONVERSATION", "humour_or_joke", [
        "Well, at least the plants are still alive",
        "My cooking has improved from terrible to merely bad",
        "I am told I have a face for radio",
    ], [
        "على الأقل النباتات ما زالت حية",
        "تحسن طبخي من فظيع إلى سيئ فقط",
        "يقولون لي إن وجهي يصلح للإذاعة",
    ]),
]

