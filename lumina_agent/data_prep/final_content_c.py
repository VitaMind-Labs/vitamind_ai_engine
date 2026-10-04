"""Authored response library (part C) and Arabic reply-guard patterns.

The library replaces "79 fixed sentences" with components that are composed per
turn: opener + body + one question. Nothing here is free generation. Bodies that
mention the patient's own pattern exist in a `measured` and a `heard` form,
because Lumina may not assert a change it has not measured (response.py, HEARD).
Arabic is written gender-neutral for the patient; the assistant refers to itself
in the masculine, as the existing AR templates do.
All text needs clinician review before use.
"""

OPENERS = {
    "en": ["Thanks for telling me.", "I'm glad you said that.", "I hear you.",
           "Thank you for being open about that.", "I appreciate you putting that into words.",
           "Okay, I'm following.", "Thanks for sharing that with me.",
           "That's useful for me to know.", "I'm listening.", "Got it."],
    "ar": ["شكراً لأنك أخبرتني.", "يسعدني أنك قلت ذلك.", "أسمعك.",
           "أقدّر أنك وضعت هذا في كلمات.", "حسناً، أتابعك.", "شكراً لمشاركتي ذلك.",
           "هذا مفيد لي أن أعرفه.", "أنا أُصغي.", "جيد أن أعرف كيف يسير الأمر.",
           "وصلتني رسالتك."],
}

# strategy -> lang -> {bodies | bodies_measured+bodies_heard, questions}
STRATEGIES = {
    "ACKNOWLEDGE": {
        "en": {"bodies": ["Nothing more is needed from you right now.",
                          "We can leave it there for the moment.", "It's noted."],
               "questions": ["Is there anything you'd like to add?",
                             "What would be most useful to talk about next?",
                             "Is there something on your mind today?"]},
        "ar": {"bodies": ["لا يلزمك شيء آخر الآن.", "يمكننا التوقف هنا للحظة.",
                          "تم تدوين ذلك."],
               "questions": ["هل تود إضافة شيء؟", "ما الذي سيكون أنفع أن نتحدث عنه بعد ذلك؟",
                             "هل هناك ما يشغل بالك اليوم؟"]},
    },
    "CLARIFY": {
        "en": {"bodies": ["I want to make sure I follow you properly.",
                          "I'd like to understand this better before suggesting anything.",
                          "I don't want to guess what you mean."],
               "questions": ["Can you say a little more about what happened?",
                             "What part of this matters most to you right now?",
                             "When did you first notice it?",
                             "How has it been across today so far?",
                             "What would you like to be different?"]},
        "ar": {"bodies": ["أريد أن أتأكد أنني أتابعك جيداً.",
                          "أود أن أفهم هذا أفضل قبل أي اقتراح.",
                          "لا أريد أن أخمّن ما تقصده."],
               "questions": ["هل يمكنك أن تشرح أكثر ما حدث؟", "ما الجزء الأهم بالنسبة لك الآن؟",
                             "متى لاحظت ذلك لأول مرة؟", "كيف كان الأمر خلال اليوم حتى الآن؟",
                             "ما الذي تتمنى أن يكون مختلفاً؟"]},
    },
    "REFLECT": {
        "en": {"bodies": ["That sounds like a lot to carry.", "That sounds tiring.",
                          "It sounds like this has been weighing on you.",
                          "There's a lot in what you just described.",
                          "That sounds hard to sit with."],
               "questions": ["Which part feels heaviest right now?",
                             "What has been the hardest part?",
                             "What do you need most at the moment?",
                             "What would help even a little today?"]},
        "ar": {"bodies": ["يبدو هذا كثيراً للحمل.", "يبدو هذا مرهقاً.",
                          "يبدو أن هذا يثقل عليك.", "في كلامك الكثير من الأشياء.",
                          "يبدو أن هذا صعب."],
               "questions": ["أي جزء يثقل عليك أكثر الآن؟", "ما الأصعب في الأمر؟",
                             "ما الذي تحتاجه أكثر في هذه اللحظة؟",
                             "ما الذي قد يساعد ولو قليلاً اليوم؟"]},
    },
    "NORMALIZE_WITHOUT_MINIMIZING": {
        "en": {"bodies": ["Many people go through something like this, and that doesn't make it lighter for you.",
                          "You're not the only one who has felt this, and it still counts.",
                          "It's common, and it can still be hard."],
               "questions": ["What has this been like for you?",
                             "How has it been affecting your days?"]},
        "ar": {"bodies": ["يمرّ كثيرون بأمر مشابه، وهذا لا يجعله أخف عليك.",
                          "لست وحدك في هذا الشعور، ومع ذلك يبقى مهماً.",
                          "هذا شائع، ومع ذلك قد يكون صعباً."],
               "questions": ["كيف كان هذا الأمر بالنسبة لك؟", "كيف يؤثر على أيامك؟"]},
    },
    "MICRO_ACTION": {
        "en": {"bodies": ["Let's make the first step smaller. {action}",
                          "Starting is often the hardest part, so let's shrink it. {action}",
                          "One small step is enough for now. {action}"],
               "questions": ["Does that feel doable?", "Want to try it and tell me how it goes?",
                             "What would make that step easier?"]},
        "ar": {"bodies": ["لنجعل الخطوة الأولى أصغر. {action}",
                          "البداية غالباً هي الأصعب، فلنصغّرها. {action}",
                          "خطوة صغيرة واحدة تكفي الآن. {action}"],
               "questions": ["هل يبدو هذا ممكناً؟", "هل تود تجربته وإخباري كيف سار؟",
                             "ما الذي سيجعل هذه الخطوة أسهل؟"]},
    },
    "TASK_BREAKDOWN": {
        "en": {"bodies": ["There's a lot here. Let's pick just one piece. {action}",
                          "Let's narrow this to one thing for now. {action}",
                          "We don't have to do it all today. {action}"],
               "questions": ["Which piece feels smallest?", "What would you like to start with?",
                             "What has to happen first?"]},
        "ar": {"bodies": ["الأمور كثيرة، فلنختر جزءاً واحداً فقط. {action}",
                          "لنضيّق الأمر إلى شيء واحد الآن. {action}",
                          "لا يلزم أن ننجز كل شيء اليوم. {action}"],
               "questions": ["أي جزء يبدو الأصغر؟", "بماذا تود أن نبدأ؟",
                             "ما الذي يجب أن يحدث أولاً؟"]},
    },
    "GROUNDING": {
        "en": {"bodies": ["Let's slow things down for a moment. {action}",
                          "Let's pause together before anything else. {action}",
                          "Let's let your body settle first. {action}"],
               "questions": ["How does that feel after a minute?",
                             "Can you tell me what changes, if anything?",
                             "Would you like to try it once more?"]},
        "ar": {"bodies": ["لنُبطئ الإيقاع لحظة. {action}", "لنتوقف معاً قبل أي شيء آخر. {action}",
                          "لندع جسدك يهدأ أولاً. {action}"],
               "questions": ["كيف تشعر بعد دقيقة؟", "هل يمكنك إخباري بما يتغير، إن تغيّر شيء؟",
                             "هل تود تجربته مرة أخرى؟"]},
    },
    "ROUTINE_SUPPORT": {
        "en": {"bodies_measured": ["Your routine looks less settled than usual. One fixed point is enough to start. {action}"],
               "bodies_heard": ["A steadier day can start from one fixed point. {action}",
                                "Routine doesn't have to be big; one anchor helps. {action}"],
               "questions": ["Which part of the day feels least steady?",
                             "What could be the one fixed point?"]},
        "ar": {"bodies_measured": ["يبدو روتينك أقل استقراراً من المعتاد. نقطة ثابتة واحدة تكفي للبداية. {action}"],
               "bodies_heard": ["يمكن أن يبدأ يوم أكثر ثباتاً من نقطة ثابتة واحدة. {action}",
                                "الروتين لا يحتاج أن يكون كبيراً؛ مرساة واحدة تكفي. {action}"],
               "questions": ["أي جزء من اليوم يبدو الأقل ثباتاً؟", "ما النقطة الثابتة الممكنة؟"]},
    },
    "SLEEP_SUPPORT": {
        "en": {"bodies_measured": ["Your sleep has been shorter than your own usual pattern. {action}"],
               "bodies_heard": ["Sleep trouble makes everything else feel heavier. {action}",
                                "Nights like that wear you down. {action}"],
               "questions": ["What has been getting in the way of sleep?", "How did last night go?",
                             "What does your evening look like before bed?"]},
        "ar": {"bodies_measured": ["نومك كان أقصر من نمطك المعتاد. {action}"],
               "bodies_heard": ["اضطراب النوم يجعل كل شيء أثقل. {action}", "ليالٍ كهذه مرهقة. {action}"],
               "questions": ["ما الذي يعيق نومك؟", "كيف مرّت ليلة أمس؟",
                             "كيف يبدو مساؤك قبل النوم؟"]},
    },
    "STATE_MONITORING": {
        "en": {"bodies_measured": ["Compared with your own recent pattern, {observation}. I'm not drawing a conclusion from that. {action}"],
               "bodies_heard": ["Energy is worth paying attention to.",
                                "It's worth keeping an eye on how this moves over the day."],
               "questions": ["How has it moved across your day?", "What have you noticed yourself?"]},
        "ar": {"bodies_measured": ["مقارنةً بنمطك الأخير، {observation}. لا أستنتج شيئاً من ذلك. {action}"],
               "bodies_heard": ["الطاقة تستحق الانتباه.", "يستحق الأمر أن نراقب كيف يتغير خلال اليوم."],
               "questions": ["كيف تغيّرت خلال يومك؟", "ما الذي لاحظته أنت؟"]},
    },
    "ENCOURAGE_SUPPORT_CONNECTION": {
        "en": {"bodies_measured": ["You've had less contact with people than usual. {action}"],
               "bodies_heard": ["Feeling on your own is hard, and I'm glad you said it. {action}",
                                "Connection matters, even in small doses. {action}"],
               "questions": ["Who would be easiest to reach out to?",
                             "What kind of contact feels manageable today?"]},
        "ar": {"bodies_measured": ["تواصلك مع الناس كان أقل من المعتاد. {action}"],
               "bodies_heard": ["الشعور بالوحدة صعب، ويسعدني أنك قلته. {action}",
                                "التواصل مهم ولو بقدر بسيط. {action}"],
               "questions": ["من الأسهل أن تتواصل معه؟", "أي نوع من التواصل يبدو ممكناً اليوم؟"]},
    },
    "FOLLOW_UP": {
        "en": {"bodies": ["Last time we tried something together.",
                          "I wanted to come back to what we tried."],
               "questions": ["How did it go?", "What did you notice afterwards?",
                             "Did anything about it help, even a little?"]},
        "ar": {"bodies": ["في المرة الماضية جرّبنا شيئاً معاً.", "أردت أن أعود إلى ما جرّبناه."],
               "questions": ["كيف سار الأمر؟", "ما الذي لاحظته بعد ذلك؟",
                             "هل ساعد شيء منه ولو قليلاً؟"]},
    },
    "UNKNOWN_SUPPORT": {
        "en": {"bodies": ["I'm not sure I follow yet.", "I may have missed what you meant.",
                          "I don't want to assume."],
               "questions": ["Can you say it another way?", "Can you tell me a bit more?",
                             "What's the main thing you want me to understand?"]},
        "ar": {"bodies": ["لست متأكداً أنني فهمت بعد.", "ربما فاتني ما تقصده.",
                          "لا أريد أن أفترض."],
               "questions": ["هل يمكنك قولها بطريقة أخرى؟", "هل يمكنك أن تخبرني بالمزيد؟",
                             "ما أهم شيء تريدني أن أفهمه؟"]},
    },
}

# Replies for the new meta intents. Each is complete; pick one, avoid the last N.
INTENT_REPLIES = {
    "GREETING": {
        "en": ["Hello. How are you doing today?", "Hi. How has today been?",
               "Hello again. How are things right now?", "Hi. What's on your mind today?",
               "Good to hear from you. How are you?", "Hello. How has your day been so far?"],
        "ar": ["مرحباً. كيف حالك اليوم؟", "أهلاً بك. كيف كان يومك حتى الآن؟",
               "أهلاً من جديد. كيف الأمور الآن؟", "مرحباً. ما الذي يشغل بالك اليوم؟",
               "يسعدني سماعك. كيف حالك؟", "أهلاً. كيف مرّ يومك حتى الآن؟"],
    },
    "GOODBYE": {
        "en": ["Take care. I'm here whenever you want to come back.",
               "Bye for now. Be gentle with yourself today.",
               "See you next time. Thanks for checking in.", "Talk soon. Rest well.",
               "Goodbye for now. I'm here when you need me.",
               "Take care of yourself. Until next time."],
        "ar": ["أنا هنا متى أردت العودة. أتمنى لك يوماً هادئاً.",
               "إلى اللقاء الآن. أتمنى أن يكون يومك لطيفاً.",
               "إلى المرة القادمة. شكراً على تسجيل حالتك.", "إلى لقاء قريب. أتمنى لك راحة جيدة.",
               "مع السلامة الآن. أنا هنا حين تحتاجني.", "أتمنى لك الخير. إلى اللقاء."],
    },
    "THANKS": {
        "en": ["You're welcome. I'm glad it helped.", "Anytime. Tell me how it goes.",
               "You're welcome. Is there anything else on your mind?",
               "Glad to help. We can keep going if you like.",
               "Happy to. What would you like to do next?"],
        "ar": ["عفواً. يسعدني أنه ساعد.", "في أي وقت. أخبرني كيف يسير الأمر.",
               "عفواً. هل هناك شيء آخر يشغل بالك؟", "يسعدني ذلك. يمكننا المتابعة إن أردت.",
               "بكل سرور. ما الذي تود فعله بعد ذلك؟"],
    },
    "ABOUT_BOT": {
        "en": ["I'm Lumina, an AI assistant. I'm not a person and not a doctor.",
               "I'm an AI. I can help you keep track of how you're doing and work through small steps, but I don't diagnose or give advice about medication.",
               "I'm a computer program made to support you between appointments. Your care team is the right place for medical questions.",
               "I can help you check in on sleep, mood and energy, break tasks into smaller steps, and notice patterns over time.",
               "I'm not a real person, but I'm here to listen and help where I can.",
               "Your messages can be part of the summaries your care team sees, so I can't promise they stay only between us."],
        "ar": ["أنا لومينا، مساعد ذكاء اصطناعي. لست شخصاً ولا طبيباً.",
               "أنا ذكاء اصطناعي. أستطيع مساعدتك في متابعة حالك وتقسيم الخطوات، لكنني لا أشخّص ولا أنصح بشأن الدواء.",
               "أنا برنامج حاسوبي صُمم لدعمك بين المواعيد. فريق رعايتك هو المكان المناسب للأسئلة الطبية.",
               "أستطيع مساعدتك في متابعة النوم والمزاج والطاقة، وتقسيم المهام، وملاحظة الأنماط مع الوقت.",
               "لست شخصاً حقيقياً، لكنني هنا لأُصغي وأساعد قدر ما أستطيع.",
               "قد تكون رسائلك جزءاً من الملخصات التي يراها فريق رعايتك، لذلك لا أستطيع أن أعدك بأنها تبقى بيننا فقط."],
        "needs_product_review": "The last line in each language makes a statement about data visibility; "
                                "confirm it against the real product before shipping.",
    },
    "BOT_FEEDBACK": {
        "en": ["You're right, that wasn't helpful. Let me try again: what's the main thing on your mind right now?",
               "Sorry, I've been repeating myself. Tell me in your own words what you need most.",
               "I hear that I'm not getting it. What should I have understood?",
               "Fair point. Let me slow down and listen. What happened?",
               "I'm sorry that missed the mark. What would be useful right now?",
               "Thanks for telling me. I'll try a different approach. What would you like me to do differently?"],
        "ar": ["معك حق، لم يكن ذلك مفيداً. لنحاول من جديد: ما أهم شيء في بالك الآن؟",
               "آسف، كنت أكرر نفسي. أخبرني بكلماتك ما الذي تحتاجه أكثر.",
               "أسمع أنني لا أفهمك. ما الذي كان يجب أن أفهمه؟",
               "ملاحظة في محلها. لنُبطئ قليلاً وأُصغي. ماذا حدث؟",
               "آسف أن ذلك لم يصب الهدف. ما الذي سيفيد الآن؟",
               "شكراً لإخباري. سأجرب طريقة مختلفة. ماذا تود أن أفعل بشكل مختلف؟"],
    },
    "DISENGAGE": {
        "en": ["That's okay. We don't have to talk about it.", "Understood. We can leave that for now.",
               "No pressure at all. We can talk about something else, or just pause.",
               "Thanks for telling me. I'll step back from that topic.",
               "That's fine. I'm here if you want to come back to it.",
               "Okay. What would you rather do right now?"],
        "ar": ["لا بأس. لا يلزمنا أن نتحدث عن ذلك.", "مفهوم. يمكننا ترك ذلك الآن.",
               "لا ضغط أبداً. يمكننا الحديث عن شيء آخر أو التوقف قليلاً.",
               "شكراً لإخباري. سأبتعد عن هذا الموضوع.", "لا مشكلة. أنا هنا إن أردت العودة إليه.",
               "حسناً. ما الذي تفضّل أن نفعله الآن؟"],
    },
}

# Track overlays: moves to avoid, plus extra lines a track may add to a strategy.
TRACK_OVERLAYS = {
    "SCHIZOPHRENIA": {
        "avoid": ["agreeing that a described belief or experience is real",
                  "telling the person it is not real or that they are imagining it",
                  "asking for detail about the content of a delusional belief",
                  "open-ended guided imagery or meditation unless a clinician approved it"],
        "REFLECT": {
            "en": {"bodies": ["That sounds frightening to live with.", "That sounds very unsettling.",
                              "It sounds like this feels very real to you."],
                   "questions": ["How has sleep been over the last few days?",
                                 "Is there someone you trust nearby?",
                                 "How is your body feeling right now?"]},
            "ar": {"bodies": ["يبدو هذا مخيفاً للعيش معه.", "يبدو هذا مقلقاً جداً.",
                              "يبدو أن هذا حقيقي جداً بالنسبة لك."],
                   "questions": ["كيف كان نومك خلال الأيام الماضية؟", "هل هناك شخص تثق به قريب منك؟",
                                 "كيف يشعر جسدك الآن؟"]}},
    },
    "BIPOLAR": {
        "avoid": ["praising unusually high energy or very little sleep as positive",
                  "encouraging new large plans, purchases or commitments",
                  "naming an episode or saying the person seems manic or hypomanic"],
        "MICRO_ACTION": {
            "en": {"bodies": ["Nothing has to be decided tonight. {action}",
                              "Big ideas can wait a day; one calm step is enough. {action}"],
                   "questions": ["What would a quiet evening look like for you?"]},
            "ar": {"bodies": ["لا شيء يجب أن يُحسم الليلة. {action}",
                              "الأفكار الكبيرة تستطيع أن تنتظر يوماً؛ خطوة هادئة واحدة تكفي. {action}"],
                   "questions": ["كيف يبدو مساء هادئ بالنسبة لك؟"]}},
    },
    "ADHD": {
        "avoid": ["long lists or several instructions at once",
                  "shaming language such as 'just focus' or 'you should have'",
                  "open-ended planning questions with no small first step"],
        "TASK_BREAKDOWN": {
            "en": {"bodies": ["Let's do one tiny thing, then stop and check. {action}",
                              "Just one step, nothing else yet. {action}"],
                   "questions": ["What is the smallest version of that step?"]},
            "ar": {"bodies": ["لنفعل شيئاً صغيراً واحداً ثم نتوقف ونراجع. {action}",
                              "خطوة واحدة فقط، ولا شيء آخر بعد. {action}"],
                   "questions": ["ما أصغر نسخة من هذه الخطوة؟"]}},
    },
}

# ---------------------------------------------------------------------------
# Arabic reply guard. response.py's PROHIBITED_PATTERNS are English-only, so an
# Arabic reply is currently never checked. Names match the English rules so the
# same `prohibited` lists apply; ar_guard_patch.extend() merges them by name.
# ---------------------------------------------------------------------------
AR_PROHIBITED = {
    "diagnosis": (
        r"(?:أنت|انت|إنت)\s+(?:مصاب|مصابة|تعاني|تعانين)\s+(?:ب|من)?\s*"
        r"(?:ال)?(?:فصام|اكتئاب|ثنائي القطب|هوس|ذهان|اضطراب|وسواس)"
        r"|(?:هذه|هذي|هاي)\s+(?:هي\s+)?(?:علامات|أعراض)\s+(?:كلاسيكية\s+)?"
        r"(?:لل|ل\s*(?:ال)?|على\s*(?:ال)?)\s*(?:اكتئاب|فصام|ثنائي القطب|هوس|ذهان)"
        r"|(?:لديك|عندك)\s+(?:اضطراب|اكتئاب|فصام|ذهان|هوس)"),
    "episode_naming": r"نوبة\s+(?:هوس|اكتئاب|ذهان)|حلقة\s+(?:هوس|اكتئاب|ذهان)",
    "medication_advice": (
        r"(?:توقف|أوقف|اترك|قلل|خفف|زد|ضاعف|تخطَّ|تخط)\s*(?:عن)?\s*"
        r"(?:ال)?(?:دواء|أدوية|ادوية|جرعة|الجرعة|حبوب|حبة)"
        r"|(?:خذ|تناول)\s+(?:حبة|حبتين|جرعة)"),
    "mania_claim": r"(?:أنت|يبدو أنك|تبدو)\s+(?:في\s+)?(?:حالة\s+)?(?:هوس|هوسي|مهووس)",
    "validate_unverified_interpretation": (
        r"(?:نعم|فعلاً|فعلا)[،, ]*(?:هم|هو|هي|الناس)\s+"
        r"(?:يراقبونك|يراقبك|يتجسسون|يتجسس|يتتبعونك|يتتبعك)"),
    "confront_or_dispute_belief": (
        r"(?:هذا|ذلك)\s+(?:غير حقيقي|ليس حقيقي|ليس حقيقياً|لم يحدث)"
        r"|أنت\s+(?:تتخيل|تتوهم)"),
    "treatment_replacement": r"(?:بدلاً|بدلا)\s+(?:من|عن)\s+(?:ال)?(?:علاج|دواء|طبيب)",
}

# Unit tests for the Arabic guard: (text, rule that must fire or None)
AR_GUARD_TESTS = [
    ("هذه علامات كلاسيكية للاكتئاب", "diagnosis"),
    ("أنت تعاني من اضطراب ثنائي القطب", "diagnosis"),
    ("لديك اضطراب في الشخصية", "diagnosis"),
    ("هذه أعراض للفصام", "diagnosis"),
    ("هذه علامات على الاكتئاب", "diagnosis"),
    ("توقف عن الدواء فوراً", "medication_advice"),
    ("زد الجرعة قليلاً", "medication_advice"),
    ("يبدو أنك في حالة هوس", "mania_claim"),
    ("نعم، هم يراقبونك فعلاً", "validate_unverified_interpretation"),
    ("أنت تتخيل هذا", "confront_or_dispute_belief"),
    ("هذا غير حقيقي", "confront_or_dispute_belief"),
    ("بدلاً من العلاج جرّب هذا", "treatment_replacement"),
    ("يبدو هذا مرهقاً.", None),
    ("ما الذي يثقل عليك أكثر الآن؟", None),
    ("يمكنك أن تسأل طبيبك عن أي سؤال يخص دوائك", None),
    ("نومك كان أقصر من نمطك المعتاد.", None),
]
