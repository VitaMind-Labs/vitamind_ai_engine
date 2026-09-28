"""Deterministic, auditable extraction of high-signal prototype features."""

from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class FeatureRule:
    domain: str
    feature: str
    present: tuple[str, ...]
    absent: tuple[str, ...] = ()
    confidence: float = 0.75


RULES = (
    # ── Attention / ADHD ──────────────────────────────────────────────────
    # Priorite: hyperactivity_impulsivity, thought_disorganization, social_withdrawal couverts en AR.
    FeatureRule("developmental_history", "childhood_onset", (
        "since school", "since childhood", "as a child", "at school",
        "منذ المدرسة", "منذ الطفولة", "منذ الصغر", "في المدرسة منذ",
    )),
    FeatureRule("attention", "distractibility", (
        "distracted", "can't focus", "cannot focus", "lose focus", "attention drifts", "lose track", "hard to concentrate", "trouble concentrating", "difficulty focusing",
        # AR
        "مشتت", "لا أستطيع التركيز", "لا استطيع التركيز", "أفقد التركيز", "أفقد تركيزي", "صعوبة التركيز", "تشتت الانتباه",
    )),
    FeatureRule("attention", "forgetfulness", (
        "forget things", "forgetful", "forget appointments", "forget what", "lose things", "losing things",
        # AR
        "أنسى", "نسيان", "أنسى المواعيد", "أنسى الأشياء", "أضيع الأشياء", "أنسى ما",
    )),
    FeatureRule("attention", "procrastination", (
        "procrastinate", "put things off",
        # AR
        "أسوف", "أؤجل", "المماطلة", "أؤجل المهام", "أسوف المهام",
    )),
    FeatureRule("attention", "hyperactivity_impulsivity", (
        "can't sit still", "cannot sit still", "restless", "fidget", "act without thinking", "impulsive decisions",
        "لا أستطيع الجلوس", "لا استطيع الجلوس", "أتململ", "أتصرف بدون تفكير", "قرارات اندفاعية", "فرط الحركة", "لا أستطيع البقاء ساكنا",
    )),

    # ── Sleep ─────────────────────────────────────────────────────────────
    FeatureRule("sleep", "reduced_sleep", (
        "sleep 3", "sleep less", "barely sleep", "little sleep", "three or four hours", "two to four hours",
        "أنام 3 ساعات", "أنام أقل", "قليل النوم", "ساعتين نوم", "ساعتين", "ثلاث أو أربع ساعات", "ساعات قليلة", "بالكاد أنام", "أنام ساعتين",
    )),
    FeatureRule("sleep", "decreased_need_for_sleep", (
        "still felt full of energy", "without feeling tired",
        "don't need sleep", "don't feel tired", "didn't feel tired",
        "no fatigue", "feel like i could run",
        "still feel full of energy", "wake up full of energy",
        "بدون الشعور بالتعب", "بدون تعب", "من غير تعب", "بلا تعب", "لا أحتاج للنوم", "لا أشعر بالتعب", "لم أشعر بالتعب",
        "لا اشعر بالتعب", "ما زلت مليان طاقة", "أستيقظ مليان طاقة", "بدون إرهاق", "من غير إرهاق",
    ), ("exhausted", "tired", "مرهق", "متعب", "تعب", "إرهاق", "ارهاق"), 0.9),

    # ── Mood / Bipolar ────────────────────────────────────────────────────
    FeatureRule("mood", "elevated_mood", (
        "unusually energized", "full of energy", "high mood",
        "very happy", "on top of the world", "felt amazing",
        "unusually high", "best i've ever felt", "nothing can stop me",
        "مبتهج بشكل غير عادي", "مليان طاقة", "مليئة بالطاقة", "مزاج عالي", "سعيد جدا", "أشعر أنني في القمة", "أشعر أنني مذهل", "طاقة عالية",
    )),
    FeatureRule("mood", "grandiosity", (
        "smartest person", "could do anything", "invincible",
        "special abilities", "i am the best", "i can do anything",
        "felt like a god", "superior to everyone",
        "عبقري", "أذكى شخص", "أستطيع فعل أي شيء", "لا يقهر", "قدرات خارقة", "أنا الأفضل", "أنا الأذكى", "أشعر أنني إله", "متفوق على الجميع",
    )),
    FeatureRule("mood", "racing_thoughts", (
        "racing thoughts", "mind is racing", "thoughts racing", "mind was going so fast",
        # AR
        "أفكاري تتسارع", "عقلي يتسارع", "تسارع الأفكار", "أفكاري تتدفق", "عقلي لا يتوقف",
    )),
    FeatureRule("mood", "pressured_speech", (
        "talk too fast", "can't stop talking", "people tell me i talk", "pressured speech", "talking nonstop",
        "أتكلم بسرعة", "اتكلم بسرعة", "كلامي سريع", "لا أستطيع التوقف عن الكلام", "يقولون أنني أتحدث بسرعة", "أتحدث بسرعة كبيرة", "أتحدث بدون توقف",
    )),
    FeatureRule("mood", "flight_of_ideas", (
        "jump from topic", "flight of ideas", "mind going so fast", "one idea to the next", "topic to topic",
        # AR
        "أقفز من موضوع لآخر", "تطاير الأفكار", "من فكرة لأخرى", "من موضوع لموضوع", "أفكاري تقفز",
    )),
    FeatureRule("mood", "impulsive_spending", (
        "spent too much", "spending spree", "spent a lot", "bought things i didn't need", "bought things i did not need",
        # AR
        "أنفقت كثيرا", "نوبة إنفاق", "أنفقت الكثير", "اشتريت أشياء لا أحتاجها", "اشتريت اشياء لا احتاجها", "إنفاق اندفاعي",
    )),
    FeatureRule("mood", "increased_goal_directed_activity", (
        "started many projects", "many projects", "take on too much", "took on too much",
        # AR
        "بدأت مشاريع كثيرة", "مشاريع كثيرة", "أتحمل الكثير", "أخذت على عاتقي الكثير", "بدأت الكثير من المشاريع",
    )),

    # ── Episode history ───────────────────────────────────────────────────
    FeatureRule("episode_history", "episodic_pattern", (
        "come and go", "periods of", "on and off", "cycles", "episodes", "this pattern repeats",
        "تأتي وتذهب", "فترات", "دورات", "نوبات", "هذا النمط يتكرر", "تتكرر", "بشكل دوري",
    )),
    FeatureRule("episode_history", "depression_alternation", (
        "crash into", "deep depression", "then i crash",
        "followed by depression", "can't get out of bed",
        "cannot get out of bed", "feel worthless", "feel empty",
        "انهيار", "اكتئاب عميق", "ثم انهار", "لا أستطيع النهوض من السرير", "لا استطيع النهوض", "أشعر بعدم القيمة", "أشعر بالفراغ", "لا أستطيع الخروج من السرير",
    )),

    # ── Psychosis ─────────────────────────────────────────────────────────
    FeatureRule("psychosis", "auditory_perceptual_experience", (
        "hear voices", "hearing voices", "voices when alone", "voices told me", "voices telling me", "whispering", "voices in my head",
        # AR
        "أسمع أصواتا", "أسمع أصوات", "سماع أصوات", "أصوات عندما أكون وحيدا", "أصوات تخبرني", "همس", "أصوات في رأسي", "أسمع اصواتا",
    ), ("never heard voices", "لم أسمع أصواتا", "لم اسمع اصواتا")),
    FeatureRule("psychosis", "persecutory_ideas", (
        "people are watching me", "being watched", "following me", "out to get me", "plotting against me", "someone is following me", "paranoia", "paranoid", "talking about me",
        # AR
        "الناس يراقبونني", "يتم مراقبتي", "يتبعونني", "يريدون إيذائي", "يتآمرون ضدي", "جنون الارتياب", "أشعر أنني مراقب", "يتحدثون عني",
    )),
    FeatureRule("psychosis", "thought_disorganization", (
        "can't organize my thoughts", "thoughts are jumbled", "nothing makes sense", "confused all the time", "thought disorder",
        "لا أستطيع ترتيب أفكاري", "لا استطيع ترتيب افكاري", "أفكاري مشوشة", "افكاري مشوشة", "لا شيء منطقي", "مشوش طوال الوقت", "اضطراب الفكر", "أفكاري مبعثرة",
    )),
    FeatureRule("psychosis", "social_withdrawal", (
        "stopped seeing friends", "isolate myself", "withdraw from people", "don't want to see anyone", "stay in my room",
        "توقفت عن رؤية الأصدقاء", "أنعزل", "انعزل", "انسحبت من الناس", "انسحبت من الأصدقاء", "لا أريد رؤية أحد", "أبقى في غرفتي", "أنطوي على نفسي",
    )),
)



from .language import normalize, patient_clauses, negated_before, occurrences

class FeatureExtractor:
    def extract_into_state(self, state, text, chapter=None, message_id=None):
        extracted = []
        segments = list(patient_clauses(text))
        for rule in RULES:
            polarities = set()
            for clause, own in segments:
                if not own: continue
                positives = [(p, m) for p in rule.present for m in occurrences(clause, p)]
                negatives = [(p, m) for p in rule.absent for m in occurrences(clause, p)]
                for phrase, match in positives:
                    polarity = 'absent' if negated_before(clause, match.start()) else 'present'
                    polarities.add(polarity)
                # An explicit absence phrase has meaning even without a positive trigger.
                for phrase, match in negatives:
                    if any(a.start() <= match.start() and a.end() >= match.end() for _, a in positives): continue
                    if not negated_before(clause, match.start()): polarities.add('absent')
            # A statement of energy alone cannot establish reduced sleep need.
            if rule.feature == 'decreased_need_for_sleep' and 'present' in polarities:
                sleep_reduced = state.get_status('sleep', 'reduced_sleep').value == 'present'
                explicit = any(normalize(p) in normalize(text) for p in ("don't need sleep", 'لا احتاج للنوم'))
                if not sleep_reduced and not explicit: polarities.discard('present')
            for polarity in sorted(polarities):
                extracted.append(state.add_observation(rule.domain, rule.feature, polarity, rule.confidence, text[:500], chapter, message_id))
        return extracted
