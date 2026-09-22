"""Deterministic, auditable extraction of high-signal prototype features."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .patient_state import PatientState


@dataclass(frozen=True)
class FeatureRule:
    domain: str
    feature: str
    present: tuple[str, ...]
    absent: tuple[str, ...] = ()
    confidence: float = 0.75


# Negation tokens: if one of these appears within 5 words BEFORE an absent
# trigger, the absent trigger is contextually negated and should be ignored.
# Tache 3: ajout des declencheurs arabes (MSA) en priorite securite.
_NEGATION_WINDOW = re.compile(
    r"(?:don't|dont|do not|didn't|didnt|did not|doesn't|doesnt|does not|"
    r"not|never|without|no|"
    # AR negation (MSA) — precede fatigue words like من غير تعب / بدون تعب
    r"لا|لم|لن|ليس|لست|بدون|من غير|بلا)\s",
    re.IGNORECASE,
)


def _absent_is_contextually_negated(normalized: str, absent_phrase: str) -> bool:
    """Return True when the absent trigger appears inside a negation context.

    E.g. "don't feel tired" contains "tired" but the negation means the user
    is saying they are NOT tired — so the absent trigger should not fire.
    """
    idx = normalized.find(absent_phrase)
    if idx < 0:
        return False
    # Look at the 60 characters before the absent phrase for negation words
    window_start = max(0, idx - 60)
    window = normalized[window_start:idx]
    return bool(_NEGATION_WINDOW.search(window))


RULES = (
    # ── Attention / ADHD ──────────────────────────────────────────────────
    # Tache 3: regles AR ajoutees depuis arabic_raw (verifiees MSA), sans traduction inventee.
    # Priorite: hyperactivity_impulsivity, thought_disorganization, social_withdrawal couverts en AR.
    FeatureRule("developmental_history", "childhood_onset", (
        "since school", "since childhood", "as a child", "at school",
        # AR MSA + dialectal-source (arabic_raw: منذ المدرسة 12 hits)
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
        # AR — arabic_raw: لا أستطيع الجلوس 9, أنعزل 40 mais ici hyperactivite
        "لا أستطيع الجلوس", "لا استطيع الجلوس", "قلق", "أتململ", "أتصرف بدون تفكير", "قرارات اندفاعية", "فرط الحركة", "لا أستطيع البقاء ساكنا",
    )),

    # ── Sleep ─────────────────────────────────────────────────────────────
    FeatureRule("sleep", "reduced_sleep", (
        "sleep 3", "sleep less", "barely sleep", "little sleep", "three or four hours", "two to four hours",
        # AR — raw: قليل النوم 8, ساعتين 69 (69 occurrences de ساعتين)
        "أنام 3 ساعات", "أنام أقل", "قليل النوم", "قليلا", "ساعتين نوم", "ساعتين", "ثلاث أو أربع ساعات", "ساعات قليلة", "بالكاد أنام", "أنام ساعتين",
    )),
    FeatureRule("sleep", "decreased_need_for_sleep", (
        "still felt full of energy", "without feeling tired",
        "don't need sleep", "don't feel tired", "didn't feel tired",
        "no fatigue", "felt rested", "feel like i could run",
        "still feel full of energy", "wake up full of energy",
        # AR — raw: بدون تعب 2, مليان طاقة 2, plus MSA standard
        "بدون الشعور بالتعب", "بدون تعب", "من غير تعب", "بلا تعب", "لا أحتاج للنوم", "لا أشعر بالتعب", "لم أشعر بالتعب",
        "لا اشعر بالتعب", "ما زلت مليان طاقة", "أستيقظ مليان طاقة", "أشعر بالنشاط", "بدون إرهاق", "من غير إرهاق",
    ), ("exhausted", "tired", "مرهق", "متعب", "تعب", "إرهاق", "ارهاق"), 0.9),

    # ── Mood / Bipolar ────────────────────────────────────────────────────
    FeatureRule("mood", "elevated_mood", (
        "unusually energized", "full of energy", "high mood",
        "very happy", "on top of the world", "felt amazing",
        "unusually high", "best i've ever felt", "nothing can stop me",
        # AR — raw: مليان طاقة 2
        "مبتهج بشكل غير عادي", "مليان طاقة", "مليئة بالطاقة", "مزاج عالي", "سعيد جدا", "أشعر أنني في القمة", "أشعر أنني مذهل", "طاقة عالية",
    )),
    FeatureRule("mood", "grandiosity", (
        "smartest person", "could do anything", "invincible",
        "special abilities", "i am the best", "i can do anything",
        "felt like a god", "superior to everyone",
        # AR — raw: عبقري 7
        "عبقري", "أذكى شخص", "أستطيع فعل أي شيء", "لا يقهر", "قدرات خارقة", "أنا الأفضل", "أنا الأذكى", "أشعر أنني إله", "متفوق على الجميع",
    )),
    FeatureRule("mood", "racing_thoughts", (
        "racing thoughts", "mind is racing", "thoughts racing", "mind was going so fast",
        # AR
        "أفكاري تتسارع", "عقلي يتسارع", "تسارع الأفكار", "أفكاري تتدفق", "عقلي لا يتوقف",
    )),
    FeatureRule("mood", "pressured_speech", (
        "talk too fast", "can't stop talking", "people tell me i talk", "pressured speech", "talking nonstop",
        # AR — raw: أتكلم بسرعة 5, كلامي سريع 2
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
        # AR — raw: نوبات 1421 (tres frequent)
        "تأتي وتذهب", "فترات", "دورات", "نوبات", "هذا النمط يتكرر", "تتكرر", "بشكل دوري",
    )),
    FeatureRule("episode_history", "depression_alternation", (
        "crash into", "deep depression", "then i crash",
        "followed by depression", "can't get out of bed",
        "cannot get out of bed", "feel worthless", "feel empty",
        # AR — raw: لا أستطيع النهوض 8, افكاري مشوشة 9 etc.
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
        # AR — raw: أفكاري مشوشة 9
        "لا أستطيع ترتيب أفكاري", "لا استطيع ترتيب افكاري", "أفكاري مشوشة", "افكاري مشوشة", "لا شيء منطقي", "مشوش طوال الوقت", "اضطراب الفكر", "أفكاري مبعثرة",
    )),
    FeatureRule("psychosis", "social_withdrawal", (
        "stopped seeing friends", "isolate myself", "withdraw from people", "don't want to see anyone", "stay in my room",
        # AR — raw: أنعزل 40, انسحبت 13, لا أستطيع الجلوس 9 (overlap)
        "توقفت عن رؤية الأصدقاء", "أنعزل", "انعزل", "انسحبت من الناس", "انسحبت من الأصدقاء", "لا أريد رؤية أحد", "أبقى في غرفتي", "أنطوي على نفسي",
    )),
)


class FeatureExtractor:
    def extract_into_state(self, state: PatientState, text: str, chapter: str | None = None, message_id: str | None = None) -> list:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        extracted = []
        for rule in RULES:
            present_hit = next((phrase for phrase in rule.present if phrase in normalized), None)
            absent_hit = next((phrase for phrase in rule.absent if phrase in normalized), None)
            if not present_hit and not absent_hit:
                continue
            # ── Substring-safe absent logic ───────────────────────────────
            # If the absent phrase appears inside a present phrase or inside
            # a negation context ("don't feel tired"), discard the absent hit.
            if absent_hit:
                if present_hit and absent_hit in present_hit:
                    absent_hit = None
                elif _absent_is_contextually_negated(normalized, absent_hit):
                    absent_hit = None
            polarity = "absent" if absent_hit and not present_hit else "present"
            excerpt = text[:240]
            extracted.append(state.add_observation(rule.domain, rule.feature, polarity, rule.confidence, excerpt, chapter, message_id))
        return extracted