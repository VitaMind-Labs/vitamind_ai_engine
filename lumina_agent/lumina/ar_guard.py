"""Arabic reply guard.

response.py checks rendered text against English-only regexes, so an Arabic reply
is never checked. This module adds Arabic patterns under the SAME rule names, so
every existing `prohibited` tuple (track rules included) now covers both languages.

Wire-up (once, in lumina/response.py, right after PROHIBITED_PATTERNS is defined):

    from .ar_guard import extend as _extend_arabic_guard
    _extend_arabic_guard(PROHIBITED_PATTERNS)

Patterns were written without a native clinical reviewer; review before relying on them.
"""
import re

AR_PROHIBITED = {
    "diagnosis": "(?:أنت|انت|إنت)\\s+(?:مصاب|مصابة|تعاني|تعانين)\\s+(?:ب|من)?\\s*(?:ال)?(?:فصام|اكتئاب|ثنائي القطب|هوس|ذهان|اضطراب|وسواس)|(?:هذه|هذي|هاي)\\s+(?:هي\\s+)?(?:علامات|أعراض)\\s+(?:كلاسيكية\\s+)?(?:لل|ل\\s*(?:ال)?|على\\s*(?:ال)?)\\s*(?:اكتئاب|فصام|ثنائي القطب|هوس|ذهان)|(?:لديك|عندك)\\s+(?:اضطراب|اكتئاب|فصام|ذهان|هوس)",
    "episode_naming": "نوبة\\s+(?:هوس|اكتئاب|ذهان)|حلقة\\s+(?:هوس|اكتئاب|ذهان)",
    "medication_advice": "(?:توقف|أوقف|اترك|قلل|خفف|زد|ضاعف|تخطَّ|تخط)\\s*(?:عن)?\\s*(?:ال)?(?:دواء|أدوية|ادوية|جرعة|الجرعة|حبوب|حبة)|(?:خذ|تناول)\\s+(?:حبة|حبتين|جرعة)",
    "mania_claim": "(?:أنت|يبدو أنك|تبدو)\\s+(?:في\\s+)?(?:حالة\\s+)?(?:هوس|هوسي|مهووس)",
    "validate_unverified_interpretation": "(?:نعم|فعلاً|فعلا)[،, ]*(?:هم|هو|هي|الناس)\\s+(?:يراقبونك|يراقبك|يتجسسون|يتجسس|يتتبعونك|يتتبعك)",
    "confront_or_dispute_belief": "(?:هذا|ذلك)\\s+(?:غير حقيقي|ليس حقيقي|ليس حقيقياً|لم يحدث)|أنت\\s+(?:تتخيل|تتوهم)",
    "treatment_replacement": "(?:بدلاً|بدلا)\\s+(?:من|عن)\\s+(?:ال)?(?:علاج|دواء|طبيب)"
}


def extend(prohibited_patterns):
    for rule, arabic in AR_PROHIBITED.items():
        existing = prohibited_patterns.get(rule)
        if existing is None:
            prohibited_patterns[rule] = re.compile(arabic)
        else:
            prohibited_patterns[rule] = re.compile(existing.pattern + "|" + arabic,
                                                   existing.flags | re.UNICODE)
    return prohibited_patterns
