"""Fixed development templates. Require clinical and Arabic-language review."""
TEMPLATES = {
    "en":{
        "none":"Thank you for making a little space for yourself today.",
        "low":"It sounds like today took some energy. Would you like to name one thing you need right now?",
        "moderate":"It sounds heavy right now. Would it help to pause and notice a few things around you, or reach out to someone you trust?",
        "moderate_flagged":"This sounds difficult. Are you feeling safe right now? You can keep writing, or reach out to someone you trust.",
        "high":"I'm glad you wrote this down. If you might act on these thoughts or cannot stay safe, contact local emergency services now. If you can, ask someone you trust to stay with you. You can keep writing here.",
        "other":"That sounds worrying for someone you care about. If they may be in immediate danger, contact local emergency services. You can also encourage them to reach a trusted person or their clinician.",
        "paranoia":"That sounds unsettling. Would it help to notice where your feet meet the floor and name a few things you can see? You can also reach someone you trust.",
        "unavailable":"I couldn't complete the journal check. You can keep writing. If you feel unsafe, contact local emergency services or someone you trust.",
    },
    "ar":{
        "none":"شكرًا لأنك خصصت لنفسك مساحة للكتابة اليوم.",
        "low":"يبدو أن اليوم استهلك بعض طاقتك. هل تحب أن تذكر شيئًا واحدًا تحتاجه الآن؟",
        "moderate":"يبدو أن الوضع ثقيل الآن. هل يساعدك التوقف قليلًا وملاحظة أشياء حولك، أو التواصل مع شخص تثق به؟",
        "moderate_flagged":"يبدو هذا صعبًا. هل تشعر أنك بأمان الآن؟ يمكنك متابعة الكتابة أو التواصل مع شخص تثق به.",
        "high":"شكرًا لأنك كتبت ما تمر به. إذا كنت قد تتصرف بناءً على هذه الأفكار أو لا تستطيع البقاء بأمان، فاتصل بخدمات الطوارئ المحلية الآن. إن أمكن، اطلب من شخص تثق به أن يبقى معك. يمكنك متابعة الكتابة هنا.",
        "other":"يبدو أنك قلق على شخص تهتم به. إذا كان في خطر فوري، اتصل بخدمات الطوارئ المحلية. يمكنك أيضًا تشجيعه على التواصل مع شخص يثق به أو مع طبيبه.",
        "paranoia":"يبدو هذا مزعجًا. هل يساعدك أن تشعر بقدميك على الأرض وتذكر بعض الأشياء التي تراها؟ يمكنك أيضًا التواصل مع شخص تثق به.",
        "unavailable":"لم أتمكن من إكمال تحليل اليوميات. يمكنك متابعة الكتابة. إذا كنت لا تشعر بالأمان، تواصل مع خدمات الطوارئ المحلية أو شخص تثق به.",
    }
}

def response(tier,lang,categories,subject,degraded=False):
    key="unavailable" if degraded else "other" if subject=="other" and tier in ("moderate","moderate_flagged","high") else "paranoia" if "paranoia" in categories and tier in ("moderate","moderate_flagged") else tier
    return {"template_id":f"journal-v1.{lang}.{key}","text":TEMPLATES[lang][key],"language":lang,"clinician_reviewed":False}
