"""Explicit interview slots. Labels are internal; EN/AR text is patient-facing."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Question:
    key: str
    en: str
    ar: str
    target: tuple | None = None
    kind: str = 'boolean'
    group: str = 'GENERAL'

    def text(self, language):
        return self.ar if language == 'ar' else self.en

QUESTIONS = [
 Question('concern', 'What has been troubling you recently? Please describe your own experience in your words.', 'ما الذي يزعجك مؤخراً؟ صف تجربتك الشخصية بكلماتك.', kind='narrative'),
 Question('attention', 'Do you often struggle to keep your attention on a task, even when you want to focus?', 'هل تجد صعوبة متكررة في إبقاء انتباهك على مهمة حتى عندما تريد التركيز؟', ('attention','distractibility'), group='ADHD'),
 Question('sleep_need', 'During a change in your energy, have you needed much less sleep than usual and still felt rested? Or were you tired from losing sleep?', 'خلال تغير طاقتك، هل احتجت نوماً أقل بكثير من المعتاد مع بقاء نشاطك، أم كنت متعباً بسبب قلة النوم؟', ('sleep','decreased_need_for_sleep'), kind='sleep', group='BIPOLAR_SPECTRUM'),
 Question('voices', 'Have you heard voices that other people around you could not hear?', 'هل سمعت أصواتاً لم يستطع الأشخاص حولك سماعها؟', ('psychosis','auditory_perceptual_experience'), group='PSYCHOSIS_SPECTRUM'),
 Question('beliefs', 'Have you felt watched or threatened in a way that people you trust do not share? Please describe what happened without assuming an explanation.', 'هل شعرت بالمراقبة أو التهديد بطريقة لا يشاركك فيها أشخاص تثق بهم؟ صف ما حدث دون افتراض تفسير.', ('psychosis','persecutory_ideas'), group='PSYCHOSIS_SPECTRUM'),
 Question('childhood', 'Were these attention difficulties already present during childhood?', 'هل كانت صعوبات الانتباه هذه موجودة بالفعل في طفولتك؟', ('developmental_history','childhood_onset'), group='ADHD'),
 Question('settings', 'Do these difficulties affect more than one setting, such as both home and work or school?', 'هل تؤثر هذه الصعوبات في أكثر من مكان، مثل المنزل والعمل أو المدرسة؟', group='ADHD'),
 Question('energy', 'Have there been periods when your energy or mood was clearly higher than your usual self?', 'هل مررت بفترات كانت فيها طاقتك أو حالتك المزاجية أعلى بوضوح من المعتاد بالنسبة لك؟', ('mood','elevated_mood'), group='BIPOLAR_SPECTRUM'),
 Question('episodes', 'Do these changes happen in distinct periods that differ from your usual self?', 'هل تحدث هذه التغيرات في فترات واضحة تختلف فيها عن حالتك المعتادة؟', ('episode_history','episodic_pattern'), group='BIPOLAR_SPECTRUM'),
 Question('duration', 'How long have these difficulties been happening? If they come in periods, how long does one period last?', 'منذ متى تحدث هذه الصعوبات؟ وإذا كانت تأتي في فترات، فكم تدوم الفترة الواحدة؟', kind='duration'),
 Question('impairment', 'Have these experiences caused difficulties at work, school, home or in relationships?', 'هل سببت لك هذه التجارب صعوبات في العمل أو الدراسة أو المنزل أو العلاقات؟'),
 Question('confounders', 'Could recent changes in medicines, alcohol or other substances, sleep schedule, or a physical illness be involved? You can say no or describe a change.', 'هل توجد تغيرات حديثة في الأدوية أو الكحول أو مواد أخرى أو مواعيد النوم، أو مرض جسدي قد يكون له علاقة؟ يمكنك قول لا أو وصف التغير.', kind='context'),
]
BY_KEY = {q.key:q for q in QUESTIONS}
