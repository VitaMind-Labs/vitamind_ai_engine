"""Readable report rendering shared by terminal and API, with no ML dependency."""
from dataclasses import asdict
from .interview import DEFAULT_SESSION_QUESTIONS

LABELS={
 'ADHD':('ADHD','اضطراب نقص الانتباه وفرط الحركة'),
 'BIPOLAR_SPECTRUM':('Bipolar-spectrum patterns','أنماط طيف الاضطراب ثنائي القطب'),
 'PSYCHOSIS_SPECTRUM':('Psychosis-related experiences','تجارب مرتبطة بأعراض ذهانية'),
 'distractibility':('Difficulty maintaining attention','صعوبة الحفاظ على الانتباه'),
 'forgetfulness':('Forgetfulness','النسيان'),
 'procrastination':('Putting tasks off','تأجيل المهام'),
 'hyperactivity_impulsivity':('Restlessness or impulsivity','التململ أو الاندفاع'),
 'childhood_onset':('Difficulties dating to childhood','صعوبات منذ الطفولة'),
 'reduced_sleep':('Reduced sleep','قلة النوم'),
 'decreased_need_for_sleep':('Less need for sleep with preserved energy','انخفاض الحاجة للنوم مع استمرار النشاط'),
 'elevated_mood':('Unusually elevated mood or energy','ارتفاع غير معتاد في المزاج أو الطاقة'),
 'grandiosity':('Unusually heightened self-confidence','ارتفاع غير معتاد في تقدير القدرات الذاتية'),
 'racing_thoughts':('Racing thoughts','تسارع الأفكار'),
 'pressured_speech':('Rapid or difficult-to-stop speech','كلام سريع أو يصعب إيقافه'),
 'flight_of_ideas':('Rapid shifts between ideas','انتقال سريع بين الأفكار'),
 'impulsive_spending':('Impulsive spending','إنفاق اندفاعي'),
 'increased_goal_directed_activity':('Increased goal-directed activity','زيادة النشاط الموجه نحو أهداف'),
 'episodic_pattern':('Distinct periods of change','فترات واضحة من التغير'),
 'depression_alternation':('Reported low-mood periods','فترات مزاج منخفض أُبلغ عنها'),
 'auditory_perceptual_experience':('Reported voice-hearing experience','تجربة سماع أصوات أُبلغ عنها'),
 'persecutory_ideas':('Feeling watched or threatened','الشعور بالمراقبة أو التهديد'),
 'thought_disorganization':('Difficulty organizing thoughts','صعوبة تنظيم الأفكار'),
 'social_withdrawal':('Social withdrawal','الانسحاب الاجتماعي'),
 'duration':('Duration and course','المدة ومسار التغيرات'),
 'impairment':('Impact on daily life','الأثر على الحياة اليومية'),
 'confounders':('Medicines, substances, sleep or physical-health context','سياق الأدوية أو المواد أو النوم أو الصحة الجسدية'),
 'childhood_onset_unconfirmed':('Childhood history not confirmed','تاريخ الطفولة غير مؤكد'),
 'multiple_settings_unconfirmed':('Impact in multiple settings not confirmed','الأثر في أكثر من مكان غير مؤكد'),
 'elevated_mood_unconfirmed':('Elevated mood/energy not confirmed','ارتفاع المزاج أو الطاقة غير مؤكد'),
 'episodic_pattern_unconfirmed':('Distinct episodes not confirmed','الفترات الواضحة غير مؤكدة'),
 'functional_impact_unconfirmed':('Functional impact not confirmed','الأثر الوظيفي غير مؤكد'),
 'medication_context':('Reported medicine-related context','سياق متعلق بالأدوية أُبلغ عنه'),
 'substance_context':('Reported substance-related context','سياق متعلق بمواد أُبلغ عنه'),
 'medical_context':('Reported physical-health context','سياق للصحة الجسدية أُبلغ عنه'),
 'sleep_loss_with_fatigue':('Sleep loss with fatigue','قلة النوم مع التعب'),
}

def label(key,ar):
    if key.startswith('conflicting:'):
        return ('متعارض: ' if ar else 'Conflicting: ')+label(key.split(':',1)[1],ar)
    return LABELS.get(key,(key.replace('_',' '),key.replace('_',' ')))[ar]

def format_report(report):
    ar=report.get('language')=='ar'
    def t(en,arabic): return arabic if ar else en
    def items(keys): return '\n'.join('- '+label(k,ar) for k in dict.fromkeys(keys)) or t('- None recorded','- لا توجد معلومات مسجلة')
    incomplete=report['assessment_status']=='inconclusive'
    candidates=report.get('candidate_patterns',[])
    lines=[t('MIRA ASSESSMENT REPORT','تقرير تقييم ميرا'),
           t('Status: ','الحالة: ')+(t('incomplete or inconclusive','غير مكتمل أو غير حاسم') if incomplete else t('screening pattern for clinician review','نمط أولي يحتاج مراجعة مختص')),
           t('Answers received: ','عدد الإجابات: ')+str(report.get('answers_received',0)),
           '',t('Patterns to discuss with a clinician','أنماط لمناقشتها مع مختص'),
           items(candidates) if candidates else t('- No clear target pattern from the available information; this does not rule out a condition.','- لا يوجد نمط واضح من المعلومات المتاحة؛ وهذا لا يستبعد وجود اضطراب.'),
           '',t('Supporting reported observations','ملاحظات أُبلغ عنها تدعم المتابعة'),items(report.get('supporting_features',[])),
           '',t('Denied or conflicting observations','ملاحظات منفية أو متعارضة'),items(report.get('contradictory_features',[])),
           '',t('Information still needed','معلومات ما زالت مطلوبة'),items(report.get('missing_information',[])),
           '',t('Other reported context','سياق آخر أُبلغ عنه'),items(report.get('reported_context',[])),
           '',t('Next step: discuss this report with a qualified clinician.','الخطوة التالية: ناقش هذا التقرير مع مختص مؤهل.'),
           t('This screening report does not confirm or rule out a diagnosis.','هذا التقرير الأولي لا يؤكد تشخيصاً ولا يستبعده.')]
    return '\n'.join(lines)

def reply_payload(session,reply):
    result=asdict(reply)
    result['assistant_message']=result.pop('text')
    result['session_id']=session.session_id
    result['report_ready']=bool(reply.result)
    return result

def session_payload(session):
    return {'session_id':session.session_id,'language':session.language,
            'assessment_complete':session.complete,'complete':session.complete,
            'chapter_progress':1.0 if session.complete else min(.99,session.messages/DEFAULT_SESSION_QUESTIONS),
            'assistant_message':session.assistant_message,'chapter':session.chapter,
            'result':session.result,'report_ready':bool(session.result),'input_enabled':True,
            'safety':{'level':'urgent' if session.safety_mode else 'needs_clarification' if session.safety_check_pending else 'routine',
                      'flags':list(dict.fromkeys(x['feature'] for x in session.state.safety_snapshot()['active_features']))},
            'model_assessment':session.model_assessment}
