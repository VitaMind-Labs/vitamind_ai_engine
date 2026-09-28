"""Integrated local screening conversation. No autonomous diagnosis or treatment."""
from dataclasses import dataclass,field
from uuid import uuid4
import secrets
from ..clinical.patient_state import PatientState
from ..clinical.feature_extractor import FeatureExtractor
from ..clinical.assessment.assessment_engine import AssessmentEngine
from ..clinical.differential import DifferentialEngine
from ..clinical.language import normalize,patient_clauses
from ..safety.detector import SafetyDetector
from ..ml.integrated import IntegratedModel
from .interview import InterviewPlanner,DEFAULT_SESSION_QUESTIONS
from .questions import Question
from .answers import interpret
from .reporting import format_report

@dataclass
class MiraReply:
    text:str
    chapter:str
    chapter_progress:float
    assessment_complete:bool=False
    safety:dict=field(default_factory=dict)
    result:dict|None=None
    language:str='en'
    model_assessment:dict=field(default_factory=dict)
    input_enabled:bool=True

    @property
    def pathway(self):
        return (self.result or {}).get('recommended_pathway')

    @property
    def match_strength(self):
        return (self.result or {}).get('match_strength')

    @property
    def recommended_test(self):
        return (self.result or {}).get('recommended_test')

@dataclass
class MiraSession:
    session_id:str
    state:PatientState
    language:str
    session_token:str=field(default_factory=lambda:secrets.token_urlsafe(32))
    chapter:str='ASSESSMENT'
    messages:int=0
    complete:bool=False
    result:dict|None=None
    assistant_message:str|None=None
    pending:Question|None=None
    answers:dict=field(default_factory=dict)
    skipped:set=field(default_factory=set)
    attempts:dict=field(default_factory=dict)
    history:list=field(default_factory=list)
    model_assessment:dict=field(default_factory=dict)
    safety_mode:bool=False
    safety_check_pending:bool=False

class MiraAgent:
    def __init__(self,debug=False,model=None):
        self.extractor=FeatureExtractor()
        self.assessments=AssessmentEngine()
        self.differential=DifferentialEngine()
        self.safety=SafetyDetector()
        self.interview=InterviewPlanner()
        self.model=model or IntegratedModel()

    def create_session(self,age_group='adult',language='en',country_context=None):
        if language not in {'en','ar'}: raise ValueError("language must be 'en' or 'ar'")
        if age_group!='adult': raise ValueError('This development interview supports adults only.')
        s=MiraSession(uuid4().hex,PatientState.new(age_group,language=language,country_context=country_context),language)
        s.pending=self.interview.plan(s.state,s.answers,{})
        intro=('I’m Mira. I can help organize your experiences for a clinician; I cannot confirm a diagnosis. You can ask why, skip a question, or request a summary. '
               if language=='en' else 'أنا ميرا. أساعدك في تنظيم تجربتك لمناقشتها مع مختص، ولا أؤكد تشخيصاً. يمكنك السؤال عن السبب أو تخطي سؤال أو طلب ملخص. ')
        return s,self._reply(s,intro+s.pending.text(language))

    def _reply(self,s,text,safety=None):
        s.assistant_message=text
        s.history.append({'role':'assistant','text':text})
        return MiraReply(text,s.chapter,min(.99,s.messages/DEFAULT_SESSION_QUESTIONS) if not s.complete else 1,
                         s.complete,safety or {'level':'routine','flags':[]},s.result,s.language,s.model_assessment)

    def _support(self,s,urgent=True):
        s.complete=False
        s.result=None
        s.chapter='SAFETY'
        if urgent:
            s.safety_mode=True
            flags=[x['feature'] for x in s.state.safety_snapshot()['active_features']]
            text=('I’m here to listen, and you can keep writing. If you might act on these thoughts or are in immediate danger, contact local emergency services now. Can you reach a trusted person who can stay with you? I cannot contact anyone for you.'
                  if s.language=='en' else 'أنا هنا للاستماع ويمكنك مواصلة الكتابة. إذا كنت قد تتصرف وفق هذه الأفكار أو كنت في خطر مباشر، اتصل بخدمات الطوارئ المحلية الآن. هل تستطيع التواصل مع شخص تثق به ليبقى معك؟ لا أستطيع الاتصال بأحد نيابة عنك.')
            return self._reply(s,text,{'level':'urgent','flags':flags,'requires_human_support':True,'notification_sent':False})
        s.safety_check_pending=True
        text=('I want to understand what you mean. Are you thinking about harming yourself or someone else right now? You can keep writing.'
              if s.language=='en' else 'أريد فهم ما تقصده. هل تفكر في إيذاء نفسك أو شخص آخر الآن؟ يمكنك مواصلة الكتابة.')
        return self._reply(s,text,{'level':'needs_clarification','flags':[]})

    def _report(self,s,completion_reason='requested'):
        assessment=self.assessments.assess(s.state)
        differential=self.differential.analyze(s.state,assessment)
        candidates=[k for k,a in assessment.assessments.items() if a.raw_score >= (1 if k=='PSYCHOSIS_SPECTRUM' else 2)]
        missing=[k for k in ('duration','impairment','confounders') if k not in s.answers]
        if 'ADHD' in candidates:
            if s.state.get_status('developmental_history','childhood_onset').value!='present': missing.append('childhood_onset_unconfirmed')
            if s.answers.get('settings',{}).get('value') is not True: missing.append('multiple_settings_unconfirmed')
        if 'BIPOLAR_SPECTRUM' in candidates:
            for d,f in [('mood','elevated_mood'),('episode_history','episodic_pattern')]:
                if s.state.get_status(d,f).value!='present': missing.append(f+'_unconfirmed')
        conflicts=[f for a in assessment.assessments.values() for f in a.contradictory_evidence if f.startswith('conflicting:')]
        if s.answers.get('impairment',{}).get('value') is not True: missing.append('functional_impact_unconfirmed')
        ml=s.model_assessment
        disagreement=bool(ml.get('suggested_condition') and candidates and ml['suggested_condition'] not in candidates)
        uncertainty=bool(missing or conflicts or differential.alternatives or len(candidates)!=1 or disagreement or ml.get('status')=='unavailable')
        names={'ADHD':('ADHD','اضطراب نقص الانتباه وفرط الحركة'),
               'BIPOLAR_SPECTRUM':('bipolar-spectrum patterns','أنماط ضمن طيف الاضطراب ثنائي القطب'),
               'PSYCHOSIS_SPECTRUM':('psychosis-related experiences','تجارب مرتبطة بأعراض ذهانية')}
        labels=[names[k][s.language=='ar'] for k in candidates]
        description=', '.join(labels) if labels else ('no clear target pattern from the information collected' if s.language=='en' else 'لا يوجد نمط واضح من المعلومات التي جُمعت')
        text=(f'Here is the screening summary: the experiences you reported warrant discussion about {description}. '
              +('Important information is still missing, conflicting, or needs interpretation. ' if uncertainty else '')
              +'This does not confirm or rule out a diagnosis. A qualified clinician should review your history and possible alternative explanations. You can add or correct anything.'
              if s.language=='en' else f'هذا ملخص التقييم الأولي: تستحق التجارب التي وصفتها مناقشة {description} مع مختص. '
              +('توجد معلومات ناقصة أو متعارضة أو تحتاج إلى تفسير. ' if uncertainty else '')
              +'هذا لا يؤكد تشخيصاً ولا يستبعده. ينبغي أن يراجع مختص تاريخك والتفسيرات المحتملة الأخرى. يمكنك إضافة معلومات أو تصحيحها.')
        by_condition={k:a.supporting_evidence for k,a in assessment.assessments.items()}
        contradictory={k:a.contradictory_evidence for k,a in assessment.assessments.items()}
        pathway={'ADHD':'ADHD_focused_clinical_assessment','BIPOLAR_SPECTRUM':'mood_disorder_clinical_assessment',
                 'PSYCHOSIS_SPECTRUM':'psychosis_spectrum_clinical_assessment'}.get(candidates[0]) if len(candidates)==1 else (
                 'qualified_clinical_assessment' if candidates else 'NO_STRONG_TARGET_SIGNAL')
        s.result={'assessment_complete':True,'assessment_status':'inconclusive' if uncertainty else 'screening_pattern_for_clinician_review',
                  'report_ready':True,'completion_reason':completion_reason,'answers_received':s.messages,
                  'candidate_patterns':candidates,'recommended_pathway':pathway,
                  'match_strength':'LOW' if uncertainty else 'MODERATE',
                  'match_strength_meaning':'prototype_evidence_completeness_not_diagnostic_confidence',
                  'recommended_test':'clinical consultation' if s.language=='en' else 'استشارة مختص',
                  'orientation':description,'condition_scores':{k.lower().replace('_spectrum',''):round(a.raw_score/len(a.supporting_evidence+a.contradictory_evidence+a.missing_information),3) for k,a in assessment.assessments.items()},
                  'score_meaning':'fraction_of_prototype_features_present_not_disease_probability',
                  'supporting_features':sorted({f for features in by_condition.values() for f in features}),
                  'contradictory_features':sorted({f for features in contradictory.values() for f in features}),
                  'supporting_features_by_condition':by_condition,
                  'contradictory_features_by_condition':contradictory,
                  'missing_information_by_condition':{k:a.missing_information for k,a in assessment.assessments.items()},
                  'missing_information':missing,'unresolved_conflicts':conflicts,'reported_context':differential.alternatives,
                  'other_signals':[label.upper() for label in differential.alternatives],
                  'model_assessment':ml,'model_disagrees':disagreement,'requires_clinician_review':True,
                  'clinical_validation':False,'safety':{'level':'routine','flags':[]},'language':s.language,
                  'disclaimer':'Not a diagnosis.' if s.language=='en' else 'ليس تشخيصاً.'}
        s.complete=True
        s.chapter='SUMMARY'
        s.result['report_text']=format_report(s.result)
        return self._reply(s,text+'\n\n'+s.result['report_text'])

    def respond(self,s,text):
        if not text.strip(): raise ValueError('message text cannot be empty')
        if len(text)>10000: raise ValueError('message too long')
        s.history.append({'role':'user','text':text})
        value=normalize(text).strip(' .?!؟')
        # Safety still runs after a summary or during an ongoing support conversation.
        safety=self.safety.detect(s.state,text)
        if safety.urgent or s.safety_mode: return self._support(s)
        if s.safety_check_pending:
            state,answer=interpret(Question('safety','',''),text)
            if state=='answered' and answer is True:
                s.state.add_observation('safety','immediate_danger','present',1,text[:500],'SAFETY')
                return self._support(s)
            if state=='answered' and answer is False:
                s.safety_check_pending=False
                s.chapter='ASSESSMENT'
                text_out=('Thank you for clarifying. You can tell me if that changes. ' if s.language=='en' else 'شكراً للتوضيح. يمكنك إخباري إذا تغير ذلك. ')
                return self._reply(s,text_out+(s.pending.text(s.language) if s.pending else ''))
            else: return self._support(s,False)
        elif safety.needs_clarification: return self._support(s,False)
        if value in {'hello','hi','hey','مرحبا','اهلا','السلام عليكم'}:
            greeting=('Hello. I’m here to help you describe what you have been experiencing. ' if s.language=='en'
                      else 'أهلاً بك. أنا هنا لمساعدتك في وصف ما تمر به. ')
            return self._reply(s,greeting+(s.pending.text(s.language) if s.pending else ''))
        if value in {'what can you do','who are you','help','ماذا تستطيع ان تفعل','ماذا يمكنك ان تفعل','من انت','مساعدة'}:
            explanation=('I can ask assessment questions, keep track of your answers, and prepare a report for a clinician. Type report whenever you want a summary. '
                         if s.language=='en' else 'أستطيع طرح أسئلة تقييم ومتابعة إجاباتك وإعداد تقرير لمناقشته مع مختص. اكتب تقرير عندما تريد الملخص. ')
            return self._reply(s,explanation+(s.pending.text(s.language) if s.pending else ''))
        if value in {'why','why this question','why are you asking','لماذا','لماذا هذا السؤال'}:
            explanation=('I ask about timing, daily impact and context to avoid jumping to a conclusion. You may skip it. '
                         if s.language=='en' else 'أسأل عن التوقيت والأثر اليومي والسياق لتجنب الاستنتاج المتسرع. يمكنك تخطي السؤال. ')
            return self._reply(s,explanation+(s.pending.text(s.language) if s.pending else ''))
        if value in {'what do you mean','i do not understand',"i don't understand",'repeat','لم افهم','ماذا تقصد','اعد السؤال'}:
            explanation=('You can describe a concrete example, say whether it happens to you, or skip. I will not treat an unclear answer as a symptom. '
                         if s.language=='en' else 'يمكنك وصف مثال محدد أو توضيح هل يحدث ذلك لك أو التخطي. لن أسجل الإجابة غير الواضحة كعرض. ')
            return self._reply(s,explanation+(s.pending.text(s.language) if s.pending else ''))
        if any(phrase in value for phrase in ['do i have bipolar','do i have adhd','do i have psychosis','diagnose me','هل لدي اضطراب','هل انا مصاب','شخص حالتي']):
            explanation=('I cannot determine a diagnosis from this chat. I can organize the patterns you report and the questions that need a clinician’s assessment. '
                         if s.language=='en' else 'لا أستطيع تأكيد تشخيص من هذه المحادثة. يمكنني تنظيم الأنماط التي تصفها والنقاط التي تحتاج إلى تقييم مختص. ')
            return self._reply(s,explanation+(s.pending.text(s.language) if s.pending else ''))
        if value in {'summary','finish','show summary','report','show report','generate report','finish assessment',
                     'give me a report','give me the report','can you give me a report',
                     'ملخص','انهاء','اعرض الملخص','تقرير','التقرير','اعرض التقرير','انشاء تقرير','اعطني تقرير','اريد تقرير'}:
            return self._report(s)
        refresh_report=s.complete
        s.complete=False
        s.result=None
        s.chapter='ASSESSMENT'
        s.messages+=1
        question=s.pending
        status,answer=interpret(question,text) if question else ('unknown',None)
        observations=self.extractor.extract_into_state(s.state,text,s.chapter,str(s.messages))
        if question:
            if status=='answered':
                s.answers[question.key]={'value':answer,'evidence':text[:500]}
                if question.target and isinstance(answer,bool):
                    if question.kind=='clarify': s.state.resolve_feature(*question.target,'present' if answer else 'absent',text[:500],str(s.messages))
                    else: s.state.add_observation(*question.target,'present' if answer else 'absent',1,text[:500],s.chapter,str(s.messages))
                if question.key=='confounders' and answer!='denied': s.state.add_observation('context','reported_confounder','present',1,text[:500],s.chapter,str(s.messages))
            elif question.target and s.state.get_status(*question.target).value in {'present','absent'}:
                s.answers[question.key]={'value':s.state.get_status(*question.target).value=='present','evidence':text[:500]}
            else:
                s.attempts[question.key]=s.attempts.get(question.key,0)+1
                if status=='skip' or s.attempts[question.key]>=2: s.skipped.add(question.key)
        s.model_assessment=self.model.assess(s.state,text=text)
        s.pending=self.interview.plan(s.state,s.answers,s.model_assessment,s.skipped)
        if refresh_report: return self._report(s,'updated_with_additional_information')
        if s.messages>=DEFAULT_SESSION_QUESTIONS: return self._report(s,'answer_limit')
        if s.pending is None: return self._report(s,'questions_addressed')
        if safety.context=='other_person_or_educational' or any(not own for _,own in patient_clauses(text)):
            prefix=('I will not count someone else’s experience as your symptoms. Please describe your own experience. '
                    if s.language=='en' else 'لن أسجل تجربة شخص آخر كأعراض لديك. يرجى وصف تجربتك الشخصية. ')
        elif status=='unknown' and not observations:
            prefix=('I could not confidently interpret that answer. You can give an example, say yes or no where appropriate, or skip. '
                    if s.language=='en' else 'لم أتمكن من فهم الإجابة بثقة. يمكنك إعطاء مثال أو قول نعم أو لا عند ملاءمة ذلك أو التخطي. ')
        elif status=='skip': prefix='We can leave that unanswered. ' if s.language=='en' else 'يمكننا ترك هذا السؤال دون إجابة. '
        else: prefix='Thank you. ' if s.language=='en' else 'شكراً لك. '
        return self._reply(s,prefix+s.pending.text(s.language))
