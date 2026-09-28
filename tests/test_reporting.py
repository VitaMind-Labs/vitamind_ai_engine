"""Report regressions. Run with stdlib unittest, or as part of pytest.

The model is stubbed because these tests target reporting, not learned inference.
The production conversation, feature extraction and serialization are used.
"""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from vitamind.mira.agent import MiraAgent
from vitamind.mira.reporting import reply_payload,session_payload

class FixedModel:
    def __init__(self): self.classifier=self
    def _load(self): return {}
    def assess(self,state,text=None):
        return {'status':'available','suggested_condition':None,'input_type':'structured_evidence_projection'}

class ReportTests(unittest.TestCase):
    def agent(self,language='en'):
        a=MiraAgent(model=FixedModel())
        s,_=a.create_session(language=language)
        return a,s

    def test_automatic_report_after_ten_answers(self):
        a,s=self.agent()
        for turn in range(1,11):
            reply=a.respond(s,'ok')
            self.assertEqual(reply.assessment_complete,turn==10)
            self.assertEqual(reply.chapter_progress,turn/10)
        self.assertEqual(reply.result['answers_received'],10)
        self.assertEqual(reply.result['completion_reason'],'answer_limit')
        self.assertEqual(reply.result['assessment_status'],'inconclusive')
        self.assertIn('MIRA ASSESSMENT REPORT',reply.text)

    def test_incomplete_answers_still_produce_a_report(self):
        a,s=self.agent()
        for _ in range(10): reply=a.respond(s,"I don't know")
        self.assertTrue(reply.result['report_ready'])
        self.assertTrue(reply.result['missing_information'])
        self.assertEqual(reply.result['candidate_patterns'],[])
        self.assertEqual(reply.match_strength,'LOW')

    def test_immediate_english_report_aliases(self):
        for command in ['report','show report','generate report','give me the report','summary','finish assessment']:
            with self.subTest(command=command):
                a,s=self.agent(); reply=a.respond(s,command)
                self.assertTrue(reply.assessment_complete)
                self.assertEqual(reply.result['completion_reason'],'requested')
                self.assertEqual(reply.result['answers_received'],0)

    def test_arabic_report_commands(self):
        for command in ['تقرير','التقرير','اعرض التقرير','إنشاء تقرير','ملخص']:
            with self.subTest(command=command):
                a,s=self.agent('ar');reply=a.respond(s,command)
                self.assertTrue(reply.assessment_complete)
                self.assertIn('تقرير تقييم ميرا',reply.text)
                self.assertNotIn('MIRA ASSESSMENT REPORT',reply.text)

    def test_legacy_report_fields_and_types(self):
        a,s=self.agent()
        a.respond(s,'I have been distracted since childhood and I forget appointments.')
        reply=a.respond(s,'report'); r=reply.result
        for key in ['recommended_pathway','orientation','match_strength','condition_scores',
                    'supporting_features','contradictory_features','other_signals','missing_information',
                    'recommended_test','requires_clinician_review','disclaimer','language','safety']:
            self.assertIn(key,r)
        for key in ['supporting_features','contradictory_features','other_signals','missing_information']:
            self.assertIsInstance(r[key],list)
        self.assertEqual(set(r['condition_scores']),{'adhd','bipolar','psychosis'})
        self.assertIn('distractibility',r['supporting_features'])
        self.assertEqual(reply.pathway,'ADHD_focused_clinical_assessment')
        self.assertEqual(reply.recommended_test,'clinical consultation')
        self.assertIn('not_diagnostic_confidence',r['match_strength_meaning'])

    def test_per_condition_details_are_retained(self):
        a,s=self.agent();a.respond(s,'I hear voices when I am alone.')
        r=a.respond(s,'report').result
        self.assertIn('auditory_perceptual_experience',r['supporting_features_by_condition']['PSYCHOSIS_SPECTRUM'])
        self.assertIn('BIPOLAR_SPECTRUM',r['missing_information_by_condition'])

    def test_report_text_contains_actual_findings(self):
        a,s=self.agent();a.respond(s,'I have been distracted since childhood.')
        r=a.respond(s,'report').result
        for section in ['Supporting reported observations','Information still needed','Next step:']:
            self.assertIn(section,r['report_text'])
        self.assertIn('Difficulty maintaining attention',r['report_text'])
        self.assertNotIn('Reported voice-hearing experience',r['report_text'])

    def test_get_and_message_payloads_expose_same_report(self):
        a,s=self.agent()
        for _ in range(10): reply=a.respond(s,'ok')
        post=reply_payload(s,reply); get=session_payload(s)
        self.assertTrue(get['complete'])
        for key in ['assessment_complete','result','chapter_progress','report_ready','assistant_message']:
            self.assertEqual(post[key],get[key])
        self.assertNotIn('session_token',get)

    def test_why_does_not_count_as_assessment_answer(self):
        a,s=self.agent()
        for _ in range(9): a.respond(s,'ok')
        self.assertFalse(a.respond(s,'why').assessment_complete)
        self.assertEqual(s.messages,9)
        self.assertTrue(a.respond(s,'ok').assessment_complete)

    def test_additional_evidence_refreshes_report(self):
        a,s=self.agent();a.respond(s,'report')
        reply=a.respond(s,'I hear voices when I am alone.')
        self.assertTrue(reply.assessment_complete)
        self.assertEqual(reply.result['completion_reason'],'updated_with_additional_information')
        self.assertIn('auditory_perceptual_experience',reply.result['supporting_features'])

    def test_crisis_does_not_force_assessment_report(self):
        a,s=self.agent()
        for _ in range(9): a.respond(s,'ok')
        for text in ['I want to kill myself.','report','Please stay with me.']:
            reply=a.respond(s,text)
            self.assertEqual(reply.safety['level'],'urgent')
            self.assertFalse(reply.assessment_complete)
            self.assertTrue(reply.input_enabled)
            self.assertIsNone(reply.result)

    def test_crisis_after_report_remains_open(self):
        a,s=self.agent();a.respond(s,'report')
        reply=a.respond(s,'I want to kill myself.')
        self.assertFalse(reply.assessment_complete)
        self.assertIsNone(session_payload(s)['result'])

    def test_terminal_prints_full_report(self):
        import run_mira
        output=io.StringIO()
        with patch.object(run_mira,'MiraAgent',side_effect=lambda:MiraAgent(model=FixedModel())), \
             patch('sys.argv',['run_mira.py']),patch('builtins.input',side_effect=['ok']*10+['exit']),redirect_stdout(output):
            run_mira.main()
        self.assertIn('MIRA ASSESSMENT REPORT',output.getvalue())
        self.assertIn('Answers received: 10',output.getvalue())

if __name__=='__main__': unittest.main()
