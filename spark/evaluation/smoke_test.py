"""Core integration checks using runtime dependencies only; no test framework needed."""
import json,tempfile,unittest
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
from uuid import uuid4
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.adhd.executive_function.schemas import OrganizeRequest
from lumina.adhd.executive_function.errors import AssistantError
from lumina.adhd.executive_function.time_engine import parse_temporal,available_minutes
from lumina.calendar import LocalCalendar

NOW=datetime(2026,9,29,14,30,tzinfo=timezone.utc)
PATIENT='ae3e23cc-5b26-468f-a8df-708519a4a144'

def request(text='Help me prioritize today',lang='EN',**kwargs):
    data={'requestId':str(uuid4()),'patient':{'id':PATIENT,'language':lang},'message':{'text':text},'timeContext':{'referenceDate':'2026-09-29','availableMinutes':20}}
    data.update(kwargs)
    return OrganizeRequest.model_validate(data)

class CoreSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.agent=LuminaADHD()

    def test_report_metrics(self):
        from .categorical_metrics import accuracy_score,f1_score
        expected=['a','a','b','b']; predicted=['a','a','a','b']
        self.assertEqual(accuracy_score(expected,predicted),.75)
        self.assertAlmostEqual(f1_score(expected,predicted),(.8+2/3)/2)

    def test_bilingual_examples(self):
        root=Path(__file__).resolve().parents[1]
        for language in ('english','arabic'):
            payload=json.loads((root/'examples'/f'{language}.json').read_text(encoding='utf-8'))
            result=self.agent.organize(payload,NOW)
            self.assertEqual(result.safety.level,'NORMAL')
            self.assertIsNotNone(result.plan.primaryTask)
            self.assertIsNone(result.plan.primaryTask.durationMinutes)

    def test_arabic_task_difficulty_separated(self):
        result=self.agent.organize(request('عندي تقرير لازم أخلصه بس ما قدرت أبدأ','AR'),NOW)
        self.assertEqual(result.plan.primaryTask.title,'اخلص تقرير')

    def test_arabic_time(self):
        parsed=parse_temporal('اكلم البنك بكرة الساعة ٣ مساء لمدة ١٠ دقيقة',NOW.date())
        self.assertEqual(parsed['uncertainty'],[])
        self.assertEqual(parsed['date'],date(2026,9,30))
        self.assertEqual(parsed['start'].hour,15)
        self.assertEqual(parsed['duration'],10)
        self.assertEqual(available_minutes('عندي عشرة دقائق'),10)

    def test_ambiguous_day(self):
        for query in ('Prioritize next week','Help me prioritize on 03/04','Help me prioritize tomorrow'):
            result=self.agent.organize(request(query,timeContext={},tasks=[{'title':'finish report'}]),NOW)
            self.assertTrue(result.analysis.needsClarification)
            self.assertEqual(result.plan.strategy,'CLARIFY')
            self.assertIsNone(result.plan.nextAction)
            self.assertIsNone(result.focusSession)

    def test_prior_day_dependency(self):
        result=self.agent.organize(request(tasks=[{'temporaryId':'a','title':'write report','scheduledDate':'2026-09-28','status':'DONE'},{'temporaryId':'b','title':'send report','scheduledDate':'2026-09-29','dependencies':['a']}]),NOW)
        self.assertEqual(result.plan.primaryTask.temporaryId,'b')

    def test_calendar_workflow(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calendar.db'
            calendar=LocalCalendar(path)
            agent=LuminaADHD(calendar)
            payload=request('I need to email James tomorrow',calendar={'commit':True})
            result=agent.organize(payload,NOW)
            self.assertEqual(result.taskOperations[0].status,'APPLIED')
            self.assertIsNone(result.focusSession)
            self.assertEqual(agent.organize(payload,NOW),result)
            self.assertEqual(len(LocalCalendar(path).tasks(PATIENT)),1)
            self.assertFalse(calendar.tasks(uuid4()))
            today=agent.organize(request(),NOW)
            self.assertIsNone(today.plan.primaryTask)
            tomorrow=agent.organize(request('Help me prioritize tomorrow'),NOW)
            self.assertEqual(tomorrow.plan.primaryTask.title,'email James')
            self.assertIsNone(tomorrow.focusSession)
            moved=agent.organize(request('Reschedule email James to 2026-10-03',calendar={'commit':True}),NOW)
            self.assertEqual(calendar.tasks(PATIENT)[0].scheduledDate,date(2026,10,3))
            self.assertIsNone(moved.focusSession)
            self.assertEqual(moved.followUp.type,'NONE')

    def test_reduced_capacity(self):
        result=self.agent.organize(request('I need to finish the report',currentState={'capacity':'REDUCED'},tasks=[{'temporaryId':'b','title':'buy bread'}]),NOW)
        self.assertEqual(result.plan.secondaryTasks,[])
        self.assertLessEqual(result.focusSession.minutes,5)
        self.assertIsNone(result.plan.nextAction.estimatedMinutes)

    def test_no_fabricated_metadata(self):
        result=self.agent.organize(request('I need to email James'),NOW)
        self.assertIsNone(result.plan.primaryTask.deadline)
        self.assertIsNone(result.plan.primaryTask.durationMinutes)
        self.assertEqual(result.plan.primaryTask.importance,'UNKNOWN')

    def test_crisis_overrides(self):
        for text in ('I want to die','الأصوات تأمرني أؤذي نفسي','I forgot what I was doing. I want to die.','Help me prioritize on 03/04. I want to die.'):
            result=self.agent.organize(request(text,tasks=[{'title':'finish report'}]),NOW)
            self.assertEqual(result.safety.level,'CRISIS')
            self.assertEqual(result.plan.strategy,'SAFETY_HANDOFF')
            self.assertEqual(result.taskOperations,[])
            self.assertIsNone(result.focusSession)

    def test_benign_recovery(self):
        result=self.agent.organize(request('I forgot what I was doing.',tasks=[{'temporaryId':'a','title':'finish presentation'}],recentContext=[{'taskId':'a','timestamp':NOW.isoformat(),'nextStep':'Open slide 4.'}]),NOW)
        self.assertEqual(result.plan.nextAction.text,'Open slide 4.')

    def test_unsafe_recovery(self):
        result=self.agent.organize(request('I forgot what I was doing.',tasks=[{'temporaryId':'a','title':'finish report'}],recentContext=[{'taskId':'a','timestamp':NOW.isoformat(),'nextStep':'double my medication dose'}]),NOW)
        self.assertEqual(result.safety.level,'ELEVATED')
        self.assertIsNone(result.plan.nextAction)

    def test_future_outcome_rejected(self):
        with self.assertRaises(AssistantError):
            self.agent.organize(request(outcomes=[{'attemptId':str(uuid4()),'patientId':PATIENT,'timestamp':(NOW+timedelta(days=1)).isoformat(),'outcome':'DONE'}]),NOW)

if __name__=='__main__': unittest.main(verbosity=2)
