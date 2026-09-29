# -*- coding: utf-8 -*-
"""The second half of the job: the patient says what they are struggling with.

The friction classifier already named the block; before this, that finding was
computed and thrown away, and every struggle message got the same generic
"What is one task you want help with?".
"""
from datetime import datetime, timezone
from uuid import uuid4, UUID
from datetime import date
import pytest
from lumina.adhd.executive_function.schemas import OrganizeRequest
from lumina.adhd.executive_function.task_extractor import extract_tasks
from lumina.adhd.executive_function.friction_support import leading, ask_for_task, adapt_step, ASK, STEP, PRIORITY
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.calendar import LocalCalendar

PATIENT = 'ae3e23cc-5b26-468f-a8df-708519a4a144'
NOW = datetime(2026, 9, 29, 14, 30, tzinfo=timezone.utc)
REF = date(2026, 9, 29)
RID = UUID('11111111-2222-3333-4444-555555555555')


def request(text, lang='EN', **kwargs):
    value = {'requestId': str(uuid4()), 'patient': {'id': PATIENT, 'language': lang},
             'message': {'text': text}, 'timeContext': {'referenceDate': '2026-09-29'}}
    value.update(kwargs)
    return OrganizeRequest.model_validate(value)


@pytest.fixture(scope='module')
def agent():
    return LuminaADHD()


GENERIC_EN = 'What is one task you want help with?'
GENERIC_AR = 'ما المهمة الواحدة التي تريد المساعدة فيها؟'

STRUGGLES_EN = ["I can't get started on anything.",
                "too much stuff, I don't know where to start",
                'the report feels way too big',
                "I'm overwhelmed, I have like 15 things to do",
                'I have no energy today',
                "I keep putting off my taxes",
                "I've been avoiding calling the dentist for weeks"]

STRUGGLES_AR = ['ما اقدر ابدا بشي', 'عندي وايد شغلات ومب عارف من وين ابدا', 'ما عندي طاقة اليوم']


@pytest.mark.parametrize('text', STRUGGLES_EN)
def test_struggle_gets_a_specific_reply_en(agent, text):
    result = agent.organize(request(text), NOW)
    assert result.response.text != GENERIC_EN, f'{text!r} still got the generic prompt'
    assert result.safety.level == 'NORMAL'


@pytest.mark.parametrize('text', STRUGGLES_AR)
def test_struggle_gets_a_specific_reply_ar(agent, text):
    result = agent.organize(request(text, lang='AR'), NOW)
    assert result.response.text != GENERIC_AR, f'{text!r} still got the generic prompt'


def test_overwhelm_is_answered_as_overwhelm(agent):
    result = agent.organize(request("I'm overwhelmed, I have like 15 things to do"), NOW)
    assert result.response.text == ASK['OVERWHELM'][0]


def test_low_energy_is_answered_as_low_energy(agent):
    result = agent.organize(request('I have no energy today'), NOW)
    assert result.response.text == ASK['LOW_ENERGY'][0]


# --- A named struggle carries the task itself ------------------------------------
@pytest.mark.parametrize('text,title', [
    ('I keep putting off my taxes', 'my taxes'),
    ("I've been avoiding calling the dentist for weeks", 'calling the dentist'),
    ('the report feels way too big', 'report'),
    ("I'm dreading the performance review", 'the performance review'),
    ('التقرير كبير علي', 'التقرير')])
def test_stuck_on_names_the_task(text, title):
    tasks, _ = extract_tasks(text, RID, REF)
    assert [t.title for t in tasks] == [title]


def test_struggle_saves_the_task_and_adapts_the_step(tmp_path):
    calendar = LocalCalendar(tmp_path / 'c.sqlite3')
    agent = LuminaADHD(calendar)
    result = agent.organize(request('I keep putting off my taxes', calendar={'commit': True}), NOW)
    assert [t.title for t in calendar.tasks(PATIENT)] == ['my taxes']
    assert result.response.text.startswith(STEP['AVOIDANCE'][0])


def test_too_big_gets_the_first_slice(tmp_path):
    calendar = LocalCalendar(tmp_path / 'c.sqlite3')
    agent = LuminaADHD(calendar)
    result = agent.organize(request('the report feels way too big', calendar={'commit': True}), NOW)
    assert result.response.text.startswith(STEP['TOO_BIG'][0])


# --- The block is inferred from the intent when the classifier finds nothing -----
def test_intent_supplies_the_block_when_friction_is_unknown():
    assert leading(['UNKNOWN'], 'BREAK_DOWN_TASK') == 'TOO_BIG'
    assert leading(['UNKNOWN'], 'PROCRASTINATION') == 'AVOIDANCE'
    assert leading([], None) is None


def test_unclear_never_shapes_a_reply():
    """UNCLEAR is the classifier's catch-all; it fires on plain queries."""
    assert 'UNCLEAR' not in PRIORITY
    assert leading(['UNCLEAR']) is None


def test_every_prioritised_block_has_both_templates():
    for key in PRIORITY:
        assert key in ASK and key in STEP
        for index in (0, 1):
            assert ASK[key][index].strip() and STEP[key][index].strip()


def test_both_languages_are_distinct():
    for key in PRIORITY:
        assert ask_for_task([key], 'EN') == ASK[key][0]
        assert ask_for_task([key], 'AR') == ASK[key][1]
        assert adapt_step([key], 'EN', 'X') == STEP[key][0] + 'X'


def test_no_block_means_no_invention():
    assert ask_for_task(['UNKNOWN'], 'EN') is None
    assert adapt_step(['UNKNOWN'], 'EN', 'X') is None


# --- The safety fix must still hold for struggle vocabulary ----------------------
@pytest.mark.parametrize('text', ["I'm overwhelmed, I have like 15 things to do",
                                  'everything is piling up and I cannot focus',
                                  "I'm so anxious about this deadline",
                                  'I feel exhausted and sad today'])
def test_struggle_vocabulary_is_not_a_crisis(agent, text):
    result = agent.organize(request(text), NOW)
    assert result.safety.level == 'NORMAL', f'{text!r} escalated to {result.safety.level}'


# --- A pointer is not a name ------------------------------------------------------
@pytest.mark.parametrize('text', ['The task feels too big.', 'I keep avoiding this task.',
                                  "I'm dreading it", 'المهمة كبيرة علي'])
def test_placeholder_is_not_saved_as_a_task(text):
    """"this task" points at something the message never names; saving it is clutter."""
    tasks, uncertainty = extract_tasks(text, RID, REF)
    assert tasks == []
    assert 'TASK_NOT_NAMED' in uncertainty


def test_placeholder_still_gets_the_block_specific_question(agent):
    result = agent.organize(request('The task feels too big.'), NOW)
    assert result.analysis.taskCount == 0
    assert result.response.text == ASK['TOO_BIG'][0]
