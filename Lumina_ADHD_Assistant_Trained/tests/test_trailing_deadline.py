# -*- coding: utf-8 -*-
"""A deadline stated in its own clause.

"I keep putting off my taxes, they're due Friday" splits into a task clause and a
clause carrying only the date. The date used to be dropped. It is now attached, but
only when there is exactly one task it could belong to -- a pronoun is never
resolved by guessing, and the tasks are saved either way.
"""
from datetime import date, datetime, timezone
from uuid import uuid4, UUID
import pytest
from lumina.adhd.executive_function.schemas import OrganizeRequest
from lumina.adhd.executive_function.task_extractor import extract_tasks
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.calendar import LocalCalendar

PATIENT = 'ae3e23cc-5b26-468f-a8df-708519a4a144'
NOW = datetime(2026, 9, 29, 14, 30, tzinfo=timezone.utc)
REF = date(2026, 9, 29)          # a Tuesday
FRIDAY = date(2026, 10, 2)
RID = UUID('11111111-2222-3333-4444-555555555555')


def request(text, lang='EN', **kwargs):
    value = {'requestId': str(uuid4()), 'patient': {'id': PATIENT, 'language': lang},
             'message': {'text': text}, 'timeContext': {'referenceDate': '2026-09-29'}}
    value.update(kwargs)
    return OrganizeRequest.model_validate(value)


# --- Attached when exactly one task could own it ---------------------------------
@pytest.mark.parametrize('text,title,due', [
    ("I keep putting off my taxes, they're due Friday.", 'my taxes', FRIDAY),
    ("I have to finish the report, it's due 2026-10-05", 'finish the report', date(2026, 10, 5)),
    ('I need to submit the form, deadline Friday', 'submit the form', FRIDAY),
    ('I have to finish the report, it needs to be done Thursday', 'finish the report', date(2026, 10, 1)),
    ('لازم اخلص التقرير، موعدها الخميس', 'اخلص التقرير', date(2026, 10, 1))])
def test_trailing_deadline_is_attached(text, title, due):
    tasks, uncertainty = extract_tasks(text, RID, REF)
    assert [t.title for t in tasks] == [title]
    assert tasks[0].deadline == due and tasks[0].scheduledDate == due
    assert not [u for u in uncertainty if u.startswith('DEADLINE_CLAUSE')]


# --- Never guessed -----------------------------------------------------------------
def test_ambiguous_deadline_is_not_guessed():
    """Two tasks and one "they're due Friday" gives no honest way to pick."""
    tasks, uncertainty = extract_tasks(
        "I have to call the bank, buy groceries, they're due Friday", RID, REF)
    assert [t.title for t in tasks] == ['call the bank', 'buy groceries']
    assert all(t.deadline is None and t.scheduledDate is None for t in tasks)
    assert 'DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS' in uncertainty


def test_unresolvable_deadline_is_not_guessed():
    tasks, uncertainty = extract_tasks("I keep putting off my taxes, they're due next week", RID, REF)
    assert [t.title for t in tasks] == ['my taxes']
    assert tasks[0].deadline is None
    assert 'DATE_NEEDS_CLARIFICATION' in uncertainty


def test_deadline_clause_alone_creates_no_task():
    tasks, uncertainty = extract_tasks("they're due Friday", RID, REF)
    assert tasks == [] and 'DEADLINE_CLAUSE_WITHOUT_TASK' in uncertainty


def test_trailing_clause_never_overrides_a_stated_date():
    tasks, uncertainty = extract_tasks("I need to email James tomorrow, it's due Friday", RID, REF)
    assert tasks[0].scheduledDate == date(2026, 9, 30)   # tomorrow, as stated
    assert tasks[0].deadline is None
    assert 'DEADLINE_CLAUSE_CONFLICTS_WITH_TASK_DATE' in uncertainty


# --- Existing parsing is untouched -------------------------------------------------
def test_same_clause_deadline_still_works():
    tasks, _ = extract_tasks('I have to pay the bill due Friday', RID, REF)
    assert [t.title for t in tasks] == ['pay the bill']
    assert tasks[0].deadline == FRIDAY


def test_plain_task_list_is_unaffected():
    tasks, uncertainty = extract_tasks(
        'I have to finish the report, call the bank, and buy groceries', RID, REF)
    assert [t.title for t in tasks] == ['finish the report', 'call the bank', 'buy groceries']
    assert uncertainty == []


# --- An uncertain due date must never cost the user the tasks ---------------------
@pytest.fixture
def calendar(tmp_path):
    return LocalCalendar(tmp_path / 'cal.sqlite3')


def test_ambiguous_deadline_still_saves_every_task(calendar):
    agent = LuminaADHD(calendar)
    result = agent.organize(request("I have to call the bank, buy groceries, they're due Friday",
                                    calendar={'commit': True}), NOW)
    assert not result.analysis.needsClarification
    assert {t.title for t in calendar.tasks(PATIENT)} == {'call the bank', 'buy groceries'}
    assert all(t.deadline is None for t in calendar.tasks(PATIENT))


def test_ambiguous_deadline_asks_which_one(calendar):
    agent = LuminaADHD(calendar)
    result = agent.organize(request("I have to call the bank, buy groceries, they're due Friday",
                                    calendar={'commit': True}), NOW)
    assert 'due date' in result.response.text
    assert 'DEADLINE_CLAUSE_AMBIGUOUS_MULTIPLE_TASKS' in result.analysis.uncertainty


def test_attached_deadline_reaches_the_calendar(calendar):
    agent = LuminaADHD(calendar)
    agent.organize(request("I keep putting off my taxes, they're due Friday.",
                           calendar={'commit': True}), NOW)
    stored = calendar.tasks(PATIENT)
    assert [(t.title, t.deadline) for t in stored] == [('my taxes', FRIDAY)]


def test_day_query_finds_the_attached_deadline(calendar):
    agent = LuminaADHD(calendar)
    agent.organize(request("I keep putting off my taxes, they're due Friday.",
                           calendar={'commit': True}), NOW)
    result = agent.organize(request('What do I have on 2026-10-02?'), NOW)
    assert result.plan.primaryTask.title == 'my taxes'
