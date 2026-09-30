# -*- coding: utf-8 -*-
"""Spark must understand a plan written the way people write it.

Found in the patient app: "i am going to the doctor and i have to pray and i have to
study and see my friend so help to organise them by priorite" saved nothing and got
"What is one task you want help with?", so the task list stayed empty and no focus
attempt (and so no learned pattern) could ever follow.
"""
from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.adhd.executive_function.schemas import OrganizeRequest
from lumina.adhd.executive_function.task_extractor import extract_tasks

PATIENT = 'ae3e23cc-5b26-468f-a8df-708519a4a144'
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
REF = date(2026, 9, 30)
MESSAGE = ('i am going to the doctor and i have to pray and i have to study and see my '
           'friend so help to organise them by priorite')


def request(text, lang='EN', **kwargs):
    value = {'requestId': str(uuid4()), 'patient': {'id': PATIENT, 'language': lang},
             'message': {'text': text}, 'timeContext': {'referenceDate': '2026-09-30'}}
    value.update(kwargs)
    return OrganizeRequest.model_validate(value)


@pytest.fixture(scope='module')
def agent():
    return LuminaADHD()


def titles(text):
    return [t.title for t in extract_tasks(text, uuid4(), REF)[0]]


def test_the_reported_message_yields_four_tasks_and_no_task_from_the_request():
    assert titles(MESSAGE) == ['go to the doctor', 'pray', 'study', 'see my friend']


@pytest.mark.parametrize('text, expected', [
    ('I have to pray, study and see my friend. Can you prioritize them?', ['pray', 'study', 'see my friend']),
    ("then i'm going to study, and later i will visit my mom", ['study', 'visit my mom']),
    ('I need to email James and call the bank', ['email James', 'call the bank']),
    ('I need to do my taxes', ['do my taxes']),
    ('Help me prioritize my day', []),
    ('help me organise them', []),
    # The verb inside a request is not a task.
    ('please organise my day', []),
])
def test_extraction_variants(text, expected):
    assert titles(text) == expected


def test_arabic_list_with_a_request_tail():
    assert titles('لازم اروح للدكتور واصلي وادرس واشوف صديقي ورتبهم لي حسب الاولوية') == [
        'اروح للدكتور', 'اصلي', 'ادرس', 'اشوف صديقي']


def test_the_reported_message_is_saved_and_ordered(agent):
    result = agent.organize(request(MESSAGE), NOW)
    assert [op.operation for op in result.taskOperations] == ['ADD'] * 4
    order = [r.task.title for r in result.plan.rankedTasks]
    assert order == ['go to the doctor', 'pray', 'study', 'see my friend']
    text = result.response.text
    # Every task is shown, in order, even though the plan's secondary list holds two.
    assert [line for line in text.splitlines() if line[:2] in ('1.', '2.', '3.', '4.')] == [
        '1. Go to the doctor', '2. Pray', '3. Study', '4. See my friend']
    assert 'Check the time and place' in text
    assert 'I ordered by what has a fixed time first' in text


def test_the_reason_for_the_order_is_in_the_factors(agent):
    result = agent.organize(request(MESSAGE), NOW)
    reasons = {r.task.title: [f.name for f in r.factors] for r in result.plan.rankedTasks}
    assert 'fixed_time' in reasons['go to the doctor'] and 'fixed_time' in reasons['pray']
    assert 'flexible_social' in reasons['see my friend']


def test_a_stated_time_outranks_a_named_event(agent):
    result = agent.organize(request(
        'I have to study for my exam, buy groceries, see the dentist at 4pm and call mom, '
        'can you prioritize them?'), NOW)
    assert [r.task.title for r in result.plan.rankedTasks][:2] == ['see the dentist', 'study for my exam']
    # A stated clock time is the patient's own information, so no "I guessed" note.
    assert 'I ordered by' not in result.response.text


def test_a_stated_deadline_still_outranks_the_names(agent):
    result = agent.organize(request(
        'I need to see my friend today and I have to pray and finish the report by 2026-09-30, '
        'prioritize them'), NOW)
    assert result.plan.rankedTasks[0].task.title == 'finish the report'


def test_planning_with_an_empty_list_asks_for_the_list(agent):
    for text in ('Help me prioritize my day', 'what should I do first today?'):
        result = agent.organize(request(text), NOW)
        assert result.response.text.startswith('Tell me what is on your plate today')
        assert result.taskOperations == []


def test_arabic_planning_with_an_empty_list(agent):
    result = agent.organize(request('رتب لي يومي', lang='AR'), NOW)
    assert result.response.text.startswith('اخبرني بما عليك اليوم')


def test_adding_tasks_confirms_what_was_saved(agent):
    result = agent.organize(request('I need to email James and call the bank'), NOW)
    assert result.response.text.startswith('Added: Email James; Call the bank.')


def test_low_capacity_shortens_an_explicit_ordering_request_to_three(agent):
    result = agent.organize(request(MESSAGE, currentState={'energy': 'LOW', 'stress': 'HIGH'}), NOW)
    assert '3. Study' in result.response.text
    assert '4. See my friend' not in result.response.text


def test_low_capacity_with_a_single_task_is_still_one_small_start(agent):
    result = agent.organize(request('I need to email James', currentState={'energy': 'LOW', 'stress': 'HIGH'}), NOW)
    assert 'Just one small start.' in result.response.text or 'One thing for now.' in result.response.text


@pytest.mark.parametrize('text', ['order them', 'help me', 'help me organise them',
                                  MESSAGE + ' and i want to order them'])
def test_bare_requests_are_never_tasks(text):
    assert not any(t.title in ('order them', 'help me') for t in extract_tasks(text, uuid4(), REF)[0])


def test_order_them_orders_the_tasks_already_on_the_list(agent):
    tasks = [{'temporaryId': f't{i}', 'title': title} for i, title in
             enumerate(['see my friend', 'study', 'go to the doctor', 'pray'])]
    result = agent.organize(request('order them', tasks=tasks), NOW)
    assert result.intent == 'PRIORITIZE'
    assert '1. Go to the doctor' in result.response.text and '4. See my friend' in result.response.text


def test_bare_help_asks_for_the_list(agent):
    assert agent.organize(request('help me'), NOW).response.text.startswith('Tell me what is on your plate')


@pytest.mark.parametrize('title, expected', [
    ('go to the doctor', 'Check the time and place'),
    ('pray', 'get ready for prayer'),
    ('see my friend', 'Send a short message'),
    ('call the doctor', 'Find the contact details'),
])
def test_first_step_fits_the_task(agent, title, expected):
    result = agent.organize(request(f'I need to {title}'), NOW)
    assert expected in result.plan.nextAction.text


def test_tasks_are_written_in_ranked_order(agent):
    result = agent.organize(request(MESSAGE), NOW)
    written = [op.task.title for op in result.taskOperations if op.operation == 'ADD']
    assert written == ['go to the doctor', 'pray', 'study', 'see my friend']


@pytest.mark.parametrize('text', [
    'no i want you to order my day i have a workout and school and praying',
    'going to workout and school and praying',
])
def test_verbless_commitments_are_saved_and_ordered(agent, text):
    result = agent.organize(request(text), NOW)
    assert sorted(op.task.title.lower() for op in result.taskOperations) == ['praying', 'school', 'workout']
    assert '1. School' in result.response.text and '3. Workout' in result.response.text


def test_adding_a_task_orders_the_whole_open_day(agent):
    stored = [{'temporaryId': f't{i}', 'title': title} for i, title in
              enumerate(['study', 'go to the doctor', 'pray'])]
    result = agent.organize(request('see the dentist', tasks=stored), NOW)
    text = result.response.text
    assert text.startswith('Added: See the dentist.')
    assert [line for line in text.splitlines() if line[:2] in ('1.', '2.', '3.', '4.')] == [
        '1. Go to the doctor', '2. Pray', '3. See the dentist', '4. Study']


def test_restating_a_task_does_not_duplicate_it(agent):
    stored = [{'temporaryId': 't0', 'title': 'pray'}, {'temporaryId': 't1', 'title': 'go to the doctor'}]
    result = agent.organize(request('going to workout and praying and see the doctor', tasks=stored), NOW)
    assert [op.task.title for op in result.taskOperations if op.operation == 'ADD'] == ['workout']
