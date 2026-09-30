# -*- coding: utf-8 -*-
"""A completion reported inside a longer message closes the task it names.

Found in a two-turn walk-through: after "call my mom" was saved, "I called my mom. Also study
for the exam again and phone mom tonight." added a task but left "call my mom" open, because a
completion was only recognised when the whole message was labelled as one.
"""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from lumina.adhd.executive_function.language import completion_clauses
from lumina.adhd.executive_function.orchestrator import LuminaADHD
from lumina.adhd.executive_function.schemas import OrganizeRequest

PATIENT = 'ae3e23cc-5b26-468f-a8df-708519a4a144'
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
OPEN = [{'temporaryId': 'task_call', 'title': 'call my mom'}, {'temporaryId': 'task_study', 'title': 'study for my exam'},
        {'temporaryId': 'task_pray', 'title': 'pray'}]


def turn(text, tasks=OPEN):
    payload = {'requestId': str(uuid4()), 'patient': {'id': PATIENT, 'language': 'EN'}, 'message': {'text': text},
               'timeContext': {'referenceDate': '2026-09-30'}, 'tasks': tasks}
    return LuminaADHD().organize(OrganizeRequest.model_validate(payload), NOW)


def closed(result):
    return sorted(o.taskId for o in result.taskOperations if o.operation == 'COMPLETE')


def added(result):
    return [o.task.title for o in result.taskOperations if o.operation == 'ADD']


def test_a_completion_and_a_new_task_in_one_message():
    result = turn('I called my mom. Also study for the exam again and phone mom tonight.')
    assert closed(result) == ['task_call']
    assert 'study for the exam' not in ' '.join(added(result))      # already on the list, not a second copy


@pytest.mark.parametrize('text', [
    'I need to call my mom', "I haven't called my mom", "I didn't finish studying for my exam", 'call my mom tomorrow',
])
def test_plans_and_negations_are_not_completions(text):
    assert closed(turn(text)) == []


def test_only_the_named_task_is_closed():
    assert closed(turn('I prayed and I have to buy milk')) == ['task_pray']


def test_a_task_added_in_the_same_message_is_not_closed_by_it():
    result = turn('I finished my report. I need to email James', tasks=[])
    assert closed(result) == [] and 'email James' in added(result)


def test_clauses():
    assert completion_clauses('I called my mom. Also study again') == ['i called my mom']
    assert completion_clauses('I have to call mom') == []
