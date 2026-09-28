# Mira report interpretation

## Status of the displayed report

The displayed report is structurally valid for the data that reached the AI service.
It is an orientation report, not a diagnosis.

The two supporting observations shown are `decreased_need_for_sleep` and `grandiosity`.
The bipolar raw score is therefore `2 / 10 = 0.20`, which maps to `LOW` under the
existing assessment engine. The `mood_disorder_clinical_assessment` pathway identifies
the area for qualified clinical follow-up; it does not establish a condition.

The long missing-information list is expected when the conversation does not provide
the corresponding evidence. A low score with missing information must not be read as
a negative clinical conclusion.

## Question count

The default session closes after 10 user responses. The opening prompt is sent when
the session starts and is not counted as a user response. Each later POST to
`/message` increments `session.messages` by one. The progress denominator is
`DEFAULT_SESSION_QUESTIONS`, currently 10.

Therefore, a report generated after fewer than 10 user responses indicates one of
these situations:

- the session was explicitly completed by an urgent safety event;
- the session was extended or resumed from an earlier stored session;
- the visible conversation did not include every request/response;
- an older client or proxy stopped displaying messages while the backend continued.

The backend now returns `chapter_progress`, `assistant_message`, `language`, `safety`,
and `result` through the session endpoints so the client can resynchronize instead
of estimating the count locally.

## Arabic sessions

A session language is fixed at creation. The frontend stores the AI `session_id` by
chat ID, calls GET session on reload, reads the authoritative session language, and
uses that language for every subsequent message. Changing the global interface
preference cannot change the language of an existing Mira session.

An Arabic session must therefore have:

- an Arabic opening question;
- Arabic follow-up questions when Arabic text exists in the question bank;
- `language: "ar"` on start, message, GET, and result payloads;
- RTL direction in the chat and report.

If an Arabic question-bank entry has no Arabic text, the AI service logs a warning
and does not serve its English text. The current report generator still emits
English clinical pathway labels and disclaimer text because no Arabic report-text
contract exists in the source-of-truth assessment output. This is a documented data
contract gap, not a frontend translation.

## Contract errors

`language does not match session` is a correct AI-service rejection when a client
sends a language different from the language stored on the session. The frontend
fix prevents this error during normal reloads and language switching by sending the
stored session language.
