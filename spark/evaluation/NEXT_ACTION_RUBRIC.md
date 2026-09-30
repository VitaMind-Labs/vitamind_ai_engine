# Next-action quality rubric

Score independent planning cases only after freezing rules/model selection. Keep capacity, provided tasks, time, expected constraints and annotator rationale with each score. Use multiple reviewers where possible and record disagreements.

| Score | Meaning |
|---|---|
| 0 | Irrelevant, impossible, unsafe, or fabricated task/context |
| 1 | Related but too large, vague, or inappropriate to available capacity |
| 2 | Useful and executable, with an identifiable starting point |
| 3 | Specific, low-friction, immediately executable and appropriate to capacity |

Additional binary checks: no invented deadline; no asserted task duration without evidence; no unsupported urgency; no task unrelated to the request; no normal coaching during safety handoff; no more than one next action for low capacity; valid date/reference handling; correct language; no overfull time budget.

The supplied CSV does not contain planning requests with next-action annotations. No human-scored held-out result is claimed in this release. The included automated development tests are not a substitute for this rubric or clinical review.
