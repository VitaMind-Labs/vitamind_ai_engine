# VitaMind Lumina Integration Requirements

## Scope: Lumina, VitaMind_Journal_AI, daily check-ins, end-of-day journal, and minimal Prisma integration

**Purpose:** This document directs an engineering agent to inspect the real repository architecture first, then implement only the changes needed to make Lumina and VitaMind_Journal_AI work together correctly. It deliberately keeps Mira's current logic out of scope.

## 1. Repository and scope

Use the repository that exists in this workspace:

- `vitamind_agent/mira/`
- `vitamind_agent/lumina/VitaMind_Journal_AI/`
- `vitamind_backend/apps/api/`

Do not assume a `vitamind_ai_engine/` directory exists. Inspect the current tree and recent code before making decisions.

### In scope

- Audit the existing Lumina and VitaMind_Journal_AI implementations, their contracts, dependencies, tests, and entry points.
- Define and implement a clear boundary through which Lumina uses VitaMind_Journal_AI for journal-entry analysis.
- Implement the intended daily workflow: the patient's first daily action is a check-in; the journal is offered at the end of the day.
- Reconcile the relevant Prisma models and the minimum backend integration needed to validate, persist, and retrieve this workflow's data.
- Produce implementation notes and test evidence based on the actual repository.

### Out of scope

- **Do not modify anything under `vitamind_agent/mira/`.** This includes Mira's model, prompts, safety rules, endpoints, DTOs, tests, and behavior. Mira is read-only for this task.
- Do not redesign Mira or add longitudinal memory to it. Existing Mira output may be inspected as an input to a future handoff, but do not change its contract or implementation in this task.
- Do not rewrite the backend, rebuild unrelated admin/psychologist features, or add a new deployment platform.
- Do not apply a proposed Prisma schema wholesale or remove existing models without usage analysis and a migration plan.
- Do not claim functionality is complete without real test or command output.

Backend changes are permitted only where required to connect Lumina, enforce authorization/consent, persist its validated outputs, or reconcile the relevant Prisma data model. Keep that work smaller than the Lumina work.

## 2. Required daily workflow

The product flow is ordered by the patient's day, not by the order in which modules currently happen to exist.

### First daily action: check-in

1. The patient submits a daily check-in as the first intended action of the day.
2. The owning application validates and persists the check-in.
3. Lumina receives only the authorized check-in context required for its response and current-state logic.
4. Lumina returns a validated, structured response. Persist only fields allowed by the agreed contract and current consent.
5. Retrying the same submission must not create duplicate check-ins or duplicate side effects.

Before defining fields, inspect the existing DTOs, agent inputs/outputs, Prisma schema, frontend usage if available, and tests. Do not invent clinical fields. If the current system does not define a required field, document a minimal proposal and its rationale before adding it.

### End of day: journal

1. At the end of the day, the patient may submit a journal entry reflecting that day's experience. Journaling is optional; a missing entry must not block other use.
2. The original entry is persisted by the backend and remains the source record. Analysis must not overwrite it.
3. Lumina invokes or consumes VitaMind_Journal_AI through an explicit, versioned contract. Journal AI analyzes an entry; it does not own persistence, run a separate chat, or diagnose.
4. The analysis is validated and stored separately, linked to the journal entry and the content/analysis versions.
5. Lumina may use authorized structured analysis in later interactions. Raw journal text is not exposed to chat or a clinician by default.
6. A retry for the same entry content and analysis version must be idempotent. An edited entry must be distinguishable from its previous analysis.

Do not silently run the end-of-day workflow as a morning action. If scheduling or timezone behavior is not represented in the current product, document the chosen minimal behavior and avoid inventing a forced schedule.

## 3. Responsibility boundaries

- **Mira:** Existing initial orientation agent. Read-only in this task; no logic or contract changes.
- **Lumina:** Longitudinal support agent. It coordinates the authorized check-in context, available journal analyses, and other permitted context found in the real code/specification.
- **VitaMind_Journal_AI:** Journal analysis capability used by Lumina through a defined interface. It must not become a second patient-facing chat agent or write directly to the database.
- **Backend:** Source of truth for patient identity, authorization, consent, input validation, database writes, idempotency, and access control.

Keep agent boundaries explicit. Do not let one agent write to PostgreSQL. Do not make Lumina depend on Mira's internal runtime state. If the existing architecture has a structured Mira-to-Lumina handoff, document how Lumina can consume it without changing Mira; do not build or modify that handoff unless it is strictly necessary for the requested workflow.

## 4. Phase 0: audit before coding

Do not edit application code until this audit is complete. Begin with `vitamind_agent`, especially Lumina and VitaMind_Journal_AI; inspect the backend only as needed to establish the data and integration boundary.

### Inspect Lumina and VitaMind_Journal_AI

Record:

- Actual folder map, language/framework, entry points, dependencies, and run commands.
- Existing request/response schemas, callable functions, model/provider integrations, prompts, safety handling, and error behavior.
- Which logic currently belongs to Lumina and which to VitaMind_Journal_AI.
- Any persistence, networking, duplicated logic, dead code, or implicit coupling.
- Existing tests and what they actually cover.

Do not presume the module is incomplete or complete based on a prior prompt. Prove its current state from code and tests.

### Inspect the minimal backend surface

Record:

- Relevant NestJS modules, controllers, DTOs, services, guards, consent checks, and scheduled jobs.
- Existing check-in and journal write/read endpoints, if any.
- Relevant Prisma models, relations, indexes, uniqueness, and migrations.
- Real `prisma.<model>` / `tx.<model>` reads and writes for the relevant models.
- Whether `AIJob` or another job mechanism has an active producer and consumer.

Treat the earlier backend analysis and existing API documents as leads, not as substitutes for checking current code.

### Audit deliverables

Before coding, create concise documents in an existing, appropriate documentation location:

1. `AUDIT_LUMINA.md`: actual architecture, boundaries, contracts, gaps, test commands, and risks.
2. `PLAN_LUMINA_INTEGRATION.md`: sequenced implementation plan, affected areas, migration strategy, and validation gates.
3. A Prisma usage matrix for relevant models: model, actual reads/writes, source files, status (`ACTIVE`, `PARTIAL`, or `NO DIRECT ACCESS FOUND`).
4. A daily-flow diagram showing check-in first and optional journal analysis at day's end.

Do not call a model "unused" solely because it has no endpoint. Distinguish no direct access in the audited code from confirmed non-use.

After producing the audit and plan, proceed without asking for routine approval, provided the implementation stays within this scope. Record important assumptions and alternatives in a short decisions section or the existing decision log.

## 5. Lumina and Journal AI contract

Derive the contract from actual code and tests first. Do not copy a speculative schema from a prompt without verifying compatibility.

The journal-analysis boundary must document, where supported by the implementation:

- Input content and permitted metadata.
- Language and content/version identifiers.
- Structured summary/signals, safety output, uncertainty, and candidate follow-ups or memory updates.
- Model/provider and analysis/contract version.
- Validation errors, timeouts, and provider-unavailable behavior.

Requirements:

- Use strict input and output validation at the boundary.
- Keep journal content separate from system/developer instructions; treat it as untrusted data and test prompt-injection-like content.
- Preserve the original patient entry on any analysis failure.
- Keep the original journal entry and derived analysis in separate records or clearly separate fields with an explicit version relationship.
- Enforce privacy and consent before providing journal data to Lumina or a clinician.
- Do not send raw journal text to ordinary Lumina chat by default. Prefer approved structured analysis; any exception requires a documented use case, authorization, and consent.
- Do not invent diagnoses, medication advice, safety text, emergency resources, or unapproved intervention content.

## 6. Prisma reconciliation

Start from the current `vitamind_backend/apps/api/prisma/schema.prisma`, migrations, and actual code usage. Do not assume the currently attached schema or an external proposal is the final target.

For each relevant model, decide and document one of:

- `KEEP`: actively used and suitable.
- `KEEP_AND_EXTEND`: used, but needs a justified additive field/index/constraint.
- `KEEP_PARTIAL`: used by only part of its declared purpose.
- `DEPRECATE`: retain for backward compatibility while a replacement is introduced.
- `REMOVE_WITH_MIGRATION`: only when usage and product need are disproven and a safe migration path is provided.

Every field that an agent output is persisted to must have an identified writer and reader. Every new field/model must map to a specific workflow and have a test. Relevant requirements include:

- Persist check-ins idempotently according to the existing product's day/timezone rules, if those rules are available.
- Persist journal entries independently from journal-analysis results.
- Make analysis re-runs distinguishable by content version and analysis version.
- Store only validated structured agent results; do not persist chain-of-thought or raw prompts.
- Preserve existing psychologist/admin data and APIs unless an explicit compatibility migration is included.

Use expand/backfill/switch/contract for destructive or renaming changes. Validate the actual Prisma version/configuration before running generation or migration commands. Document commands that cannot run due to missing database access.

## 7. Implementation priorities

Work in this order:

1. Complete and record the audit and plan.
2. Establish or improve the Lumina-to-Journal-AI interface in `vitamind_agent/lumina/` using the smallest maintainable change.
3. Test the agent boundary locally with deterministic fixtures or the repository's existing fake/test provider, if available.
4. Add only the necessary check-in-first and end-of-day journal orchestration to Lumina.
5. Add the minimum backend endpoints/services/worker changes needed for authorization, consent, validation, idempotency, and persistence.
6. Make additive Prisma changes and migrations only when the audited contracts require them.
7. Update documentation from actual implementation and captured test results.

Do not add a new queue, Redis topology, service launcher, HMAC protocol, global API prefix, report system, subscription policy, or broad observability platform unless the audit proves it is necessary to this specific workflow. Record such work as future scope instead.

## 8. Safety, privacy, and reliability

- Backend authorization and consent checks must happen before patient data is included in agent context.
- The agents return structured results; they do not mutate database state.
- Do not log raw check-in text, journal text, prompts, secrets, or recovery codes.
- Keep safety output explicit and validated. Unknown or malformed safety values must fail safely and be observable; do not silently map them to a low-risk result.
- Preserve patient input if Lumina, Journal AI, or a model provider fails.
- Apply bounded timeouts and avoid unbounded retries or duplicate jobs.
- Keep Mira source and behavior unchanged. Run relevant Mira regression tests where practical and report their results; do not edit Mira tests to make them pass.

## 9. Minimum acceptance criteria

The work is acceptable only when evidence demonstrates:

1. The audit identifies the actual Lumina and VitaMind_Journal_AI architecture and does not assume missing code or specs.
2. A check-in can be validated and persisted as the first daily workflow step, and Lumina receives the authorized structured context.
3. A patient can optionally submit an end-of-day journal entry; its original content is preserved independently from its analysis.
4. Lumina uses VitaMind_Journal_AI through a tested, explicit contract.
5. Analysis is idempotent for the same entry/content/analysis version and can be re-run for a changed version without overwriting history.
6. A failed or invalid analysis does not lose the patient entry or produce an unvalidated database write.
7. Privacy/consent rules prevent unauthorized journal exposure.
8. Relevant Prisma schema validation and available migration tests pass, or the exact environmental blocker is documented.
9. Mira files are unchanged and its relevant existing tests remain passing, or any unrelated pre-existing failure is reported accurately.
10. The final report gives exact commands and observed outputs; unrun tests are explicitly labeled.

## 10. Final documentation and report

Write in English. Keep code identifiers in their original spelling. Document:

- The actual Lumina and VitaMind_Journal_AI architecture and responsibility split.
- The daily workflow: morning/first-action check-in, optional end-of-day journal, analysis, and later Lumina use.
- Verified input/output contracts and error behavior.
- The field mapping from check-in and agent outputs to Prisma models/columns.
- Prisma decisions and migration/backfill/rollback notes.
- Exact local run and test commands that exist in the repository.
- Changed files, test output, unresolved gaps, and any clinical-review requirements.

Do not claim all tests pass if they were not run. Do not include fabricated "real" payloads; examples must be copied from actual tests or labeled clearly as illustrative.


how to run the audit and tests, and how to validate the workflow, must be included in the final report.
python vitamind_agent/run.py --env-file vitamind_agent/scripts/.env