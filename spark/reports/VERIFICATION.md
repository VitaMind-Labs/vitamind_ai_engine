# Verification — trained edition

117 tests passed on 2026-09-29 with the actual trained intent and friction artifacts loaded. The suite covers model-export parity, both language paths, artifact corruption rejection, API readiness/version fields, dataset provenance/group boundaries, calendar persistence/isolation/idempotency, ambiguous dates, recovery, safety precedence and an offline startup tripwire. One Starlette/AnyIO dependency deprecation warning was emitted.

The earlier dependency-access limitation was resolved for this run by the granted network-enabled environment. The previous folder's incomplete rerun status no longer applies to this trained edition.

The held-out models were frozen before scoring. `planning-heldout.json` is the classifier evaluation; `evaluation.json` contains known development probes and latency. Synthetic results are not independent patient validation. There is no evidence that 117 software checks establish clinical reliability.

ZIP portability verification is recorded by the packaging step below.

## ZIP portability — passed

Extracted to a fresh temporary directory. All 117 tests passed there with the shipped models, four CLI JSON examples passed, and an interactive save/plan/JSON workflow completed. Manifest hashes matched. Rebuilding the dataset preserved its exact SHA-256 hash. Normal inference did not require training dependencies, API keys or downloaded models. Dependencies themselves are installed separately, not bundled. The final ZIP adds only this report and the captured test log after verification.
