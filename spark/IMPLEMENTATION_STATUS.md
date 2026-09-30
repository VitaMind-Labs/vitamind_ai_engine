# Delivered status

This is a separate folder based on the previous standalone assistant. Mira, VitaMind repositories and the older delivered folder were not modified.

| Area | Current implementation |
|---|---|
| Intent | Trained local linear SVM for EN/AR plus an explicit-command guard |
| Difficulty/friction | Trained local multi-label logistic regression for EN/AR |
| Mixed language | Rule fallback; not promoted as trained coverage |
| Data | 7,844 public/synthetic labeled records with grouped partitions and provenance |
| Extraction/dates | Existing bounded bilingual rules; no trained slot extractor |
| Planning/replies | Explained prioritization, focus selection and controlled templates |
| Calendar | Optional local SQLite, patient scoping, idempotent requests |
| Interruption/outcomes | Supplied context and repeated structured evidence; not unrestricted conversation memory |
| Safety | Existing local journal classifier and rules, unchanged weights; important false-positive/false-negative limitations |
| API | Local FastAPI with active model versions and readiness status |
| UI/backend integration | None |
| Clinical validation | None |

This update improves classification over the rules on the recorded holdout. It does not demonstrate arbitrary language understanding, accurate diagnosis, clinical benefit or a fully autonomous conversational planner. Task facts are still not inferred when missing.
