# VitaMind Mira V5 — Steps 1 to 8

This package now contains the core non-frontend AI architecture for Mira.

## Architecture

The service is split into four boundaries:

- `vitamind/clinical`: patient state, extraction, assessments, differential, and uncertainty
- `vitamind/safety`: independent urgent-safety detection
- `vitamind/mira`: session application service and interview planning
- `app.py`: FastAPI adapter only; it owns HTTP validation and session lifecycle wiring
- `vitamind/ml`: lazy adapter for the optional persisted sklearn ensemble

The HTTP contract is:

```text
GET    /health
POST   /api/v1/mira/session
POST   /api/v1/mira/session/{session_id}/message
GET    /api/v1/mira/session/{session_id}
DELETE /api/v1/mira/session/{session_id}
```

## Implemented

### Step 1 — Patient State
`vitamind/clinical/patient_state.py`

Canonical clinical memory.

### Step 2 — Feature Extractor
`vitamind/clinical/feature_extractor.py`

Converts user language into structured clinical observations and writes them
to PatientState. Includes explicit negation and cross-feature logic.

### Step 3 — Condition Assessments

- `vitamind/clinical/assessment/adhd.py`
- `vitamind/clinical/assessment/bipolar.py`
- `vitamind/clinical/assessment/psychosis.py`

Each produces:
- supporting evidence
- contradictory evidence
- missing information
- prototype raw score

### Step 4 — Assessment Engine
`vitamind/clinical/assessment/assessment_engine.py`

Runs all three target assessments together and identifies the current leading
target condition.

### Step 5 — Adaptive Interview Planner
`vitamind/mira/interview.py`

Chooses the next clinically useful question based on missing information and
competing target conditions.

### Step 6 — Safety Detector
`vitamind/safety/detector.py`

Runs independently from diagnosis logic and can flag urgent safety content.

### Step 7 — Differential Engine
`vitamind/clinical/differential.py`

Checks anxiety, depression, sleep deprivation, substances, medications, and
medical/neurologic explanations.

### Step 8 — Uncertainty Engine
`vitamind/clinical/uncertainty.py`

Allows Mira to abstain instead of forcing ADHD/Bipolar/Psychosis when evidence
is weak, conflicting, incomplete, or better explained elsewhere.

## Run tests

```powershell
cd apps/ai-service
python -m pip install -e ".[test]"
python -m pytest -q
```

## Run the service

```powershell
cd apps/ai-service
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Important

These scores are prototype engineering signals only.
They are not calibrated medical probabilities and must not be used as a
clinically validated diagnosis.

The persisted model was trained with scikit-learn 1.7.2. Install the optional
`ml` extra with that version when reproducible model inference is required.
