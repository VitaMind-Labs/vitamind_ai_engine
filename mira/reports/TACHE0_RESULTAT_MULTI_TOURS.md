# TÂCHE 0 — Résultat du test d'intégration multi-tours (avant / après)

**Objectif:** Prouver que le fix Workstream A/B (bipolaire 10/10) fonctionne à travers l'API réelle, tour par tour, pas seulement en bloc isolé.

**Transcript utilisé:** `BIPOLAR_TRANSCRIPT` 8 messages (issu de `scratch/diagnose_bipolar.py` / `tests/test_bipolar_regression.py`), chaque message envoyé via `POST /api/v1/mira/session/{id}/message` séparé.

**Attendu isolé (diagnostic direct via `FeatureExtractor` + `AssessmentEngine`):**
- `BIPOLAR_SPECTRUM` raw_score = 10.0 / 10
- `supporting_evidence` = 10/10: `decreased_need_for_sleep`, `elevated_mood`, `grandiosity`, `racing_thoughts`, `pressured_speech`, `flight_of_ideas`, `impulsive_spending`, `increased_goal_directed_activity`, `episodic_pattern`, `depression_alternation`
- `condition_scores.bipolar` = 1.0
- `contradictory_features` ne contient pas `decreased_need_for_sleep` (bug fixé: crash-phase fatigue ne déclasse plus le sleep)

---

## Sortie brute AVANT correction T2 (hardcode 8)

```bash
python -m pytest tests/test_tache0_bipolar_multitour.py -v  # avant T2
```
```
=== ISOLATED ===
leading BIPOLAR_SPECTRUM
BIPOLAR_SPECTRUM raw 10.0 supporting [...]  # 10/10
bipolar isolated 10.0
=== MULTI-TOUR ===
session dc788... 
Tour 1 complete False progress 0.125 GET 0.125
...
Tour 8 complete True progress 1.0 GET 1.0 complete True
FINAL scores {'adhd': 0.0, 'bipolar': 1.0, 'psychosis': 0.0}
supporting [...] 10/10
contradictory []
# COMPARISON: Isolated 10.0 == Multi-tour 10.0 -> Ecart 0
```
**Verdict AVANT:** ✅ Vert, 10/10, 1.0 atteint, écart 0 entre isolé et multi-tours.  
La façon dont `PatientState` accumule les observations entre tours (`session.messages` + `observations` list + `get_status` agrégé) est correcte. Aucune correction nécessaire dans `interview.py` / `orchestrator` pour T0.

**Écart détecté dans T0 (signalé, non corrigé silencieusement):**
- `GET /api/v1/mira/session/{id}` calcule `min(1.0, messages / 8)` avec 8 hardcodé.
- `MiraAgent.respond()` fait `messages >= 8` et `min(0.99, messages / 8)`.
- Or Tâche 2 révèle que la banque offre 17 free_text utilisables (19 total), donc default ≈10 plus cohérent. **Écart noté pour T2.**

---

## Sortie brute APRÈS correction T2 (default 10, plafond 17)

Après T2, le dénominateur passe à `DEFAULT_SESSION_QUESTIONS = 10` (choix documenté: 17 free_text utilisables → 10 ≈ moitié). Le test T0 est mis à jour pour refléter le nouveau contrat: 8 messages donnent déjà 10/10 en interne, mais la session ne se clot qu'à 10 (sans phrase d'extension).

```bash
python -m pytest tests/test_tache0_bipolar_multitour.py -v  # après T2
```
```
tests/test_tache0_bipolar_multitour.py::test_bipolar_isolated_score PASSED
tests/test_tache0_bipolar_multitour.py::test_bipolar_multitour_via_api_matches_isolated PASSED
2 passed in 1.45s
```

Détail du tour après T2:
```
Tour 1 progress 0.1 GET 0.1 complete False
...
Tour 8 progress 0.8 GET 0.8 complete False   # score déjà 1.0 en interne, mais pas encore complete
Tour 9 (padding) progress 0.9
Tour 10 (padding) progress 1.0 complete True -> result bipolar 1.0, supporting 10/10
```
**Comparaison isolé vs multi-tours après T2:**
- Isolé: 10.0 /10 =1.0
- Multi-tours (8 tours +2 padding pour atteindre seuil 10): 10.0 /10 =1.0
- **Écart 0, même cause racine validée.** La correction T2 n'a pas introduit de régression sur l'extraction; elle a seulement déplacé le seuil de cloture.

**Action prise sur chapitre_progress:**
- `app.py` GET: `messages / 8` → `messages / DEFAULT_SESSION_QUESTIONS (10)` (import depuis `interview.py`)
- `agent.py`: `messages >=8` → `messages >= MAX (17)` / `DEFAULT (10)` avec logique d'extension, `min(0.99, messages/10)`
- **Justification listée dans rapport final (Tâche 2).** Aucun seuil d'`uncertainty.py` touché (non nécessaire pour T0).

---

## Conclusion T0

- ✅ Test d'intégration vert avant et après, sans Docker, via `TestClient` en process.
- ✅ Aucune correction de `PatientState` nécessaire — l'accumulation était déjà correcte.
- ✅ Écart `hardcode 8` signalé et corrigé en T2 de façon documentée (pas silencieuse).
