# TÂCHE 1 — RAPPORT D'ANALYSE DES DONNÉES (préalable au code)

**Date:** 2026-09-22  
**Service:** ai-service (`vitamind_backend/apps/ai-service`)  
**Auteur:** Muse Spark (audit local, sans Docker)  
**Principe:** aucune ligne de code métier modifiée avant ce rapport (seul test T0 ajouté pour validation).

---

## 1. Inventaire des fichiers `data/`

```
data/
├── arabic_raw/
│   ├── train.csv        (47.6 MB, 21780 lignes)
│   ├── validation.csv   (8.4 MB, 3844 lignes)
│   └── test.csv         (9.9 MB, 4522 lignes)
├── Mira_Bilingual_Dynamic_Question_Bank.json  (33.3 KB, 43 questions)
├── Mira_V1_Data_Dictionary.csv               (12.8 KB, 61 lignes)
└── Mira_V1_Synthetic_Training_Dataset.csv     (157.8 MB, 30000 lignes)
```

---

## 2. Analyse détaillée par fichier

### 2.1 `arabic_raw/{train,validation,test}.csv`

**Structure exacte:**
| colonne | type | exemple |
|---------|------|---------|
| `text` | string | conversation complète patient+bot en arabe, concaténée avec `|` |
| `dialect` | string | `MSA`, `EGY`, `GLF`, `LEV`, `IRQ`, `MAG`… |
| `chatbot_related_questions` | bool string | `True`/`False` (rare: 196/21780 en train) |
| `Label` | int 0-5 | 6 classes équilibrées (~3610 chacune) |
| `cleaned_text` | string | même contenu que `text`, nettoyé |
| `char_count` | int | ~500-600 |
| `word_count` | int | ~100-120 |

**Volumes:**
- train: 21780 (6 labels × ~3612, équilibré)
- validation: 3844 (6 labels × ~638)
- test: 4522 (6 labels × ~750)
- **Total: 30146 lignes** — toutes en arabe (100%)

**Format de texte:**
- Phrases complètes, style conversationnel MSA / dialectal mélangé.
- Exemple `train[0]` (MSA): « مرينا بظروف صعبة في الغربة وظهرت عندي أعراض جسدية شديدة... » + réponses bot empathiques + suites patient (« نعم. طبيب الأعصاب أجرى تخطيطا... »).
- **Pas de mots-clés isolés**, mais de vrais échanges multi-tours (patient 2-4 tours par ligne, total ~100 mots).
- Dialectes: MSA dominant (11463/21780 = 52.6% train), suivi EGY (23%), GLF/LEV (~11% chacun), puis minoritaires MAG, IRQ, NORD, etc. **Pas de FR/TN (Derja Arabizi latin)** — l'arabe est en script arabe standard.

**Labels de condition:**
- **Présence:** oui, colonne `Label` 0-5 sur chaque ligne.
- **Absence de mapping explicite:** aucune doc ne lie 0-5 à ADHD / bipolar / psychosis. Audit du contenu montre que `Label` ne correspond PAS à nos 3 conditions cibles (ex. Label=1 contient des textes sur « trouble du sommeil » et « anxiété » mélangés). Il s'agit d'un dataset arabe **générique santé mentale / chatbot**, pas du labelling Mira V1 (qui utilise `target_training_label` anglophone distinct).
- **Conséquence:** `arabic_raw` **ne fournit pas directement** de supervision ADHD/bipolar/psychose par ligne. Il peut servir de **source de phrases-déclencheurs (triggers) réelles en arabe** après filtrage manuel par mots-clés (ex. « قلة النوم بدون تعب » → sleep), mais pas de fine-tuning supervisé sans re-labellisation.

**Qualité pour Tâche 3:**
- ✅ Directement exploitable comme **réservoir de formulations naturelles** pour enrichir les regex arabes de `feature_extractor.py` (patterns déjà présents mais perfectibles).
- ⚠️ Nécessite **transformation**: filtrage par regex arabes préexistants pour extraire les snippets pertinents, puis validation manuelle (pas de label clinique fiable).
- ❌ Non exploitable tel quel pour de l'apprentissage supervisé condition-spécifique sans mapping 0-5 → dictionnaire.

---

### 2.2 `Mira_Bilingual_Dynamic_Question_Bank.json`

**Structure exacte (43 entrées, JSON array):**
```json
{
  "question_id": "U01A",          // string unique, ex. U01A, B02A, S01B
  "module": "UNIVERSAL",          // UNIVERSAL|ADHD|BIPOLAR|PSYCHOSIS|DIFFERENTIAL|SAFETY|COMPLETION
  "question_group": "OPEN_STORY", // sous-groupe clinique
  "input_type": "free_text",      // free_text|single_select|multi_select|story_select|slider
  "display_style": "story_prompt",// story_prompt|large_cards|slider|...
  "question_en": "What made you decide...",
  "question_ar": "ما الذي جعلك تقرر...",
  "options_en": "Focus, memory|Big changes...", // vide si free_text
  "options_ar": "التركيز أو الذاكرة|تغيرات كبيرة...", // vide si free_text
  "target_feature": "chief_concern;free_narrative", // domaine clinique visé
  "branching_note": "",          // vide dans ce dump
  "required": "yes"              // yes|no
}
```

**Décompte:**
- Total: **43 questions**
  - Par `module`: UNIVERSAL 9, ADHD 9, BIPOLAR 8, PSYCHOSIS 8, DIFFERENTIAL 4, SAFETY 3, COMPLETION 2
  - Par `input_type`: free_text **19**, single_select 17, multi_select 4, story_select 2, slider 1
  - Par `question_group`: OPEN_STORY 3, BRANCH_CHOOSER 2, TIMELINE 2, IMPAIRMENT 2, ADHD_FOCUS 3, etc. (21 groupes, 2 questions par groupe sauf OPEN_STORY=3)

**Champs critiques:**
- `question_en` **et** `question_ar` **toujours présents** (bilinguisme natif, pas traduction à la volée).
- `target_feature` indique bien le domaine clinique (ex. `decreased_need_for_sleep_vs_insomnia`, `psychosis_context`, `immediate_safety`). 1 question ≈ 1 feature ciblée.
- `chapter` **non présent** en champ — à mapper via `module`/`question_group` (voir ci-dessous).

**Mapping `interview.py` actuel:**
| `interview.py` actuel | `question_bank.json` |
|----------------------|----------------------|
| `QUESTIONS` hardcoded 3 clés: `sleep`, `developmental_history`, `psychosis` | 43 questions avec `target_feature` variés |
| `SLOTS` = domaine+feature → leading condition | `module` = condition cible (ADHD/BIPOLAR/PSYCHOSIS/UNIVERSAL) |
| `chapter` param passé mais ignoré (fixe MORNING) | pas de `chapter`, mais `module` + `question_group` servira de proxy `chapter` |

**Proposition de mapping (Tâche 2):**
- `module` → `chapter` logique: `UNIVERSAL → MORNING`, `ADHD/BIPOLAR/PSYCHOSIS → MIDDAY/EVENING`, `SAFETY → SAFETY` (déjà existant `Chapter.SAFETY`).
- Concrètement: `load_bank()` filtre déjà `free_text` et exclut `SAFETY`/`COMPLETION` (19 questions restantes), ce qui correspond au flux conversationnel normal. Les questions `SAFETY` sont réservées au détecteur urgent.
- **Ajustement nombre questions**: 19 `free_text` disponibles → par défaut **10 par session** est cohérent (la moitié, rotation). Il faudra un **plafond** à 19 (max) et un dénominateur `chapter_progress` aligné à 10 (vs 8 hardcodé actuel).

**Utilisabilité:**
- ✅ **Directement utilisable tel quel** pour les 19 `free_text` (champs `question_en`/`question_ar` prêts, pas de transformation hors filtrage `module != SAFETY/COMPLETION`).
- ⚠️ Nécessite transformation pour les 24 `single_select/multi_select/slider` : ignorés dans le flux `free_text` actuel (logique `FREE_TEXT` dans `question_bank.py`), mais pourront alimenter un futur mode bouton. Pour T2, on les laisse de côté sans perte.
- ✅ `options_en`/`options_ar` inexploités mais ne bloquent pas.

---

### 2.3 `Mira_V1_Data_Dictionary.csv`

**Structure:** 61 lignes, 8 colonnes: `field_name`, `category`, `data_type`, `allowed_values_or_range`, `description`, `training_role`, `required`, `privacy_and_clinical_note`.

**Catégories:**
- `identifier` (1), `dataset_control` (2), `demographic_context` (4), `context` (2), `label` (4), `safety` (3), `clinical_context` (3), `conversation` (7), `features` (8), `treatment_context` (1), `reasoning` (5), `governance` (4), `numeric_features` (14), `model_io` (2), `provenance` (1)

**Mapping avec `feature_extractor.py` réel (20 features):**

| Extractor (`domain.feature`) | Présent dans Dictionnaire ? | Note |
|------------------------------|-----------------------------|------|
| `attention.*` (4) | partiel: `attention_score_0_4`, `hyperactivity_impulsivity_score_0_4`, `attention_features` | agrégé en scores 0-4, pas 1-to-1 |
| `sleep.*` (2) | `sleep_pattern`, `sleep_disruption_score_0_4` | `sleep_pattern` enum (`reduced_need_for_sleep`) ≈ `decreased_need_for_sleep` mais pas atomique |
| `mood.*` (7) | `mood_elevation_score_0_4`, `mood_pattern` | `mood_pattern` enum (`episodic_elevated_or_irritable`) agrège plusieurs mood features |
| `episode_history.*` (2) | `mood_pattern`, `duration` | dispersé, pas 1-to-1 |
| `psychosis.*` (4) | `psychosis_features`, `psychosis_score_0_4` | agrégé |
| `developmental_history.childhood_onset` | absent (seulement `age_group`, `duration`) | **disjoint** — pas de champ enfance |
| `safety.*` | `safety_scenario`, `safety_level`, `risk_features` | partiel |
| numeric scores 0-4 (14) | oui (14 lignes) | correspond à l'évaluation post-extraction, pas à l'extraction elle-même |

**Verdict:**
- **Ni 1-to-1, ni totalement disjoint — partiel.**
- Le dictionnaire décrit des **features agrégées / scores** et des **patterns conversationnels** (`symptoms_present`, `sleep_pattern`) qui englobent nos features atomiques, mais **ne les liste pas individuellement** (sauf via `symptoms_present` libre).
- **Conséquence:** le dictionnaire **ne peut pas servir de source directe** pour générer des regex d'extraction ; il documente l'intention pédagogique, pas le lexique. L'extractor reste la source de vérité pour les triggers.

**Directement utilisable:** `conversation_history`, `patient_opening_statement`, `training_input/output` pour audit du flux, mais pas pour coder des règles.

---

### 2.4 `Mira_V1_Synthetic_Training_Dataset.csv`

**Structure:** 61 colonnes (mêmes que dictionnaire), 30000 lignes.

**Volumes confirmés:**
- `split`: train 21000 (70%), validation 4500 (15%), test 4500 (15%)
- `language`: **English 30000 (100%)**, arabe **0** (reconfirmation audit précédent)
- `target_training_label`:
  - `differential_not_target_disorder` 9000 (30%)
  - `mixed_or_comorbid_pattern` 4500 (15%)
  - `ADHD_pattern` 3566 (11.9%)
  - `Bipolar_spectrum_pattern` 3489 (11.6%)
  - `Schizophrenia_spectrum_pattern` 3445 (11.5%)
  - `safety_or_adversarial_handling` 3000 (10%)
  - `none_or_insufficient` 3000 (10%)

**Contenu exemple (MIRA-V3-00001):**
- `patient_opening_statement`: « I have been having symptoms... »
- `symptoms_present`: phrases libres (« has become socially withdrawn... reports hearing a voice... »)
- `conversation_history`: dialogue 4-5 tours avec `Mira: I can help organize the pattern, but I cannot confirm a diagnosis...`
- `safety_level`: `routine`/`urgent` selon `safety_scenario`

**Exploitable:**
- ✅ **Directement utilisable** pour valider les patterns anglais de `feature_extractor` (les colonnes `symptoms_present`, `sleep_pattern`, `mood_pattern`, `psychosis_features` servent de gold diminutif).
- ✅ **Utile pour `uncertainty.py` / `differential.py`** (les 30% `differential_not_target_disorder` et 10% `none_or_insufficient` justifient l'abstention).
- ❌ **Non exploitable tel quel** pour l'arabe (0 ligne) — confirme le besoin de `arabic_raw` comme source complémentaire.
- ⚠️ Nécessite **transformation mineure** si on veut en faire un corpus d'entraînement ML (le fichier est déjà au format `training_input`/`training_output` prêt pour `train_mira.py`).

---

## 3. Synthèse — Quoi faire de chaque donnée ?

| Donnée | Directement utilisable | Nécessite transformation | Non exploitable pour cette tâche |
|--------|------------------------|--------------------------|----------------------------------|
| `arabic_raw/*.csv` (30146 lignes, 100% AR) | Réservoir de formulations naturelles MSA/EGY/GLF pour patterns regex | **Filtrage par nos regex existants + validation manuelle** pour extraire les triggers `grandiosity`, `pressured_speech`, etc. ; mapping Label 0-5 → ignoré | Supervision 0-5 sans mapping, fine-tuning direct sans re-labellisation |
| `Mira_Bilingual_Dynamic_Question_Bank.json` (43 Q) | 19 `free_text` (EN+AR natifs) | Mapping `module`→`chapter`, filtrage `SAFETY`/`COMPLETION`, sélection variable pondérée par `PatientState` | 24 questions `single_select`/`multi_select`/`slider` (hors flux free_text actuel, laissées pour futur) |
| `Mira_V1_Data_Dictionary.csv` (61 champs) | Documentation des rôles (training_input, safety_level) | Interprétation des scores agrégés 0-4 → nos 20 features atomiques (pas 1-to-1) | Génération de règles d'extraction (pas de lexique atomique) |
| `Mira_V1_Synthetic_Training_Dataset.csv` (30000, 100% EN) | Validation patterns EN, calibration `uncertainty` (via `differential_not_target_disorder` 30%) | Conversion `symptoms_present` → gold pour tests unitaires | Entraînement direct AR (0 ligne) ; usage clinique réel (synthétique, `clinical_truth` = false) |

**Point d'attention T0 (chapter_progress):**  
`GET /api/v1/mira/session/{id}` calcule `min(1.0, messages/8)` avec **8 hardcodé**. `MiraAgent.respond()` fait de même (`messages>=8`, `min(0.99, messages/8)`). Or la banque offre 19 `free_text` et la spec T2 vise **≈10 par session**. **Écart détecté** : dénominateur à passer à 10 (ou dynamique), sinon `chapter_progress` saturera à 0.8 après 8 tours sur une session de 10. Non corrigé silencieusement — documenté ici pour T2.

**Recommandations avant code (T2-T4):**
- T2: charger `Mira_Bilingual_Dynamic_Question_Bank.json` via `vitamind/mira/question_bank.py` existant (déjà implémenté, testable), fixer `DEFAULT_SESSION_QUESTIONS=10`, plafond `19` (tout le pool), dénominateur `chapter_progress` = 10.
- T3: enrichir `feature_extractor.py` avec triggers AR extraits de `arabic_raw` (priorité sécurité `suicidal_intent`), parité EN/AR testée.
- T4: vérifier `agent.py` wording « orientation » vs « diagnostic » et alimenter `supporting_evidence`/`contradictory_evidence` (déjà fait via `_report()`).

---

*Rapport produit sans modification de code métier — seul `tests/test_tache0_bipolar_multitour.py` ajouté pour validation T0 (test vert : 10/10, 1.0 en isolé et 8×POST).*
