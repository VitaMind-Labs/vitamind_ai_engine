# RAPPORT FINAL — Renforcement de Mira (ai-service)

**Date:** 2026-09-22  
**Version service:** 1.0.0 → 1.1.0 (T2-T4)  
**Mode exécution:** pytest + TestClient FastAPI, en process, sans Docker  
**Logs et code en anglais (exigence)**

---

## 1. Synthèse exécutive

Toutes les tâches 0-4 sont vertes, sans Docker, sans logique non demandée silencieuse.

| Tâche | Statut | Preuve |
|-------|--------|--------|
| **T0** multi-tours 8→10 | ✅ Vert | `tests/test_tache0_bipolar_multitour.py` (2 passed, 10/10, 1.0) |
| **T1** analyse données | ✅ Livré avant code | `reports/TACHE1_RAPPORT_ANALYSE_DONNEES.md` |
| **T2** banque bilingue EN/AR | ✅ 10 défaut, 17 plafond, variation, langue stricte | `tests/test_tache2_bilingual_bank.py` (7 passed) |
| **T3** extraction AR | ✅ 10 features + sécurité, parité AR/EN | `tests/test_tache3_arabic_extraction.py` (7 passed) |
| **T4** orientation & traçabilité | ✅ wording orientation, supporting/contradictory | `tests/test_tache4_orientation.py` (5 passed) |
| **Total** | **56 passed, 0 failed** | `python -m pytest -q` |

---

## 2. Diff clair par tâche

### Tâche 0 — Validation fix bipolaire en conditions réelles

**Aucun code métier modifié** (seul test ajouté). Le diagnostic isolé et le flux 8×POST donnaient déjà 10/10, 1.0, écart 0. L'accumulation `PatientState.add_observation` est correcte.

**Fichiers:**
- Nouveau: `tests/test_tache0_bipolar_multitour.py` (voir `reports/TACHE0_RESULTAT_MULTI_TOURS.md` pour avant/après)
- **Aucun changement de seuil `uncertainty.py`** (non nécessaire, comme demandé).

**Écart signalé (non corrigé silencieusement):**
- `GET /session/{id}` hardcode `8` vs banque 17 → documenté pour T2.

---

### Tâche 1 — Rapport avant code

**Livraison:** `reports/TACHE1_RAPPORT_ANALYSE_DONNEES.md` avant toute modification métier.

**Points clés (résumé):**
- `arabic_raw`: 30146 lignes, 100% AR, 6 labels équilibrés 0-5 sans mapping ADHD/bipolar/psychose → utilisable comme réservoir de triggers après filtrage, pas supervision directe.
- `Mira_Bilingual_Dynamic_Question_Bank.json`: 43 questions, 19 free_text (17 utilisables hors SAFETY/COMPLETION), champs `question_en`/`question_ar`/`target_feature` complets → remplace hardcoded `interview.py` via mapping `module`→`chapter`.
- `Mira_V1_Data_Dictionary.csv`: 61 champs, mapping avec `feature_extractor.py` (20 features) partiel (scores 0-4 agrégés, pas 1-to-1).
- `Mira_V1_Synthetic_Training_Dataset.csv`: 30000 lignes, 100% English, 0 arabe (reconfirmé), répartition 30% differential, 12% ADHD/Bipolar/Psychose chacun.

---

### Tâche 2 — Banque bilingue EN/AR, variation, plafond

**Choix documentés:**

1. **Nombre par défaut ≈10** — Confirmé par T1: 17 free_text utilisables → 10 ≈ moitié, assez pour variation inter-session, pas trop long pour l'utilisateur. **Plafond 17** (tout le pool utilisable) — choix raisonnable, documenté dans `interview.py:DEFAULT_SESSION_QUESTIONS=10`, `MAX_SESSION_QUESTIONS=17`.
2. **Extension à la demande:** Détection de phrases explicites (`EXTENSION_PHRASES` FR/EN/AR) — ex. « puis-je répondre à d'autres questions », « I have more to say », « مزيد من الأسئلة ». Si `wants_more` et `messages < MAX`, on prolonge; sinon on clôt à `DEFAULT`. Plafond dur 17.
3. **Variation inter-session:** Sélection aléatoire pondérée par infos manquantes. Pour chaque `entry`, poids 2 si son `target_feature` contient un feature encore `UNKNOWN` dans `PatientState`, sinon 1. Tirage `random.choice` parmi pool pondéré → deux sessions même chapitre n'ont pas même séquence. Logique documentée dans `interview.py:_bank_question`.
4. **Langue stricte EN/AR uniquement:** `text_for(entry, language)` sert `question_ar` si `language.startswith("ar")`, sinon `question_en`. Jamais de traduction à la volée. Fallback bilingue: si banque manquante, `QUESTIONS` fallback est désormais `{en, ar}` strict. `language: ar` → question arabe, `en` → anglaise, aucun mélange.
5. **Couverture:** Banque couvre tous les chapitres demandés en AR (UNIVERSAL 4, ADHD 3, BIPOLAR 4, PSYCHOSIS 4). Aucun manque AR détecté (`describe_coverage` → `missing_ar: []`). Si un jour `question_ar` vide, log warning et fallback EN documenté comme gap (pas d'improvisation).

**Changements de contrat (justifiés, listés):**
- `vitamind/mira/interview.py`: ajout constantes + `is_extension_requested()` + pondération + fallback bilingue.
- `vitamind/mira/question_bank.py`: `text_for` stricte + `describe_coverage()`.
- `vitamind/mira/agent.py`: import `DEFAULT/MAX`, logique de fin de session avec extension, `progress = messages/10`, message de cloture orientation.
- `app.py`: `GET /session` utilise `DEFAULT_SESSION_QUESTIONS` (fin du hardcode `8`).
- **Tests mis à jour** pour refléter nouveau seuil 10 (voir diff ci-dessous) — évolution justifiée, non silencieuse.

**Diff T2 (extraits):**

```diff
# interview.py
+DEFAULT_SESSION_QUESTIONS = 10
+MAX_SESSION_QUESTIONS = 17
+EXTENSION_PHRASES = ("plus de questions", "more questions", "مزيد من الأسئلة", ...)
+def is_extension_requested(text): ...
-QUESTIONS = {"sleep": "How much..."} 
+QUESTIONS = {"sleep": {"en": "How much...", "ar": "خلال هذه الفترات..."}}
-        entry = pool[0]
+        weighted_pool = [] # pondere par UNKNOWN
+        entry = rng.choice(weighted_pool)

# agent.py
-from .interview import InterviewPlanner
+from .interview import DEFAULT_SESSION_QUESTIONS, MAX_SESSION_QUESTIONS, is_extension_requested
-        session.complete = session.messages >= 8
+        if session.messages >= MAX_SESSION_QUESTIONS: complete True
+        elif session.messages >= DEFAULT_SESSION_QUESTIONS and not wants_more: complete True
-        min(0.99, session.messages / 8)
+        min(0.99, session.messages / DEFAULT_SESSION_QUESTIONS)

# app.py
+from vitamind.mira.interview import DEFAULT_SESSION_QUESTIONS
-        min(1.0, session.messages / 8)
+        min(1.0, session.messages / DEFAULT_SESSION_QUESTIONS)
```

---

### Tâche 3 — Extraction clinique arabe depuis `arabic_raw`

**Démarche:** `arabic_raw` analysé comme réservoir de formulations (pas de labels directs). Comptes vérifiés: `ساعتين` 69 hits, `نوبات` 1421, `أنعزل` 40, `أفكاري مشوشة` 9, `لا أستطيع الجلوس` 9, `عبقري` 7, `قليل النوم` 8, `بدون تعب` 2 → ces triggers ont été intégrés en priorité.

**Règles ajoutées (10 features listées + sécurité):**

| Feature | Triggers AR ajoutés (extraits `arabic_raw` + MSA standard) |
|---------|---------------------------------------------------------------|
| `grandiosity` | `عبقري`, `أذكى شخص`, `أستطيع فعل أي شيء`, `لا يقهر`, `قدرات خارقة` |
| `pressured_speech` | `أتكلم بسرعة`, `كلامي سريع`, `لا أستطيع التوقف عن الكلام` |
| `flight_of_ideas` | `أقفز من موضوع لآخر`, `تطاير الأفكار`, `من فكرة لأخرى` |
| `decreased_need_for_sleep` | `بدون الشعور بالتعب`, `من غير تعب`, `بدون تعب`, `لا أشعر بالتعب`, `ما زلت مليان طاقة` (+ absent `مرهق`, `تعب`) |
| `elevated_mood` | `مبتهج بشكل غير عادي`, `مليان طاقة`, `مزاج عالي` |
| `episodic_pattern` | `تأتي وتذهب`, `نوبات`, `دورات`, `هذا النمط يتكرر` |
| `depression_alternation` | `لا أستطيع النهوض من السرير`, `أشعر بعدم القيمة`, `انهيار` |
| `hyperactivity_impulsivity` | `لا أستطيع الجلوس`, `فرط الحركة`, `قرارات اندفاعية` |
| `thought_disorganization` | `أفكاري مشوشة`, `لا أستطيع ترتيب أفكاري`, `لا شيء منطقي` |
| `social_withdrawal` | `أنعزل`, `توقفت عن رؤية الأصدقاء`, `أبقى في غرفتي` |
| `safety` (priorité) | Déjà couvert dans `safety/detector.py` (EN + AR MSA), vérifié en premier |

**Negation AR:** `_NEGATION_WINDOW` étendue avec `لا|لم|لن|ليس|بدون|من غير|بلا` pour que `بدون تعب` ne déclenche pas `tired` comme absent.

**Parité AR/EN:** Tests dans `tests/test_tache3_arabic_extraction.py` — 10 paires EN/AR même sens → même `present`. Ex. `I sleep 3 hours without feeling tired` ↔ `أنام 3 ساعات بدون الشعور بالتعب` → `decreased_need_for_sleep:present` dans les deux langues.

**Ce qui a été laissé de côté (manque de données suffisantes):**
- `arabic_raw` Labels 0-5 ne mappent pas à ADHD/bipolar/psychose → pas utilisé pour supervision.
- Dialectes Derja Arabizi latin (ex. `ma 7assitech b ta3ab`) présents dans `services/feature_extractor.py` (ancien) mais absents de `arabic_raw` (100% script arabe) → non enrichi ici, documenté comme gap (pas de Derja dans `arabic_raw`).
- Certaines nuances fines `arabic_raw` (ex. `مزيد من الأسئلة` pour extension) sont MSA, pas dialectal tunisien — noté.

---

### Tâche 4 — Score et orientation, traçabilité

**Wording:** `_report()` ajoute `orientation` (ex. `orientation: mood disorder clinical assessment with a psychiatrist/psychologist`) et `disclaimer: "This is an orientation for clinical follow-up, not a diagnosis."`. `MiraReply` de cloture contient désormais `This is an orientation (not a diagnosis) toward {orientation} with confidence {HIGH/MODERATE/LOW}`. Aucun mot `diagnostic` trompeur hors disclaimer. Vérifié via `vitamind/mira/agent.py:298`.

**Traçabilité:**
- `condition_scores` (adhd/bipolar/psychosis) + `match_strength` (confiance) toujours retournés.
- `supporting_features` et `contradictory_features` alimentés depuis `assessment_engine` (vérifié après T2/T3: test `test_traceability_scores_and_evidence` montre 10 supporting pour bipolaire, pas de `decreased_need_for_sleep` en contradictory).
- `missing_information` listée.
- `GET /session` et `POST /message` cohérents: même `chapter_progress` (10) et `safety` snapshot; `POST` final contient `result` avec tout l'arbre, `GET` contient `complete`/`safety`.

**Seuils `uncertainty.py`:** Inchangés (`top <2 or margin <1`), comme exigé — T0 n'a pas démontré besoin de les toucher.

**Ne pas rendre plus sensible:** Court/vague (`ok`, `مرحبا`) reste `NO_STRONG_TARGET_SIGNAL`, `LOW`, `abstain True` — test `test_short_vague_no_strong_signal_is_expected` vert.

---

## 3. Ce qui a été laissé de côté (manque de données)

- `arabic_raw` Labels 0-5: pas de mapping clinique → non exploitable pour fine-tuning sans re-labellisation.
- 24 questions `single_select/multi_select/slider` de la banque: hors flux `free_text` actuel → ignorées, documentées pour futur mode bouton.
- `Mira_V1_Data_Dictionary` 61 champs: pas de lexique atomique → non utilisé pour générer des règles, seulement pour audit.
- `Mira_V1_Synthetic_Training_Dataset` 0 ligne AR → non exploitable pour AR, confirme besoin `arabic_raw`.
- Dialecte tunisien Derja latin absent de `arabic_raw` → pas de nouveaux triggers Derja, gap documenté.
- Si banque `question_ar` manquante pour un module: log warning, pas d'invention (actuellement 0 manquant).

## 4. Décisions de conception prises sans instruction précise (à valider)

| Décision | Justification | Plafond / valeur choisie |
|----------|---------------|---------------------------|
| `DEFAULT=10`, `MAX=17` | 17 free_text utilisables → 10 ≈ moitié, variation assurée, 17 = tout le pool | À valider |
| `EXTENSION_PHRASES` liste FR/EN/AR 18 phrases | Couvre « puis-je répondre à d'autres questions », « I have more to say », « مزيد من الأسئلة » | À valider — ajouter/supprimer des variantes dialectales? |
| Pondération aléatoire `weight=2` si UNKNOWN | Simple, évite déterministe `pool[0]`, reste pondéré par `PatientState` | À valider — alternative: softmax sur `missing_information` |
| Fallback bilingue `{en,ar}` | Assure langue stricte même si banque indisponible | À valider |
| `orientation` + `disclaimer` nouvelles clés | Rendent le wording orientation explicite, sans casser ancien `recommended_pathway` | À valider — garder ou renommer? |
| `safety` prioritaire via `SafetyDetector` avant extraction clinique | Déjà existant, confirmé en premier dans `agent.py:respond()` | À valider |

## 5. Écarts de contrat détectés et actions prises

| Écart | Action |
|-------|--------|
| `GET /session` hardcode `8` vs banque 17 | Corrigé → `DEFAULT_SESSION_QUESTIONS=10`, listé ci-dessus |
| `MiraAgent` hardcode `8` | Corrigé avec extension `10→17`, listé |
| Tests historiques attendaient `8` (ex. `test_api.py` 8 loops) | Mis à jour à `10` (avec padding) — évolution justifiée, non silencieuse |
| `question_bank` 19 vs 17 utilisables | Plafond fixé à 17 (filtre SAFETY/COMPLETION), documenté |
| FR/TN (Derja) mentionné dans d'anciens fichiers (`core/config` FR/derja) | Hors scope EN/AR uniquement — noté, pas implémenté |
| `Mira_V1_Synthetic_Training_Dataset` 0 AR | Reconfirmé, noté comme non exploitable pour AR |

## 6. Livrables attendus — Check

1. ✅ Rapport T1 avant code: `reports/TACHE1_RAPPORT_ANALYSE_DONNEES.md`
2. ✅ T0 sortie brute avant/après: `reports/TACHE0_RESULTAT_MULTI_TOURS.md`
3. ✅ Code modifié diff par tâche: ce rapport §2 + `git diff vitamind/mira/interview.py vitamind/mira/question_bank.py vitamind/mira/agent.py vitamind/clinical/feature_extractor.py app.py`
4. ✅ Nouveaux tests + outputs: `tests/test_tache0_bipolar_multitour.py`, `tests/test_tache2_bilingual_bank.py`, `tests/test_tache3_arabic_extraction.py`, `tests/test_tache4_orientation.py` — tous verts (`pytest -q` 56 passed)
5. ✅ Rapport final: ce fichier, listant laissé de côté, décisions, écarts.

## 7. Definition of Done — Vérification

- [x] **T0:** test vert, 10/10, 1.0 via multi-tours (8 isolé, 10 via API après T2, écart 0) — `pytest tests/test_tache0_bipolar_multitour.py`
- [x] **T1:** rapport livré, aucune ligne métier avant — `reports/TACHE1_RAPPORT_ANALYSE_DONNEES.md` timestamp avant `interview.py` edits
- [x] **T2:** questions depuis banque, langue respectée, sélection variable (random pondéré), plafond 17 documenté — `tests/test_tache2_bilingual_bank.py`
- [x] **T3:** règles AR pour 10 features + sécurité prioritaire, parité AR/EN verts — `tests/test_tache3_arabic_extraction.py`
- [x] **T4:** orientation sans diagnostic trompeur, supporting/contradictory alimentés — `tests/test_tache4_orientation.py`
- [x] Aucun Docker, aucune logique non demandée silencieuse (toutes les additions documentées ici)

## 8. Commandes de vérification (sans Docker, en process)

```bash
cd apps/ai-service
python -m pytest -q                          # 56 passed
python -m pytest tests/test_tache0_bipolar_multitour.py -v
python -m pytest tests/test_tache2_bilingual_bank.py -v
python -m pytest tests/test_tache3_arabic_extraction.py -v
python -m pytest tests/test_tache4_orientation.py -v
```

**Logs en anglais dans le code** (exigence): `logger.info("question bank loaded: ...", ...)`, `logger.warning("missing AR text...")`.

---

*Rapport final — à valider avant de considérer les décisions de conception comme acquises.*
