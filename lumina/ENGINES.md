# Lumina — moteurs déterministes

Les modèles appris fournissent des *signaux*. Ces moteurs prennent les
*décisions*. C'est la règle centrale de la spec (§2, §13, §50, §51) : Lumina est
un système de décision qui parle, pas un chatbot qui connaît la santé mentale.

État : prototype, **non validé cliniquement**. 110 tests passent.

```bash
cd vitamind_ai_engine/lumina
python -m pytest -q
```

## Le tour complet — `lumina/orchestrator.py`

```
texte / check-in
   ↓ state          construire l'instantané (observé / estimé / manquant)
   ↓ intent         routage par règles EN/AR
   ↓ understanding  émotion + acte (chat uniquement)
   ↓ safety         AVANT toute génération
   ↓ baseline       comparaison à la baseline personnelle
   ↓ changes        écarts, avec leur arithmétique
   ↓ capacity       plafonnée par la sécurité
   ↓ tracks         ADHD / BIPOLAR / SCHIZOPHRENIA, séparés
   ↓ memory         récupération bornée, ACTIVE seulement
   ↓ decision       moteur de règles — décide CE QUI est permis
   ↓ response       gabarits — décide COMMENT c'est dit
   → enveloppe structurée + reason codes + hints de persistance
```

Latence mesurée : **0.4–2.1 ms** par tour. Aucune écriture en base : le backend
décide ce qui est réellement persisté.

## Ordre de priorité (spec §52), imposé par `decision.py`

```
CRISIS > HIGH SAFETY > SAFETY UNKNOWN > TRACK > INTENT > GENERAL SUPPORT
```

Rien en aval ne peut réordonner cela. Vérifié par test.

## Les moteurs

| Module | Rôle | Garantie clé |
|---|---|---|
| `state.py` | instantané 0–10, 8 dimensions | ne fabrique jamais une mesure : `missing` reste `missing`, `distress=None` si rien n'est rapporté |
| `baseline.py` | baseline personnelle + détection d'écart | `INSUFFICIENT_DATA` sous 5 observations ; chaque écart stocke son calcul |
| `capacity.py` | HIGH/NORMAL/REDUCED/VERY_LOW/UNKNOWN | décision UX, jamais clinique ; CRISIS ⇒ VERY_LOW |
| `memory.py` | mémoire typée, sourcée, versionnée | correction patient > inférence modèle ; le trivial est refusé |
| `tracks.py` | trois tracks isolés | un track ne se réassigne jamais : plafond = `TRACK_REVIEW_FLAG` |
| `interventions/` | catalogue + filtres + apprentissage | jamais d'exercice inventé ; les filtres ne peuvent que restreindre |
| `decision.py` | moteur de règles | reason codes uniquement, jamais de raisonnement stocké |
| `response.py` | gabarits EN/AR | garde-fous post-rendu : une réponse fautive lève, elle n'est pas envoyée |
| `intent.py` | routage EN/AR par règles | n'influence jamais la sécurité |
| `journal.py` | **Journal AI à l'intérieur de l'agent** | ne renvoie jamais de texte brut ; rien n'y devient un fait |

### Baseline — deux bugs corrigés en cours de route

1. **Fenêtre contaminée.** La baseline incluait les jours mêmes qu'on testait,
   gonflant l'écart-type et masquant le changement. Corrigé par `exclude_recent`.
2. **Variance nulle.** Un patient parfaitement stable qui décroche ne déclenchait
   rien, car `sigma = delta/0` était traité comme « pas d'estimation ». Une
   variance nulle est la *preuve la plus forte* d'un écart, pas la plus faible.

Après correction, le motif bipolaire classique est détecté avec sa traçabilité :

```
sleep    DECREASE SIGNIFICANT_CHANGE  delta=-4.70  sigma=18.2  persist=3
energy   INCREASE SIGNIFICANT_CHANGE  delta=+3.32  sigma= 9.4  persist=3
→ TRACK_REVIEW_FLAG  not_a_diagnosis: true  track_unchanged: true
```

Jamais « épisode maniaque ». Jamais de changement de track.

### Sécurité — une correction importante

Le tier `moderate_flagged` du journal était mappé sur `HIGH`. Or dans la
sémantique du journal c'est un marqueur de *revue* sur un signal modéré, pas une
escalade. Résultat : l'exemple ADHD de la spec elle-même (§33, « dix choses à
faire et je n'en fais aucune ») déclenchait un workflow de sécurité.

Mapping corrigé selon les *actions* du journal :

| tier journal | action journal | niveau Lumina |
|---|---|---|
| `none` / `low` | réflexion optionnelle | NORMAL |
| `moderate` | panneau de soutien | ELEVATED |
| `moderate_flagged` | idem + à revoir | ELEVATED + `needs_review` |
| `high` | soutien de crise | **CRISIS** |

## Journal AI — à l'intérieur de l'agent

`journal_ai` est un **analyseur**, pas un second assistant (§44). Il
n'écrit rien, ne répond jamais au patient, ne décide rien. `lumina/journal.py` est
la frontière qui l'y contraint.

```
texte de l'entrée
   → JournalSentinel      (lexique + son propre classifieur + portée par clause)
   → JournalAnalysisResult (contrat §45)
   → Lumina : signaux d'état · fusion de sécurité · candidats de mémoire
```

Deux points d'entrée dans l'agent :

```python
L = Lumina.load()

# 1. analyser une entrée
env = L.analyze_journal(texte, entry_id="e42", content_version=1)
#    → journalAnalysis, safety, persistence, idempotencyKey

# 2. injecter l'analyse dans un tour
analysis = L.journal.analyze(texte, entry_id="e42")
out = L.turn(checkin={...}, journal_signals=analysis, track="ADHD")
```

L'analyseur **partage la tête émotion** de l'agent : chat et journal ne peuvent
pas être en désaccord sur la même phrase.

### Le texte brut ne sort jamais du module (§46)

Le résultat porte un résumé *structurel* et des signaux numériques — jamais une
phrase écrite par le patient. Vérifié par test : aucune expression distinctive de
l'entrée n'apparaît dans l'enveloppe du tour.

| Entrée | Résumé renvoyé | Signaux |
|---|---|---|
| deadlines, rien fini, pas de concentration | `cues for: overload` | stress 7.5 · focus 3.0 · task_completion 3.0 |
| rien ne s'améliorera, je suis un poids | `cues for: hopelessness` | mood 1.5 · energy 3.0 |
| à peine dormi, je me sens inarrêtable | `cues for: elevated` | energy 8.0 · mood 6.5 |
| la radio m'envoie des messages secrets | `cues for: paranoia` | stress 7.5 · social_connection 3.0 |
| promenade agréable, bien dormi | `no support-relevant cues` | *(aucun)* |

Une dimension qu'aucune catégorie ne touche est **absente**, pas mise à une
valeur neutre — `build_state` la marque donc `missing`. Les signaux issus du
journal sont `estimated` et **un check-in les écrase toujours**.

### Rien n'y devient un fait (§41, §46)

Les candidats de mémoire sortent en `CANDIDATE`, avec une confiance
**délibérément sous le seuil de promotion** du magasin de mémoire — un seul
journal est un indice. Les indices attribués à autrui, au passé, niés ou
idiomatiques n'engendrent **aucun** candidat.

### Sécurité — escalade uniquement

| Situation | Résultat |
|---|---|
| journal CRISIS, tour sans texte libre | tour → CRISIS, `decided_by: journal_escalation` |
| chat CRISIS + journal calme | reste **CRISIS** |
| risque attribué à un frère | < CRISIS |
| injection + risque réel dans l'entrée | CRISIS, track inchangé |

### Idempotence (§90)

`journal:{entry_id}:{content_version}:{analysis_version}` — une reprise de job ne
peut pas double-écrire, et une entrée éditée produit une nouvelle clé.

### Confidentialité

Une entrée `is_private` sans `analysis_consent` lève `PermissionError`. Le backend
reste l'autorité, mais refuser ici aussi empêche un appelant mal câblé de
contourner la règle silencieusement.

## Intent — pourquoi les règles, pas le modèle

Aucun corpus de `data/` ne porte d'étiquettes d'intention Lumina. J'ai donc écrit
un jeu de semences (`data_prep/intent_seed.py`, 17 intentions × 3 significations
× EN/AR). Avec 3 significations par intention, le seul découpage groupé possible
est 1/1/1 — soit **88 lignes d'entraînement pour 17 classes**.

Comparaison sur le **même split retenu** :

| Routeur | macro-F1 (test) | Exactitude | Abstention |
|---|---|---|---|
| **Règles** | **0.549** | 0.521 | 31 % |
| Modèle appris | 0.000 | 0.000 | **100 %** |
| Règles + modèle | 0.549 | 0.521 | 31 % |

Le modèle s'abstient sur *tout* — comportement correct pour un modèle qui ne sait
rien, et inutile comme routeur. Les règles sont donc câblées ; le modèle est
enregistré comme `candidate` et **désactivé par défaut**. Il reprendra la main
quand il sera réentraîné sur des données réelles revues et battra les règles.

`SAFETY` et `CRISIS` sont volontairement **absents** du jeu d'intentions : un
second chemin non calibré vers la décision la plus lourde du système serait un
risque, pas une fonctionnalité.

## Comportement vérifié (extraits)

| Entrée | Décision | Intervention |
|---|---|---|
| `I have ten things to do and I am doing none of them` | SUPPORT / MICRO_ACTION | `adhd_micro_start_01` |
| `I barely slept last night` (BIPOLAR) | SUPPORT / SLEEP_SUPPORT | `bipolar_sleep_consistency_01` |
| `لا أستطيع التركيز اليوم` (ADHD) | SUPPORT / MICRO_ACTION | `adhd_environment_reset_01` |
| `I started a new medication last week` | SUPPORT / ACKNOWLEDGE | aucune — jamais de conseil médicamenteux |
| check-in stable | GENERAL_SUPPORT / ACKNOWLEDGE | aucune |
| `I want to kill myself` | CRISIS_WORKFLOW | aucune, texte approuvé fixe |
| injection + risque | CRISIS_WORKFLOW, track inchangé | aucune |

## Limites

1. **Aucune validation clinique.** Les 12 interventions sont
   `DRAFT_NEEDS_CLINICAL_REVIEW` ; `assert_production_ready()` refuse de démarrer
   dessus.
2. **Seuils non validés.** Ceux de `baseline.py` et `capacity.py` sont des
   valeurs d'ingénierie, versionnées mais non étayées cliniquement.
3. **Réponses par gabarits** — fiables et testables, mais répétitives sur la
   durée.
4. **Intent à 0.549** sur des phrases écrites pour l'occasion, pas du langage
   patient réel.
5. **Signaux du journal non validés.** Le mapping catégorie → dimension d'état
   (`CATEGORY_SIGNALS`) est un jugement d'ingénierie : « l'entrée mentionne le
   stress » est un indice de stress, pas une mesure.
6. **Pas encore construit** : service HTTP, persistance, rapports hebdo/mensuel,
   résumé de fil roulant, handoff Mira→Lumina côté backend.
