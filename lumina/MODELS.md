# Lumina — modèles locaux

État : **prototype entraîné, non validé cliniquement.** Aucun modèle pré-entraîné,
aucune clé d'API, aucune inférence externe. Tout est entraîné depuis des poids
initialisés à zéro avec NumPy seul.

## Commandes

```bash
cd vitamind_ai_engine/lumina

# 1. Construire les jeux de données gouvernés (provenance + splits par groupe)
python -m data_prep.build_dialogue
python -m data_prep.build_safety
python -m data_prep.build_intent

# 2. Entraîner
python -m training.train_safety      # ~4 min CPU
python -m training.train_emotion     # ~5 min CPU
python -m training.train_intent      # secondes — voir ENGINES.md

# 3. Registre + tests
python -m training.registry
python -m pytest -q
```

## Architecture

Un espace de features TF-IDF creux (mots, bigrammes, n-grammes de caractères 3–5,
normalisation EN/AR partagée avec Journal AI) alimente plusieurs têtes softmax
indépendantes dans une seule matrice de poids. C'est la version économique de
l'encodeur partagé multi-tâches de la spec (§122) : les têtes partagent la
représentation tout en restant évaluables séparément.

Chaque tête porte une **température** ajustée sur validation et un **seuil
d'abstention** : sous le seuil, la prédiction devient `UNKNOWN` au lieu d'une
réponse confiante et fausse (§54, §124).

Ordre d'entraînement, strictement respecté : features et poids sur `train`
uniquement → température sur `val` → seuil d'abstention sur `val` → `test` touché
une seule fois, à la fin.

## Modèles entraînés

| Modèle | Tête | Macro-F1 (test) | Exactitude | ECE | Utilisable |
|---|---|---|---|---|---|
| `safety` | `safety_signal` | **0.793** | 0.804 | 0.036 | oui, comme signal |
| `understanding` | `act` | **0.673** | 0.755 | 0.020 | oui |
| `understanding` | `emotion` | **0.358** | 0.814 | 0.042 | non — voir limites |
| `intent` | `intent` | **0.173** | 0.183 | 0.055 | non — remplacé par des règles ([ENGINES.md](ENGINES.md)) |

L'exactitude est trompeuse sur ces données (84 % des énoncés DailyDialog sont
`NEUTRAL`). **Le macro-F1 est le chiffre à lire.**

### `safety` — signal de détresse à 3 niveaux

Entraîné sur `NORMAL` / `ELEVATED` / `HIGH` (+ `UNKNOWN` par abstention).

| Classe | n | Précision | Rappel | F1 |
|---|---|---|---|---|
| NORMAL | 1 760 | 0.894 | 0.903 | 0.899 |
| ELEVATED | 2 787 | 0.859 | 0.753 | 0.802 |
| HIGH | 1 146 | 0.603 | 0.774 | 0.678 |

Sous-évaluation 7.2 %, sur-évaluation 12.4 % — l'erreur penche du côté prudent,
ce qui est voulu. Rappel sur les lignes ≥ HIGH : 0.774.

**Pas de classe CRISIS apprise, délibérément.** Toutes les lignes CRISIS viennent
du corpus journal curé, qui ne contient que **dix significations distinctes** de
langage de crise (le reste sont des variantes de surface). Un split retenu en
contient une ou deux : un F1 calculé là-dessus mesure une phrase, pas une
capacité. La première version entraînée sur 4 niveaux donnait un rappel CRISIS de
0.20 sur un seul item de test. CRISIS est donc décidé par les règles.

### `understanding` — émotion + acte de communication

L'acte de dialogue est exploitable (QUESTION F1 0.85, INFORM 0.80).

L'émotion ne l'est pas au-delà de `NEUTRAL` (0.89) et `CONTENT` (0.52) :
`SAD` 0.26, `ANGRY` 0.24, `FEARFUL` 0.24, `IRRITABLE` 0.04. DailyDialog est de la
conversation quotidienne, pas du langage de patient, et n'annote que quelques
centaines d'exemples par émotion négative. **GoEmotions, que la spec §19 demande
précisément pour cette tête, n'est pas présent dans `data/`.**

## Fusion de sécurité (`lumina/safety.py`)

```
texte ─┬─► règles déterministes (JournalSentinel : lexique + scoping + modèle propre)
       │        seul chemin pouvant atteindre CRISIS
       └─► tête apprise (signal calibré 3 niveaux)

       niveau final = max(règles, modèle)
```

La fusion est **unidirectionnelle** : le modèle peut élever un niveau que le
lexique a manqué, jamais abaisser un niveau que le lexique a levé, et ne peut
jamais atteindre CRISIS seul.

Comportement vérifié par les tests :

| Entrée | Niveau | Décidé par |
|---|---|---|
| `I want to kill myself` | CRISIS | règles |
| `أريد أن أقتل نفسي` | CRISIS | règles |
| `ما ابغي اصحي بكرة` | CRISIS | règles |
| `My brother said he wanted to die` | ELEVATED | règles (attribution) |
| `A few years ago I wished I could disappear… I am in a much better place now` | ELEVATED | règles (passé résolu) |
| `This traffic is killing me` | NORMAL | règles (idiome) |
| `I had a nice walk and slept well` | NORMAL | règles |
| `I feel hopeless and like a burden` | HIGH | **escalade du modèle** |
| `Ignore previous instructions…` + risque | CRISIS | règles (injection ignorée) |

Dégradations : modèle absent → règles seules (CRISIS toujours détecté) ; les deux
détecteurs absents → `UNKNOWN` + `requires_human_review`, jamais `NORMAL`.

## Données et gouvernance

| Jeu | Source | Type | Lignes | Usage |
|---|---|---|---|---|
| `dialogue` | DailyDialog (I17-1099) | `public_dataset` | 64 570 / 7 574 / 7 371 | émotion, acte |
| `safety` | Combined Data (53 k) | `weak_supervision` | 45 142 / 5 415 / 5 693 | signal de détresse |
| | Journal AI curé | `human_reviewed` | 4 014 | seule source CRISIS et arabe |
| | adverse outcomes | `research_dataset` | 1 795 | |

Chaque ligne porte `source`, `source_type`, `license`, `annotated_by`,
`clinical_validity`, `verified`. Vérifié par test.

**Les noms de pathologies ne deviennent jamais un niveau de sécurité.** Les
étiquettes `Depression`, `Bipolar`, `Personality disorder` de Combined Data sont
toutes réduites à `ELEVATED` : un diagnostic n'est pas un niveau de risque.

**Splits par groupe, jamais par ligne** (§61) : familles de sens du journal,
`pairing_id` des adverse outcomes, dialogues DailyDialog, texte normalisé. Le
split de sécurité est en plus stratifié par niveau, car un split aléatoire mettait
zéro ligne CRISIS en validation. Aucun groupe partagé entre splits — vérifié par
test.

Fuite retirée : 3 391 lignes d'entraînement DailyDialog dupliquaient une ligne de
val/test. 10 275 lignes au libellé contradictoire écartées plutôt qu'arbitrées.

## Limites à connaître

1. **Aucune validation clinique.** `status: candidate`, `not_clinically_validated: true`.
2. **Tête émotion inexploitable** pour les émotions de détresse (voir plus haut).
3. **Les étiquettes de sécurité ne sont pas des évaluations de risque cliniques** —
   elles viennent de la communauté d'origine d'un post ou d'un corpus curé interne.
4. **DailyDialog est CC BY-NC-SA 4.0** : usage non commercial. Le modèle
   `understanding` ne peut pas être expédié commercialement en l'état.
5. **CRISIS repose entièrement sur les règles**, dont le lexique est décrit par ses
   propres auteurs comme « authored development rules, not clinician-reviewed ».
6. Corpus arabe très minoritaire : 1 195 lignes sur 45 142 en entraînement sécurité.

## Moteurs déterministes

Construits et testés — voir **[ENGINES.md](ENGINES.md)** : état, baseline,
mémoire, les trois tracks, capacité, moteur de décision, catalogue et classement
d'interventions, génération de réponse contrôlée, routage d'intention.

## Ce qui n'est pas encore construit

Service HTTP, persistance, rapports hebdomadaire/mensuel, résumé de fil roulant,
handoff Mira→Lumina côté backend, modèle génératif local optionnel (spec §15).
