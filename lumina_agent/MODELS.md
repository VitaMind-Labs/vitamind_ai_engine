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
| `safety` | `safety_signal` | **0.797** | 0.803 | 0.026 | oui, comme signal |
| `understanding` | `emotion` | **0.538** | 0.720 | 0.034 | avis seulement — voir limites |
| `intent` | `intent` | **0.494** | 0.498 | 0.060 | oui, en repli derrière les règles ([ENGINES.md](ENGINES.md)) |

⚠️ **Le macro-F1 d'`intent` est à lire comme 0.45 ± 0.04, pas comme 0.494.** Le jeu
de test ne fait que 267 lignes et le découpage est groupé par famille de sens, donc
le score dépend beaucoup de *quelles* familles tombent en test. Mesuré sur quatre
graines de découpage : 0.494 / 0.484 / 0.420 / 0.407, soit moyenne 0.451, écart-type
0.044. La graine livrée (42) est la plus favorable des quatre. Une comparaison entre
deux modèles n'est donc valable que sur un jeu d'évaluation **identique**.

L'exactitude est trompeuse sur ces données (`CONTENT` et `NEUTRAL` représentent
67 % du test émotion). **Le macro-F1 est le chiffre à lire.**

La tête `act` de DailyDialog n'est plus entraînée : la tête émotion est maintenant
apprise sur GoEmotions + tweets arabes, sans corpus d'actes de dialogue apparié.
`orchestrator.py` renvoie `act: None` dans ce cas, sans échouer.

Historique des deux têtes corrigées dans cette itération :

| Tête | Avant | Après | Cause du changement |
|---|---|---|---|
| `intent` | 0.093 | **0.494** | 3 → 17 familles de sens par intention (en deux passes) ; les deux jeux d'intentions fusionnés en un seul ; seuil d'abstention visant 0.70 et non 0.80 |
| `safety` | 0.782 | **0.797** | entraînement porté de 3 à 10 époques — voir plus bas, le chiffre qui compte n'est pas le macro-F1 |
| `emotion` | 0.534 | **0.538** | poids 0.5 réellement appliqué aux lignes synthétiques ; 80 k features ; plafond de pondération de classe 10 |

Le gain sur `intent` vient des **données**, pas du modèle. Preuve sur un jeu
d'évaluation *constant* (le même `val`, en ne variant que le volume
d'entraînement) : 982 lignes → 0.387, 1 554 → 0.412, 1 940 → 0.440. La courbe
monte encore.

### Ce qui a été essayé et ne marche pas

Mesuré avant d'écrire la moindre ligne de données supplémentaire, et conservé ici
pour que personne ne recommence :

| Piste | `intent` | `emotion` | Verdict |
|---|---|---|---|
| SGD maison (actuel) | **0.505** | **0.552** | référence |
| TF-IDF + régression logistique | 0.477 | 0.543 | perd, même avec les mêmes poids de classe |
| TF-IDF + LinearSVC | — | 0.499 | perd, et mal calibré (ECE 0.15–0.22) |
| Architecture fastText (rang réduit, écrite en numpy) | — | 0.473 | perd ; le score *monte* avec le rang, donc le goulot ne fait que retirer de la capacité |
| Moyennage sur 5 graines | 0.505 | — | identique : le SGD a convergé |
| Espace de features plus large / plus étroit | ≤ 0.505 | — | aucun gain |

Deux conclusions utiles. La première : **l'optimiseur n'est pas le levier.** Un
solveur convergé (lbfgs) fait *moins* bien que ce SGD partiellement convergé, sur
les deux têtes et à pondération de classe identique — la convergence partielle
joue ici un rôle de régularisation. La seconde : l'architecture fastText n'apporte
rien parce que son atout principal, la généralisation par sous-mots, est **déjà
présent** — les n-grammes de caractères de `lumina/features.py` valent 0.13 de
macro-F1 à eux seuls (les retirer fait tomber `intent` de 0.505 à 0.373).

`fasttext` lui-même n'est pas installé et n'a pas été testé : il faudrait une
dépendance C++ supplémentaire et un téléchargement, et le test d'architecture
ci-dessus suggère qu'il n'y a rien à y gagner.

Le quasi-non-gain sur `emotion` est documenté dans `training/train_emotion.py` — la
contrainte est la table de correspondance GoEmotions → vocabulaire Lumina, pas le
modèle ni l'optimiseur.

### `safety` — signal de détresse à 3 niveaux

Entraîné sur `NORMAL` / `ELEVATED` / `HIGH` (+ `UNKNOWN` par abstention).

| Classe | n | Précision | Rappel | F1 |
|---|---|---|---|---|
| HIGH | 1 906 | 0.572 | **0.848** | 0.683 |

Sous-évaluation **5.3 %** (contre 7.1 % avant), sur-évaluation 14.4 %. L'erreur
penche du côté prudent, ce qui est voulu. Rappel sur les lignes ≥ HIGH : 0.848.

Les époques ont été choisies sur `val` contre le **rappel HIGH et le taux de
sous-évaluation**, pas contre le macro-F1 : sous-évaluer le risque est l'erreur qui
atteint le patient. Passer de 3 à 10 époques fait tomber de 78 à 42 le nombre de
lignes réellement HIGH notées NORMAL. Vingt époques gagnent 0.006 de macro-F1 mais
reperdent du rappel HIGH (0.814) : ce n'est pas l'arbitrage à prendre ici.

#### ⚠️ Défaut connu, non corrigé : la surcharge exécutive lue comme de la détresse

La tête note HIGH des phrases de surcharge de tâches sans aucun contenu de risque —
« I have so much to do and I cannot start any of it » à 0.864, « I have ten things to
do and I am doing none of them » à 0.556. Son corpus est du texte Reddit
depression / SuicideWatch où rien n'est formulé ainsi : le langage de dysfonction
exécutive (la piste ADHD) est **hors distribution** pour elle.

Ce défaut **préexiste** : la tête précédente notait les mêmes phrases 0.848, 0.694,
0.543 et 0.499. Elle passait les tests seulement parce que les deux phrases qu'ils
utilisaient tombaient juste sous le seuil d'abstention — l'une à 0.007 près.

Deux corrections ont été essayées et **rejetées** :

1. *Un plancher de confiance sur le HIGH de la tête.* Les plages se recouvrent et ne
   sont pas séparables : surcharge bénigne à 0.556 / 0.623 / 0.682 / 0.864, détresse
   réelle à 0.624 (« worthless and everything is pointless ») et 0.890 (« hopeless
   and like a burden »). Tout plancher qui attrape le 0.864 bénigne jette le 0.624
   authentique.
2. *Un motif de surcharge, appliqué seulement si le lexique ne signale aucun indice.*
   La porte n'est pas fiable, parce que le lexique est précisément ce qui a manqué le
   risque : il ne signalait aucun indice sur « I cannot start anything anymore and I
   want it to end », et le garde-fou a alors supprimé un HIGH à 0.966 sur un message
   qui le mérite clairement. Supprimer un risque sur la foi du détecteur qui vient
   d'échouer n'est pas un échange acceptable.

La correction est de la **supervision**, pas un seuil ni une regex : le corpus de
sécurité a besoin de langage de surcharge exécutive étiqueté, avec validation
clinicienne de chaque ligne HIGH. Suivi par cinq tests `xfail(strict=True)` —
`tests/test_recent_safety.py`, `tests/test_lumina_engines.py`,
`tests/test_answers_what_was_said.py` — qui échoueront dès que le comportement sera
correct, pour signaler qu'il faut les supprimer.

**Pas de classe CRISIS apprise, délibérément.** Toutes les lignes CRISIS viennent
du corpus journal curé, qui ne contient que **dix significations distinctes** de
langage de crise (le reste sont des variantes de surface). Un split retenu en
contient une ou deux : un F1 calculé là-dessus mesure une phrase, pas une
capacité. La première version entraînée sur 4 niveaux donnait un rappel CRISIS de
0.20 sur un seul item de test. CRISIS est donc décidé par les règles.

### `understanding` — émotion

Entraîné sur `emotion_en` (GoEmotions agrégé par accord d'annotateurs),
`emotion_ar` (tweets) et `emotion_synth` (lignes rédigées, `train` seulement,
poids 0.5). Évalué sur `emotion_en` + `emotion_ar` uniquement.

| Classe | n | Précision | Rappel | F1 |
|---|---|---|---|---|
| CONTENT | 1 537 | 0.863 | 0.850 | **0.856** |
| FEARFUL | 168 | 0.890 | 0.816 | **0.851** |
| NEUTRAL | 955 | 0.637 | 0.764 | 0.695 |
| DISTRESSED | 67 | 0.694 | 0.642 | 0.667 |
| ANGRY | 263 | 0.688 | 0.586 | 0.632 |
| HOPEFUL | 114 | 0.647 | 0.579 | 0.611 |
| ANXIOUS | 9 | 0.556 | 0.556 | 0.556 |
| SAD | 222 | 0.548 | 0.518 | 0.532 |
| MOTIVATED | 51 | 0.447 | 0.412 | 0.429 |
| CONFUSED | 101 | 0.388 | 0.307 | 0.343 |
| IRRITABLE | 148 | 0.416 | 0.284 | 0.337 |
| FRUSTRATED | 71 | 0.355 | 0.310 | 0.331 |
| CALM | 9 | 0.250 | 0.111 | 0.154 |

Abstention sous 0.55 : couverture 78 %, exactitude 0.80 quand la tête répond.

**Le plafond est la table de correspondance, pas le modèle.** GoEmotions est
replié sur le vocabulaire Lumina : six étiquettes source deviennent `CONTENT`,
tandis que trois étiquettes voisines (*annoyance* / *anger* / *disappointment*)
sont séparées en `IRRITABLE` / `ANGRY` / `FRUSTRATED` — précisément les trois
scores les plus bas, avec `CALM` (*relief*) et `MOTIVATED` (*excitement*) que le
manifeste de `emotion_en` qualifie lui-même de correspondances approximatives.

Un balayage sur validation (époques, features, pas d'apprentissage,
régularisation, plafond de pondération, emphase par classe, moyennage des poids)
a déplacé le macro-F1 de 0.02 au total. Aller plus loin demande de revoir la
correspondance à partir des CSV GoEmotions bruts, **absents de `data/`**.

`OVERWHELMED`, `LONELY` et `LOW_ENERGY` n'ont aucune ligne d'évaluation réelle :
seules des lignes rédigées les couvrent, en `train`. Elles sont donc entraînées
et **non mesurées** — le macro-F1 ci-dessus porte sur les 13 classes présentes.

**Avis seulement.** `lumina/decision.py` ne lit pas cette tête, et un test vérifie
que la décision est identique quelle que soit l'émotion prédite.

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
