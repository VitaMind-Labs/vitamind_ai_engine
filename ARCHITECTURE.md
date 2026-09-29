# vitamind-ai-engine — architecture

## Aujourd'hui : les deux agents tournent ensemble

```bash
cp .env.example .env    # puis renseigner LUMINA_SERVICE_SECRET
python run.py           # == make dev == python -m scripts.run_all
```

```
SERVICE    URL                        HEALTH    READY     VERSION
-----------------------------------------------------------------
lumina     http://127.0.0.1:8102      ok        ready     0.1.0 (lumina-contract-1)
mira       http://127.0.0.1:8101      ok        ready     2.0.0 (mira-contract-1)

both agents up. Ctrl+C to stop.
```

**Processus OS séparés, pas une app ASGI qui monte les deux.** C'est la décision
structurante : montés ensemble, une exception au démarrage, un chargement de modèle
bloquant ou un pic mémoire de l'un emporte l'autre — alors que tout l'intérêt de la
séparation est qu'une panne de Mira n'interrompe pas le suivi des patients déjà
onboardés.

Garanties vérifiées :

| Test | Résultat |
|---|---|
| deux agents simultanés, ports distincts | ✅ 8101 + 8102, `ready` tous les deux |
| `kill -9` sur Mira | ✅ redémarrée (backoff 1 s) ; **Lumina jamais relancée** |
| lancement avec ports identiques | ✅ refusé avant tout démarrage |
| lancement sans `LUMINA_SERVICE_SECRET` | ✅ refusé, avec la sortie de secours nommée |
| launcher tué de force (`-Force`, aucun handler) | ✅ **0 orphelin**, deux ports libérés |

Ce dernier point était un vrai bug : sans garde, les deux agents survivaient en
gardant leurs ports, et le lancement suivant échouait en « address in use ».
Corrigé par un Job Object Windows (`KILL_ON_JOB_CLOSE`) et `PR_SET_PDEATHSIG` sur
POSIX.

---

## Le problème d'architecture réel : trois normaliseurs qui ne s'accordent pas

Ce n'est pas une remarque esthétique. Mesuré :

| Entrée | mira | lumina | journal |
|---|---|---|---|
| `I can't sleeeeep at all` | `sleeeeep` | `sleep` | `sleep` |
| `أنا مؤكد أني سأموت` | `موكد` | `مؤكد` | `مؤكد` |
| `ئيس الأفكار` | `ييس` | `ئيس` | `ئيس` |
| `ٱلحمد` | `ٱلحمد` | `الحمد` | `الحمد` |
| `so tiiiired` | `tiiiired` | `tiired` | `tiired` |

**Mira diverge sur les cinq cas.** Conséquences concrètes :

- une entrée de lexique de sécurité écrite pour Lumina **rate silencieusement**
  dans Mira, et réciproquement ;
- l'allongement expressif (`sleeeeep`, un marqueur de détresse fréquent en
  écriture réelle) est absorbé par Lumina et conservé par Mira ;
- `ؤ`/`ئ` sont repliés par Mira, pas par Lumina : le même mot arabe devient deux
  tokens différents selon l'agent ;
- `ٱ` (alef wasla) n'est normalisé que par Lumina.

Un système qui prétend appliquer « la même politique de sécurité » dans les deux
agents ne le fait pas aujourd'hui.

Second duplicat, même nature : **deux vocabulaires de sécurité**
(`routine`/`needs_clarification`/`urgent` chez Mira, `NORMAL`/`ELEVATED`/`HIGH`/
`CRISIS`/`UNKNOWN` chez Lumina, `none`→`high` dans le journal). Le backend a déjà
dû écrire un mapper pour les réconcilier — ce qui prouve que la frontière est au
mauvais endroit.

---

## Architecture cible

```
vitamind_ai_engine/
├── Makefile · scripts/run_all.py · .env.example · docker-compose.yml
│
├── shared/                          ← LE SEUL code importé par les deux agents
│   └── vitamind_shared/
│       ├── text.py                  UN normaliseur EN/AR versionné
│       ├── safety.py                UN vocabulaire + mapper (agent → canonique)
│       ├── contracts.py             handoff Mira→Lumina, enveloppes, versions
│       └── service.py               /health /ready /version, HMAC, request-id
│
├── mira/                            service :8101 — orientation, sans état long
│   ├── app.py                       (utilise shared/service.py)
│   └── vitamind/                    clinical, ml, safety, mira
│
└── lumina/                          service :8102 — suivi longitudinal
    ├── service/                     app, contracts, auth
    ├── lumina/                      state, baseline, memory, tracks, capacity,
    │                                decision, response, intent, journal, safety
    ├── journal_ai/                  ← remonté de VitaMind_Journal_AI/
    ├── data_prep/ training/ evaluation/ artifacts/ datasets/
    └── tests/
```

### Les quatre règles qui tiennent l'ensemble

1. **`shared/` ne contient que ce qui doit être identique.** Normalisation,
   vocabulaire de sécurité, contrats, plomberie de service. Rien de clinique, rien
   de spécifique à un agent. Si un seul agent en a besoin, ça n'y va pas.

2. **Les agents ne s'importent jamais l'un l'autre.** Le handoff Mira→Lumina passe
   par le backend, sous forme d'objet persisté et versionné. `shared/contracts.py`
   définit sa forme ; aucun des deux n'importe le code de l'autre.

3. **Aucun agent n'écrit en base.** Ils reçoivent un bundle de contexte autorisé et
   renvoient une enveloppe structurée avec des `persistence[]`. Le backend décide.

4. **La sécurité est déterministe et ne descend jamais.** Les règles décident, le
   modèle ne peut qu'élever un niveau. Déjà en place dans Lumina ; à porter dans
   Mira via `shared/safety.py`.

### Pourquoi pas un monorepo Python avec un seul package

Tentant, et ce serait une régression : un seul package installable pousse vers un
seul processus et un seul cycle de release. Or Mira et Lumina ont des rythmes
différents (Mira est stable, Lumina évolue vite) et des profils de risque
différents (Mira voit des visiteurs anonymes, Lumina des patients abonnés). Deux
services + un socle partagé garde les deux propriétés.

### Pourquoi pas un service unique avec deux routeurs

C'est exactement ce que la vérification ci-dessus interdit : l'isolation de panne
est une exigence produit, pas une préférence de déploiement.

---

## Migration — par ordre de valeur, du plus sûr au plus invasif

### Étape 1 — `shared/vitamind_shared/text.py` (le plus de valeur, risque réel)

Un normaliseur, une `VERSION`. C'est le seul changement qui **corrige un bug de
sécurité** plutôt que de ranger du code.

Attention : les modèles entraînés stockent `normalization: "nfkc-ar-en-v1"` et
**refusent de charger** en cas de non-correspondance — c'est voulu. Donc :

1. écrire `shared/text.py` en partant de la version Lumina/journal (la plus
   complète : allongement, `ٱ`) **plus** les caractères que seule Mira replie
   (`ؤ`, `ئ`) ;
2. bumper la version → `nfkc-ar-en-v2` ;
3. **réentraîner les trois modèles Lumina et le modèle journal** (5 min CPU) ;
4. relancer les suites ; les régressions arabes de Mira sont les tests à surveiller.

Ne pas faire l'étape 1 sans le réentraînement : les modèles refuseront de charger,
et c'est le comportement correct.

### Étape 2 — `shared/safety.py`

Porter le mapper déjà écrit côté backend
(`src/common/utils/safety-level.util.ts`) en Python, et faire de
`SafetyLevel {NORMAL, ELEVATED, CRISIS}` le vocabulaire des deux agents. Mira
continue d'exposer ses chaînes historiques en sortie HTTP pour ne pas casser le
backend ; la conversion se fait en un point.

### Étape 3 — `shared/service.py`

Mira n'a aujourd'hui **aucune authentification service-à-service** ; Lumina en a.
Factoriser `/health`, `/ready`, `/version`, la vérification HMAC et le request-id,
puis les appliquer à Mira. C'est un durcissement, pas un refactor cosmétique.

### Étape 4 — remonter `journal_ai/` d'un niveau — DONE

> **Status (2026-09-29):** done. The package now lives at `lumina/journal_ai/`
> (code, `training/`, `tests/`, `data/`, `reports/`, `examples/`), the
> `sys.path` hacks in `lumina/journal.py` and `lumina/safety.py` are gone, its
> tests run in Lumina's pytest suite, and `MANIFEST.json` was regenerated. It
> still has its own normalizer until step 1 lands. The checksum failure was
> caused by CRLF checkouts (`core.autocrlf=true`), now prevented by the root
> `.gitattributes`.

`lumina/VitaMind_Journal_AI/` est un dépôt vendored avec son propre `pyproject`,
son propre normaliseur et un test de checksum `MANIFEST` **actuellement en échec**
(les fichiers ont été déplacés sans régénérer les empreintes). Après l'étape 1 il
n'a plus de normaliseur propre ; le remonter en `lumina/journal_ai/` et régénérer
le manifeste supprime la dernière ambiguïté sur « où vit le journal ».

### Étape 5 — réparer ou retirer les 22 tests Mira obsolètes

Ils échouent parce qu'ils précèdent le refactor (pas de `session_token`,
`InterviewPlanner.next_question` renommé en `.plan`). Une suite rouge en permanence
ne protège plus rien. Réparer ceux qui décrivent le comportement voulu, supprimer
les autres — mais ne pas les laisser en l'état.

### Étape 6 — `docker-compose.yml`

Même topologie en conteneurs : `mira`, `lumina`, et le backend en profil optionnel.
Le launcher reste l'outil de développement ; compose est la cible de déploiement.

---

## Ce qu'il ne faut pas faire

- **Ne pas fusionner les jeux de données.** `datasets/{dialogue,safety,intent}`
  restent séparés avec leur manifeste de provenance. Un `lumina.csv` unique perd
  la licence, la provenance et la validité clinique de chaque ligne.
- **Ne pas faire de `shared/` un fourre-tout.** Dès qu'un seul agent l'utilise,
  c'est du code d'agent.
- **Ne pas laisser Mira devenir longitudinale.** Elle oriente ; c'est tout. Y
  ajouter de la mémoire recrée Lumina en double.
- **Ne pas promouvoir un modèle automatiquement.** Tous sont `candidate` ; la
  promotion est un acte explicite via `training/registry.py --promote`.

---

## Dette connue, non résolue

1. **Trois normaliseurs** — étape 1 ci-dessus. C'est la dette la plus coûteuse.
2. **Mira sans auth service-à-service** — étape 3.
3. **22 tests Mira en échec** (préexistants, inchangés par ce travail).
4. ~~**Checksum `MANIFEST` du journal en échec** après le déplacement des fichiers.~~
   Resolved — see step 4.
5. **DailyDialog est CC BY-NC-SA 4.0** : le modèle `understanding` n'est pas
   expédiable commercialement en l'état.
6. **Aucune validation clinique.** Les 12 interventions sont
   `DRAFT_NEEDS_CLINICAL_REVIEW` ; `ALLOW_UNREVIEWED_SAFETY_CONTENT=false` refuse
   le démarrage dessus, et c'est le réglage de production.
7. **Pas de `docker-compose.yml`** encore.
