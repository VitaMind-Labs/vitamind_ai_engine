# Données retirées de `lumina/data/`

Retiré le 2026-09-29. Chaque entrée est publique et re-téléchargeable : rien
d'irremplaçable n'a été supprimé, et aucune ligne n'était référencée par
`data_prep/`, `training/`, `lumina/` ou les tests (vérifié par `grep` avant
suppression).

| Chemin | Taille | Pourquoi retiré | Récupération |
|---|---|---|---|
| `ds003944-download/` | 6.2 Go | EEG au repos (premier épisode psychotique vs témoins). 82 enregistrements `.eeg`/`.vhdr` + échelles cliniques. Lumina n'a aucun chemin d'entrée EEG : ni l'état, ni la baseline, ni les tracks, ni la sécurité ne consomment de signal électrophysiologique. Les TSV de phénotype (BPRS, SANS, SAPS, SFS) sont des instruments cliniques administrés, pas des auto-évaluations quotidiennes, donc inutilisables pour la baseline personnelle. | OpenNeuro, CC0, `doi:10.18112/openneuro.ds003944.v1.0.1` |
| `adhd-info.csv` | 4 Ko | 45 sujets, scores tabulaires d'instruments validés (MDQ, WURS, ASRS, MADRS, HADS). Aucun texte. Trop peu de lignes pour entraîner quoi que ce soit, et Lumina ne doit pas implémenter d'instrument diagnostique (spec §80). | source d'origine du dépôt |
| `I17-1099.Datasets/__MACOSX/` | — | Artefacts d'archive macOS. | aucune |
| `I17-1099.Datasets/EMNLP_dataset/.DS_Store` | — | Artefact macOS. | aucune |

## Conservé

| Chemin | Usage |
|---|---|
| `I17-1099.Datasets/EMNLP_dataset/` | DailyDialog → `datasets/dialogue` (émotion, acte) |
| `Combined Data.csv/` | → `datasets/safety` (signal de détresse) |
| `adverse_outcomes.csv` | → `datasets/safety` |
| `bipolar_dataset.csv/` | → `datasets/track_language` (signal de revue de track, non diagnostique) |
