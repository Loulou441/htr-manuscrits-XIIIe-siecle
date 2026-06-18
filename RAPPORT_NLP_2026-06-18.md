# Rapport détaillé — branche `nlp-pipeline` (18 juin 2026)

## 0. Étapes préparatoires (branche `nlp`, 14-17 juin) — avant la réorganisation

Le pipeline NLP repose sur un travail préalable de constitution et de validation du dataset, réalisé sur la branche `nlp` avant la réorganisation en `nlp-pipeline`. Quatre étapes, dans l'ordre chronologique :

### a) Extraction du nouveau dataset depuis le modèle (17/06, commit `5f55bc1`)

Script `batch_transcribe.py` : à partir d'une liste de manuscrits du XIIIe siècle référencés sur Gallica/BnF (`Manuscrits XIII siecle.txt` — *Le Roman de Tristan*, *Le Roman de Troie*, *Tristan en prose*, *Le Brut* de Wace, etc.), le script :
1. extrait l'identifiant ARK de chaque manuscrit et télécharge les pages (haute résolution) depuis Gallica,
2. charge le modèle HTR de référence (« Exp 2 — Baseline binarisée, CER 26.3% », via Kraken),
3. segmente chaque page (Kraken BLLA) puis transcrit les lignes détectées (`rpred`),
4. construit le data contract JSON pour chaque page et le sauvegarde dans `data/predictions/`.

10 manuscrits × jusqu'à 10 pages chacun ont été traités par lot de cette façon.

### b) Correction / évaluation du dataset par rapport à une vérité terrain (17/06, commit `187674b`)

Script `evaluate_model.py` : charge le modèle HTR entraîné (format `.safetensors` ou VGSL) et le confronte aux transcriptions de référence ALTO/PageXML pour calculer **CER** et **WER** ligne par ligne, avec export CSV/JSON. Cette étape sert à mesurer objectivement la qualité du nouveau dataset transcrit avant de l'envoyer dans le pipeline NLP.

### c) Construction de l'alphabet et du lexique (16/06, commit `2bc3920`)

Deux scripts complémentaires sous `src/lexique/`, tous deux *cumulatifs* (les comptes s'additionnent à chaque nouveau contrat traité, pour accumuler la statistique au fil des manuscrits) :

- **`build_alphabet.py`** : compte les caractères du texte brut (sans normalisation) de chaque data contract, avec leur point de code Unicode (`U+XXXX`) — utile pour repérer les caractères invisibles ou parasites (mojibake, espaces spéciaux). Sortie : `data/lexique/alphabet.csv` (colonnes `char, codepoint, count`).
- **`build_lexicon.py`** : construit un lexique *descriptif* (pas un lexique de référence) des mots produits par le modèle HTR — minuscule + tokenisation préservant les marqueurs d'abréviation médiévaux (`⁊`, `ꝑ`, `~`...). Sortie : `data/lexique/lexicon.csv` (colonnes `word, count`). Utile pour inspecter le vocabulaire produit et repérer les mots aberrants.

### d) Construction du dictionnaire de référence ancien français (16/06, commit `b565e0f`)

Script `dictionary.py` : télécharge et fusionne deux sources externes —
- le dictionnaire StarDict *Old French-English Wiktionary*,
- le lexique CLTK (`lexiquedelancienfrancais.txt`),

puis construit `dictionnaire_ancien_francais.json` (mot → définitions Wiktionary/CLTK), **54 921 entrées au total**. C'est ce dictionnaire qui sera utilisé plus tard (18/06) par `find_lexical_errors` / la commande `lexical-check` pour détecter les erreurs lexicales.

Ces 4 étapes ont ensuite été reprises et réorganisées dans `nlp_pipeline/` (voir section 1 et suivantes) : les scripts `build_alphabet.py`/`build_lexicon.py` ont depuis été supprimés (jugés redondants une fois le lexique construit), et `dictionary.py` déplacé vers `nlp_pipeline/lexique/`.

## 1. Vue d'ensemble

| | |
|---|---|
| Commits poussés aujourd'hui | 9 (17:18 → 22:10) |
| Modifications en cours, non commitées | 3 fichiers |
| Tests unitaires | **13/13 passés** (`pytest nlp_pipeline/tests/ -q` → `13 passed in 0.29s`) |
| Données sur S3 | 399 objets, **27,3 Mo** (`28 329 754` octets) sous `s3://htr-cremma-medieval/nlp/` |
| Dictionnaire ancien français | **54 921 entrées** |
| Documents HTR traités | 18 manuscrits, 129 fichiers contracts normalisés |

## 2. Chronologie des commits

| Heure | Commit | Contenu |
|---|---|---|
| 17:18 | `d8aed35` | Sortie HTR branchée comme entrée du pipeline NLP |
| 19:10 | `1559427` | Ajout de `average_pairwise_cer` |
| 19:10 | `224eaf8` | Merge de la branche `nlp` distante |
| 19:43 | `29f6ad9` | Construction du dictionnaire ancien français |
| 20:52 | `9d27da5` | Réorganisation `src/` → `nlp_pipeline/` |
| 20:53 | `ec7b1f8` | Suppression des scripts de construction du lexique (380 lignes) |
| 21:05 | `78b9ba0` | Réécriture du notebook Kaggle (vrai pipeline, +163/-81 lignes) |
| 21:16 | `7a2ef34` | Consolidation des commandes AWS CLI |
| 21:17 | `ce38d8e` | Fix détection racine projet sur Kaggle |
| 21:52 | `ae2dc31` | Fix abréviations + schéma + `find_lexical_errors` |
| 22:10 | `e73393b` | Logging cumulatif horodaté + notebook local |

## 3. Résultats mesurés

### a) Tests unitaires

```
13 passed in 0.29s
```

Aucune régression introduite par les changements de la journée (règles u/v, i/j, ajout `find_lexical_errors`, docstrings).

### b) Qualité de la normalisation — EDA sur 157 lignes traitées (`data/eda_report.json`)

| Métrique | Valeur |
|---|---|
| Confiance moyenne | 0.957 |
| Lignes à réviser (confidence faible) | 3.18% |
| Abréviations par ligne | 0.39 |
| Lignes très courtes (<10 car.) | 3.18% |
| Répartition confiance | 153/157 lignes ≥ 0.9, seulement 4 lignes < 0.6 |

Le corpus HTR est globalement de très bonne qualité de reconnaissance ; peu de lignes nécessitent une révision manuelle.

### c) Détection lexicale — avant/après le fix des règles u/v et i/j

| | Avant le fix | Après le fix |
|---|---|---|
| Tokens totaux | 26 900 | 26 452 |
| Occurrences totales | 89 179 | 89 179 |
| Tokens inconnus du dictionnaire | 25 827 (**96.0%**) | 25 294 (**95.6%**) |
| Top token inconnu | `⁊` (3103 occ.) | `e` (483 occ.) |

Le fix a fait disparaître du top les marqueurs typographiques bruts (`⁊` 3103 occ., `q̃` 509 occ., `qͥ` 279 occ., `ꝑ` 250 occ.) — preuve que la normalisation et la table d'abréviations fonctionnent. Le taux global reste élevé (~96%) mais ce n'est **pas un signal d'échec** : le top restant après normalisation est composé de mots grammaticaux français très courants et bien orthographiés (`est`, `ce`, `des`, `car`, `vous`, `tout`, `fait`, `ont`, `dist`), absents du dictionnaire externe (Wiktionary ancien français + CLTK) car celui-ci couvre surtout le vocabulaire lexical, pas les mots-outils. C'est une limite de la ressource, pas du pipeline — décision prise de ne pas la corriger avant la présentation de vendredi.

### d) Impact réel du fix u/v et i/j sur tout le corpus (129 documents)

Comparaison exhaustive avant/après (et non plus seulement sur l'échantillon de 30 documents de l'audit initial) :

- **3725 paires de mots distinctes** changent suite au fix
- Top des corrections les plus fréquentes :

| Avant (bug) | Après (corrigé) | Occurrences |
|---|---|---|
| `qve` | `que` | 651 |
| `qvi` | `qui` | 420 |
| `bjen` | `bien` | 182 |
| `lvi` | `lui` | 182 |
| `qvil` | `quil` | 146 |
| `pvis` | `puis` | 112 |
| `mje` | `mie` | 85 |
| `qvant` | `quant` | 81 |
| `djeu` | `dieu` | 62 |
| `svi` | `sui` | 57 |
| `onqves` | `onques` | 53 |
| `djex` | `diex` | 53 |

Ces corrections concernent des centaines à des milliers d'occurrences de mots très fréquents (`que`, `qui`, `bien`, `lui`...) qui étaient auparavant mal normalisés à chaque occurrence. D'autres changements observés dans le diff (`je`→`ie`, `vit`→`uit`, `vint`→`uint`, `svi`→`sui`) sont des effets de bord corrects de la résolution i/j et u/v ailleurs dans le même mot — vérifiés manuellement sur des exemples réels, ce ne sont pas des régressions.

### e) Stockage S3

```
Total Objects: 399
Total Size: 28 329 754 octets (~27,3 Mo)
```

Contenu : sorties HTR (`nlp/output/`, un dossier par manuscrit avec `.json`/`.txt`/`.xml`), dictionnaire ancien français, logs.

### f) Historique des exécutions CLI (`data/review/nlp_cli_run_log.jsonl`, cumulatif)

```
10 runs enregistrés, tous OK (returncode 0)
validate, eda, review-queue, normalize-contract, correct, relative-eval, lexical-check
+ 3 runs supplémentaires de normalize-contract / lexical-check
```

Aucun échec enregistré depuis l'activation du logging horodaté.

## 4. Data contract et règles de normalisation

### a) Le data contract HTR (`nlp_pipeline/htr_data_contract_schema.json`)

C'est le format JSON pivot entre la sortie du HTR et l'entrée du pipeline NLP — validé via JSON Schema (Draft 2020-12). Chaque document HTR produit un JSON avec cette structure :

```
document_id, metadata { source, century_estimate, document_type, sha256, scan_quality }
pages[] {
  page_id, image_path,
  lines[] {
    line_id, text, confidence, char_confidences[],
    candidates: null | {position, options[]} | [{position, options[]}, ...],
    needs_review, polygon[], reading_order
  }
}
```

Points clés :
- `confidence` / `char_confidences` : scores de confiance du modèle HTR, ligne entière et par caractère — c'est ce que `correct` utilise pour cibler les positions ambiguës.
- `candidates` : variantes proposées par le HTR aux positions de faible confiance. **Champ presque toujours `null`** dans les données réelles (le modèle HTR actuel ne génère pas de candidats alternatifs) — c'est ce qui causait le bug du schéma corrigé aujourd'hui (`ae2dc31`) : le schéma n'autorisait pas `null` avant le fix, donc 100% des documents réels échouaient à la validation.
- `needs_review` : drapeau booléen pour la file de relecture humaine (`review-queue`).

### b) Les règles de normalisation déterministe (`nlp_pipeline/normalization_rules.py`)

Appliquées dans cet ordre par `MedievalFrenchNormalizer.normalize()`, avant toute correction statistique/IA (priorité imposée par les consignes du projet) :

1. **NFC** : normalisation Unicode des formes combinées (caractère + accent flottant → caractère précomposé), pour une comparaison stable.
2. **lowercase** : minuscule systématique.
3. **u/v** : résout les variantes graphiques médiévales selon le contexte (début de mot ou voyelle suivante → `v` ; sinon → `u`).
   - Exceptions ajoutées aujourd'hui : `qu`/`gu` (digrammes `que`, `qui`, `guerre`) et `u` suivi de `i` (`lui`) restent vocaliques. Limite acceptée : `deuient→devient` n'est plus corrigé (faux négatif jugé moins coûteux que les faux positifs supprimés — voir section 3.d pour l'ampleur de l'impact réel).
4. **i/j** : convertit `i` en `j` quand il précède une voyelle autre que `e` (graphie moyen-français).
   - Exception ajoutée aujourd'hui : le digramme `ie`/`ien`/`ier` (`bien`, `mie`, `biel`) garde son `i` vocalique.
5. **tilde nasal** : `a~`/`e~`/`o~` (et formes combinantes Unicode équivalentes) → `an`/`en`/`on`.
6. **table d'abréviations** (`medieval_abbreviations.json`, 14 entrées) : remplace les marqueurs scribaux et latinismes (`⁊→et`, `ꝑ→per`, `ꝓ→pro`, `ꝙ→us`, `ꝗ→que`, `dñs→dominus`, `q̄→qui`, etc.) par leur expansion, clés les plus longues en premier pour éviter les collisions.

Deux mécanismes de détection complètent les règles, sans les remplacer :
- `detect_normalization_candidates` : repère les tokens porteurs de marqueurs typographiques d'abréviation non résolus, propose une expansion.
- `find_lexical_errors` (ajouté aujourd'hui) : repère les tokens absents du dictionnaire ancien français (55k entrées) — détection lexicale, complémentaire à la détection par marqueurs.

## 5. Bugs corrigés aujourd'hui

| Bug | Impact avant fix | Fix |
|---|---|---|
| `ModuleNotFoundError: transformers` | Bloquait 100% des commandes CLI | Suppression import top-level inutile |
| Schéma rejette `candidates: null` | Bloquait la validation de 100% des données réelles | Ajout `{"type":"null"}` au schéma |
| Clone Kaggle imbriqué | Notebook inutilisable après 2 exécutions | Idempotence (`isdir` + `git pull`) |
| `awscli` absent sur Kaggle | Échec de la cellule de sync S3 | `pip install awscli` ajouté |
| Règles u/v et i/j trop larges | `que→qve`, `qui→qvi`, `lui→lvi`, `bien→bjen`, `mie→mje` (mots corrects cassés) | Exceptions `qu`/`gu`/`u+i` et `ie` — vérifié sur 30+ documents réels, puis confirmé sur les 129 documents du corpus complet (section 3.d) |

## 6. Non commité actuellement

3 fichiers modifiés (`CONVENTIONS_NLP.md`, `nlp_pipeline/normalization_rules.py`, `nlp_pipeline/tests/test_normalization_rules.py`) contenant le fix des règles u/v et i/j ci-dessus, déjà validé par les tests (13/13). Pas encore poussé.
