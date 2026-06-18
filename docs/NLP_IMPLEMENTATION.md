# NLP - Implementation dans le projet

Ce document explicite comment l'utilisation du NLP a ete applique au projet, sans suppression de fichiers existants.

## 1. Validation du data contract HTR

- Schema JSON ajoute: `config/htr_data_contract_schema.json`
- Validation schema + controles logiques (taille `char_confidences` == taille `text`):
  - `src/htr_data_contract.py` -> `validate_contract()`
- Commandes:

```bash
python src/nlp_cli.py validate --input data/contracts/htr_contract.json
python src/nlp_cli.py validate --input nlp/output
```

## 2. EDA corpus HTR

Metriques implementees (cours J1):

- confiance moyenne
- mediane de longueur de ligne
- taux `needs_review`
- taux de lignes courtes `< 10`
- abreviations residuelles par ligne (`~`, `ꝑ`, `ꝗ`, `ꝓ`, `ꝙ`)

Code:

- `src/htr_data_contract.py` -> `compute_eda()`

Commandes:

```bash
python src/nlp_cli.py eda --input data/contracts/htr_contract.json --output reports/eda_day1.json
python src/nlp_cli.py eda --input nlp/output --output reports/eda_nlp_output.json
```

## 3. Strategie de triage confidence / needs_review

Regles implementees:

- `confidence < 0.60` -> exclusion auto
- `0.60 <= confidence < 0.90` -> review
- `confidence >= 0.90` -> ingestion directe
- override review si:
  - `needs_review == true`
  - ecart-type `char_confidences > 0.2`

Sorties:

- CSV de revue humain-in-the-loop
- JSON des buckets `direct/review/exclude`

Code:

- `src/htr_data_contract.py` -> `split_review_buckets()`, `export_review_csv()`

Commandes:

```bash
python src/nlp_cli.py review-queue --input data/contracts/htr_contract.json
python src/nlp_cli.py review-queue --input nlp/output
```

## 4. Normalisation par regles

Normaliseur en classe independante, regles activables/desactivables (ablation possible):

- NFC Unicode
- minuscule
- regles `u/v`
- regles `i/j`
- expansion tilde (`a~`, `e~`, `o~`)
- table d'abreviations JSON

Code:

- `src/normalization_rules.py` -> `NormalizerConfig`, `MedievalFrenchNormalizer`
- table par defaut: `data/abbreviations/medieval_abbreviations.json`

Commandes:

```bash
python src/nlp_day1_cli.py normalize --text "Et li cuens prist la d~e"
python src/nlp_day1_cli.py normalize --csv-input data/input.csv --csv-output data/normalized/output.csv
```

## 5. CER et tableau d'ablation

Code CER:

- `src/cer_utils.py` -> `cer()`

Ablation (avant/apres normalisation):

```bash
python src/nlp_day1_cli.py ablation --csv-input data/reference_200.csv --reference-col reference --hypothesis-col text
```

## 6. Correction contextuelle guidee par confiance

Implementation operationnelle pour J1:

- detection des positions ambiguës selon `char_confidences` + `candidates`
- selection de variante par scorer contextuel (heuristique par defaut)
- trace JSONL des corrections
- mise a jour du contrat corrige

Code:

- `src/confidence_correction.py` -> `ConfidenceGuidedCorrector`

Commandes:

```bash
python src/nlp_cli.py correct --input data/contracts/htr_contract.json --output data/contracts/htr_contract.corrected.json --log-output data/review/correction_log.jsonl
python src/nlp_cli.py correct --input nlp/output --output-dir nlp/output_corrected --log-output data/review/correction_log.jsonl
```

Note:
- Le cours mentionne CamemBERT MLM pour le scoring (J2 detaille). Ici, un scorer heuristique est fourni pour que la fonctionnalite soit exploitable immediatement sans poids lourds.

## 7. Split stratifie + test set scelle

Implementation:

- stratification sur `(century_estimate, document_type)`
- generation `train/val/test`
- scellement du test set (`test_sealed.json`) et hash SHA-256 (`test_set.sha256`)

Code:

- `src/htr_data_contract.py` -> `stratified_split_records()`, `seal_test_set()`

Commande:

```bash
python src/nlp_day1_cli.py split --records data/documents_metadata.json --output-dir data/splits_nlp
```

## 8. Tests automatiques

Nouveaux tests:

- `tests/test_normalization_rules.py`
- `tests/test_htr_data_contract.py`

Lancer:

```bash
pytest -q
```

## 9. Dependances ajoutees

`requirements.txt`:

- `jsonschema>=4.21`
- `pytest>=8.0`
