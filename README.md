# HTR CREMMA Medieval 2026 — Fine-tuning Kraken

Projet de **Reconnaissance Automatique d'Écriture Manuscrite (HTR)** sur le corpus CREMMA Medieval (ancien français + latin, XIIIe–XVe siècle). Fine-tuning du modèle `cremma-generic` avec Kraken 7.x sur GPU cloud (Kaggle / Colab).

**Meilleur CER obtenu à ce jour : 26.3% (Run 4 — Kaggle T4)**  
**Objectif : CER < 15% (validation) → CER < 8% (excellence)**  
**Modèles publiés : [legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval)**

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Méthodologie](#2-méthodologie)
3. [Résultats des expériences](#3-résultats-des-expériences)
4. [Pipeline](#4-pipeline)
5. [Structure du projet](#5-structure-du-projet)
6. [Installation](#6-installation)
7. [Utilisation](#7-utilisation)
8. [Infrastructure cloud](#8-infrastructure-cloud)
9. [Références](#9-références)

---

## 1. Contexte et objectifs

Les modèles HTR génériques CREMMA atteignent ~95% de précision sur leurs corpus de validation, mais leur utilisation pour de la transcription fine de manuscrits médiévaux requiert un **fine-tuning spécialisé**.

Ce projet part du modèle `cremma-generic-1.0.1` (Zenodo 7631619) et tente de l'améliorer par fine-tuning sur l'intégralité du corpus CREMMA Medieval (213 documents ALTO, ~19 800 lignes de texte courant).

### Métriques cibles

| Niveau | CER | val_accuracy |
|--------|:---:|:------------:|
| Baseline actuelle | 26.3% | 73.7% |
| Objectif validation | < 15% | > 85% |
| Objectif excellence | < 8% | > 92% |

---

## 2. Méthodologie

### Vue d'ensemble

```
Corpus CREMMA Medieval (ALTO XML + JPEG)
    │
    ├─ pre_traitement.py ──── Deskew + CLAHE + filtres → mode L (grayscale)
    │
    ├─ ketos compile ─────── Arrow binaire (train.arrow / dev.arrow)
    │
    ├─ compile_arrow.py ──── Arrow filtré sans zones bruit (train_clean.arrow)
    │
    └─ ketos train ───────── Fine-tuning depuis cremma-generic-1.0.1
```

### Prétraitement des images (`pre_traitement.py`)

Pipeline de 4 étapes, chacune précédée d'un diagnostic automatique :

1. **Deskew** — correction d'inclinaison (FFT / projection / Hough), seuil 0.3°–10°
2. **CLAHE** — amélioration du contraste local (clipLimit=2.0, tileGrid=8×8)
3. **Filtre médian** — bruit sel-et-poivre (ksize=3 pour préserver les déliés gothiques)
4. **Filtre gaussien** — bruit de fond uniforme (sigma=0.8–1.2)

Sortie : images en **mode L (grayscale 8-bit)** — critique pour la compatibilité avec `cremma-generic` entraîné en mode L.

### Compilation Arrow (`compile_arrow.py`)

Deux stratégies de compilation :

| Arrow | Contenu | Lignes | Taille |
|-------|---------|:------:|:------:|
| `train.arrow` | Toutes zones ALTO | ~19 800 | 939 MB |
| `train_clean.arrow` | Zones MainZone + MarginTextZone uniquement | 18 769 | 914 MB |

`train_clean.arrow` exclut les `MusicZone`, `DropCapitalZone`, `InterlinearLine` — zones parasites représentant ~4.4% des lignes du corpus.

### Fine-tuning Kraken

```bash
ketos train \
  -f binary \
  -i cremma-generic-1.0.1.mlmodel \
  --resize union \
  --augment \
  --lag 10 \
  --precision 16-mixed \
  -b 8 \
  --workers 4 \
  -t train.arrow -e dev.arrow
```

---

## 3. Résultats des expériences

### Tableau de bord

| Run | Date | Plateforme | Données | Modèle base | CER | Stages | Statut |
|-----|------|-----------|---------|-------------|:---:|:------:|--------|
| 1 | 11 juin | Local Windows | binarisé mode 1 | cremma-medieval_best | ~30% | — | Bloqué (workers Windows) |
| 2 | 12 juin | Kaggle T4 x2 | binarisé mode 1 | cremma_generic | ~27% | — | Logs partiels |
| 3 | 13 juin | Colab A100 | binarisé mode 1 | cremma-generic-1.0.1 | 28.1% | 24 | Stagnation stage 12 |
| **4** | **13 juin** | **Kaggle T4 x2** | **binarisé mode 1** | **cremma_generic** | **26.3%** | **37** | **Meilleur run** |
| 5 | 14 juin | Kaggle T4 x2 | binarisé mode 1 | cremma-generic-1.0.1 | 26.3% | 37 | Identique Run 4 |
| 6 | 14 juin | Colab T4 | binarisé mode 1 | cremma_generic | ~26.5% | 14 | Aborté — mismatch confirmé |

### Diagnostic : le plafond à ~74%

Toutes les runs 1–6 utilisent des données **binarisées (mode 1)** alors que `cremma-generic` a été entraîné sur des images **grayscale (mode L)**. Ce mismatch crée un plafond artificiel à ~74% de `val_accuracy`.

Preuves :
- `WARNING training set contains mode 1 data` présent dès la Run 2
- Changer de modèle de base (Run 5 vs Run 4) ne change rien — même CER, même stage 27
- `train.arrow` sur S3 s'avère binarisé malgré la vérification initiale

### Expériences en cours / planifiées

| # | Hypothèse | Impact estimé | Statut |
|---|-----------|:-------------:|--------|
| Exp 3 | Arrow filtré grayscale (`train_clean.arrow`) | **majeur** (+10–15 pts CER) | Données prêtes |
| Exp 4 | TrOCR vs Kraken (LoRA) | comparaison | Planifiée |

### Modèles publiés (HuggingFace)

[legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval) — licence CC-BY 4.0

| Fichier | Expérience | CER |
|---------|-----------|:---:|
| `exp2_binarise_20260613.safetensors` | Baseline binarisée (Run 4/5) | 26.3% |
| `exp3_clean_arrow_20260613.safetensors` | Arrow filtré grayscale | en cours |

---

## 4. Pipeline

### Étape 1 — Prétraitement

```bash
python src/pre_traitement.py data/repos/ --output data/preprocessed_grayscale/
```

### Étape 2 — Compilation Arrow filtrée

```bash
python src/compile_arrow.py \
  --splits data/splits/train.txt data/splits/dev.txt \
  --output data/splits/arrow_clean/
```

### Étape 3 — Entraînement (Kaggle / Colab)

Utiliser les notebooks dans `notebooks/` :

| Notebook | Plateforme | Expérience |
|----------|-----------|-----------|
| `notebooks/kaggle_exp3_clean_arrow.ipynb` | Kaggle T4 x2 | Exp 3 — Arrow filtré |
| `notebooks/colab_exp3_clean_arrow.ipynb` | Colab A100 / T4 | Exp 3 — Arrow filtré |
| `notebooks/colab_exp2_grayscale.ipynb` | Colab T4 | Exp 2 — Arrow S3 brut (aborté) |

Les notebooks téléchargent les données depuis S3 via **AWS Secrets** (Kaggle Secrets / Colab Secrets — jamais de credentials hardcodés).

---

## 5. Structure du projet

```
htr-cremma-medieval-2026/
│
├── src/
│   ├── pre_traitement.py        ← Pipeline prétraitement images (deskew, CLAHE, filtres)
│   ├── compile_arrow.py         ← Compilation Arrow filtré (sans zones bruit)
│   ├── train.py                 ← Script entraînement local (référence)
│   └── aws_sagemaker_launch.py  ← Orchestrateur SageMaker (optionnel)
│
├── notebooks/
│   ├── kaggle_exp3_clean_arrow.ipynb   ← Exp 3 — Kaggle T4
│   ├── colab_exp3_clean_arrow.ipynb    ← Exp 3 — Colab
│   └── colab_exp2_grayscale.ipynb      ← Exp 2 — aborté
│
├── experiments/
│   ├── EXPERIMENT_LOG.md        ← Journal des hypothèses et décisions
│   └── journal.jsonl            ← Logs structurés machine-readable (une ligne par run)
│
├── tests/
│   └── test_pretraitement.py    ← Tests non-régression pipeline image
│
├── data/
│   ├── splits/                  ← Fichiers .txt (train/dev/test) + Arrow compilés
│   └── repos/                   ← Clones des corpus HTR-United (gitignore)
│
├── models/                      ← Modèles téléchargés localement (gitignore)
│
├── docs/
│   └── SAGEMAKER_ARCHITECTURE.md
│
├── README.md
├── MODEL_CARD.md                ← Fiche modèle officielle
├── TRAINING_RUNS.md             ← Historique détaillé des runs
├── DATA_SOURCES.md              ← Sources corpus + SHA-256 + liens HuggingFace
├── CONVENTIONS_TRANSCRIPTION.md ← Règles de transcription CREMMA
└── requirements.txt
```

---

## 6. Installation

```bash
git clone https://github.com/legb78/htr-cremma-medieval-2026.git
cd htr-cremma-medieval-2026

python -m venv cremma
# Windows
cremma\Scripts\activate
# Linux/macOS
source cremma/bin/activate

pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.23
opencv-python>=4.5
kraken>=4.3
lxml>=4.9
tqdm>=4.64
torch>=2.4.0
boto3>=1.26
sagemaker>=2.0
huggingface_hub>=0.23
```

---

## 7. Utilisation

### Lancer les tests

```bash
pytest tests/
```

### Compiler un Arrow filtré localement

```bash
python src/compile_arrow.py \
  --splits data/splits/train.txt data/splits/dev.txt \
  --output data/splits/arrow_clean/ \
  --upload  # optionnel — upload sur S3
```

### Vérifier le mode d'un Arrow

```python
from kraken.lib import train
import pyarrow as pa

reader = pa.ipc.open_file("data/splits/arrow_clean/train_clean.arrow")
sample = reader.get_batch(0)
# Doit afficher mode L, pas mode 1
```

### Uploader un modèle sur HuggingFace

```bash
hf auth login
hf upload legb/htr-cremma-medieval models/mon_modele.safetensors mon_modele.safetensors
```

---

## 8. Infrastructure cloud

### Amazon S3 (privé)

```
s3://htr-cremma-medieval/
├── base-model/
│   ├── cremma_generic.mlmodel           (21.8 MB — Zenodo 7234166)
│   └── cremma-generic-1.0.1.mlmodel    (21.7 MB — Zenodo 7631619)
├── splits/
│   ├── train.arrow                      (939 MB — binarisé mode 1)
│   ├── dev.arrow                        (144 MB — binarisé mode 1)
│   ├── train_clean.arrow                (914 MB — grayscale mode L filtré)
│   └── dev_clean.arrow                  (144 MB — grayscale mode L filtré)
└── output/
    └── (modèles fine-tunés)
```

### Credentials AWS

Les credentials AWS ne doivent **jamais** être hardcodés. Utiliser :
- **Kaggle** : `UserSecretsClient()` (Kaggle Secrets)
- **Colab** : `userdata.get('AWS_ACCESS_KEY_ID')` (Colab Secrets)

### Plateformes GPU utilisées

| Plateforme | GPU | Batch | Precision | Durée/run |
|-----------|-----|:-----:|:---------:|:---------:|
| Kaggle T4 x2 | 2× T4 16 GB | 8 | 16-mixed | ~2h30 |
| Colab A100 | A100 40 GB | 16 | bf16-mixed | ~5h30 |
| Colab T4 | T4 16 GB | 8 | 16-mixed | ~3h |

---

## 9. Références

### Corpus et données

- **CREMMA-Medieval** — HTR-United / ENC-PSL. <https://github.com/HTR-United/cremma-medieval>
- **CREMMA-Medieval-LAT** — HTR-United. <https://github.com/HTR-United/CREMMA-Medieval-LAT>
- **HTRomance Medieval French** — <https://github.com/HTRomance-Project/medieval-french>
- **HTRomance Medieval Latin** — <https://github.com/HTRomance-Project/medieval-latin>

### Modèles de base

- **cremma_generic** — Pinche, A. (2022). Zenodo. DOI: [10.5281/zenodo.7234166](https://doi.org/10.5281/zenodo.7234166)
- **cremma-generic-1.0.1** — Zenodo. DOI: [10.5281/zenodo.7631619](https://doi.org/10.5281/zenodo.7631619)

### Framework

- **Kraken** — Kiessling, B. (2019). *Kraken — an Universal Text Recognizer for the Humanities*. DH2019. <https://github.com/mittagessen/kraken>
- **SegmOnto** — <https://segmonto.github.io>
- **Pinche, A.** (2022). *Guide de transcription pour les manuscrits du Xe au XVe siècle*. HAL.

### Citation

```bibtex
@misc{htr-cremma-medieval-2026,
  title  = {HTR CREMMA Medieval 2026 — Fine-tuning Kraken sur manuscrits médiévaux},
  author = {Ouazar, Djamal},
  year   = {2026},
  url    = {https://github.com/legb78/htr-cremma-medieval-2026}
}
```
