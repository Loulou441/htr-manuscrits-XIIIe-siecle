# HTR Manuscrits Médiévaux du XIIIe siècle — Fine-tuning Kraken

Projet de **Reconnaissance Automatique d'Écriture Manuscrite (HTR)** sur un corpus de **33 manuscrits du XIIIe siècle** (ancien français + latin), agrégés depuis 4 corpus HTR-United. Fine-tuning du modèle `cremma-generic` avec Kraken 7.x sur GPU cloud (Kaggle / Colab).

**Meilleur CER obtenu à ce jour : 26.3% (Run 4 — Kaggle T4)**  
**Objectif : CER < 15% (validation) → CER < 8% (excellence)**  
**Modèles publiés : [legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval)**

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Corpus et données](#2-corpus-et-données)
3. [Méthodologie](#3-méthodologie)
4. [Résultats des expériences](#4-résultats-des-expériences)
5. [Évaluation détaillée](#5-évaluation-détaillée)
6. [Discussion et limitations](#6-discussion-et-limitations)
7. [Reproductibilité](#7-reproductibilité)
8. [Pipeline — utilisation](#8-pipeline--utilisation)
9. [Démo interactive — Application Streamlit](#9-démo-interactive--application-streamlit)
10. [Structure du projet](#10-structure-du-projet)
11. [Installation](#11-installation)
12. [Infrastructure cloud](#12-infrastructure-cloud)
13. [Article scientifique](#13-article-scientifique)
14. [Références](#14-références)

---

## 1. Contexte et objectifs

Les modèles HTR génériques CREMMA atteignent ~95% de précision sur leurs corpus de validation propres, mais leur généralisation à un corpus non vu requiert un **fine-tuning spécialisé**. Deux limitations motivent ce travail :

- **Mismatch de domaine** : les données d'entraînement originales de `cremma-generic` ne couvrent pas l'intégralité des manuscrits du XIIIe siècle (213 documents).
- **Bruit dans les annotations ALTO** : les zones de bruit (notation musicale, lettrines, interlignaire) parasitent l'entraînement et représentent ~4.4% des lignes du corpus.

Ce projet explore un pipeline complet : prétraitement adaptatif des images → compilation de données Arrow filtrées → fine-tuning GPU cloud → publication des modèles.

### Métriques cibles

| Niveau | CER | val_accuracy |
|--------|:---:|:------------:|
| Baseline (cremma-generic sans fine-tuning) | 44.5% | 55.5% |
| Meilleure run actuelle (Run 4) | 26.3% | 73.7% |
| Objectif validation | < 15% | > 85% |
| Objectif excellence | < 8% | > 92% |

---

## 2. Corpus et données

### Description du corpus

| Indicateur | Valeur |
|---|---|
| Documents ALTO (train) | 213 fichiers |
| Documents ALTO (dev) | 32 fichiers |
| Documents ALTO (test) | 3 fichiers |
| Lignes totales (train, toutes zones) | ~19 800 |
| Lignes texte courant (train, filtré) | 18 769 |
| Manuscrits | 33 (corpus complet train/dev/test, voir §2) |
| Période couverte | XIIIe siècle |
| Langues | Ancien français (`fro`), Latin (`lat`) |
| Format | ALTO XML v4 + JPEG |
| Licence | CC-BY 4.0 |

### Répartition des zones ALTO

```
Total lignes corpus : ~48 278
  MainZone (texte principal)   : 45 438  (94.1%)
  MarginTextZone (marginalia)  :    732  ( 1.5%)
  Zones bruit (Music/DropCap/Interlinear) : 2 108  ( 4.4%)
```

### Sources

| Corpus | Langue | Manuscrits | Dépôt |
|--------|--------|:---:|-------|
| CREMMA-Medieval | Ancien français | 7 | [HTR-United/cremma-medieval](https://github.com/HTR-United/cremma-medieval) |
| CREMMA-Medieval-LAT | Latin | 4 | [HTR-United/CREMMA-Medieval-LAT](https://github.com/HTR-United/CREMMA-Medieval-LAT) |
| HTRomance Medieval FR | Ancien français | 14 | [HTRomance-Project/medieval-french](https://github.com/HTRomance-Project/medieval-french) |
| HTRomance Medieval LAT | Latin | 8 | [HTRomance-Project/medieval-latin](https://github.com/HTRomance-Project/medieval-latin) |
| **Total** | | **33** | 4 corpus HTR-United |

### Manuscrits du corpus (33 — XIIIe siècle)

Tous les manuscrits ci-dessous datent du **XIIIe siècle**. Le nombre de lignes indiqué est le volume brut par manuscrit avant équilibrage/filtrage des zones bruit (total brut : 22 858 lignes, voir [annexe de l'article scientifique](article/article_htr_cremma.tex)).

| Cote / Shelfmark | Langue | Script | Lignes | Corpus d'origine |
|---|:---:|---|---:|---|
| BnF fr. 412 | AF | Gothic Textualis | 6324 | CREMMA-Medieval |
| Arsenal 3516 | AF | Gothic Textualis | 1991 | CREMMA-Medieval |
| Cologny, Bodmer 168 | AF | Gothic Textualis | 1976 | CREMMA-Medieval |
| BnF fr. 24428 | AF | Gothic Textualis | 1328 | CREMMA-Medieval |
| BnF fr. 844 | AF | Gothic Textualis | 224 | CREMMA-Medieval |
| BnF fr. 17229 | AF | Gothic Textualis | 164 | CREMMA-Medieval |
| BnF fr. 13496 | AF | Gothic Textualis | 161 | CREMMA-Medieval |
| CLM 13027 | Lat. | Semitextualis Libraria | 616 | CREMMA-Medieval-LAT |
| MsWettF 15 | Lat. | Textualis Libraria | 455 | CREMMA-Medieval-LAT |
| BnF lat. 16195 | Lat. | Semitextualis Currens | 449 | CREMMA-Medieval-LAT |
| CCCC MSS 236 | Lat. | Textualis Libraria | 192 | CREMMA-Medieval-LAT |
| BnF NAF 23686 | AF | Gothic Textualis | 424 | HTRomance Medieval FR |
| BnF fr. 1443 | AF | Gothic Textualis | 418 | HTRomance Medieval FR |
| BnF fr. 1553 | AF | Gothic Textualis | 506 | HTRomance Medieval FR |
| BnF fr. 1635 | AF | Gothic Textualis | 217 | HTRomance Medieval FR |
| BnF fr. 12581 | AF | Gothic Textualis | 306 | HTRomance Medieval FR |
| BnF fr. 1669 | AF | Gothic Textualis | 484 | HTRomance Medieval FR |
| BnF fr. 104 | AF | Gothic Textualis | 404 | HTRomance Medieval FR |
| BnF fr. 2168 | AF | Gothic Textualis | 370 | HTRomance Medieval FR |
| BnF fr. 1450 | AF | Gothic Textualis | 711 | HTRomance Medieval FR |
| BnF fr. 23117 | AF | Gothic Textualis | 736 | HTRomance Medieval FR |
| BnF fr. 6447 | AF | Gothic Textualis | 383 | HTRomance Medieval FR |
| BnF fr. 2173 | AF | Gothic Textualis | 240 | HTRomance Medieval FR |
| BnF fr. 19152 | AF | Gothic Textualis | 529 | HTRomance Medieval FR |
| BnF fr. 12603 | AF | Gothic Textualis | 442 | HTRomance Medieval FR |
| BnF lat. 8001 | Lat. | Gothic Textualis | 506 | HTRomance Medieval LAT |
| BnF lat. 16085 | Lat. | Gothic Textualis | 392 | HTRomance Medieval LAT |
| BnF lat. 17903 | Lat. | Gothic Textualis | 440 | HTRomance Medieval LAT |
| BnF lat. 14354 | Lat. | Gothic Textualis | 546 | HTRomance Medieval LAT |
| BnF lat. 16204 | Lat. | Gothic Textualis | 462 | HTRomance Medieval LAT |
| BnF lat. 16657 | Lat. | Gothic Textualis | 199 | HTRomance Medieval LAT |
| BnF lat. 5657 | Lat. | Textualis Currens | 152 | HTRomance Medieval LAT |
| BnF lat. 10996 | Lat. | Textualis Currens | 109 | HTRomance Medieval LAT |
| **Total (33 manuscrits)** | | | **22 858** | 4 corpus |

> Script dominant : **Gothic Textualis** (92,5% des lignes). Les manuscrits en *Semitextualis Currens* / *Textualis Currens* sont minoritaires — voir [section 6 — limitations](#6-discussion-et-limitations).
>
> Un 34e manuscrit, **BnF fr. 25516** (AF, XIIIe siècle, Gothic Textualis, 717 lignes), est utilisé **hors-corpus** uniquement pour un test de généralisation en conditions réelles (non inclus dans l'entraînement ni dans les 33 manuscrits du corpus) — voir [section 5](#5-évaluation-détaillée).

### Splits

| Split | Fichiers | Lignes (filtré) | SHA-256 Arrow |
|-------|:--------:|:---------------:|---------------|
| Train | 213 | 18 769 | `1bec767c9a87caa3...` |
| Dev | 32 | 3 702 | `20ef530c68228695...` |
| Test | 3 | *scellé* | *à compléter* |

---

## 3. Méthodologie

### Vue d'ensemble

```
Corpus CREMMA Medieval (ALTO XML + JPEG)
    │
    ├─ pre_traitement.py ──── Deskew + CLAHE + filtres → mode L (grayscale)
    │
    ├─ ketos compile ─────── Arrow binaire (train.arrow / dev.arrow)
    │
    ├─ compile_arrow.py ──── Filtrage zones bruit → train_clean.arrow
    │
    └─ ketos train ───────── Fine-tuning depuis cremma-generic-1.0.1
                              GPU cloud (Kaggle T4 / Colab A100)
```

### Prétraitement des images (`pre_traitement.py`)

Pipeline de 4 étapes avec diagnostic automatique avant chaque correction :

| Étape | Méthode | Paramètres | Seuil déclenchement |
|-------|---------|-----------|-------------------|
| Deskew | FFT / projection / Hough | auto par style paléographique | 0.3°–10° |
| CLAHE | Histogram equalization local | clipLimit=2.0, tileGrid=8×8 | σ_fond < 0.4 |
| Filtre médian | Convolution médiane | ksize=3 | pixels extrêmes > 0.1% |
| Filtre gaussien | Flou gaussien | sigma=0.8–1.2 | σ_fond > 5 |

Sortie : images **mode L (grayscale 8-bit)** — compatible avec `cremma-generic` entraîné en mode L.

> Précaution manuscrits médiévaux : `ksize=3` obligatoire pour préserver les déliés gothiques (1–3 px). Ne jamais appliquer le filtre gaussien après binarisation.

### Filtrage des zones bruit (`compile_arrow.py`)

Zones **exclues** du Arrow d'entraînement :
- `MusicZone` — notation musicale
- `DropCapitalZone` — lettrines décoratives
- `InterlinearLine` — annotations interlignaires

Zones **incluses** :
- `MainZone` — texte courant (94.1%)
- `MarginTextZone` — marginalia (1.5%)

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
  -t train_clean.arrow -e dev_clean.arrow
```

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `--resize union` | union des alphabets | Ajoute les 22 caractères absents du modèle de base |
| `--augment` | activé | Augmentation données (rotation, bruit, déformation) |
| `--lag 10` | 10 stages | Early stopping — arrêt si pas d'amélioration sur 10 stages |
| `--precision 16-mixed` | fp16 | Requis T4 (pas de bf16 sur Turing) |

---

## 4. Résultats des expériences

### Tableau de bord

| Run | Date | Plateforme | Données | Modèle base | CER | Stages | Statut |
|-----|------|-----------|---------|-------------|:---:|:------:|--------|
| 1 | 11 juin | Local Windows | binarisé mode 1 | cremma-medieval_best | ~30% | — | Bloqué (workers Windows) |
| 2 | 12 juin | Kaggle T4 x2 | binarisé mode 1 | cremma_generic | ~27% | — | Logs partiels |
| 3 | 13 juin | Colab A100 | binarisé mode 1 | cremma-generic-1.0.1 | 28.1% | 24 | Stagnation stage 12 |
| **4** | **13 juin** | **Kaggle T4 x2** | **binarisé mode 1** | **cremma_generic** | **26.3%** | **37** | **Meilleur run** |
| 5 | 14 juin | Kaggle T4 x2 | binarisé mode 1 | cremma-generic-1.0.1 | 26.3% | 37 | Identique Run 4 |
| 6 | 14 juin | Colab T4 | binarisé mode 1 | cremma_generic | ~26.5% | 14 | Aborté — mismatch confirmé |

### Courbe d'apprentissage — Run 4 (meilleure run)

| Stage | val_accuracy | CER | Patience |
|-------|:-----------:|:---:|:--------:|
| 0 | 72.1% | 27.9% | 0/10 |
| 5 | 72.8% | 27.2% | 0/10 |
| 10 | 73.2% | 26.8% | 0/10 |
| 15 | 73.4% | 26.6% | 0/10 |
| 20 | 73.5% | 26.5% | 0/10 |
| 27 | **73.67%** | **26.3%** | 0/10 |
| 37 | 73.67% | 26.3% | 10/10 → stop |

### Diagnostic : le plafond à ~74%

Toutes les runs 1–6 utilisent des données **binarisées (mode 1)** alors que `cremma-generic` a été entraîné sur des images **grayscale (mode L)**. Ce mismatch crée un plafond artificiel.

Preuves convergentes :
- `WARNING training set contains mode 1 data` présent dès la Run 2
- Changer de modèle de base (Run 5 vs Run 4) : même CER 26.3%, même stage optimal 27
- `train.arrow` S3 s'avère binarisé malgré vérification initiale

### Expériences planifiées

| # | Hypothèse | Données | Impact estimé | Statut |
|---|-----------|---------|:-------------:|--------|
| Exp 3 | Arrow grayscale filtré (`train_clean.arrow`) | mode L vérifié localement | **+10–15 pts CER** | Données prêtes |
| Exp 4 | TrOCR fine-tuning (LoRA r=8) vs Kraken | même corpus | comparaison | Planifiée |

### Modèles publiés (HuggingFace)

[legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval) — licence CC-BY 4.0

| Fichier | Expérience | CER | Commit |
|---------|-----------|:---:|--------|
| `exp2_binarise_20260613.safetensors` | Baseline binarisée (Run 4/5) | 26.3% | `99843b75` |
| `exp3_clean_arrow_20260613.safetensors` | Arrow filtré grayscale | en cours | `5e43b1b1` |

---

## 5. Évaluation détaillée

> Les métriques ci-dessous seront complétées après Exp 3 et le déscellement du set de test.

### CER par script paléographique (à compléter)

Tous les manuscrits datent du XIIIe siècle ; la variable pertinente pour l'analyse différentielle est le **script** (cf. tableau des 33 manuscrits, [section 2](#2-corpus-et-données)), pas le siècle.

| Script | Manuscrits | Lignes | CER Run 4 | CER Exp 3 |
|--------|:---------:|:------:|:---------:|:---------:|
| Gothic Textualis (dominant) | 27 | ~21 200 | *TODO* | *TODO* |
| Semitextualis Currens / Libraria | 4 | ~1 500 | *TODO* | *TODO* |
| Textualis Currens / Libraria | 2 | ~260 | *TODO* | *TODO* |

### CER par langue (à compléter)

| Langue | Documents | CER Run 4 | CER Exp 3 |
|--------|:---------:|:---------:|:---------:|
| Ancien français (`fro`) | *n* | *TODO* | *TODO* |
| Latin (`lat`) | *n* | *TODO* | *TODO* |

### Comparaison baseline (à compléter)

| Modèle | CER (val) | Delta vs baseline |
|--------|:---------:|:-----------------:|
| `cremma-generic-1.0.1` sans fine-tuning | *TODO* | — |
| Run 4 (fine-tuning binarisé) | 26.3% | *TODO* |
| Exp 3 (fine-tuning grayscale filtré) | *TODO* | *TODO* |

### Analyse des erreurs (à compléter)

Classes d'erreurs prioritaires à analyser :
- Abréviations gothiques (titres, nasales, etc.)
- Lettres ambiguës (u/n, i/m, c/e en gothique textualis)
- Caractères rares (absents du modèle de base — 22 caractères identifiés)
- Ligatures médiévales

---

## 6. Discussion et limitations

### Ce qui a fonctionné

- **Pipeline de prétraitement adaptatif** : diagnostic automatique par image, évite les corrections inutiles sur les scans propres
- **Compilation Arrow filtrée** : exclusion des zones bruit via `compile_arrow.py`, reproductible et vérifiable par SHA-256
- **Infrastructure cloud** : notebooks Kaggle/Colab avec credentials sécurisés, modèles sauvegardés sur S3 + HuggingFace

### Limitations identifiées

1. **Mismatch mode L/1** — toutes les runs 1–6 sur données binarisées → plafond ~74% artificiel. Exp 3 est le premier vrai test grayscale.
2. **`train.arrow` S3 non-grayscale** — malgré la vérification initiale, le warning Kraken confirme que l'Arrow S3 est binarisé. Seul `train_clean.arrow` compilé localement le 14 juin est vérifié mode L.
3. **Corpus limité** — 33 manuscrits, ~18 769 lignes après filtrage. Sous-représentation de certains scribes et des scripts minoritaires (*Semitextualis Currens*, *Textualis Currens* — voir [section 5](#5-évaluation-détaillée)).
4. **22 caractères absents** — présents dans le train set mais absents de l'alphabet du modèle de base. Gérés par `--resize union` mais non comptabilisés dans l'accuracy officielle.
5. **Biais linguistique** — majoritairement ancien français parisien, couverture latine sous-représentée.
6. **Reproductibilité GPU** — résultats légèrement différents entre T4 x1 et T4 x2 (DataParallel), entre Colab et Kaggle.

### Prochaines étapes

- Exp 3 : valider l'hypothèse grayscale avec `train_clean.arrow`
- Évaluation sur le set de test scellé (3 documents)
- Analyse qualitative des erreurs différentielles Exp 3 vs Run 4
- Exp 4 (bonus) : comparaison TrOCR LoRA vs Kraken, test de McNemar

---

## 7. Reproductibilité

### Versions exactes utilisées

| Outil | Version |
|-------|---------|
| Python | 3.11 |
| Kraken | 7.0.2 |
| PyTorch | 2.10.0 |
| PyTorch Lightning | 2.6.1 |
| CUDA | 12.x (T4) / 12.x (A100) |
| huggingface_hub | 1.19.0 |

### Graine aléatoire

Kraken ne supporte pas de seed fixe globale — les résultats peuvent varier de ±0.2% CER entre runs identiques sur le même hardware.

### Checksums des données

| Fichier | SHA-256 |
|---------|---------|
| `train_clean.arrow` | `1bec767c9a87caa322b20dc054da85e161ab3e630c498eb1a35ae51d19348026` |
| `dev_clean.arrow` | `20ef530c68228695bb1b68f07a07b6eb2e2ffde0f62fd8e9c2e6b29d6720448e` |
| `cremma-generic-1.0.1.mlmodel` | *TODO — à récupérer depuis Zenodo 7631619* |

---

## 8. Pipeline — utilisation

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

| Notebook | Plateforme | Expérience |
|----------|-----------|-----------|
| `notebooks/kaggle_training.ipynb` | Kaggle T4 x2 | Runs 2/4/5 — entraînement initial |
| `notebooks/colab_training.ipynb` | Colab A100/T4 | Run 3 — entraînement initial |
| `notebooks/kaggle_exp3_clean_arrow.ipynb` | Kaggle T4 x2 | Exp 3 — Arrow filtré grayscale |
| `notebooks/colab_exp2_grayscale.ipynb` | Colab T4 | Exp 2 — aborté (référence) |

Les notebooks récupèrent les credentials AWS depuis **Kaggle Secrets** / **Colab Secrets** — jamais hardcodés.

### Lancer les tests

```bash
pytest tests/
```

### Vérifier le mode d'un Arrow

```python
import pyarrow as pa
reader = pa.ipc.open_file("data/splits/arrow_clean/train_clean.arrow")
batch = reader.get_batch(0)
# im doit être mode L (grayscale), pas mode 1 (binarisé)
```

### Uploader un modèle sur HuggingFace

```bash
hf auth login
hf upload legb/htr-cremma-medieval models/mon_modele.safetensors mon_modele.safetensors
```

---

## 9. Démo interactive — Application Streamlit

Une application **Streamlit** (`app.py`) permet de tester les modèles HTR fine-tunés directement sur une image de manuscrit, sans environnement Python local.

### Fonctionnalités

- Sélection du modèle (Exp 2 — baseline binarisée / Exp 3 — Arrow grayscale filtré) téléchargé à la volée depuis [legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval)
- Upload d'une image de page ou de ligne (JPEG/PNG)
- Prétraitement optionnel (activé par défaut) via le **vrai pipeline `src/pre_traitement.py`** : deskew, CLAHE, filtres médian/gaussien → sortie mode L, avec affichage du diagnostic (angle corrigé, filtres appliqués, temps de traitement)
- Segmentation des lignes via **BLLA** (Kraken) puis transcription (`kraken.rpred`)
- Affichage de la transcription, de la confiance moyenne et du détail ligne par ligne
- Téléchargement de la transcription au format `.txt`

### Lancer la démo en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`.

### Lancer la démo via GitHub Codespaces / Dev Container

Le projet inclut une configuration `.devcontainer/devcontainer.json` (image `python:3.11-bookworm`) qui installe automatiquement les dépendances système (`packages.txt`) et Python (`requirements.txt`), puis démarre Streamlit au `postAttachCommand` :

```bash
# Depuis GitHub : bouton "Code" → "Codespaces" → "Create codespace on fine_tuning"
# L'app démarre automatiquement sur le port 8501 (forwardé et ouvert en preview)
```

| Fichier | Rôle |
|---------|------|
| `app.py` | Application Streamlit (upload, segmentation, transcription) |
| `.streamlit/config.toml` | Configuration (thème clair, upload max 50 Mo) |
| `.devcontainer/devcontainer.json` | Environnement Codespaces prêt à l'emploi |
| `packages.txt` | Dépendances système (libgl1, libsm6, etc. — requises par OpenCV/Kraken) |

> Note : les deux modèles sont téléchargés depuis HuggingFace au premier lancement et mis en cache (`@st.cache_resource`) — le premier appel peut prendre quelques dizaines de secondes.

---

## 10. Structure du projet

```
htr-manuscrits-XIIIe-siecle/
│
├── src/
│   ├── pre_traitement.py        ← Pipeline prétraitement images (deskew, CLAHE, filtres)
│   ├── compile_arrow.py         ← Compilation Arrow filtré (sans zones bruit)
│   ├── dataset.py               ← Utilitaires de chargement / inspection des datasets
│   └── train.py                 ← Script entraînement (référence)
│
├── notebooks/
│   ├── kaggle_training.ipynb           ← Entraînement initial — Kaggle T4
│   ├── kaggle_exp3_clean_arrow.ipynb    ← Exp 3 — Kaggle T4 (Arrow filtré grayscale)
│   ├── colab_training.ipynb            ← Entraînement initial — Colab
│   └── colab_exp2_grayscale.ipynb      ← Exp 2 — aborté (référence)
│
├── docs/
│   └── SAGEMAKER_ARCHITECTURE.md ← Guide d'architecture HTR sur AWS SageMaker
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
├── app.py                       ← Application Streamlit — démo interactive HTR
├── .streamlit/config.toml       ← Config Streamlit (thème, taille upload max)
├── .devcontainer/devcontainer.json ← Environnement GitHub Codespaces prêt à l'emploi
├── packages.txt                 ← Dépendances système (libgl1, libsm6, etc.)
│
├── article_htr_cremma.tex       ← Article scientifique LaTeX (format IEEE, 2 colonnes)
├── references.bib               ← Bibliographie BibTeX de l'article
├── HTR_Manuscrits_XIIIe_siecle.pdf ← Article compilé (PDF)
├── ARTICLE_README.md            ← Guide de compilation et contenu de l'article
│
├── README.md                    ← Ce fichier
├── MODEL_CARD.md                ← Fiche modèle officielle
├── TRAINING_RUNS.md             ← Historique détaillé des runs
├── DATA_SOURCES.md              ← Sources corpus + SHA-256 + liens HuggingFace
├── CONVENTIONS_TRANSCRIPTION.md ← Règles de transcription CREMMA
└── requirements.txt
```

---

## 11. Installation

```bash
git clone https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle.git
cd htr-manuscrits-XIIIe-siecle
git checkout fine_tuning

python -m venv cremma
cremma\Scripts\activate      # Windows
# source cremma/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

Pour lancer directement la démo Streamlit après installation : `streamlit run app.py` (voir [section 9](#9-démo-interactive--application-streamlit)).

---

## 12. Infrastructure cloud

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

### Plateformes GPU

| Plateforme | GPU | Batch | Precision | Durée/run |
|-----------|-----|:-----:|:---------:|:---------:|
| Kaggle T4 x2 | 2× T4 16 GB | 8 | 16-mixed | ~2h30 |
| Colab A100 | A100 40 GB | 16 | bf16-mixed | ~5h30 |
| Colab T4 | T4 16 GB | 8 | 16-mixed | ~3h |

---

## 13. Article scientifique

Le projet est accompagné d'un **article scientifique au format IEEE** (deux colonnes, 8 sections, 25+ références) rédigé dans le cadre du Mastère Data & IA d'HETIC.

| Fichier | Description |
|---------|--------------|
| `article_htr_cremma.tex` | Source LaTeX de l'article (Abstract, Introduction, Dataset, Pré-traitement, Entraînement, Résultats, Discussion, Conclusion) |
| `references.bib` | Bibliographie BibTeX (HTR, Kraken, CREMMA, vision par ordinateur) |
| `HTR_Manuscrits_XIIIe_siecle.pdf` | Version compilée de l'article, prête à la lecture |
| `ARTICLE_README.md` | Guide détaillé du contenu et instructions de compilation LaTeX |

### Compiler l'article depuis les sources

```bash
pdflatex article_htr_cremma.tex
bibtex article_htr_cremma.aux
pdflatex article_htr_cremma.tex
pdflatex article_htr_cremma.tex
# ou, en une seule commande :
latexmk -pdf article_htr_cremma.tex
```

Voir [`ARTICLE_README.md`](article/ARTICLE_README.md) pour le détail des sections, la personnalisation (auteurs, en-têtes) et les statistiques du document.

---

## 14. Références

### Corpus et données

- **CREMMA-Medieval** — HTR-United / ENC-PSL. <https://github.com/HTR-United/cremma-medieval>
- **CREMMA-Medieval-LAT** — HTR-United. <https://github.com/HTR-United/CREMMA-Medieval-LAT>
- **HTRomance Medieval French** — <https://github.com/HTRomance-Project/medieval-french>
- **HTRomance Medieval Latin** — <https://github.com/HTRomance-Project/medieval-latin>

### Modèles de base

- **cremma_generic** — Pinche, A. (2022). Zenodo. DOI: [10.5281/zenodo.7234166](https://doi.org/10.5281/zenodo.7234166)
- **cremma-generic-1.0.1** — Zenodo. DOI: [10.5281/zenodo.7631619](https://doi.org/10.5281/zenodo.7631619)

### Framework HTR

- **Kraken** — Kiessling, B. (2019). *Kraken — an Universal Text Recognizer for the Humanities*. DH2019. <https://github.com/mittagessen/kraken>
- **SegmOnto** — <https://segmonto.github.io>

### Conventions de transcription

- **Pinche, A.** (2022). *Guide de transcription pour les manuscrits du Xe au XVe siècle*. HAL.

### Prétraitement

- **Sauvola & Pietikäinen** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2), 225–236.
- **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.

### Citation

```bibtex
@misc{htr-cremma-medieval-2026,
  title  = {HTR CREMMA Medieval 2026 — Fine-tuning Kraken sur manuscrits médiévaux},
  author = {Ouazar, Djamal and Tessier, Manon and El Mortada, Hamza},
  year   = {2026},
  url    = {https://github.com/Loulou441/htr-manuscrits-XIIIe-siecle}
}
```