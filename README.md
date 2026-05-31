# HTR — Manuscrits du XIIIe siècle

Projet d'entraînement d'un modèle de **Reconnaissance Automatique d'Écriture Manuscrite (HTR)** spécialisé sur les manuscrits du XIIIe siècle en latin et ancien français, via la plateforme **eScriptorium / Kraken**.

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Dataset](#2-dataset)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Démarrage rapide](#5-démarrage-rapide)
6. [Étape 1 — Compilation du dataset](#6-étape-1--compilation-du-dataset)
7. [Étape 2 — Prétraitement des images](#7-étape-2--prétraitement-des-images)
8. [Structure du projet](#8-structure-du-projet)
9. [Conventions de transcription](#9-conventions-de-transcription)
10. [Références](#10-références)

---

## 1. Contexte et objectifs

Les modèles HTR génériques existants (CREMMA Bicerin, Cortado) atteignent 95–95,5 % de précision sur leurs corpus de validation, mais présentent deux limitations majeures pour une utilisation ciblée sur le XIIIe siècle :

- **Monolinguisme** : CREMMA Medieval est intégralement en ancien français — aucune couverture du latin médiéval.
- **Déséquilibre temporel** : les données XIIIe ne représentent que 57 % du corpus CREMMA, dilué sur trois siècles.

Ce projet construit un dataset XIIIe-strict en latin et ancien français, entraîne un modèle fine-tuné depuis le CREMMA Generic, et vise un **CER < 5 %** (objectif cible : CER < 3 %).

---

## 2. Dataset

### Vue d'ensemble

Le dataset agrège des sources provenant de **4 corpus HTR-United**, tous en format **ALTO XML** et suivant la **convention graphématique CREMMA** (abréviations conservées, u/v non distingués).

| Indicateur | Valeur |
|---|---|
| Lignes totales (brut) | ~25 000 |
| Manuscrits | 38 |
| Période | XIIIe siècle strict |
| Langues | Ancien français + Latin |
| Format | ALTO XML |
| Licence | CC-BY 4.0 |

### Répartition par corpus

| Corpus | Langue | Manuscrits | Lignes |
|---|---|---|---|
| CREMMA-Medieval | Ancien français | 8 | ~12 885 |
| HTRomance Medieval FR | Ancien français | 15 | ~7 107 |
| CIHAM Fabliaux | Ancien français | 3 | ~450 |
| CREMMA-Medieval-LAT | Latin | 4 | ~1 712 |
| HTRomance Medieval LAT | Latin | 8 | ~2 806 |

### Liste complète des manuscrits

| Cote | Langue | Script | Lignes | Corpus |
|---|---|---|---|---|
| BnF fr. 412 | Ancien français | Gothic Textualis | 6 324 | CREMMA-Medieval |
| Arsenal 3516 | Ancien français | Gothic Textualis | 1 991 | CREMMA-Medieval |
| Cologny, Bodmer 168 | Ancien français | Gothic Textualis | 1 976 | CREMMA-Medieval |
| BnF fr. 24428 | Ancien français | Gothic Textualis | 1 328 | CREMMA-Medieval |
| BnF fr. 25516 | Ancien français | Gothic Textualis | 717 | CREMMA-Medieval |
| BnF fr. 844 | Ancien français | Gothic Textualis | 224 | CREMMA-Medieval |
| BnF fr. 17229 | Ancien français | Gothic Textualis | 164 | CREMMA-Medieval |
| BnF fr. 13496 | Ancien français | Gothic Textualis | 161 | CREMMA-Medieval |
| BnF NAF 23686 | Ancien français | Gothic Textualis | 424 | HTRomance Medieval FR |
| BnF fr. 1443 | Ancien français | Gothic Textualis | 418 | HTRomance Medieval FR |
| BnF fr. 1553 | Ancien français | Gothic Textualis | 506 | HTRomance Medieval FR |
| BnF fr. 1635 | Ancien français | Gothic Textualis | 217 | HTRomance Medieval FR |
| BnF fr. 12581 | Ancien français | Gothic Textualis | 306 | HTRomance Medieval FR |
| BnF fr. 1669 | Ancien français | Gothic Textualis | 484 | HTRomance Medieval FR |
| BnF fr. 104 | Ancien français | Gothic Textualis | 404 | HTRomance Medieval FR |
| BnF fr. 2168 | Ancien français | Gothic Textualis | 370 | HTRomance Medieval FR |
| BnF fr. 1450 | Ancien français | Gothic Textualis | 711 | HTRomance Medieval FR |
| BnF fr. 23117 | Ancien français | Gothic Textualis | 736 | HTRomance Medieval FR |
| BnF fr. 6447 | Ancien français | Gothic Textualis | 383 | HTRomance Medieval FR |
| BnF fr. 2173 | Ancien français | Gothic Textualis | 240 | HTRomance Medieval FR |
| BnF fr. 19152 | Ancien français | Gothic Textualis | 529 | HTRomance Medieval FR |
| BnF fr. 12603 | Ancien français | Gothic Textualis | 442 | HTRomance Medieval FR |
| BnF fr. 837 | Ancien français | Gothic Textualis | 150 | CIHAM Fabliaux |
| BnF fr. 1593 | Ancien français | Gothic Textualis | 150 | CIHAM Fabliaux |
| Mazarine 1553 | Ancien français | Gothic Textualis | 150 | CIHAM Fabliaux |
| CLM 13027 | Latin | S. Textualis Libraria | 616 | CREMMA-Medieval-LAT |
| MsWettF 15 | Latin | Textualis Libraria | 455 | CREMMA-Medieval-LAT |
| BnF lat. 16195 | Latin | Semitextualis Currens | 449 | CREMMA-Medieval-LAT |
| CCCC MSS 236 | Latin | Textualis Libraria | 192 | CREMMA-Medieval-LAT |
| BnF lat. 8001 | Latin | Gothic Textualis | 506 | HTRomance Medieval LAT |
| BnF lat. 16085 | Latin | Gothic Textualis | 392 | HTRomance Medieval LAT |
| BnF lat. 17903 | Latin | Gothic Textualis | 440 | HTRomance Medieval LAT |
| BnF lat. 14354 | Latin | Gothic Textualis | 546 | HTRomance Medieval LAT |
| BnF lat. 16204 | Latin | Gothic Textualis | 462 | HTRomance Medieval LAT |
| BnF lat. 16657 | Latin | Gothic Textualis | 199 | HTRomance Medieval LAT |
| BnF lat. 5657 | Latin | Textualis Currens | 152 | HTRomance Medieval LAT |
| BnF lat. 10996 | Latin | Textualis Currens | 109 | HTRomance Medieval LAT |

### Rééquilibrage 60/40 (optionnel)

Le dataset brut est déséquilibré (~80 % AF / ~20 % LAT). L'option `--balance` dans `dataset.py` applique un plafonnement par manuscrit pour atteindre **60 % ancien français / 40 % latin**, soit ~14 700 lignes.

---

## 3. Prérequis

- **Python** ≥ 3.10
- **Git** (pour le clonage des dépôts HTR-United)
- **OpenCV** ≥ 4.5 (`opencv-python`)
- **NumPy** ≥ 1.23
- **Kraken** ≥ 4.3 (pour l'entraînement HTR)
- RAM : 8 Go minimum (16 Go recommandé pour le prétraitement en batch)

---

## 4. Installation

```bash
# 1. Cloner ce dépôt
git clone https://github.com/votre-org/htr-xiii.git
cd htr-xiii

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### requirements.txt

```
numpy>=1.23
opencv-python>=4.5
kraken>=4.3
jdeskew>=0.10        # deskewing alternatif (optionnel)
lxml>=4.9            # lecture des fichiers ALTO XML
tqdm>=4.64           # barres de progression
```

---

## 5. Démarrage rapide

```bash
# Étape 1 — Télécharger le dataset complet (~25 000 lignes)
python dataset.py --output ./data/dataset

# Étape 1 (variante) — Dataset rééquilibré 60/40 (~14 700 lignes)
python dataset.py --output ./data/dataset --balance

# Étape 2 — Prétraiter les images
python pre_traitement.py ./data/dataset --output ./data/preprocessed

# Vérifier le résultat sur une image unique avant de traiter tout le corpus
python pre_traitement.py ./data/dataset/fro/BnF_fr_412/page001.jpg \
    --diagnose-only --verbose
```

---

## 6. Étape 1 — Compilation du dataset

### Script : `dataset.py`

Le script `dataset.py` télécharge automatiquement les ALTO XML et images depuis
les 4 dépôts GitHub HTR-United, puis les organise dans le dossier de sortie.

```
data/dataset/
├── fro/                          ← Ancien français
│   ├── BnF_fr_412/
│   │   ├── page001.xml           ← ALTO XML
│   │   ├── page001.jpg           ← image associée
│   │   └── ...
│   └── ...
├── lat/                          ← Latin
│   ├── CLM_13027/
│   └── ...
└── manifest.json                 ← Inventaire du dataset
```

### Options disponibles

| Option | Description | Défaut |
|---|---|---|
| `--output` | Dossier de destination | `./data/dataset` |
| `--balance` | Active le rééquilibrage 60/40 | désactivé |
| `--af-ratio` | Proportion cible AF (si --balance) | `0.60` |
| `--repos-dir` | Cache des clones Git | `<output>/../repos` |
| `--dry-run` | Affiche le plan sans télécharger | désactivé |
| `--seed` | Graine aléatoire (reproductibilité) | `42` |
| `--verbose` | Messages DEBUG | désactivé |

### Exemples

```bash
# Dataset complet (recommandé pour un premier entraînement)
python dataset.py --output ./data/dataset

# Dataset équilibré 60/40 (recommandé pour production)
python dataset.py --output ./data/dataset_balanced --balance

# Vérifier le plan sans rien télécharger
python dataset.py --dry-run --balance

# Personnaliser le ratio (55/45)
python dataset.py --balance --af-ratio 0.55 --output ./data/dataset_55_45
```

---

## 7. Étape 2 — Prétraitement des images

---

## 7. Étape 2 — Prétraitement des images

### Script : `pre_traitement.py`

Le pipeline de prétraitement est conforme au cours 4.1 « Prétraitement de scans de manuscrits ». Il applique quatre corrections dans l'ordre suivant, chacune précédée d'un **diagnostic automatique**. 

*Note : Afin de fiabiliser les calculs d'angles sur les manuscrits médiévaux, le diagnostic d'inclinaison s'effectue automatiquement sur une région nettoyée via une binarisation locale adaptative restreinte aux 60 % centraux du scan (évitant les bruits géométriques de bord de page ou de reliure).*

```
Image brute
    │
    ├─ 1. DESKEWING ─────────────────── Correction de l'inclinaison
    │       Diagnostic : angle d'inclinaison
    │       Méthode    : FFT (défaut) | Profils de projection | Hough
    │       Seuils     : < 0.3° → skip | ≤ 10° → corriger | > 10° → manuel
    │
    ├─ 2. CLAHE ─────────────────────── Amélioration du contraste local
    │       Diagnostic : uniformité du fond (σ_fond / 255)
    │       Seuils     : < 0.4 → nécessaire | 0.4–0.7 → léger | > 0.7 → skip
    │
    ├─ 3a. FILTRE MÉDIAN ─────────────── Bruit sel-et-poivre
    │       Diagnostic : fraction de pixels extrêmes (< 2 ou > 253)
    │       Seuils     : < 0.1% → skip | ≤ 1% → ksize=3 | > 1% → ksize=5
    │
    └─ 3b. FILTRE GAUSSIEN ──────────── Bruit de fond (grain uniforme)
            Diagnostic : σ_fond (écart-type zone claire)
            Seuils     : < 5 → skip | ≤ 15 → sigma=0.8 | > 15 → sigma=1.2
```

### Options disponibles

| Option | Description | Défaut |
|---|---|---|
| `--output` | Image ou dossier de sortie | `<input>_preprocessed` |
| `--methode-deskew` | `projection` \| `hough` \| `fft` | `projection` |
| `--no-auto-methode` | Désactive la sélection de méthode par style paléographique | sélection automatique activée |
| `--diagnose-only` | Diagnostic sans modification | désactivé |
| `--force-all` | Forcer toutes les corrections | désactivé |
| `--verbose` | Messages détaillés + rapports | désactivé |

### Exemples

```bash
# Traiter tout le dataset avec les paramètres automatiques (recommandé)
# Par défaut, le script consulte manifest.json et choisit la meilleure méthode (ex: projection pour la Textualis)
python pre_traitement.py ./data/dataset --output ./data/preprocessed

# Désactiver l'aiguillage automatique pour appliquer une méthode fixe sur tout le lot
python pre_traitement.py ./data/dataset --output ./data/preprocessed \
    --no-auto-methode --methode-deskew hough

# Diagnostic sans modification (inspecter les décisions avant de lancer)
python pre_traitement.py ./data/dataset --diagnose-only --verbose

# Forcer toutes les corrections (utile si les diagnostics sont trop conservateurs)
python pre_traitement.py ./data/dataset --output ./data/preprocessed --force-all

# Tester sur une seule image avant de traiter tout le corpus
python pre_traitement.py ./data/dataset/fro/BnF_fr_412/page001.jpg \
    --output /tmp/test_preproc.jpg --verbose
```

### Précautions spécifiques aux manuscrits médiévaux

**Filtre médian** : les déliés gothiques et points diacritiques mesurent 1–3 pixels.
Toujours utiliser `ksize=3` par défaut. `ksize=5` peut effacer les signes diacritiques.
Vérifier visuellement sur une zone d'écriture dense avant de traiter en batch.

**Filtre gaussien** : ne jamais appliquer après binarisation.
S'applique uniquement sur l'image en niveaux de gris, avant toute binarisation.

**CLAHE** : les encres colorées médiévales (cinabre, azurite, or) peuvent réagir
différemment à l'égalisation de contraste. En cas de rubriques rouges importantes,
vérifier que le CLAHE ne sature pas ces zones.

**Deskewing** : la méthode FFT est la plus robuste pour les manuscrits bruités,
mais elle échoue sur les courbures de reliure (déformation non uniforme).
Dans ce cas, préférer `--methode-deskew projection` ou corriger manuellement.

---

## 8. Structure du projet

```
htr-xiii/
├── dataset.py              ← Téléchargement et compilation du dataset
├── pre_traitement.py       ← Pipeline de prétraitement des images
├── README.md               ← Ce fichier
├── requirements.txt        ← Dépendances Python
│
├── data/
│   ├── dataset/            ← Dataset brut compilé (gitignore)
│   │   ├── fro/            ← Ancien français
│   │   ├── lat/            ← Latin
│   │   └── manifest.json
│   ├── preprocessed/       ← Images prétraitées (gitignore)
│   └── repos/              ← Cache des clones GitHub (gitignore)
│
└── models/                 ← Modèles entraînés (gitignore)
```

---

## 9. Conventions de transcription

Toutes les sources du dataset suivent la **convention graphématique CREMMA**
(Pinche 2022, disponible sur HAL) :

- **Abréviations conservées** : les abréviations sont transcrites avec leurs signes
  tels qu'ils apparaissent dans le manuscrit (pas de développement).
- **u/v non distingués** : `u` et `v` sont normalisés selon l'usage du scribe,
  sans distinction systématique.
- **i/j non distingués** : même principe que u/v.
- **Segmentation SegmOnto** : les zones et lignes sont annotées selon le
  vocabulaire SegmOnto (MainZone, MarginTextZone, DropCapitalZone…).
- **Format** : ALTO XML v4, compatible eScriptorium et Kraken.

Ces conventions garantissent la compatibilité directe entre les 4 corpus sources
et permettent un entraînement Kraken sans conversion préalable.

---

## 10. Références

### Dataset et corpus

- **CREMMA-Medieval** — HTR-United, ENC/PSL. <https://github.com/HTR-United/cremma-medieval>
- **CREMMA-Medieval-LAT** — HTR-United. <https://github.com/HTR-United/CREMMA-Medieval-LAT>
- **HTRomance Medieval French** — HTRomance Project. <https://github.com/HTRomance-Project/medieval-french>
- **HTRomance Medieval Latin** — HTRomance Project. <https://github.com/HTRomance-Project/medieval-latin>

### Modèles de référence

- **Bicerin / Cortado** — Pinche, A. (2022). *CREMMA Medieval models*. Zenodo. DOI: 10.5281/zenodo.6669553
- **CREMMA Generic** — Zenodo. DOI: 10.5281/zenodo.7234166
- **TRIDIS v2** — Zenodo. DOI: 10.5281/zenodo.13862096

### Prétraitement

- **Sauvola & Pietikäinen** (2000). *Adaptive Document Image Binarization*. Pattern Recognition, 33(2), 225–236.
- **Otsu, N.** (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Trans. SMC, 9(1), 62–66.
- **Pizer et al.** (1987). *Adaptive Histogram Equalization and Its Variations*. CVGIP, 39(3), 355–368.
- **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.
- **Ma et al.** (2018). *DocUNet: Document Image Unwarping via a Stacked U-Net*. CVPR 2018.

### Conventions et annotation

- **Pinche, A.** (2022). *Guide de transcription pour les manuscrits du Xe au XVe siècle*. HAL.
- **SegmOnto** — <https://segmonto.github.io>
- **HTR-United catalogue** — <https://htr-united.github.io/catalog.html>