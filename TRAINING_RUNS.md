# Historique des entraînements HTR CREMMA Medieval

> Toutes les runs Kraken fine-tuning depuis le début du projet.  
> Sources : commits git, notebooks Kaggle/Colab, S3 `s3://htr-cremma-medieval/`, logs partagés en session.

---

## Résumé rapide

| # | Plateforme | Date | Modèle de base | Données | Meilleure val_acc | CER | Verdict |
|---|-----------|------|----------------|---------|:-----------------:|:---:|---------|
| 1 | Local (Windows) | 11–12 juin 2026 | cremma-medieval_best | binarisé (mode 1) | ~70% (estimé) | ~30% | Bloqué — pickling workers |
| 2 | Kaggle T4 x2 | ~12 juin 2026 | cremma_generic | binarisé (mode 1) | ~73% | ~27% | Mismatch L/1 — logs partiels |
| 3 | Colab A100 | 13 juin 2026 | cremma-generic-1.0.1 | binarisé (mode 1) | 71.9% | 28.1% | Mismatch L/1 — stagnation stage 12 |
| **4** | **Kaggle T4 x2** | **13 juin 2026** | **cremma_generic** | **binarisé (mode 1)** | **73.67%** | **26.3%** | **Mismatch L/1 — meilleur à ce jour** |
| 5 | Kaggle T4 x2 | 14 juin 2026 | cremma-generic-1.0.1 | binarisé (mode 1) | 73.67% | 26.3% | Identique Run 4 — même mismatch L/1 |
| **7 (Exp 3)** | **Kaggle T4 x2** | **15 juin 2026** | **cremma-generic-1.0.1** | **`train_clean.arrow` — grayscale mode L (vérifié)** | **73.56%** | **26.4%** | **Plafond ~26% PERSISTE malgré grayscale confirmé — cause ailleurs** |

> ⚠️ **Exp 3 (Run 7) = résultat décisif, mais cause mal interprétée.** Le run « grayscale » a reproduit *exactement* le plafond ~26% des runs 1–6, et le warning `mode 1 data` est réapparu dans les logs. **Hypothèse initiale (Arrow binarisé) RÉFUTÉE** : la vérification PIL du 15 juin 20h57 montre que `train_clean.arrow` (SHA `1bec767c…`, le fichier même utilisé par Exp 3) est **100% mode L** (508/508 lignes échantillonnées). Le SHA est resté identique après recompilation depuis `preprocessed_grayscale/` → le fichier S3 était déjà grayscale.
>
> **Conséquences :** (1) le warning `mode 1 data` de kraken **ne reflète pas le mode réel des images d'entraînement** — c'est un faux signal (vérifie probablement une métadonnée du modèle ou le 1er record, pas le dataset). (2) Le plafond ~26% n'est **pas** dû au mismatch L/1. La cause est ailleurs : corpus limité, hyperparamètres, alphabet, ou la métrique reduceonplateau jamais déclenchée (`val_metric not available`). **À réinvestiguer.**

---

## Run 1 — Local Windows (train.py)

**Date :** 11–12 juin 2026  
**Commits :** `190ff52` (refactor: trying to train model), `a6dd177`, `df99feb` (Manon — feat: train.py)  
**Plateforme :** PC local Windows  
**GPU :** aucun (CPU uniquement, ou GPU non utilisé)

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Script | `train.py` |
| Modèle de base | `cremma-medieval_best.mlmodel` (Zenodo `7234166`) |
| Format données | ALTO XML → `preprocessed/` (mode 1 binarisé) |
| Batch size | 16 (défaut) |
| Max epochs | 50 |
| Early stopping (lag) | 20 |
| Min epochs | 5 |
| Workers | **0** (obligatoire Windows — pickling ketos planterait) |
| Precision | fp32 (CPU) |
| Augmentation | oui |

### Résultats
- Entraînement très lent (CPU mono-thread)
- Aucun checkpoint final récupéré
- Problèmes : `--workers > 0` planté sous Windows (spawn + pickling du pipeline ketos)
- Abandon au profit de Kaggle/Colab

### Artefacts S3
- Aucun modèle uploadé

---

## Run 2 — Kaggle T4 x2 (première run Kaggle)

**Date :** ~12 juin 2026  
**Notebook :** `kaggle_training.ipynb`  
**Plateforme :** Kaggle — GPU T4 x2  
**Durée estimée :** ~14 min/epoch

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma_generic.mlmodel` (S3 `base-model/cremma_generic.mlmodel`) |
| Format données | Arrow — compilé depuis `preprocessed/` **(mode 1 binarisé)** |
| Batch size | 8 |
| Max epochs | 50 (`-N 50`) |
| Early stopping (lag) | 10 (`--lag 10`) |
| Min epochs | 3 |
| Workers | 4 |
| Precision | `16-mixed` (fp16, T4 Turing sm_75) |
| Learning rate | 0.0001 |
| Augmentation | oui |
| `--resize` | union |

### Résultats
Logs partiels — run arrêtée avant fin de session. Max observé ~73%.

**Meilleure val_accuracy estimée : ~73%**

### Diagnostic
- **WARNING kraken :** `training set contains mode 1 data` — modèle entraîné sur mode L
- Mismatch L/1 = plafond artificiel

### Artefacts S3
- Modèle non uploadé (session expirée)

---

## Run 3 — Colab A100 (13 juin 2026)

**Date :** 13 juin 2026, démarrage ~10h43  
**Notebook :** `colab_training.ipynb`  
**Plateforme :** Google Colab — GPU A100-SXM4-40GB (40 GB VRAM)  
**Durée :** ~14 min/epoch, run arrêtée après stage 24 (~5h30)

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma-generic-1.0.1.mlmodel` (S3 `base-model/cremma-generic-1.0.1.mlmodel`) |
| Format données | Arrow compilé **in-situ** depuis `preprocessed/` **(mode 1 binarisé)** |
| Train set | 213 fichiers ALTO → 19 793 lignes compilées |
| Dev set | 32 fichiers ALTO → 3 729 lignes compilées |
| Arrow train | 2 145 MB |
| Arrow dev | 269 MB |
| Batch size | 16 |
| Max epochs | 50 (`-N 50`) |
| Early stopping (lag) | 20 (`--lag 20`) |
| Min epochs | 3 |
| Workers | 11 |
| Precision | `bf16-mixed` (A100 Ampere sm_80) |
| Learning rate | 0.0001 |
| Augmentation | oui |
| `--resize` | union |

### Courbe d'entraînement
| Stage | val_accuracy | CER | train_loss | Patience |
|-------|:-----------:|:---:|:----------:|:--------:|
| 0 | 70.7% | 29.3% | 228.4 | 0/20 |
| 1 | 71.1% | 28.9% | 189.7 | 0/20 |
| 2 | 71.3% | 28.7% | 172.1 | 0/20 |
| 3 | 71.6% | 28.4% | 158.1 | 0/20 |
| 4 | 71.6% | 28.4% | 148.2 | 1/20 |
| 5 | 71.3% | 28.7% | 141.2 | 2/20 |
| 6 | 71.6% | 28.4% | 132.5 | 0/20 |
| 7 | 71.4% | 28.6% | 123.9 | 1/20 |
| 8 | 71.7% | 28.3% | 121.3 | 0/20 |
| 9 | 71.7% | 28.3% | 117.8 | 0/20 |
| 10 | 71.5% | 28.5% | 113.8 | 1/20 |
| 11 | 71.5% | 28.5% | 109.8 | 2/20 |
| **12** | **71.9%** | **28.1%** | 104.8 | 0/20 |
| 13 | 71.7% | 28.3% | 102.1 | 1/20 |
| 14 | 71.5% | 28.5% | 99.6 | 2/20 |
| 15 | 71.8% | 28.2% | 98.7 | 3/20 |
| 16 | 71.5% | 28.5% | 95.3 | 4/20 |
| 17 | 71.7% | 28.3% | 93.1 | 5/20 |
| 18 | 71.4% | 28.6% | 93.7 | 6/20 |
| 19 | 71.3% | 28.7% | 89.8 | 7/20 |
| 20 | 71.5% | 28.5% | 86.9 | 8/20 |
| 21 | 71.6% | 28.4% | 86.5 | 9/20 |
| 22 | 71.6% | 28.4% | 84.8 | 10/20 |
| 23 | 71.1% | 28.9% | 81.7 | 11/20 |
| 24 | 71.5% | 28.5% | 82.0 | 12/20 |

**Meilleure val_accuracy : 71.9% (stage 12) — CER 28.1%**

### Diagnostic
```
WARNING: Neural network has been trained on mode L images,
         training set contains mode 1 data.
```
- Même cause que Run 2 : Arrow compilés depuis `preprocessed/` (images binarisées mode 1)
- Le modèle `cremma-generic-1.0.1` attend des images en **mode L (niveaux de gris)**
- La train_loss descend continûment (228 → 82) mais val_accuracy plafonne à 71.9%
- **Stagnation structurelle** — pas un manque d'epochs

### Artefacts S3
- Modèle non uploadé (run arrêtée manuellement avant fin de session)

---

## Run 4 — Kaggle T4 x2 (13 juin 2026) ← meilleure run à ce jour

**Date :** 13 juin 2026, 14h50 → 17h30  
**Notebook :** `kaggle_training.ipynb`  
**Plateforme :** Kaggle — GPU T4 x2 (`CUDA_VISIBLE_DEVICES: [0,1]`)  
**Durée :** 159.4 min — 38 stages × ~4 min/stage

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma_generic.mlmodel` (S3 `base-model/cremma_generic.mlmodel`) |
| Format données | Arrow — compilé depuis `preprocessed/` **(mode 1 binarisé)** |
| Batch size | 8 |
| Max epochs | 50 (`-N 50`) |
| Early stopping (lag) | 10 (`--lag 10`) |
| Min epochs | 3 |
| Workers | 4 |
| Threads | 4 |
| Precision | `16-mixed` (fp16, T4 Turing sm_75) |
| Learning rate | 0.0001 |
| Augmentation | oui |
| `--resize` | union |

### Courbe d'entraînement complète
| Stage | val_accuracy | CER | train_loss | Patience |
|-------|:-----------:|:---:|:----------:|:--------:|
| 0 | 72.5% | 27.5% | 117.4 | 0/10 |
| 1 | 72.5% | 27.5% | 98.4 | 0/10 |
| 2 | 72.8% | 27.2% | 91.9 | 0/10 |
| 3 | 72.8% | 27.2% | 87.8 | 0/10 |
| 4 | 73.2% | 26.8% | 86.7 | 0/10 |
| 5 | 73.3% | 26.7% | 83.9 | 0/10 |
| 6 | 73.2% | 26.8% | 82.0 | 1/10 |
| 7 | 73.2% | 26.8% | 78.9 | 2/10 |
| 8 | 73.4% | 26.6% | 78.4 | 0/10 |
| 9 | 73.2% | 26.8% | 77.5 | 1/10 |
| 10 | 73.3% | 26.7% | 74.7 | 2/10 |
| 11 | 73.3% | 26.7% | 73.3 | 3/10 |
| 12 | 73.4% | 26.6% | 72.8 | 0/10 |
| 13 | 73.4% | 26.6% | 71.6 | 0/10 |
| 14 | 73.4% | 26.6% | 70.2 | 0/10 |
| 15 | 73.5% | 26.5% | 70.3 | 0/10 |
| 16 | 73.5% | 26.5% | 68.6 | 1/10 |
| 17 | 73.4% | 26.6% | 66.9 | 2/10 |
| 18 | 73.4% | 26.6% | 66.5 | 3/10 |
| 19 | 73.4% | 26.6% | 65.8 | 4/10 |
| 20 | 73.6% | 26.4% | 65.3 | 0/10 |
| 21 | 73.5% | 26.5% | 64.5 | 1/10 |
| 22 | 73.6% | 26.4% | 63.6 | 0/10 |
| 23 | 73.6% | 26.4% | 62.9 | 0/10 |
| 24 | 73.6% | 26.4% | 62.0 | 1/10 |
| 25 | 73.6% | 26.4% | 61.3 | 2/10 |
| 26 | 73.6% | 26.4% | 61.2 | 0/10 |
| **27** | **73.7%** | **26.3%** | 60.9 | 0/10 ← meilleur |
| 28 | 73.6% | 26.4% | 60.5 | 1/10 |
| 29 | 73.5% | 26.5% | 59.0 | 2/10 |
| 30 | 73.5% | 26.5% | 58.8 | 3/10 |
| 31 | 73.4% | 26.6% | 57.9 | 4/10 |
| 32 | 73.5% | 26.5% | 58.5 | 5/10 |
| 33 | 73.5% | 26.5% | 58.0 | 6/10 |
| 34 | 73.5% | 26.5% | 56.9 | 7/10 |
| 35 | 73.7% | 26.3% | 56.5 | 8/10 |
| 36 | 73.6% | 26.4% | 55.7 | 9/10 |
| 37 | 73.6% | 26.4% | 55.1 | **10/10 → ARRÊT** |

**Meilleure val_accuracy : 73.67% (stage 27) — CER : 26.33%**

### Diagnostic
```
WARNING: Neural network has been trained on mode L images,
         training set contains mode 1 data. Consider binarizing your data.
```
- Même mismatch L/1 que les runs précédentes
- Progression lente mais continue (72.5% → 73.7% sur 27 stages)
- Train_loss descend bien (117 → 55) mais val_accuracy plafonne
- **Arrêt early stopping : patience 10/10 au stage 37**

### Artefacts S3
- Checkpoint sauvegardé : `checkpoint_27-0.7367.ckpt` + `best_0.7367.safetensors`
- **Non uploadé sur S3** : cellule 7 cherchait `*.mlmodel` uniquement → bug corrigé depuis
- Note : ketos 7.x produit `.safetensors` au lieu de `.mlmodel` dans certains cas

---

## Run 5 — Kaggle T4 x2 (14 juin 2026)

**Date :** 14 juin 2026, 14h50 → 17h30  
**Notebook :** `kaggle_training.ipynb`  
**Plateforme :** Kaggle — GPU T4 x2  
**Durée :** 159.4 min — 38 stages × ~4 min/stage

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma-generic-1.0.1.mlmodel` |
| Format données | Arrow — `train.arrow` **(mode 1 binarisé — mismatch non corrigé)** |
| Batch size | 8 |
| Max epochs | 50 |
| Early stopping (lag) | 10 |
| Precision | `16-mixed` (fp16, T4) |
| Learning rate | 0.0001 |
| Augmentation | oui |

### Courbe d'entraînement complète
| Stage | val_accuracy | CER | train_loss | Patience |
|-------|:-----------:|:---:|:----------:|:--------:|
| 0 | 72.5% | 27.5% | 117.8 | 0/10 |
| 1 | 72.5% | 27.5% | 98.4 | 0/10 |
| 2 | 72.8% | 27.2% | 91.9 | 0/10 |
| 3 | 72.8% | 27.2% | 87.8 | 0/10 |
| 4 | 73.2% | 26.8% | 86.7 | 0/10 |
| 5 | 73.3% | 26.7% | 83.9 | 0/10 |
| 6 | 73.2% | 26.8% | 82.0 | 1/10 |
| 7 | 73.2% | 26.8% | 78.9 | 2/10 |
| 8 | 73.4% | 26.6% | 78.4 | 0/10 |
| 9 | 73.2% | 26.8% | 77.5 | 1/10 |
| 10 | 73.3% | 26.7% | 74.7 | 2/10 |
| 11 | 73.3% | 26.7% | 73.3 | 3/10 |
| 12 | 73.4% | 26.6% | 72.8 | 0/10 |
| 13 | 73.4% | 26.6% | 71.6 | 0/10 |
| 14 | 73.4% | 26.6% | 70.2 | 0/10 |
| 15 | 73.5% | 26.5% | 70.3 | 0/10 |
| 16 | 73.5% | 26.5% | 68.6 | 1/10 |
| 17 | 73.4% | 26.6% | 66.9 | 2/10 |
| 18 | 73.4% | 26.6% | 66.5 | 3/10 |
| 19 | 73.4% | 26.6% | 65.8 | 4/10 |
| 20 | 73.6% | 26.4% | 65.3 | 0/10 |
| 21 | 73.5% | 26.5% | 64.5 | 1/10 |
| 22 | 73.6% | 26.4% | 63.6 | 0/10 |
| 23 | 73.6% | 26.4% | 62.9 | 0/10 |
| 24 | 73.6% | 26.4% | 62.0 | 1/10 |
| 25 | 73.6% | 26.4% | 61.3 | 2/10 |
| 26 | 73.6% | 26.4% | 61.2 | 0/10 |
| **27** | **73.7%** | **26.3%** | 60.9 | 0/10 ← meilleur |
| 28 | 73.6% | 26.4% | 60.5 | 1/10 |
| 29 | 73.5% | 26.5% | 59.0 | 2/10 |
| 30 | 73.5% | 26.5% | 58.8 | 3/10 |
| 31 | 73.4% | 26.6% | 57.9 | 4/10 |
| 32 | 73.5% | 26.5% | 58.5 | 5/10 |
| 33 | 73.5% | 26.5% | 58.0 | 6/10 |
| 34 | 73.5% | 26.5% | 56.9 | 7/10 |
| 35 | 73.7% | 26.3% | 56.5 | 8/10 |
| 36 | 73.6% | 26.4% | 55.7 | 9/10 |
| 37 | 73.6% | 26.4% | 55.1 | **10/10 → ARRÊT** |

**Meilleure val_accuracy : 73.67% (stage 27) — CER : 26.33%**

### Diagnostic
```
WARNING: Neural network has been trained on mode L images,
         training set contains mode 1 data.
```
- Résultat identique à Run 4 — même mismatch L/1, même plafond
- Confirme que le plafond est structurel, pas lié au modèle de base

### Artefacts S3
- `output/best_0.7367_20260613_1738.safetensors` ← uploadé sur S3

---

## Run 7 — Exp 3 « grayscale optimisée » — Kaggle T4 x2 (15 juin 2026) ← résultat décisif NÉGATIF

**Date :** 15 juin 2026, démarrage 16:16:40
**Notebook :** `kaggle_exp3_finetune_optimise.ipynb`
**Plateforme :** Kaggle — GPU T4 x2 (`CUDA_VISIBLE_DEVICES: [0,1]`)
**Objectif annoncé :** percer le plafond CER ~26% via grayscale (mode L) + LR decay réactif

### Hypothèse testée
Le notebook combinait deux leviers : **(1) Arrow grayscale** (`train_clean.arrow`, censé être mode L) et **(2) fine-tuning doux** (LR 5e-5 + warmup 500 + reduceonplateau + freeze-backbone 2000 + NFD). Le levier (1) était présenté comme MAJEUR : *« Tout le reste n'a d'effet qu'une fois le mismatch L/1 levé. »*

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma-generic-1.0.1.mlmodel` |
| Format données | `train_clean.arrow` / `dev_clean.arrow` (SHA `1bec767c…` — **étiqueté grayscale, en réalité mode 1**) |
| Batch size | 8 |
| Max epochs | 50 (`-N 50`) |
| Early stopping (lag) | 15 (`--lag 15`) |
| Min epochs | 5 |
| Workers / Threads | 4 / 4 |
| Precision | `16-mixed` (fp16, T4 Turing sm_75) |
| Optimizer | AdamW |
| Learning rate | 0.00005 |
| Warmup | 500 samples |
| Schedule | reduceonplateau (`--sched-patience 5`) |
| Freeze-backbone | 2000 |
| Normalisation | NFD (`-u NFD`) |
| Augmentation | oui |
| `--resize` | union |

### Courbe d'entraînement (extraits, ~3 min 20 s/stage, 11.7 it/s)
| Stage | val_accuracy | CER | train_loss | Patience |
|-------|:-----------:|:---:|:----------:|:--------:|
| 0 | 72.17% | 27.83% | 145.5 | 0/15 |
| 5 | 72.95% | 27.05% | 93.5 | 0/15 |
| 10 | 73.22% | 26.78% | 87.3 | 0/15 |
| 15 | 73.32% | 26.68% | 81.0 | 0/15 |
| 20 | 73.44% | 26.56% | 77.1 | 0/15 |
| **24** | **73.56%** | **26.44%** | 75.9 | 0/15 ← meilleur |
| 25 | 73.56% | 26.44% | 73.1 | 1/15 |
| 28 | 73.56% | 26.44% | 71.8 | 4/15 |
| 30 | 73.42% | 26.58% | 71.8 | 6/15 |
| 32 | 73.56% | 26.44% | 69.8 | 8/15 |

**Meilleure val_accuracy : 73.56% (stage 24) — CER : 26.44%**

### Diagnostic
Le warning suivant est réapparu dès le chargement (16:17:06), avant le stage 0 :
```
WARNING  Neural network has been trained on mode L images,
         training set contains mode 1 data. Consider binarizing your data.
                                                           (vgsl.py:499)
```
- Le code de surveillance du notebook l'a interprété comme `WARNING MODE 1 DETECTE — l'Arrow est binarise`. **Cette interprétation s'est révélée FAUSSE.**
- **Vérification PIL (15 juin 20h57)** : `train_clean.arrow` et `dev_clean.arrow` sont **100% mode L** (508 et 529 lignes échantillonnées, aucune mode 1). Recompilés depuis `preprocessed_grayscale/`, ils donnent le **SHA identique** à la version S3 → le fichier utilisé par Exp 3 était déjà grayscale.
- Donc le warning `mode 1 data` de kraken **est un faux signal** : il ne décrit pas le mode des images du dataset (il inspecte vraisemblablement une métadonnée du modèle de base ou le premier record avant lecture complète).
- Le plateau (val_acc 73.56% dès le stage 24, train_loss continue 145 → 70) est donc **structurel mais PAS dû au L/1**.
- Autres warnings notables : `alphabet mismatch` (26 chars train-only + 2 dev-only) ; `ReduceLROnPlateau conditioned on metric val_metric which is not available` → **le scheduler reduceonplateau n'a jamais déclenché** (LR jamais réduit, fine-tuning resté à LR constant). **Piste sérieuse pour le plateau.**

### Conclusion
**Le grayscale était bien actif — l'hypothèse L/1 comme cause du plafond est RÉFUTÉE.** Le plafond ~26% a une autre origine, à investiguer :
1. **Scheduler LR non déclenché** (`val_metric not available`) → corriger le câblage de la métrique pour que reduceonplateau / cosine agisse réellement.
2. **Corpus** : 213 docs / 18 769 lignes — possible plafond de données.
3. **Alphabet** : 26 caractères train-only hors accuracy officielle.
4. Comparer au CER du modèle de base **sans** fine-tuning : vérifier que le fine-tuning apporte quelque chose.

### Artefacts S3
- Run early-stop — modèle sans gain vs baseline, mais grayscale confirmé : c'est la **vraie baseline grayscale** désormais.

---

## État de l'infrastructure S3

**Bucket :** `s3://htr-cremma-medieval/` (région `eu-west-3`)

### Modèles de base
| Fichier S3 | Taille | Date | Source |
|-----------|--------|------|--------|
| `base-model/cremma_generic.mlmodel` | 21.8 MB | 12 juin 2026 | Zenodo 7234166 |
| `base-model/cremma-generic-1.0.1.mlmodel` | 21.7 MB | 13 juin 2026 | Zenodo 7631619 |

### Données Arrow sur S3
| Fichier S3 | Taille | Date | Lignes | Description |
|-----------|--------|------|--------|-------------|
| `splits/train.arrow` | 939 MB | 13 juin 2026 | 19 797 | Grayscale mode L — toutes zones |
| `splits/dev.arrow` | 144 MB | 13 juin 2026 | 3 729 | Grayscale mode L — toutes zones |
| `splits/train_clean.arrow` | 914 MB | 14 juin 2026 | 18 769 | Grayscale mode L ✅ (vérifié PIL 15 juin : 508/508 mode L) — zones bruit filtrées (5.2%) |
| `splits/dev_clean.arrow` | 144 MB | 14 juin 2026 | 3 702 | Grayscale mode L ✅ (vérifié PIL 15 juin : 529/529 mode L) — zones bruit filtrées (0.7%) |

SHA-256 (inchangés après recompilation depuis `preprocessed_grayscale/` le 15 juin → fichiers S3 confirmés grayscale) :
- `train_clean.arrow` : `1bec767c9a87caa322b20dc054da85e161ab3e630c498eb1a35ae51d19348026` ✅ mode L confirmé
- `dev_clean.arrow` : `20ef530c68228695bb1b68f07a07b6eb2e2ffde0f62fd8e9c2e6b29d6720448e` ✅ mode L confirmé

### Splits texte (ALTO)
| Fichier | Lignes | Date |
|---------|--------|------|
| `splits/train.txt` | 213 fichiers XML | 12 juin 2026 |
| `splits/dev.txt` | 32 fichiers XML | 12 juin 2026 |
| `splits/test.txt` | 3 fichiers XML | 12 juin 2026 |

---

## Cause racine du plafond à 71–74%

Le modèle `cremma-generic-1.0.1` (et `cremma_generic`) sont entraînés sur des **images en niveaux de gris (mode L, 8 bits)**.

Les Runs 2 et 3 ont fourni des images **binarisées (mode 1, 2 valeurs)** compilées depuis l'ancien dossier `preprocessed/` (pipeline avec Sauvola activé).

**Conséquence :** Kraken émet un warning et tente d'adapter, mais les distributions de pixels sont incompatibles → plafond artificiel à ~72%.

---

## Prochaine run recommandée

**Objectif :** dépasser 80% val_accuracy  
**Action :** Utiliser les Arrow grayscale déjà sur S3

```
Ordre cellules Colab : 0 → 1 → 2 → 6a (télécharge Arrow S3) → 6b → 7
```

> Ne pas relancer les cellules 3, 4, 5 (inutiles avec le workflow Arrow)  
> Ne pas recompiler les Arrow — utiliser directement `splits/train.arrow` et `splits/dev.arrow` depuis S3

### Hyperparamètres suggérés pour la prochaine run
| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| Modèle de base | `cremma-generic-1.0.1.mlmodel` | Meilleure couverture alphabet médiéval |
| Format | Arrow grayscale (mode L) | Compatible avec le modèle de base |
| Batch | 16 (A100) / 8 (T4) | Taille VRAM |
| Precision | bf16-mixed (A100) / 16-mixed (T4) | Selon GPU |
| Lag | 5–10 | Laisser le temps de converger |
| LR | 0.0001 | Stable pour fine-tuning |
| Augmentation | oui | Améliore la généralisation |
