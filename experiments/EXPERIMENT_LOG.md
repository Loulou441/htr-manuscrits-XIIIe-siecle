# Journal des expériences — HTR Manuscrits Médievaux XIIIE siècle

> Ce document trace chaque décision, chaque run, chaque hypothèse testée.  
> Objectif : CER < 15% (validation) → CER < 8% (excellence)  
> Métrique principale : `val_accuracy` = 1 - CER

---

## État actuel

| Donnée | Valeur |
|--------|--------|
| Meilleur CER obtenu | **26.33%** (Run 4 — Kaggle T4 x2) |
| Objectif validation | CER < 15% |
| Objectif excellence | CER < 8% |
| Cause identifiée du plafond | Mismatch mode L / mode 1 |

---

## Hypothèses à tester (par ordre de priorité)

| # | Hypothèse | Impact estimé | Statut |
|---|-----------|:-------------:|--------|
| H1 | Arrow grayscale (mode L) → supprime le mismatch L/1 | **majeur** (+10–15 pts) | À tester |
| H2 | Filtrage zones bruit ALTO (MusicZone, DropCapital…) | modéré (+2–5 pts) | À tester |
| H3 | LR réduit (0.00005) pour fine-tuning plus stable | faible (+1–3 pts) | À tester |
| H4 | Lag plus grand (20) pour laisser converger | faible (+1–2 pts) | À tester |
| H5 | Comparaison TrOCR vs Kraken | — | Bonus (+1 pt note) |

---

## Expérience 1 — Baseline binarisée (CLÔTURÉE)

**Objectif** : établir la baseline avec les données existantes  
**Résultat** : CER ~26–28% sur toutes les runs (5 runs au total)  
**Conclusion** : mismatch mode L/1 = plafond artificiel, impossible à dépasser

Run 5 (14 juin, Kaggle T4) confirme : changer de modèle de base (`cremma-generic-1.0.1` au lieu de `cremma_generic`) ne change rien — même CER 26.33%, même stage 27, courbe identique à Run 4. Le plafond est dans les **données**, pas le modèle.

Voir [TRAINING_RUNS.md](TRAINING_RUNS.md) pour le détail des 5 runs.

---

## Expérience 2 — Arrow grayscale (H1) ← CLÔTURÉE (ABORTÉE)

**Date** : 14 juin 2026  
**Plateforme** : Colab T4 (1 GPU)  
**Notebook** : `notebooks/colab_exp2_grayscale.ipynb`  
**Hypothèse** : `train.arrow` sur S3 est en mode L → supprime le mismatch L/1

### Résultats (stage 0→14, aborté)
| Stage | val_accuracy | CER | train_loss | Patience |
|-------|:-----------:|:---:|:----------:|:--------:|
| 0 | 72.5% | 27.5% | 117.4 | 0/10 |
| 3 | 73.0% | 27.0% | 89.3 | 0/10 |
| 5 | 73.2% | 26.8% | 83.5 | 0/10 |
| 9 | 73.4% | 26.6% | 76.7 | 0/10 |
| 12 | 73.5% | 26.5% | 72.4 | 0/10 |
| 14 | 73.5% | 26.5% | 70.6 | 2/10 |

**Conclusion** : `train.arrow` S3 est binarisé (mode 1) — warning confirmé dès stage 0. Trajectoire identique aux runs 1–5 → même plafond ~74% inévitable. Run abortée à stage 14, aucun artefact sauvegardé.

**Hypothèse H1 invalidée sur `train.arrow` S3** — le seul Arrow grayscale vérifié est `train_clean.arrow` compilé localement (SHA-256 : `1bec767c...`). **Exp 3 est le vrai test.**

---

## Expérience 3 — Filtrage zones bruit (H2) ← DONNÉES PRÊTES

**Objectif** : exclure les lignes MusicZone, DropCapitalZone, InterlinearLine  
avant de compiler les Arrow pour ne garder que le texte courant

### Contexte
Analyse du corpus (13 juin 2026) :
```
Total lignes : 48 278
  MainZone (texte principal) : 45 438  (94.1%)
  MarginTextZone (marginalia) :    732  ( 1.5%)
  Bruit (Music/DropCap/Interl) :  2 108  ( 4.4%)
```

### Données compilées (14 juin 2026) ✓
- `src/compile_arrow.py` créé et fonctionnel
- `train_clean.arrow` : 18 769 lignes (−1 024, −5.2%) — SHA-256 : `1bec767c...`
- `dev_clean.arrow` : 3 702 lignes (−27, −0.7%) — SHA-256 : `20ef530c...`
- Uploadés sur S3 : `splits/train_clean.arrow` / `splits/dev_clean.arrow`
- Notebooks prêts : `notebooks/kaggle_exp3_clean_arrow.ipynb` + `notebooks/colab_exp3_clean_arrow.ipynb`

### Résultats
À compléter après lancement.

---

## Expérience 4 — TrOCR vs Kraken (H5 — Bonus)

**Objectif** : comparer deux architectures HTR sur le même corpus  
**Gain** : +1 point bonus sur la note finale  
**Prérequis** : avoir un CER Kraken stable (après Exp. 2 ou 3)

### Plan
- Fine-tuner `microsoft/trocr-base-handwritten` avec LoRA (r=8 puis r=16)
- Comparer sur le même jeu de dev
- Test de McNemar pour significance statistique
- Analyse qualitative des erreurs différentielles

### Résultats
À compléter.

---

## Décisions d'architecture

| Date | Décision | Raison |
|------|----------|--------|
| 12 juin 2026 | Kraken plutôt que TrOCR comme modèle principal | Modèle de base CREMMA spécialisé médiéval disponible |
| 12 juin 2026 | Arrow binaire plutôt que ALTO direct | Compilation une seule fois, chargement ~10x plus rapide |
| 13 juin 2026 | Grayscale (mode L) plutôt que binarisé (mode 1) | Modèle de base entraîné sur mode L |
| 13 juin 2026 | `cremma-generic-1.0.1` plutôt que `cremma_generic` | Meilleure couverture alphabet médiéval |
| 13 juin 2026 | Arrow compilés en local plutôt que sur Colab/Kaggle | Plus rapide, moins de risque d'erreur en session |

---

## Problèmes rencontrés et solutions

| Problème | Cause | Solution |
|----------|-------|---------|
| CER plafonné à 71–74% | Mismatch mode L/1 — données binarisées | Recompiler Arrow depuis `preprocessed_grayscale/` |
| Cellule 7 upload → `[]` | ketos 7.x produit `.safetensors` pas `.mlmodel` | Cellule 7 mise à jour pour chercher tous les formats |
| `KeyError: AWS_ACCESS_KEY_ID` sur Kaggle | `os.environ` ne contient pas les secrets Kaggle | Utiliser `UserSecretsClient()` |
| `--workers > 0` plante sous Windows | Pickling du pipeline ketos impossible avec spawn | `--workers 0` en local, workers sur GPU cloud |
| Arrow compilé depuis mauvaises données | Workflow Colab recompilait depuis `preprocessed/` binarisé | Compiler en local depuis `preprocessed_grayscale/` |

---

## Ressources S3

```
s3://htr-cremma-medieval/
├── base-model/
│   ├── cremma_generic.mlmodel          (21.8 MB — Zenodo 7234166)
│   └── cremma-generic-1.0.1.mlmodel   (21.7 MB — Zenodo 7631619)
├── splits/
│   ├── train.arrow                     (939 MB — grayscale mode L ✓)
│   ├── dev.arrow                       (144 MB — grayscale mode L ✓)
│   ├── train.txt                       (213 fichiers ALTO)
│   ├── dev.txt                         (32 fichiers ALTO)
│   └── test.txt                        (3 fichiers ALTO)
└── output/
    └── (modèles fine-tunés à venir)
```