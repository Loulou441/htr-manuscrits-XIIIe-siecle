# Model Card — HTR CREMMA Medieval 2026

## Modèle

**Nom** : `htr_cremma_ft`  
**Tâche** : Reconnaissance de texte manuscrit (HTR) — lignes individuelles  
**Architecture** : Kraken VGSL (5.7M paramètres) — fine-tuning depuis `cremma-generic-1.0.1`  
**Framework** : Kraken 7.0.x + PyTorch Lightning

## Performances (meilleure run à ce jour — Runs 4 & 5)

| Métrique | Valeur | Seuil validation | Seuil excellence |
|----------|:------:|:----------------:|:----------------:|
| val_accuracy | 73.67% | — | — |
| CER (val) | **26.33%** | < 15% | < 8% |
| WER (val) | ~46% | < 25% | < 15% |
| Stages | 37 (Run 4) / 14+ (Run 6 en cours) | — | — |

> ⚠️ Runs 1–6 utilisent toutes des données **binarisées (mode 1)** — `train.arrow` S3 s'avère binarisé malgré la vérification initiale. Le plafond ~74% est structurel. **Exp 3 (`train_clean.arrow`, mode L vérifié localement) est le prochain test décisif.**

### Format des modèles produits
- ketos < 7.0 → `.mlmodel`
- ketos 7.x → `.safetensors` (format HuggingFace, compatible `ketos train -i` et `ketos ocr`)

## Données d'entraînement

- **Corpus** : CREMMA Medieval (21 manuscrits, ancien français + latin)
- **Train** : 213 fichiers ALTO → ~19 800 lignes (Arrow compilé)
- **Dev** : 32 fichiers ALTO → ~3 700 lignes
- **Format images** : JPEG → mode L (grayscale) via `pre_traitement.py`
- **Prétraitement** : deskew + CLAHE + filtres + conversion grayscale

## Hyperparamètres (Run 4 — Kaggle T4 x2)

| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `cremma_generic.mlmodel` |
| Learning rate | 0.0001 |
| Batch size | 8 |
| Precision | 16-mixed (fp16) |
| Early stopping lag | 10 |
| Max epochs | 50 |
| Augmentation | oui |
| `--resize` | union |

## Limitations connues

1. **Mismatch mode L/1** — runs 1 à 4 entraînées sur données binarisées → plafond ~74%
2. **Corpus limité** — 213 documents train, ~19 800 lignes → sous-représentation de certains scribes
3. **Alphabet partiel** — 22 caractères du train set absents du modèle de base (non comptés dans l'accuracy)
4. **Biais temporel** — corpus XIIIe–XVe siècle principalement, sous-représentation XVe
5. **Biais linguistique** — majoritairement ancien français parisien

## Modèles publiés — HuggingFace

Dépôt public : [legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval) (licence CC-BY 4.0)

| Fichier | Expérience | CER (val) | Commit HF |
|---------|-----------|:---------:|-----------|
| `exp2_binarise_20260613.safetensors` | Exp 2 — binarisé mode 1 | 26.33% | `99843b75` |
| `exp3_clean_arrow_20260613.safetensors` | Exp 3 — Arrow filtré (à valider) | en cours | `5e43b1b1` |

## Artefacts S3 (privé)

| Fichier | Description |
|---------|-------------|
| `s3://htr-cremma-medieval/base-model/cremma-generic-1.0.1.mlmodel` | Modèle de base |
| `s3://htr-cremma-medieval/splits/train.arrow` | Arrow train binarisé (939 MB) |
| `s3://htr-cremma-medieval/splits/dev.arrow` | Arrow dev binarisé (144 MB) |
| `s3://htr-cremma-medieval/splits/train_clean.arrow` | Arrow train grayscale filtré (914 MB) |
| `s3://htr-cremma-medieval/splits/dev_clean.arrow` | Arrow dev grayscale filtré (144 MB) |
| `s3://htr-cremma-medieval/output/` | Modèles fine-tunés |

## Citation

```bibtex
@misc{htr-cremma-medieval-2026,
  title  = {HTR CREMMA Medieval 2026 — Fine-tuning Kraken sur manuscrits médiévaux},
  author = {Ouazar, Djamal and Tessier, Manon and El Mortada, Hamza},
  year   = {2026},
  url    = {https://github.com/loulou441/htr-cremma-medieval-2026}
}
```
