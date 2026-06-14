# Sources des données — HTR CREMMA Medieval 2026

## Corpus d'entraînement

### CREMMA Medieval
- **Source** : [HTR-United / cremma-medieval](https://github.com/HTR-United/cremma-medieval)
- **Licence** : CC-BY 4.0
- **Contenu** : 15 manuscrits en ancien français, XIIIe–XVe siècle
- **Format** : ALTO XML + images JPEG (Gallica / BnF)
- **Langues** : ancien français (`fro`), latin (`lat`)
- **Manuscrits inclus** :

| Manuscrit | Langue | Siècle | Lignes |
|-----------|--------|--------|--------|
| Arsenal_3516 | fro | XIIIe | ~500 |
| BnF_NAF_23686 | fro | XIIIe | ~200 |
| BnF_fr_104 | fro | XIIIe | ~300 |
| BnF_fr_12581 | fro | XIVe | ~400 |
| BnF_fr_12603 | fro | XIVe | ~400 |
| BnF_fr_13496 | fro | XIVe | ~300 |
| BnF_fr_1443 | fro | XIIIe | ~400 |
| BnF_fr_1450 | fro | XIVe | ~400 |
| BnF_fr_1553 | fro | XIVe | ~400 |
| BnF_fr_1635 | fro | XIVe | ~400 |
| BnF_fr_1669 | fro | XIVe | ~300 |
| BnF_fr_17229 | fro | XIVe | ~300 |
| BnF_fr_19152 | fro | XIVe | ~300 |
| BnF_fr_2168 | fro | XIIIe | ~300 |
| BnF_fr_2173 | fro | XIIIe | ~300 |
| BnF_fr_23117 | fro | XIVe | ~300 |
| BnF_fr_24428 | fro | XIVe | ~300 |
| BnF_fr_412 | fro | XIIIe | ~400 |
| BnF_fr_6447 | fro | XIVe | ~300 |
| BnF_fr_844 | fro | XIVe | ~400 |
| Cologny,_Bodmer_168 | fro | XIIIe | ~300 |
| BnF_lat_* (12 manuscrits) | lat | XIIIe–XVe | ~3000 |

## Modèles pré-entraînés

### cremma_generic.mlmodel
- **Source** : [Zenodo 7234166](https://zenodo.org/records/7234166)
- **Licence** : CC-BY 4.0
- **Usage** : modèle de base pour fine-tuning (latin + ancien français médiéval)

### cremma-generic-1.0.1.mlmodel
- **Source** : [Zenodo 7631619](https://zenodo.org/records/7631619)
- **Licence** : CC-BY 4.0
- **Usage** : modèle de base v1.0.1 — meilleure couverture alphabet médiéval

## Modèles publiés

### HuggingFace — legb/htr-cremma-medieval
- **Dépôt** : [huggingface.co/legb/htr-cremma-medieval](https://huggingface.co/legb/htr-cremma-medieval)
- **Licence** : CC-BY 4.0

| Fichier | Expérience | CER (val) | Commit HF |
|---------|-----------|:---------:|-----------|
| `exp2_binarise_20260613.safetensors` | Exp 2 — données binarisées (mode 1) | 26.33% | `99843b75` |
| `exp3_clean_arrow_20260613.safetensors` | Exp 3 — Arrow filtré grayscale (résultat en attente) | — | `5e43b1b1` |

## Infrastructure

### Amazon S3
- **Bucket** : `s3://htr-cremma-medieval/` (région `eu-west-3`)
- **Usage** : stockage des Arrow compilés, modèles de base, modèles entraînés
- **Accès** : credentials AWS via Kaggle Secrets / Colab Secrets (jamais hardcodés)

## Hachage SHA-256 du jeu de test

> À compléter après finalisation du split test scellé.

```
splits/test.txt : SHA-256 = TODO
```
