# Sources des données — HTR CREMMA Medieval 2026

## Corpus d'entraînement

# Sources des données — HTR Manuscrits Médiévaux du XIIIe siècle

## Corpus d'entraînement

Le corpus agrège **33 manuscrits du XIIIe siècle** (ancien français + latin) issus de 4 corpus HTR-United publics, pour un total de **22 858 lignes** (brut, avant équilibrage/filtrage des zones bruit). Liste complète issue de l'annexe A de [`article_htr_manuscrits_XIIIe_siecle.tex`](../article/article_htr_manuscrits_XIIIe_siecle.tex).

- **Format** : ALTO XML + images JPEG (Gallica / BnF)
- **Langues** : ancien français (`fro`), latin (`lat`)
- **Licence** : CC-BY 4.0

### CREMMA-Medieval (7 manuscrits, ancien français)
- **Source** : [HTR-United / cremma-medieval](https://github.com/HTR-United/cremma-medieval)

| Manuscrit | Langue | Script | Lignes |
|-----------|--------|--------|-------:|
| BnF fr. 412 | fro | Gothic Textualis | 6324 |
| Arsenal 3516 | fro | Gothic Textualis | 1991 |
| Cologny, Bodmer 168 | fro | Gothic Textualis | 1976 |
| BnF fr. 24428 | fro | Gothic Textualis | 1328 |
| BnF fr. 844 | fro | Gothic Textualis | 224 |
| BnF fr. 17229 | fro | Gothic Textualis | 164 |
| BnF fr. 13496 | fro | Gothic Textualis | 161 |

### CREMMA-Medieval-LAT (4 manuscrits, latin)
- **Source** : [HTR-United / CREMMA-Medieval-LAT](https://github.com/HTR-United/CREMMA-Medieval-LAT)

| Manuscrit | Langue | Script | Lignes |
|-----------|--------|--------|-------:|
| CLM 13027 | lat | Semitextualis Libraria | 616 |
| MsWettF 15 | lat | Textualis Libraria | 455 |
| BnF lat. 16195 | lat | Semitextualis Currens | 449 |
| CCCC MSS 236 | lat | Textualis Libraria | 192 |

### HTRomance Medieval FR (14 manuscrits, ancien français)
- **Source** : [HTRomance-Project / medieval-french](https://github.com/HTRomance-Project/medieval-french)

| Manuscrit | Langue | Script | Lignes |
|-----------|--------|--------|-------:|
| BnF NAF 23686 | fro | Gothic Textualis | 424 |
| BnF fr. 1443 | fro | Gothic Textualis | 418 |
| BnF fr. 1553 | fro | Gothic Textualis | 506 |
| BnF fr. 1635 | fro | Gothic Textualis | 217 |
| BnF fr. 12581 | fro | Gothic Textualis | 306 |
| BnF fr. 1669 | fro | Gothic Textualis | 484 |
| BnF fr. 104 | fro | Gothic Textualis | 404 |
| BnF fr. 2168 | fro | Gothic Textualis | 370 |
| BnF fr. 1450 | fro | Gothic Textualis | 711 |
| BnF fr. 23117 | fro | Gothic Textualis | 736 |
| BnF fr. 6447 | fro | Gothic Textualis | 383 |
| BnF fr. 2173 | fro | Gothic Textualis | 240 |
| BnF fr. 19152 | fro | Gothic Textualis | 529 |
| BnF fr. 12603 | fro | Gothic Textualis | 442 |

### HTRomance Medieval LAT (8 manuscrits, latin)
- **Source** : [HTRomance-Project / medieval-latin](https://github.com/HTRomance-Project/medieval-latin)

| Manuscrit | Langue | Script | Lignes |
|-----------|--------|--------|-------:|
| BnF lat. 8001 | lat | Gothic Textualis | 506 |
| BnF lat. 16085 | lat | Gothic Textualis | 392 |
| BnF lat. 17903 | lat | Gothic Textualis | 440 |
| BnF lat. 14354 | lat | Gothic Textualis | 546 |
| BnF lat. 16204 | lat | Gothic Textualis | 462 |
| BnF lat. 16657 | lat | Gothic Textualis | 199 |
| BnF lat. 5657 | lat | Textualis Currens | 152 |
| BnF lat. 10996 | lat | Textualis Currens | 109 |

**Total : 33 manuscrits — 22 858 lignes (brut)**

> **Manuscrit hors-corpus** : BnF fr. 25516 (fro, XIIIe siècle, Gothic Textualis, 717 lignes, corpus CREMMA-Medieval) n'est **pas** inclus dans les 33 manuscrits ci-dessus. Il est réservé à un test de généralisation en conditions réelles sur document jamais vu à l'entraînement (voir README, [section 5](README.md#5-évaluation-détaillée)).



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