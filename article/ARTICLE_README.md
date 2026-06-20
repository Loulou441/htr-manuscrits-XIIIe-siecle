# Article Scientifique HTR Manuscrits XIIIe siècle — Guide de Compilation

## Fichiers produits

- **article_htr_manuscrits_XIIIe_siecle.tex** — Article principal en LaTeX (8 sections)
- **references.bib** — Bibliographie (BibTeX format)

## Contenu de l'article

### 1. **Abstract**
Résumé de 200 mots couvrant objectifs, méthodologie, résultats (CER 26.3%, accuracy 73.67%), et contribution.

### 2. **Introduction**
- Contexte : HTR sur manuscrits médiévaux
- Problème : domain gap de cremma-generic
- Objectifs : pipeline complet de fine-tuning

### 3. **Dataset**
- **Composition** : 33 manuscrits (ancien français + latin, XIIIe siècle)
- **Sources** : 4 corpus HTR-United (CREMMA-Medieval, CREMMA-Medieval-LAT, HTRomance FR/LAT)
- **Statistiques** : 213 fichiers train, 32 dev, 3 test → 22 858 lignes filtrées
- **Zones ALTO** : inclusionMainZone + MarginTextZone; exclusion MusicZone, DropCapital, Interlinear
- **Alphabet** : caractères médiévaux spéciaux (ꝗ, ŧ, ᷝ, etc.)
- **Licence** : CC-BY 4.0

### 4. **Pré-traitement**
- **Deskewing** : 3 méthodes indépendantes (projection, Hough, gradient) + consensus
- **CLAHE** : Contrast Limited Adaptive Histogram Equalization (clip_limit=3.0, tile=8×8)
- **Filtrage** : Médian (5×5) + Gaussien (σ=0.5)
- **Modes** : Mode 1 binarisé vs Mode L grayscale (sans binarisation)
- **Pipeline** : Diagnostique → Décision → Appl

### 5. **Entraînement**
- **Modèle de base** : cremma_generic.mlmodel ou cremma-generic-1.0.1.mlmodel
- **Framework** : Kraken 7.x + PyTorch Lightning
- **Hyperparamètres** :
  - Learning rate : 1e-4 (fine-tuning préservant poids pré-entraînés)
  - Batch size : 8 (limites VRAM Kaggle T4)
  - Precision : 16-mixed (fp16 + loss scaling)
  - Early stopping lag : 10–20 epochs
  - Augmentation : rotation ±2°, stretch 0.9–1.1×, shear 0.1 rad
  - Resize : union (aspect ratio préservé)
- **6 runs documentées** (11–14 juin 2026) :
  1. Run 1 : Local Windows — bloquée (pickling workers)
  2. Run 2 : Kaggle T4 — ~73% val_acc, logs partiels
  3. Run 3 : Colab A100 — 71.9% val_acc
  4. **Run 4 : Kaggle T4 — 73.67% val_acc, CER 26.3% [MEILLEURE]**
  5. Run 5 : Kaggle T4 (modèle 1.0.1) — identique Run 4
- **Diagnostic majeur** : Mismatch mode L/1 identifié (données binarisées vs modèle entraîné en grayscale) — plafonne accuracy à ~74%

### 6. **Résultats et Évaluation**
- **Métriques** : CER, Accuracy, WER
- **Performances** : accuracy 73.67%, CER 26.3%, WER ~46% (Run 4)
- **Analyse par classe** : caractères simples CER ~15–18%, symboles médiévaux CER ~35–45%

### 7. **Discussion**
- **Limitations** : mismatch L/1, corpus de 19 K lignes (modeste), alphabet incomplet (22 caractères absent du vocabulaire de base)
- **Roadmap** :
  - Court terme : Exp 3 (mode L), extension corpus (25–30 K lignes)
  - Moyen terme : augmentation synthétique, ensemble de modèles
  - Long terme : fine-tuning d'architecture, données pour caractères manquants

### 8. **Conclusion**
- Résumé : pipeline complet documenté, meilleure performance 26.3% CER
- Prochaines étapes : résoudre mismatch L/1, extension corpus, ensembling
- Reproducibilité : GitHub + HuggingFace publics

## Compilation LaTeX

### Prerequis
- `pdflatex` ou `xelatex`
- `bibtex`
- Packages TeXLive/MiKTeX standard

### Instructions (Linux/macOS/Windows)

```bash
# Compilation complète (avec références)
pdflatex article_htr_manuscrits_XIIIe_siecle.tex
bibtex article_htr_manuscrits_XIIIe_siecle.aux
pdflatex article_htr_manuscrits_XIIIe_siecle.tex
pdflatex article_htr_manuscrits_XIIIe_siecle.tex

# Ou utiliser latexmk (automatisé)
latexmk -pdf article_htr_manuscrits_XIIIe_siecle.tex

# Résultat : article_htr_manuscrits_XIIIe_siecle.pdf
```

### VS Code / TeXstudio
- Ouvrir `article_htr_manuscrits_XIIIe_siecle.tex`
- Cliquer "Build" ou `Ctrl+Alt+B`
- Ou utiliser extension LaTeX Workshop (VS Code)

## Personnalisation

### Header/Footer
- Lignes 155–161 : modifier titre court et auteurs en en-tête

### Informations de première page
- Ligne 135 : "Volume 1, Issue 1, 2026"
- Ligne 136 : "HETIC. Mastère Data \& IA"
- Ligne 139–140 : dates et email correspondant

### Titre et auteurs
- Lignes 145–150 : modifier titre, noms, affiliations

## Notes

- L'article adopte le style IEEE pour la bibliographie (numérotation automatique)
- Format deux colonnes, 10pt, conforme à IEEE/HETIC
- Tableaux et figures peuvent être ajoutés via `\begin{figure}[H]...\end{figure}`
- Pour ajouter une image : `\includegraphics[width=\columnwidth]{image.pdf}`

## Statistiques du document

- Sections : 8 (Introduction, Dataset, Pré-traitement, Entraînement, Résultats, Discussion, Conclusion)
- Tableaux : 11 (composition corpus, zones ALTO, splits, hyperparamètres, runs, courbes, etc.)
- Références bibliographiques : 25+ (HTR, HTR-United, Kraken, CREMMA, vision par ordinateur)
- Longueur estimée : 5–6 pages en format deux colonnes
- Mots-clés : 6

## Historique du projet

- **Date création article** : 19 juin 2026
- **Période des expériences** : 11–14 juin 2026
- **Meilleure run** : Run 4, Kaggle T4 × 2, 13 juin 2026
- **Modèles publiés** : HuggingFace legb/htr-cremma-medieval (CC-BY 4.0)

---

*Article généré pour HETIC Mastère Data & IA. Tous les résultats basés sur expériences réelles et données du projet HTR Manuscrits XIIIe siècle.*
