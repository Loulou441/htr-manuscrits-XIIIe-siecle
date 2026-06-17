# Conventions NLP

Ce document décrit les règles de normalisation et d'évaluation utilisées dans le pipeline NLP du projet.

## 1. Normalisation déterministe

Les règles suivantes sont appliquées avant toute correction statistique/IA :

- `Unicode NFC` : normalisation des formes combinées pour assurer une comparaison stable.
- `lowercase` : conversion en minuscules pour réduire la variabilité graphique.
- `u/v` : les variantes médiévales sont harmonisées selon le contexte (début de mot / voyelle suivante).
- `i/j` : la lettre `i` est convertie en `j` lorsqu'elle précède une voyelle, comme dans les graphies moyen-français.
- `tilde nasal` : les abréviations nasales `a~`, `e~`, `o~` sont résolues en `an`, `en`, `on`.
- table d'abréviations : une table JSON spécifique aux formes médiévales et latines du corpus est appliquée.

La table `data/abbreviations/medieval_abbreviations.json` contient les correspondances préférentielles :
- `q~` → `que`
- `d~e` → `dame`
- `⁊` → `et`
- `ꝑ` → `per`
- `ꝗ` → `que`
- `ꝓ` → `pro`
- `ꝙ` → `us`
-  formes latines comme `dñs` → `dominus` et `q̄` → `qui`

## 2. Correction guidée par confiance

La correction guidée examine les positions où la confiance caractère est faible (`char_confidences < threshold`) et où des variantes sont proposées dans `candidates`.

Le principe :

- pour chaque position ambiguë, on génère des versions candidates de la ligne,
- on évalue chaque variante avec un modèle de langage en mode MLM,
- on choisit la variante la plus probable pour cette position.

Le modèle par défaut est `almanach/camembert-base` en mode Masked Language Model.

## 3. Évaluation relative

Parce qu'il n'existe pas de vérité terrain complète pour ces manuscrits, l'évaluation est comparative :

- on mesure le CER entre versions différentes d'une même transcription,
- on peut ainsi suivre l'évolution apportée par : sortie brute, normalisation par règles, correction guidée, etc.

Cette évaluation relative permet de juger des gains de qualité sans référence humaine absolue.

## 4. Mise en œuvre dans le code

- `src/normalization_rules.py` implémente les règles déterministes.
- `src/confidence_correction.py` implémente la correction guidée par confiance et par MLM.
- `src/cer_utils.py` calcule le CER et les scores relatifs entre variantes.
- `src/nlp_cli.py` expose les commandes `normalize`, `correct`, `ablation` et `relative-eval`.
