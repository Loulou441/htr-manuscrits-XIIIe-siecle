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

La table `nlp_pipeline/medieval_abbreviations.json` contient les correspondances préférentielles :
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

- on mesure le CER pairwise moyen entre plusieurs variantes d'une même ligne (`raw`, `text_normalized`, `corrected`) via `average_pairwise_cer` / la commande `relative-eval`,
- on peut ainsi suivre la stabilité du pipeline : sortie brute, normalisation par règles, correction guidée, etc.

**Ne pas utiliser `ablation` sans une vraie colonne de référence** (transcription validée manuellement, ex. XML ALTO/PageXML). Sans cela, `CER before` est trivialement 0 (comparaison du texte brut à lui-même) et `CER after` ne mesure que l'ampleur des changements introduits par la normalisation — pas un gain de qualité réel.

## 4. Détection lexicale

En complément de `detect-normalization` (qui repère des marqueurs typographiques d'abréviation : `~`, `⁊`, `ꝑ`...), la commande `lexical-check` vérifie si chaque token existe dans `dictionnaire_ancien_francais.json` (lexique Wiktionary ancien français + CLTK, ~55k entrées, téléchargé/construit par `nlp_pipeline/lexique/dictionary.py` puis synchronisé sur `s3://htr-cremma-medieval/nlp/dictionary/`). Les tokens absents du dictionnaire sont des candidats à une vraie erreur lexicale (mot mal transcrit, abréviation non résolue, terme hors corpus).

## 5. Mise en œuvre dans le code

- `nlp_pipeline/normalization_rules.py` implémente les règles déterministes, la détection de marqueurs d'abréviation (`detect_normalization_candidates`) et la détection lexicale par dictionnaire (`find_lexical_errors`).
- `nlp_pipeline/confidence_correction.py` implémente la correction guidée par confiance et par MLM.
- `nlp_pipeline/cer_utils.py` calcule le CER et les scores relatifs entre variantes.
- `nlp_pipeline/lexique/dictionary.py` construit `dictionnaire_ancien_francais.json` à partir de Wiktionary et du lexique CLTK.
- `nlp_pipeline/nlp_cli.py` expose les commandes `validate`, `eda`, `review-queue`, `normalize`, `normalize-contract`, `correct`, `ablation`, `relative-eval`, `detect-normalization`, `lexical-check` et `split`.
