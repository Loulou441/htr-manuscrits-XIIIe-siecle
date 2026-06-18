# Pipeline NLP — Normalisation de manuscrits médiévaux français

*Résultats sur le corpus complet (129 documents, 16 336 lignes) — 18 juin 2026*

## 1. Le pipeline en un coup d'œil

```
Manuscrits Gallica  →  Extraction/transcription (modèle HTR)  →  Évaluation CER/WER
        ↓                                                              ↓
Alphabet + lexique (diagnostic)              Dictionnaire ancien français (référence)
                                                              ↓
Sortie HTR (JSON)  →  Validation (data contract)  →  Normalisation par règles
                                                              ↓
                                          Détection lexicale + Évaluation relative
```

- **Constitution du dataset** (14-17 juin) : extraction de 10 manuscrits du XIIIe siècle depuis Gallica/BnF, transcription par le modèle HTR, évaluation CER/WER contre une vérité terrain, construction d'un alphabet et d'un lexique diagnostiques, construction d'un dictionnaire de référence ancien français (55k entrées).
- **Entrée du pipeline NLP** : sortie brute du modèle HTR, un JSON par page (texte, confiance par caractère, polygones).
- **Normalisation** : 6 règles déterministes (priorité fixée par les consignes du projet : règles avant IA).
- **Sortie** : texte normalisé + rapports de qualité (lexical, CER relatif).

## 1bis. Constitution du dataset — étapes préalables (14-17 juin)

| Étape | Ce qui a été fait |
|---|---|
| **Extraction depuis le modèle** | 10 manuscrits médiévaux (Tristan, Roman de Troie, Brut de Wace...) téléchargés depuis Gallica, segmentés et transcrits par le modèle HTR (Kraken, baseline binarisée CER 26.3%) |
| **Évaluation / correction du dataset** | Confrontation des transcriptions à une vérité terrain ALTO/PageXML pour calculer CER et WER ligne par ligne |
| **Alphabet** | Comptage des caractères bruts par data contract (avec point de code Unicode) pour repérer les caractères parasites |
| **Lexique** | Liste des mots produits par le modèle avec fréquences, pour inspecter le vocabulaire et repérer les anomalies |
| **Dictionnaire de référence** | Fusion Wiktionary (ancien français) + CLTK → 54 921 entrées, utilisées ensuite pour la détection lexicale (`lexical-check`) |

## 2. Le data contract

Chaque document HTR est un JSON validé par un schéma strict (`htr_data_contract_schema.json`) :

| Champ | Rôle |
|---|---|
| `text` | transcription brute du HTR |
| `confidence`, `char_confidences` | scores de confiance globaux et par caractère |
| `candidates` | variantes alternatives proposées par le HTR (souvent `null` sur ce corpus) |
| `needs_review` | drapeau pour la file de relecture humaine |

**129/129 documents valides** contre le schéma.

## 3. Les règles de normalisation

Appliquées dans l'ordre, avant toute correction statistique :

1. **NFC** — formes Unicode combinées → précomposées
2. **lowercase**
3. **u/v** — résolution contextuelle des variantes graphiques médiévales (`auant`→`avant`, `cheualier`→`chevalier`), avec exceptions pour les digrammes `qu`/`gu` et `u+i`
4. **i/j** — `i` consonantique devant voyelle (sauf digramme `ie`)
5. **tilde nasal** — `a~/e~/o~` → `an/en/on`
6. **table d'abréviations** — 14 marqueurs scribaux et latinismes (`⁊→et`, `ꝑ→per`, `dñs→dominus`...)

### Impact mesuré sur tout le corpus

| Avant | Après | Occurrences |
|---|---|---|
| `qve` | `que` | 651 |
| `qvi` | `qui` | 420 |
| `bjen` | `bien` | 182 |
| `lvi` | `lui` | 182 |
| `qvil` | `quil` | 146 |
| `pvis` | `puis` | 112 |

→ **3725 paires de mots distinctes** corrigées par les règles sur le corpus complet.

## 4. Résultats chiffrés (corpus complet, run du 18/06)

| Mesure | Valeur |
|---|---|
| Documents validés | 129 / 129 (100%) |
| Lignes analysées (EDA) | 16 336 |
| Confiance HTR moyenne | 0.793 |
| Lignes signalées pour révision | 36.8% |
| Tests unitaires | 13 / 13 |
| CER pairwise moyen (raw / normalisé / corrigé) | 0.0667 |
| Tokens couverts par le dictionnaire ancien français (55k entrées) | 4.4% |

## 5. Limites connues, à mentionner à l'oral

- **Détection lexicale (4.4% de couverture)** : le dictionnaire externe (Wiktionary ancien français + CLTK) couvre le vocabulaire lexical mais pas les mots-outils très fréquents (`est`, `ce`, `vous`, `tout`...). Le taux bas reflète une limite de la ressource, pas un échec de normalisation — la preuve : les marqueurs typographiques bruts (`⁊`, `q̃`, `ꝑ`) ont disparu du classement après normalisation.
- **Correction guidée par confiance/MLM** : codée mais non activée sur ce run (`candidates` est `null` sur les données réelles ; pas de modèle MLM branché) — 0 correction appliquée. La colonne `corrected` du CER relatif est donc actuellement identique à la normalisation par règles.
- **Règle u/v** : exclut volontairement `u+i` (ex. `lui`) pour éviter les faux positifs, ce qui empêche `deuient→devient` d'être corrigé — compromis documenté et accepté.

## 6. Prochaines étapes possibles

- Activer la correction MLM (`almanach/camembert-base`) pour aller au-delà des règles déterministes.
- Enrichir le dictionnaire de référence avec les mots-outils pour rendre `lexical-check` plus discriminant.
- Étapes NER / POS / export TEI, prévues par les consignes mais non démarrées.
