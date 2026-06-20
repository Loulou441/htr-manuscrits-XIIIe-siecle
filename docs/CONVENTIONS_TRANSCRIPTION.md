# Conventions de transcription — HTR Manuscrits XIIIe siècle

Corpus : 4 corpus HTR-United (CREMMA-Medieval, CREMMA-Medieval-LAT, HTRomance FR/LAT) (ancien français et latin, XIIIe siècle)  
Niveau de transcription : **diplomatique** (fidèle au manuscrit, sans normalisation)

## Abréviations
- Les abréviations sont conservées telles quelles (non développées)
- Les signes d'abréviation sont encodés en Unicode quand disponible (ex. ᷝ, ͫ, ᵖ)
- Les caractères spéciaux médiévaux sont préservés : ꝗ (q barré), ŧ (t barré), Ꝙ (Q barré)

## Lacunes et zones illisibles
- Zone illisible : `[...]`
- Caractère incertain : `[x?]`
- Lacune matérielle (trou, déchirure) : `[lac.]`

## Casse
- Respect de la casse originale du manuscrit
- Les majuscules de début de phrase sont conservées

## Ponctuation
- La ponctuation originale est transcrite (points, virgules médiévales)
- Pas d'ajout de ponctuation moderne

## Zones incluses dans l'entraînement
| Zone ALTO | Incluse | Raison |
|-----------|:-------:|--------|
| `MainZone` | ✓ | Texte principal |
| `MarginTextZone` | ✓ | Annotations marginales pertinentes |
| `HeadingLine` | ✓ | Titres courants |
| `DefaultLine` | ✓ | Lignes standard |
| `MusicZone` | ✗ | Portées musicales — pas de texte |
| `DropCapitalZone` | ✗ | Lettrines décorées — signal bruité |
| `DropCapitalLine` | ✗ | Idem |
| `NumberingZone` | ✗ | Numérotation de cahiers |
| `GraphicZone` | ✗ | Illustrations |
| `DamageZone` | ✗ | Zones endommagées |
| `DigitizationArtefactZone` | ✗ | Artefacts de numérisation |
| `InterlinearLine` | ✗ | Notes interlinéaires |
| `MusicLine` | ✗ | Lignes de notation musicale |

## Encodage
- UTF-8 obligatoire
- Normalisation Unicode : NFC
