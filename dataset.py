"""
dataset.py: Téléchargement et compilation du dataset HTR XIIIe siècle
--> Récupère automatiquement les manuscrits du XIIIe siècle depuis 4 corpus
HTR-United (CREMMA-Medieval, CREMMA-Medieval-LAT, HTRomance FR/LAT) et
applique optionnellement (pour l'instant) un rééquilibrage 60 % AF / 40 % LAT.

Usage :
    python dataset.py                        # dataset complet (~25 000 lignes)
    python dataset.py --balance              # rééquilibrage 60/40 (~14 700 l.)
    python dataset.py --output ./data        # dossier de sortie personnalisé
    python dataset.py --balance --dry-run    # affiche le plan sans télécharger
"""

import os
import sys
import json
import random
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Définition du catalogue

@dataclass
class Manuscript:
    shelfmark: str                      # cote du manuscrit
    lang: str                           # "fro" (ancien fr.) ou "lat"
    century: str                        # "XIII"
    script: str                         # type d'écriture
    lines_total: int                    # nombre de lignes dans le corpus source
    corpus: str                         # nom du corpus HTR-United
    repo_url: str                       # URL du dépôt GitHub
    subdir: str                         # sous-dossier dans le dépôt
    cap: Optional[int] = None           # plafond de lignes après équilibrage


# Corpus 1: CREMMA-Medieval (ancien français XIIIe)
CREMMA_REPO = "https://github.com/HTR-United/cremma-medieval"

# Corpus 2: CREMMA-Medieval-LAT (latin XIIIe)
CREMMA_LAT_REPO = "https://github.com/HTR-United/CREMMA-Medieval-LAT"

# Corpus 3: HTRomance Medieval French
HTROMANCE_FR_REPO = "https://github.com/HTRomance-Project/medieval-french"

# Corpus 4: HTRomance Medieval Latin
HTROMANCE_LAT_REPO = "https://github.com/HTRomance-Project/medieval-latin"

MANUSCRIPTS: list[Manuscript] = [

    # CREMMA-Medieval
    Manuscript(
        shelfmark="BnF fr. 412",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=6324,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_412",
    ),
    Manuscript(
        shelfmark="Arsenal 3516",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=1991,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/Arsenal_3516",
    ),
    Manuscript(
        shelfmark="Cologny, Bodmer 168",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=1976,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/Bodmer_168",
    ),
    Manuscript(
        shelfmark="BnF fr. 24428",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=1328,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_24428",
    ),
    Manuscript(
        shelfmark="BnF fr. 25516",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=717,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_25516",
    ),
    Manuscript(
        shelfmark="BnF fr. 844",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=224,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_844",
    ),
    Manuscript(
        shelfmark="BnF fr. 17229",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=164,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_17229",
    ),
    Manuscript(
        shelfmark="BnF fr. 13496",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=161,
        corpus="CREMMA-Medieval",
        repo_url=CREMMA_REPO,
        subdir="data/fr_13496",
    ),

    # CREMMA-Medieval-LAT
    Manuscript(
        shelfmark="CLM 13027",
        lang="lat", century="XIII",
        script="Southern Textualis Libraria",
        lines_total=616,
        corpus="CREMMA-Medieval-LAT",
        repo_url=CREMMA_LAT_REPO,
        subdir="data/CLM13027",
    ),
    Manuscript(
        shelfmark="MsWettF 15",
        lang="lat", century="XIII",
        script="Textualis Libraria",
        lines_total=455,
        corpus="CREMMA-Medieval-LAT",
        repo_url=CREMMA_LAT_REPO,
        subdir="data/WettF0015",
    ),
    Manuscript(
        shelfmark="BnF lat. 16195",
        lang="lat", century="XIII",
        script="Semitextualis Currens",
        lines_total=449,
        corpus="CREMMA-Medieval-LAT",
        repo_url=CREMMA_LAT_REPO,
        subdir="data/Latin16195",
    ),
    Manuscript(
        shelfmark="CCCC MSS 236",
        lang="lat", century="XIII",
        script="Textualis Libraria",
        lines_total=192,
        corpus="CREMMA-Medieval-LAT",
        repo_url=CREMMA_LAT_REPO,
        subdir="data/CCCC-MSS-236",
    ),

    # HTRomance Medieval French
    Manuscript(
        shelfmark="BnF NAF 23686",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=424,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-NAF-23686",
    ),
    Manuscript(
        shelfmark="BnF fr. 1443",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=418,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-1443",
    ),
    Manuscript(
        shelfmark="BnF fr. 1553",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=506,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-1553",
    ),
    Manuscript(
        shelfmark="BnF fr. 1635",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=217,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-1635",
    ),
    Manuscript(
        shelfmark="BnF fr. 12581",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=306,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-12581",
    ),
    Manuscript(
        shelfmark="BnF fr. 1669",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=484,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-1669",
    ),
    Manuscript(
        shelfmark="BnF fr. 104",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=404,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-104",
    ),
    Manuscript(
        shelfmark="BnF fr. 2168",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=370,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-2168",
    ),
    Manuscript(
        shelfmark="BnF fr. 1450",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=711,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-1450",
    ),
    Manuscript(
        shelfmark="BnF fr. 23117",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=736,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-23117",
    ),
    Manuscript(
        shelfmark="BnF fr. 6447",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=383,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-6447",
    ),
    Manuscript(
        shelfmark="BnF fr. 2173",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=240,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-2173",
    ),
    Manuscript(
        shelfmark="BnF fr. 19152",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=529,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-19152",
    ),
    Manuscript(
        shelfmark="BnF fr. 12603",
        lang="fro", century="XIII",
        script="Gothic Textualis",
        lines_total=442,
        corpus="HTRomance Medieval FR",
        repo_url=HTROMANCE_FR_REPO,
        subdir="data/BnF-fr-12603",
    ),

    # HTRomance Medieval Latin
    Manuscript(
        shelfmark="BnF lat. 8001",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=506,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-8001",
    ),
    Manuscript(
        shelfmark="BnF lat. 16085",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=392,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-16085",
    ),
    Manuscript(
        shelfmark="BnF lat. 17903",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=440,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-17903",
    ),
    Manuscript(
        shelfmark="BnF lat. 14354",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=546,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-14354",
    ),
    Manuscript(
        shelfmark="BnF lat. 16204",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=462,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-16204",
    ),
    Manuscript(
        shelfmark="BnF lat. 16657",
        lang="lat", century="XIII",
        script="Gothic Textualis",
        lines_total=199,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-16657",
    ),
    Manuscript(
        shelfmark="BnF lat. 5657",
        lang="lat", century="XIII",
        script="Textualis Currens",
        lines_total=152,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-5657",
    ),
    Manuscript(
        shelfmark="BnF lat. 10996",
        lang="lat", century="XIII",
        script="Textualis Currens",
        lines_total=109,
        corpus="HTRomance Medieval LAT",
        repo_url=HTROMANCE_LAT_REPO,
        subdir="data/BnF-lat-10996",
    ),
]


# Statistiques

def print_stats(manuscripts: list[Manuscript], label: str = "") -> None:
    #Affiche les statistiques du dataset
    total_fro = sum(m.lines_total for m in manuscripts if m.lang == "fro")
    total_lat = sum(m.lines_total for m in manuscripts if m.lang == "lat")
    total = total_fro + total_lat

    corpora = {}
    for m in manuscripts:
        corpora.setdefault(m.corpus, 0)
        corpora[m.corpus] += m.lines_total

    print(f"\n{'='*60}")
    print(f"  DATASET {label}")
    print(f"{'='*60}")
    print(f"  Manuscrits        : {len(manuscripts)}")
    print(f"  Ancien français   : {total_fro:>6} lignes  ({total_fro/total*100:.1f} %)")
    print(f"  Latin             : {total_lat:>6} lignes  ({total_lat/total*100:.1f} %)")
    print(f"  TOTAL             : {total:>6} lignes")
    print(f"\n  Répartition par corpus :")
    for corpus, n in sorted(corpora.items()):
        print(f"    {corpus:<30} {n:>6} lignes")
    print(f"{'='*60}\n")


# Rééquilibrage 60/40 (Optionnel pour l'instant)
def apply_balance(
    manuscripts: list[Manuscript],
    af_ratio: float = 0.60,
    random_seed: int = 42,
) -> list[Manuscript]:
    """
    Rééquilibre le dataset en plafonnant les manuscrits d'ancien français.
    --> Le latin est conservé intégralement (40 % cible).
    --> Le côté AF est sous-échantillonné par manuscrit proportionnellement,
      avec un plafond minimal de MIN_CAP lignes pour les petits manuscrits.
    --> Le plafond est calculé de façon à atteindre exactement af_ratio.

    Retourne une nouvelle liste de Manuscript avec le champ `cap` renseigné
    pour les manuscrits AF (cap=None = pas de restriction).
    """
    random.seed(random_seed)
    MIN_CAP = 100  # seuil minimal par manuscrit AF pour garder de la diversité

    lat_total = sum(m.lines_total for m in manuscripts if m.lang == "lat")
    # Si AF vise 60 % et LAT 40 %, alors total = lat_total / 0.40
    # donc AF cible = total * 0.60 = lat_total * 1.5
    af_target = int(lat_total * (af_ratio / (1 - af_ratio)))

    af_mss = [m for m in manuscripts if m.lang == "fro"]
    af_total_raw = sum(m.lines_total for m in af_mss)

    if af_total_raw <= af_target:
        log.warning(
            "Le total AF brut (%d) est déjà <= à la cible (%d). "
            "Aucun plafonnement nécessaire.",
            af_total_raw, af_target,
        )
        return manuscripts

    # Ratio de réduction global
    reduction = af_target / af_total_raw

    result = []
    for m in manuscripts:
        if m.lang != "fro":
            result.append(m)
            continue
        # Plafond proportionnel, au moins MIN_CAP
        cap = max(MIN_CAP, int(m.lines_total * reduction))
        import dataclasses
        result.append(dataclasses.replace(m, cap=cap))

    # Vérification
    af_after = sum(min(m.lines_total, m.cap or m.lines_total) for m in result if m.lang == "fro")
    lat_after = sum(m.lines_total for m in result if m.lang == "lat")
    ratio = af_after / (af_after + lat_after) * 100

    log.info("Après équilibrage : AF=%d, LAT=%d (ratio AF=%.1f%%)", af_after, lat_after, ratio)
    return result


# Clonage / mise à jour des dépôts

def _run(cmd: list[str], cwd: Optional[str] = None) -> int:
    log.debug("CMD: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("STDERR: %s", result.stderr.strip())
    return result.returncode


def clone_or_pull(repo_url: str, target_dir: Path) -> bool:
    """Clone le dépôt si absent, sinon fait un git pull."""
    if (target_dir / ".git").exists():
        log.info("Mise à jour de %s ...", target_dir.name)
        rc = _run(["git", "pull", "--quiet"], cwd=str(target_dir))
    else:
        log.info("Clonage de %s ...", repo_url)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        rc = _run(["git", "clone", "--depth=1", "--quiet", repo_url, str(target_dir)])
    return rc == 0


# Copie des données vers le dossier de sortie

def copy_manuscript_data(
    manuscript: Manuscript,
    repo_dir: Path,
    output_dir: Path,
) -> int:
    """
    Copie les fichiers ALTO XML (et images associées) d'un manuscrit
    vers output_dir/lang/shelfmark_slug/.

    Si un plafond (cap) est défini, seuls les `cap` premiers fichiers XML
    sont copiés (triés alphabétiquement pour la reproductibilité).

    Retourne le nombre de fichiers XML copiés.
    """
    src = repo_dir / manuscript.subdir
    if not src.exists():
        log.warning("  Dossier introuvable : %s", src)
        return 0

    # Slug pour le nom de dossier de destination
    slug = manuscript.shelfmark.replace(" ", "_").replace("/", "-").replace(".", "")
    dest = output_dir / manuscript.lang / slug
    dest.mkdir(parents=True, exist_ok=True)

    # Lister tous les fichiers ALTO XML
    xml_files = sorted(src.glob("**/*.xml"))
    if not xml_files:
        log.warning("  Aucun fichier XML dans %s", src)
        return 0

    # Appliquer le plafond si défini
    # NOTE : le plafond est en lignes, pas en fichiers.
    # On approxime : 1 fichier XML ≈ nombre de lignes total / nombre de fichiers.
    if manuscript.cap is not None and len(xml_files) > 0:
        lines_per_file = manuscript.lines_total / len(xml_files)
        n_files = max(1, int(manuscript.cap / lines_per_file))
        xml_files = xml_files[:n_files]
        log.debug("  Plafond %d lignes → %d fichiers retenus", manuscript.cap, n_files)

    copied = 0
    for xml_file in xml_files:
        dest_file = dest / xml_file.name
        shutil.copy2(xml_file, dest_file)
        copied += 1

        # Copier l'image associée si elle existe (même nom, extension image)
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            img = xml_file.with_suffix(ext)
            if img.exists():
                shutil.copy2(img, dest / img.name)
                break

    return copied


# Génération du manifeste JSON

def write_manifest(manuscripts: list[Manuscript], output_dir: Path) -> None:
    """Écrit un fichier manifest.json décrivant le dataset compilé."""
    data = {
        "description": "Dataset HTR XIIIe siècle — CREMMA + HTRomance",
        "total_lines": sum(
            min(m.lines_total, m.cap or m.lines_total) for m in manuscripts
        ),
        "manuscripts": [
            {
                "shelfmark": m.shelfmark,
                "lang": m.lang,
                "century": m.century,
                "script": m.script,
                "lines_total": m.lines_total,
                "lines_used": min(m.lines_total, m.cap or m.lines_total),
                "corpus": m.corpus,
                "cap": m.cap,
            }
            for m in manuscripts
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Manifeste écrit dans %s", manifest_path)


# Point d'entrée principal

def build_dataset(
    output_dir: Path,
    balance: bool = False,
    af_ratio: float = 0.60,
    repos_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    """
    Pipeline complet :
    1. Applique l'équilibrage si demandé.
    2. Clone/met à jour chaque dépôt source.
    3. Copie les fichiers ALTO XML vers output_dir.
    4. Écrit le manifeste.

    Args:
        output_dir  : dossier de destination du dataset compilé.
        balance     : active le rééquilibrage 60/40.
        af_ratio    : proportion cible d'ancien français (0.60 par défaut).
        repos_dir   : dossier de cache des clones Git (défaut : output_dir/../repos).
        dry_run     : affiche le plan sans télécharger ni copier.
    """
    if repos_dir is None:
        repos_dir = output_dir.parent / "repos"

    manuscripts = list(MANUSCRIPTS)  # copie pour ne pas muter la constante

    # Statistiques brutes
    print_stats(manuscripts, "COMPLET (brut)")

    # Équilibrage
    if balance:
        manuscripts = apply_balance(manuscripts, af_ratio=af_ratio)
        print_stats(manuscripts, f"ÉQUILIBRÉ ({int(af_ratio*100)}/{int((1-af_ratio)*100)})")

    if dry_run:
        log.info("--dry-run : aucun téléchargement ni copie effectuée.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Grouper les manuscrits par dépôt pour ne cloner qu'une fois par repo
    repos: dict[str, list[Manuscript]] = {}
    for m in manuscripts:
        repos.setdefault(m.repo_url, []).append(m)

    total_xml = 0
    for repo_url, mss in repos.items():
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo_dir = repos_dir / repo_name

        ok = clone_or_pull(repo_url, repo_dir)
        if not ok:
            log.error("Échec du clonage de %s — manuscrits ignorés.", repo_url)
            continue

        for m in mss:
            log.info("  Copie : %s (%s)", m.shelfmark, m.corpus)
            n = copy_manuscript_data(m, repo_dir, output_dir)
            log.info("    → %d fichiers XML copiés", n)
            total_xml += n

    write_manifest(manuscripts, output_dir)
    log.info("Dataset prêt dans %s (%d fichiers XML au total)", output_dir, total_xml)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile le dataset HTR XIIIe siècle depuis HTR-United."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/dataset"),
        help="Dossier de destination du dataset (défaut : ./data/dataset)",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=None,
        help="Cache des clones Git (défaut : <output>/../repos)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Active le rééquilibrage 60%% AF / 40%% LAT",
    )
    parser.add_argument(
        "--af-ratio",
        type=float,
        default=0.60,
        help="Proportion cible d'ancien français (défaut : 0.60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan sans télécharger ni copier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité (défaut : 42)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Active les messages DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    build_dataset(
        output_dir=args.output,
        balance=args.balance,
        af_ratio=args.af_ratio,
        repos_dir=args.repos_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()