"""
Compile les Arrow Kraken depuis les ALTO en filtrant les zones bruit.

Zones incluses : MainZone, MainZone#1, MainZone#2, MarginTextZone
Zones exclues  : MusicZone, DropCapitalZone, GraphicZone, NumberingZone,
                 RunningTitleZone, DamageZone, StampZone, DigitizationArtefactZone

Types de lignes exclus : MusicLine, DropCapitalLine, InterlinearLine

Usage :
    python src/compile_arrow.py
    python src/compile_arrow.py --inclure-marginalia non
    python src/compile_arrow.py --dry-run
    python src/compile_arrow.py --upload
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Force UTF-8 sur stdout/stderr (Windows cp1252 bloque les caractères Unicode)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from lxml import etree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RACINE = Path(__file__).parent.parent
GRAYSCALE_DIR = RACINE / "data" / "preprocessed_grayscale"
SPLITS_DIR = RACINE / "data" / "splits"
OUTPUT_DIR = RACINE / "data" / "splits" / "arrow_clean"

ZONES_INCLUSES = {"MainZone", "MainZone#1", "MainZone#2"}
ZONES_MARGINALIA = {"MarginTextZone"}
ZONES_EXCLUES = {
    "MusicZone", "DropCapitalZone", "GraphicZone", "NumberingZone",
    "RunningTitleZone", "DamageZone", "StampZone", "DigitizationArtefactZone",
    "TableZone", "TitlePageZone", "SealZone", "QuireMarksZone",
}
LIGNES_EXCLUES = {"MusicLine", "DropCapitalLine", "InterlinearLine"}

BUCKET = "htr-cremma-medieval"
AWS_PROFILE = "djamel_admin"


# ---------------------------------------------------------------------------
# Filtrage ALTO
# ---------------------------------------------------------------------------

def _id_to_label(tree: etree._ElementTree) -> dict[str, str]:
    mapping = {}
    for tag in tree.findall(".//{*}OtherTag") + tree.findall(".//{*}LayoutTag"):
        mapping[tag.get("ID", "")] = tag.get("LABEL", "")
    return mapping


def nettoyer_alto(alto_path: Path, inclure_marginalia: bool, dest: Path) -> dict:
    """Copie un ALTO en supprimant les blocs et lignes de bruit.

    Args:
        alto_path: Chemin vers le fichier ALTO source.
        inclure_marginalia: Si True, conserve les MarginTextZone.
        dest: Chemin de destination pour l'ALTO nettoyé.

    Returns:
        Dict avec les stats de nettoyage (lignes_avant, lignes_apres, exclus).
    """
    try:
        tree = etree.parse(str(alto_path))
    except Exception:
        return {"lignes_avant": 0, "lignes_apres": 0, "exclus": 0}

    id_label = _id_to_label(tree)
    zones_ok = ZONES_INCLUSES | (ZONES_MARGINALIA if inclure_marginalia else set())

    lignes_avant = len(tree.findall(".//{*}TextLine"))
    lignes_supprimees = 0

    for block in tree.findall(".//{*}TextBlock"):
        block_labels = {id_label.get(r, r) for r in block.get("TAGREFS", "").split()}

        # Supprimer le bloc entier si zone exclue ou non reconnue
        if block_labels & ZONES_EXCLUES or not (block_labels & zones_ok):
            nb = len(block.findall(".//{*}TextLine"))
            lignes_supprimees += nb
            block.getparent().remove(block)
            continue

        # Dans un bloc valide, supprimer les lignes bruit individuelles
        for line in block.findall(".//{*}TextLine"):
            line_labels = {id_label.get(r, r) for r in line.get("TAGREFS", "").split()}
            if line_labels & LIGNES_EXCLUES:
                line.getparent().remove(line)
                lignes_supprimees += 1

    lignes_apres = lignes_avant - lignes_supprimees
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dest), encoding="utf-8", xml_declaration=True)

    return {
        "lignes_avant": lignes_avant,
        "lignes_apres": lignes_apres,
        "exclus": lignes_supprimees,
    }


def filtrer_split(
    split_txt: Path,
    inclure_marginalia: bool = True,
    dry_run: bool = False,
) -> tuple[list[str], dict]:
    """Filtre un split en nettoyant chaque ALTO des zones bruit.

    Args:
        split_txt: Fichier .txt contenant les chemins vers les ALTO.
        inclure_marginalia: Inclure les MarginTextZone.
        dry_run: Si True, compte les lignes sans écrire les ALTO nettoyés.

    Returns:
        Tuple (liste des chemins ALTO nettoyés, statistiques globales).
    """
    with open(split_txt, encoding="utf-8") as f:
        chemins = [l.strip() for l in f if l.strip()]

    total_avant = 0
    total_apres = 0
    chemins_nettoyes = []

    for chemin in chemins:
        p = Path(chemin)
        if not p.exists():
            continue

        if dry_run:
            # Compter sans écrire
            try:
                tree = etree.parse(str(p))
                id_label = _id_to_label(tree)
                zones_ok = ZONES_INCLUSES | (ZONES_MARGINALIA if inclure_marginalia else set())
                for block in tree.findall(".//{*}TextBlock"):
                    block_labels = {id_label.get(r, r) for r in block.get("TAGREFS", "").split()}
                    for line in block.findall(".//{*}TextLine"):
                        total_avant += 1
                        line_labels = {id_label.get(r, r) for r in line.get("TAGREFS", "").split()}
                        if (block_labels & ZONES_EXCLUES or not (block_labels & zones_ok)
                                or line_labels & LIGNES_EXCLUES):
                            pass
                        else:
                            total_apres += 1
            except Exception:
                pass
        else:
            # Ecrire l'ALTO nettoyé à côté de l'image originale (même dossier),
            # suffixe _clean pour ne pas écraser l'original.
            dest = p.parent / (p.stem + "_clean.xml")
            stats = nettoyer_alto(p, inclure_marginalia, dest)
            total_avant += stats["lignes_avant"]
            total_apres += stats["lignes_apres"]
            if stats["lignes_apres"] > 0:
                chemins_nettoyes.append(str(dest))

    return chemins_nettoyes, {
        "fichiers": len(chemins),
        "lignes_avant": total_avant,
        "lignes_apres": total_apres,
        "exclus": total_avant - total_apres,
        "taux_exclusion": (total_avant - total_apres) / total_avant * 100 if total_avant > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Compilation Arrow
# ---------------------------------------------------------------------------

def compiler_arrow(
    chemins: list[str],
    output: Path,
    ketos_bin: str = "ketos",
) -> bool:
    """Lance ketos compile sur la liste de fichiers filtrés.

    Args:
        chemins: Liste des chemins ALTO à compiler.
        output: Chemin de sortie du fichier .arrow.
        ketos_bin: Chemin ou nom du binaire ketos.

    Returns:
        True si la compilation a réussi.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("\n".join(chemins))
        tmp_path = tmp.name

    try:
        cmd = [ketos_bin, "compile", "-f", "alto", "-F", tmp_path, "-o", str(output)]
        print(f"  Commande : {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    finally:
        os.unlink(tmp_path)


def sha256_fichier(path: Path) -> str:
    """Calcule le SHA-256 d'un fichier.

    Args:
        path: Chemin vers le fichier.

    Returns:
        Hash SHA-256 en hexadécimal.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Upload S3
# ---------------------------------------------------------------------------

def uploader_s3(local: Path, s3_key: str) -> bool:
    """Upload un fichier sur S3 via AWS CLI.

    Args:
        local: Chemin local du fichier.
        s3_key: Clé S3 de destination (sans s3://bucket/).

    Returns:
        True si l'upload a réussi.
    """
    cmd = [
        "aws", "s3", "cp", str(local),
        f"s3://{BUCKET}/{s3_key}",
        "--profile", AWS_PROFILE,
    ]
    print(f"  Upload : {local.name} → s3://{BUCKET}/{s3_key}")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Arrow filtrés depuis ALTO")
    parser.add_argument(
        "--inclure-marginalia", choices=["oui", "non"], default="oui",
        help="Inclure les lignes MarginTextZone (défaut: oui)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les stats sans compiler"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload les Arrow sur S3 après compilation"
    )
    parser.add_argument(
        "--ketos", default="ketos",
        help="Chemin vers le binaire ketos (défaut: ketos)"
    )
    args = parser.parse_args()

    inclure_marginalia = args.inclure_marginalia == "oui"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev"]:
        split_src = SPLITS_DIR / f"{split}_gray.txt"
        if not split_src.exists():
            # Fallback sur train.txt / dev.txt standard
            split_src = SPLITS_DIR / f"{split}.txt"
        if not split_src.exists():
            print(f"[SKIP] {split_src} introuvable")
            continue

        print(f"\n{'='*60}")
        print(f"Split : {split} ({split_src.name})")
        print(f"{'='*60}")

        gardes, stats = filtrer_split(split_src, inclure_marginalia, args.dry_run)

        print(f"  Total fichiers   : {stats['fichiers']}")
        print(f"  Lignes avant     : {stats['lignes_avant']}")
        print(f"  Lignes après     : {stats['lignes_apres']}")
        print(f"  Exclus (bruit)   : {stats['exclus']} ({stats['taux_exclusion']:.1f}%)")

        if args.dry_run:
            print("  [DRY-RUN] Compilation ignorée")
            continue

        if not gardes:
            print("  [ERREUR] Aucun fichier valide — compilation annulée")
            continue

        output = OUTPUT_DIR / f"{split}_clean.arrow"
        print(f"\n  Compilation → {output}")
        ok = compiler_arrow(gardes, output, ketos_bin=args.ketos)

        if ok and output.exists():
            taille = output.stat().st_size / 1024 / 1024
            sha = sha256_fichier(output)
            print(f"  Taille : {taille:.1f} MB")
            print(f"  SHA-256 : {sha}")

            # Sauvegarder le hash
            (OUTPUT_DIR / f"{split}_clean.sha256").write_text(
                f"{sha}  {output.name}\n", encoding="utf-8"
            )

            if args.upload:
                uploader_s3(output, f"splits/{split}_clean.arrow")
        else:
            print(f"  [ERREUR] Compilation échouée pour {split}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
