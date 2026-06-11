"""
train.py — Pipeline d'entraînement HTR XIIIe siècle
=====================================================
Fine-tuning du modèle CREMMA Generic (latin + ancien français) via Kraken,
avec data augmentation, split train/dev/test par manuscrit, et rapport
d'entraînement complet.

Pipeline :
    1. Chargement du dataset prétraité (ALTO XML)
    2. Split train / dev / test par manuscrit (pas aléatoire par page)
    3. Téléchargement du modèle de base CREMMA Generic
    4. Entraînement Kraken avec data augmentation (--augment)
    5. Évaluation sur le jeu de test
    6. Génération du rapport d'entraînement (JSON + texte)

Usage :
    python train.py                              # pipeline complet par défaut
    python train.py --input ./data/preprocessed  # dossier prétraité custom
    python train.py --split-only                 # générer le split sans entraîner
    python train.py --eval-only --model models/model_13c_best.mlmodel
    python train.py --dry-run                    # affiche la config sans lancer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

# URL du modèle CREMMA Generic (latin + ancien français) sur Zenodo
# DOI : 10.5281/zenodo.7234166
CREMMA_GENERIC_URL = (
    "https://zenodo.org/record/7234166/files/cremma_generic.mlmodel"
)
CREMMA_GENERIC_FILENAME = "cremma_generic.mlmodel"

# Dossiers par défaut (cohérents avec dataset.py et pre_traitement.py)
DEFAULT_INPUT   = Path("./data/preprocessed")
DEFAULT_MODELS  = Path("./models")
DEFAULT_SPLITS  = Path("./data/splits")
DEFAULT_REPORTS = Path("./reports")

# Hyperparamètres d'entraînement par défaut
DEFAULT_LEARNING_RATE  = 1e-4       # lr bas pour fine-tuning (préserve les poids)
DEFAULT_LAG            = 20         # early stopping : patience en epochs
DEFAULT_MIN_EPOCHS     = 5          # ne pas arrêter avant ce nombre d'epochs
DEFAULT_MAX_EPOCHS     = 50        # plafond de sécurité
DEFAULT_BATCH_SIZE     = 16
DEFAULT_DEVICE         = "cpu"      # "cpu" | "cuda" | "mps"

# Proportions du split (par manuscrit, pas par page)
SPLIT_TRAIN = 0.80
SPLIT_DEV   = 0.10
SPLIT_TEST  = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# 1. STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ManuscriptSplit:
    """Un manuscrit assigné à un split."""
    slug: str
    lang: str
    script: str
    n_files: int
    split: str          # "train" | "dev" | "test"


@dataclass
class SplitStats:
    """Statistiques d'un split."""
    name: str
    n_manuscripts: int
    n_files: int
    langs: dict[str, int]   # {lang: n_files}
    slugs: list[str]


@dataclass
class TrainingConfig:
    """Configuration complète de l'entraînement."""
    input_dir: str
    model_base: str
    output_dir: str
    splits_dir: str
    learning_rate: float
    lag: int
    min_epochs: int
    max_epochs: int
    batch_size: int
    device: str
    augment: bool
    resize: str             # "union" | "new" | "fail"
    seed: int
    timestamp: str


@dataclass
class EpochRecord:
    """Métriques d'une epoch."""
    epoch: int
    train_loss: float
    val_accuracy: float     # 1 - CER sur dev
    val_cer: float
    elapsed_s: float


@dataclass
class TrainingReport:
    """Rapport complet d'entraînement."""
    config: TrainingConfig
    split_stats: dict[str, SplitStats]
    epochs: list[EpochRecord]
    best_epoch: int
    best_val_cer: float
    best_model_path: str
    test_cer: float
    test_accuracy: float
    total_duration_s: float
    kraken_version: str
    completed: bool
    error: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 2. UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd: list[str], cwd: Optional[str] = None, capture: bool = False) -> tuple[int, str, str]:
    """Exécute une commande shell, retourne (returncode, stdout, stderr)."""
    log.debug("CMD: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0 and not capture:
        log.warning("Commande terminée avec code %d", result.returncode)
    return result.returncode, stdout, stderr


def get_kraken_version() -> str:
    """Retourne la version de Kraken installée."""
    rc, out, _ = run(["kraken", "--version"], capture=True)
    if rc == 0:
        return out.strip()
    rc, out, _ = run([sys.executable, "-m", "kraken", "--version"], capture=True)
    return out.strip() if rc == 0 else "inconnue"


def check_kraken() -> bool:
    """Vérifie que Kraken est disponible dans le PATH ou comme module Python."""
    rc, _, _ = run(["kraken", "--help"], capture=True)
    if rc == 0:
        return True
    rc, _, _ = run([sys.executable, "-m", "kraken", "--help"], capture=True)
    return rc == 0


def kraken_cmd(args: list[str]) -> list[str]:
    """Construit la commande kraken (binaire ou module Python)."""
    rc, _, _ = run(["kraken", "--help"], capture=True)
    if rc == 0:
        return ["kraken"] + args
    return [sys.executable, "-m", "kraken"] + args


def telecharger_modele_base(models_dir: Path, force: bool = False) -> Path:
    """
    Télécharge le modèle CREMMA Generic depuis Zenodo si absent.

    Le modèle est le point de départ du fine-tuning.
    DOI : 10.5281/zenodo.7234166

    Args:
        models_dir : dossier de destination.
        force      : re-télécharger même si le fichier existe déjà.

    Returns:
        Chemin vers le fichier .mlmodel.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / CREMMA_GENERIC_FILENAME

    if dest.exists() and not force:
        log.info("Modèle de base déjà présent : %s", dest)
        return dest

    log.info("Téléchargement du modèle CREMMA Generic depuis Zenodo ...")
    log.info("  URL : %s", CREMMA_GENERIC_URL)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  Progression : {pct:.1f}%  ({downloaded//1024} Ko)", end="", flush=True)

    try:
        urllib.request.urlretrieve(CREMMA_GENERIC_URL, dest, reporthook=_progress)
        print()  # newline après la barre de progression
        log.info("Modèle téléchargé : %s (%.1f Ko)", dest, dest.stat().st_size / 1024)
    except Exception as e:
        log.error("Échec du téléchargement : %s", e)
        log.error(
            "Téléchargez manuellement depuis https://zenodo.org/record/7234166 "
            "et placez le fichier dans %s", models_dir
        )
        raise

    return dest


# ══════════════════════════════════════════════════════════════════════════════
# 3. SPLIT TRAIN / DEV / TEST PAR MANUSCRIT
# ══════════════════════════════════════════════════════════════════════════════

def collecter_manuscrits(input_dir: Path) -> list[dict]:
    """
    Scanne le dossier prétraité et retourne la liste des manuscrits avec
    le nombre de fichiers XML disponibles.

    Structure attendue : input_dir / lang / slug / *.xml

    Returns:
        Liste de dicts {slug, lang, n_files, xml_files}.
    """
    manuscrits = []
    for lang_dir in sorted(input_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for slug_dir in sorted(lang_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            xml_files = sorted(slug_dir.glob("*.xml"))
            if not xml_files:
                continue
            manuscrits.append({
                "slug":      slug_dir.name,
                "lang":      lang,
                "n_files":   len(xml_files),
                "xml_files": xml_files,
            })
    return manuscrits


def construire_split(
    manuscrits: list[dict],
    train_ratio: float = SPLIT_TRAIN,
    dev_ratio:   float = SPLIT_DEV,
    seed:        int   = 42,
) -> dict[str, list[dict]]:
    """
    Construit le split train/dev/test en assignant des manuscrits entiers
    à chaque partition (pas de découpage par page).

    Les ratios train_ratio et dev_ratio (défauts 0.80 / 0.10) pilotent
    directement l'assignation : on remplit train tant que le total de ses
    pages ne dépasse pas train_ratio * total_pages, puis dev jusqu'à
    (train_ratio + dev_ratio) * total_pages, le reste va en test.

    Contrainte dure : au moins 1 manuscrit par langue en dev ET en test,
    quel que soit le ratio, pour garantir une évaluation multilingue.

    Le split par manuscrit (et non par page) est essentiel pour mesurer
    la capacité de généralisation du modèle à des mains inconnues.

    Args:
        manuscrits  : liste produite par collecter_manuscrits().
        train_ratio : proportion cible de pages en train (défaut 0.80).
        dev_ratio   : proportion cible de pages en dev   (défaut 0.10).
        seed        : graine pour la reproductibilité.

    Returns:
        {"train": [...], "dev": [...], "test": [...]}
    """
    random.seed(seed)

    test_ratio = round(1.0 - train_ratio - dev_ratio, 6)
    if test_ratio < 0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + dev_ratio ({dev_ratio}) > 1.0"
        )

    # Grouper par langue
    par_langue: dict[str, list[dict]] = {}
    for m in manuscrits:
        par_langue.setdefault(m["lang"], []).append(m)

    split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for lang, mss in par_langue.items():
        # Trier par taille décroissante : les grands manuscrits en train
        mss_sorted = sorted(mss, key=lambda x: x["n_files"], reverse=True)
        total_files = sum(m["n_files"] for m in mss_sorted)
        n = len(mss_sorted)

        if n < 3:
            # Pas assez de manuscrits pour un split complet :
            # train=tout sauf dernier, dev=dernier, test=vide
            log.warning(
                "Langue '%s' : seulement %d manuscrits — pas de jeu de test séparé.",
                lang, n,
            )
            for i, m in enumerate(mss_sorted):
                dest = "train" if i < n - 1 else "dev"
                split[dest].append({**m, "split": dest})
            continue

        # ── Contrainte dure : réserver 1 manuscrit pour dev et 1 pour test ──
        # On prend les plus petits (fin de la liste triée par taille desc.)
        # pour maximiser les données d'entraînement.
        reserved_test = mss_sorted[-1]
        reserved_dev  = mss_sorted[-2]
        restants      = mss_sorted[:-2]   # candidats pour train + ajustement

        # ── Remplissage piloté par les ratios ─────────────────────────────
        # Calculer les cibles en nombre de fichiers
        cible_train = train_ratio * total_files
        cible_dev   = dev_ratio   * total_files

        train_mss: list[dict] = []
        dev_mss:   list[dict] = []
        train_acc  = 0
        dev_acc    = 0

        for m in restants:
            if train_acc < cible_train:
                train_mss.append(m)
                train_acc += m["n_files"]
            else:
                dev_mss.append(m)
                dev_acc += m["n_files"]

        # Ajouter les manuscrits réservés
        dev_mss.append(reserved_dev)
        dev_acc += reserved_dev["n_files"]
        test_mss = [reserved_test]

        # ── Ajustement si dev dépasse largement sa cible ──────────────────
        # (peut arriver si un grand manuscrit des "restants" a basculé en dev)
        while dev_acc > cible_dev * 1.5 and len(dev_mss) > 1:
            # Déplacer le plus grand manuscrit de dev (hors réservé) vers train
            # Le réservé est toujours le dernier élément ajouté
            idx_max = max(range(len(dev_mss) - 1), key=lambda i: dev_mss[i]["n_files"])
            m_move = dev_mss.pop(idx_max)
            train_mss.append(m_move)
            train_acc += m_move["n_files"]
            dev_acc   -= m_move["n_files"]

        # ── Enregistrer dans le split global ──────────────────────────────
        for m in train_mss:
            split["train"].append({**m, "split": "train"})
        for m in dev_mss:
            split["dev"].append({**m, "split": "dev"})
        for m in test_mss:
            split["test"].append({**m, "split": "test"})

        # Log des ratios effectifs
        eff_train = train_acc / total_files * 100
        eff_dev   = dev_acc   / total_files * 100
        eff_test  = reserved_test["n_files"] / total_files * 100
        log.info(
            "  [%s] train=%.1f%%  dev=%.1f%%  test=%.1f%%  "
            "(cibles: %.0f/%.0f/%.0f)",
            lang, eff_train, eff_dev, eff_test,
            train_ratio*100, dev_ratio*100, test_ratio*100,
        )

    return split


def stats_split(split: dict[str, list[dict]]) -> dict[str, SplitStats]:
    """Calcule les statistiques de chaque partition."""
    stats = {}
    for name, mss in split.items():
        langs: dict[str, int] = {}
        for m in mss:
            langs[m["lang"]] = langs.get(m["lang"], 0) + m["n_files"]
        stats[name] = SplitStats(
            name=name,
            n_manuscripts=len(mss),
            n_files=sum(m["n_files"] for m in mss),
            langs=langs,
            slugs=[m["slug"] for m in mss],
        )
    return stats


def afficher_split(stats: dict[str, SplitStats]) -> None:
    """Affiche un résumé lisible du split."""
    total = sum(s.n_files for s in stats.values())
    print(f"\n{'='*60}")
    print("  SPLIT TRAIN / DEV / TEST")
    print(f"{'='*60}")
    for name, s in stats.items():
        pct = s.n_files / total * 100 if total else 0
        langs_str = "  ".join(f"{l}={n}" for l, n in sorted(s.langs.items()))
        print(f"  {name:<6} : {s.n_manuscripts:>3} mss  {s.n_files:>5} fichiers  "
              f"({pct:.1f}%)  [{langs_str}]")
        for slug in s.slugs:
            print(f"           - {slug}")
    print(f"{'='*60}")
    print(f"  TOTAL  : {total} fichiers XML\n")


def sauvegarder_split(split, splits_dir):  # alias conservé pour compatibilité
    return sauvegarder_split_verrouille(split, splits_dir)



SPLIT_LOCK_FILENAME = "split_lock.json"


def _hash_split(split: dict[str, list[dict]]) -> str:
    """Calcule un hash SHA-256 déterministe du split (slugs + ordre)."""
    contenu = json.dumps(
        {k: sorted(m["slug"] for m in mss) for k, mss in split.items()},
        sort_keys=True,
    )
    return hashlib.sha256(contenu.encode()).hexdigest()[:16]


def sauvegarder_split_verrouille(
    split: dict[str, list[dict]],
    splits_dir: Path,
) -> dict[str, Path]:
    """
    Écrit les fichiers .txt pour ketos train ET un split_lock.json.

    Le lock contient la date, le hash et les chemins XML complets de chaque
    partition. Il fait autorité pour tous les relancages : les pages de test
    ne seront JAMAIS vues pendant l'entraînement.

    Returns:
        {"train": path_train.txt, "dev": path_dev.txt, "test": path_test.txt}
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    lock_data: dict = {
        "created_at": datetime.now().isoformat(),
        "hash": _hash_split(split),
        "partitions": {},
    }

    for name, mss in split.items():
        txt_path = splits_dir / f"{name}.txt"
        chemins: list[str] = []
        with open(txt_path, "w", encoding="utf-8") as f:
            for m in mss:
                for xml_file in m["xml_files"]:
                    chemin = str(xml_file.resolve())
                    f.write(chemin + "\n")
                    chemins.append(chemin)
        n_files = sum(m["n_files"] for m in mss)
        log.info("Split %-6s : %d fichiers → %s", name, n_files, txt_path)
        paths[name] = txt_path
        lock_data["partitions"][name] = {
            "slugs":     [m["slug"] for m in mss],
            "n_files":   n_files,
            "xml_files": chemins,
        }

    lock_path = splits_dir / SPLIT_LOCK_FILENAME
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, ensure_ascii=False, indent=2)

    log.info("Split verrouillé → %s  (hash=%s)", lock_path, lock_data["hash"])
    log.warning(
        "Ne PAS modifier %s manuellement — ce fichier garantit "
        "que le jeu de test ne sera jamais vu à l'entraînement.",
        SPLIT_LOCK_FILENAME,
    )
    return paths


def charger_split_verrouille(splits_dir: Path) -> Optional[dict[str, Path]]:
    """
    Charge un split verrouillé existant et vérifie son intégrité.

    - Lock présent et intact  → recrée les .txt et retourne les chemins.
    - Lock absent             → retourne None (pipeline créera un nouveau split).
    - Lock corrompu           → lève RuntimeError.

    Args:
        splits_dir : dossier contenant split_lock.json.

    Returns:
        {"train": path, "dev": path, "test": path} ou None.
    """
    lock_path = splits_dir / SPLIT_LOCK_FILENAME
    if not lock_path.exists():
        return None

    with open(lock_path, encoding="utf-8") as f:
        lock = json.load(f)

    for key in ("hash", "created_at", "partitions"):
        if key not in lock:
            raise RuntimeError(
                f"split_lock.json corrompu : clé '{key}' manquante. "
                "Supprimer le lock et relancer avec --reset-split."
            )

    log.info(
        "Split verrouillé trouvé (créé le %s, hash=%s) — chargement.",
        lock["created_at"][:19], lock["hash"],
    )

    paths: dict[str, Path] = {}
    for name, data in lock["partitions"].items():
        txt_path = splits_dir / f"{name}.txt"
        chemins: list[str] = data["xml_files"]

        manquants = [c for c in chemins if not Path(c).exists()]
        if manquants:
            log.warning(
                "%d fichier(s) du split '%s' introuvables (données déplacées ?). "
                "Exemples : %s",
                len(manquants), name, manquants[:3],
            )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chemins) + "\n")

        log.info(
            "Split %-6s rechargé : %d fichiers (%d mss) → %s",
            name, data["n_files"], len(data["slugs"]), txt_path,
        )
        paths[name] = txt_path

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT KRAKEN (ketos train)
# ══════════════════════════════════════════════════════════════════════════════

def construire_commande_train(
    config: TrainingConfig,
    split_paths: dict[str, Path],
    model_base_path: Path,
) -> list[str]:
    """
    Construit la commande ketos train complète.

    Paramètres clés :
        -f alto        : format ALTO XML (notre format source)
        --resize union : intègre les nouveaux caractères latins (ligatures MUFI)
                         absents du modèle de base sans tronquer le vocabulaire
        --augment      : data augmentation (rotations, distorsions élastiques,
                         variations de contraste, bruit gaussien)
        -r lr          : learning rate bas pour fine-tuning (préserve les poids)
        --lag N        : early stopping si pas d'amélioration après N epochs
        -t train.txt   : fichier listant les ALTO XML d'entraînement
        -e dev.txt     : fichier de validation (suivi du CER pendant l'entraînement)

    Returns:
        Liste de tokens constituant la commande shell.
    """
    base = kraken_cmd([
        "-d", config.device,
        "train",
        "-f", "alto",
        "--resize", config.resize,
        "-i", str(model_base_path),
        "-o", str(Path(config.output_dir) / "model_13c"),
        "-r", str(config.learning_rate),
        "--lag", str(config.lag),
        "--min-epochs", str(config.min_epochs),
        "--max-epochs", str(config.max_epochs),
        "-t", str(split_paths["train"]),
        "-e", str(split_paths["dev"]),
    ])

    if config.augment:
        base.append("--augment")

    return base


def parser_progression_kraken(ligne: str) -> Optional[EpochRecord]:
    """
    Tente de parser une ligne de sortie Kraken pour extraire les métriques
    d'une epoch.

    Kraken affiche typiquement :
        Epoch 5/100  loss: 0.1234  val_acc: 0.9512  [00:42]
    ou (version récente) :
        stage 1/1 batch 16/16  0%|... loss 0.1234

    Returns:
        EpochRecord si la ligne contient des métriques, None sinon.
    """
    import re

    # Pattern principal : Epoch N  loss: X  val_acc: Y
    m = re.search(
        r"[Ee]poch[:\s]+(\d+).*?loss[:\s]+([\d.]+).*?val_acc[:\s]+([\d.]+)",
        ligne
    )
    if m:
        epoch = int(m.group(1))
        loss  = float(m.group(2))
        val   = float(m.group(3))
        return EpochRecord(
            epoch=epoch,
            train_loss=loss,
            val_accuracy=val,
            val_cer=round(1.0 - val, 4),
            elapsed_s=0.0,
        )

    # Pattern alternatif : accuracy seule
    m2 = re.search(r"val[_\s]acc[:\s]+([\d.]+)", ligne)
    m3 = re.search(r"[Ee]poch[:\s]+(\d+)", ligne)
    if m2 and m3:
        val = float(m2.group(1))
        return EpochRecord(
            epoch=int(m3.group(1)),
            train_loss=0.0,
            val_accuracy=val,
            val_cer=round(1.0 - val, 4),
            elapsed_s=0.0,
        )

    return None


def lancer_entrainement(
    config: TrainingConfig,
    split_paths: dict[str, Path],
    model_base_path: Path,
    dry_run: bool = False,
) -> list[EpochRecord]:
    """
    Lance l'entraînement Kraken et collecte les métriques par epoch.

    Affiche la progression en temps réel et parse les lignes de sortie
    pour alimenter le rapport.

    Args:
        config          : configuration complète.
        split_paths     : {"train": path, "dev": path, "test": path}.
        model_base_path : chemin vers le .mlmodel de base.
        dry_run         : affiche la commande sans l'exécuter.

    Returns:
        Liste des EpochRecord collectés pendant l'entraînement.
    """
    cmd = construire_commande_train(config, split_paths, model_base_path)

    log.info("Commande d'entraînement :")
    log.info("  %s", " \\\n    ".join(cmd))

    if dry_run:
        log.info("--dry-run : entraînement non lancé.")
        return []

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    epochs: list[EpochRecord] = []
    t_start = time.perf_counter()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_path = Path(config.output_dir) / "train.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        for ligne in process.stdout:
            print(ligne, end="", flush=True)
            log_file.write(ligne)

            record = parser_progression_kraken(ligne)
            if record:
                record.elapsed_s = time.perf_counter() - t_start
                epochs.append(record)

    process.wait()
    if process.returncode != 0:
        log.error("ketos train terminé avec code %d — voir %s", process.returncode, log_path)

    log.info("Log d'entraînement sauvegardé dans %s", log_path)
    return epochs


# ══════════════════════════════════════════════════════════════════════════════
# 5. ÉVALUATION SUR LE JEU DE TEST
# ══════════════════════════════════════════════════════════════════════════════

def trouver_meilleur_modele(models_dir: Path, prefix: str = "model_13c") -> Optional[Path]:
    """
    Trouve le meilleur modèle sauvegardé par Kraken dans models_dir.

    Kraken nomme les checkpoints model_13c_best.mlmodel ou
    model_13c_N.mlmodel (N = numéro d'epoch).

    Returns:
        Chemin vers le meilleur modèle, ou None si aucun trouvé.
    """
    # Chercher d'abord le fichier _best explicitement nommé
    best = models_dir / f"{prefix}_best.mlmodel"
    if best.exists():
        return best

    # Sinon, le modèle avec le numéro d'epoch le plus élevé
    candidats = sorted(models_dir.glob(f"{prefix}_*.mlmodel"))
    if candidats:
        return candidats[-1]

    # Fallback : n'importe quel .mlmodel dans le dossier
    tous = sorted(models_dir.glob("*.mlmodel"))
    return tous[-1] if tous else None


def evaluer_modele(
    model_path: Path,
    test_txt: Path,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Évalue un modèle Kraken sur le jeu de test via ketos test.

    Args:
        model_path : chemin vers le .mlmodel à évaluer.
        test_txt   : fichier texte listant les ALTO XML de test.
        device     : "cpu" | "cuda" | "mps".

    Returns:
        (cer, accuracy) — CER et accuracy moyens sur le jeu de test.
    """
    cmd = kraken_cmd([
        "-d", device,
        "test",
        "-f", "alto",
        "-m", str(model_path),
        "-e", str(test_txt),
    ])

    log.info("Évaluation sur le jeu de test : %s", model_path.name)
    rc, stdout, stderr = run(cmd, capture=True)

    output = stdout + stderr
    log.debug("Sortie ketos test :\n%s", output)

    # Parser le CER depuis la sortie
    import re
    cer = None
    for pattern in [
        r"[Cc][Ee][Rr][:\s]+([\d.]+)",
        r"character error rate[:\s]+([\d.]+)",
        r"avg\. CER[:\s]+([\d.]+)",
    ]:
        m = re.search(pattern, output)
        if m:
            cer = float(m.group(1))
            # Normaliser : Kraken peut retourner 0.05 ou 5.0 selon la version
            if cer > 1.0:
                cer = cer / 100.0
            break

    if cer is None:
        log.warning("CER non trouvé dans la sortie de ketos test — CER estimé à 1.0")
        cer = 1.0

    accuracy = round(1.0 - cer, 4)
    log.info("  CER test    : %.2f %%", cer * 100)
    log.info("  Accuracy    : %.2f %%", accuracy * 100)
    return round(cer, 4), accuracy


# ══════════════════════════════════════════════════════════════════════════════
# 6. RAPPORT D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def generer_rapport(
    rapport: TrainingReport,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """
    Génère deux fichiers de rapport :
        - rapport_YYYYMMDD_HHMMSS.json  : données brutes complètes
        - rapport_YYYYMMDD_HHMMSS.txt   : résumé lisible

    Args:
        rapport     : TrainingReport complet.
        reports_dir : dossier de destination.

    Returns:
        (json_path, txt_path)
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = rapport.config.timestamp

    # ── Rapport JSON ──────────────────────────────────────────────────────────
    json_path = reports_dir / f"rapport_{ts}.json"

    def _serialiser(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Non sérialisable : {type(obj)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(rapport), f, ensure_ascii=False, indent=2)

    # ── Rapport texte ─────────────────────────────────────────────────────────
    txt_path = reports_dir / f"rapport_{ts}.txt"
    cfg = rapport.config

    # Courbe d'apprentissage simplifiée
    courbe = ""
    if rapport.epochs:
        n = len(rapport.epochs)
        # Afficher au max 20 points répartis uniformément
        step = max(1, n // 20)
        courbe_lines = []
        for rec in rapport.epochs[::step]:
            bar_len = int(rec.val_accuracy * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            courbe_lines.append(
                f"  Epoch {rec.epoch:>4}  loss={rec.train_loss:.4f}  "
                f"CER={rec.val_cer*100:>5.2f}%  [{bar}]  {rec.val_accuracy*100:.2f}%"
            )
        courbe = "\n".join(courbe_lines)

    # Statistiques du split
    split_lines = []
    for name, stats in rapport.split_stats.items():
        langs_str = "  ".join(f"{l}={n}" for l, n in sorted(stats.langs.items()))
        split_lines.append(
            f"  {name:<6}  {stats.n_manuscripts:>3} mss  "
            f"{stats.n_files:>5} fichiers  [{langs_str}]"
        )
        for slug in stats.slugs:
            split_lines.append(f"           - {slug}")

    # Durée formatée
    def fmt_duree(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h}h {m:02d}m {sec:02d}s"

    txt = f"""
╔══════════════════════════════════════════════════════════════════╗
║          RAPPORT D'ENTRAÎNEMENT HTR — XIIIe siècle              ║
╚══════════════════════════════════════════════════════════════════╝

Date                : {ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}
Kraken              : {rapport.kraken_version}
Statut              : {"✓ Terminé" if rapport.completed else "✗ Interrompu — " + rapport.error}

──────────────────────────────────────────────────────────────────
CONFIGURATION
──────────────────────────────────────────────────────────────────
  Dossier input     : {cfg.input_dir}
  Modèle de base    : {cfg.model_base}
  Dossier output    : {cfg.output_dir}
  Device            : {cfg.device}
  Learning rate     : {cfg.learning_rate}
  Lag (early stop)  : {cfg.lag} epochs
  Min / Max epochs  : {cfg.min_epochs} / {cfg.max_epochs}
  Batch size        : {cfg.batch_size}
  Data augmentation : {"activée" if cfg.augment else "désactivée"}
  Resize mode       : {cfg.resize}
  Graine aléatoire  : {cfg.seed}

──────────────────────────────────────────────────────────────────
SPLIT TRAIN / DEV / TEST
──────────────────────────────────────────────────────────────────
{chr(10).join(split_lines)}

──────────────────────────────────────────────────────────────────
RÉSULTATS
──────────────────────────────────────────────────────────────────
  Meilleure epoch   : {rapport.best_epoch}
  Meilleur CER dev  : {rapport.best_val_cer*100:.2f} %
  Meilleur modèle   : {rapport.best_model_path}

  CER test          : {rapport.test_cer*100:.2f} %
  Accuracy test     : {rapport.test_accuracy*100:.2f} %

  Durée totale      : {fmt_duree(rapport.total_duration_s)}

──────────────────────────────────────────────────────────────────
COURBE D'APPRENTISSAGE
──────────────────────────────────────────────────────────────────
{courbe if courbe else "  (aucune epoch enregistrée)"}

──────────────────────────────────────────────────────────────────
INTERPRÉTATION
──────────────────────────────────────────────────────────────────"""

    # Interprétation automatique des résultats
    cer_pct = rapport.test_cer * 100
    if cer_pct < 3.0:
        interpretation = (
        f"  ✓ Excellent : CER={cer_pct:.2f}% — objectif cible atteint (< 3%).\n"
        "  Le modèle généralise très bien aux mains inconnues du jeu de test."
        )
    elif cer_pct < 5.0:
        interpretation = (
        f"  ✓ Bon : CER={cer_pct:.2f}% — objectif minimal atteint (< 5%).\n"
        "  Envisager un second cycle de fine-tuning à lr=5e-5 pour descendre\n"
        "  sous la barre des 3%."
        )
    elif cer_pct < 10.0:
        interpretation = (
        f"  △ Acceptable : CER={cer_pct:.2f}% — en dessous de l'objectif.\n"
        "  Pistes : (1) augmenter le dataset latin, (2) vérifier la qualité\n"
        "  du prétraitement, (3) fine-tuner sur les seuls manuscrits latins."
        )
    else:
        interpretation = (
        f"  ✗ Insuffisant : CER={cer_pct:.2f}% — loin de l'objectif.\n"
        "  Vérifier : (1) cohérence des conventions de transcription,\n"
        "  (2) qualité des images prétraitées, (3) compatibilité des\n"
        "  fichiers ALTO XML avec la version de Kraken utilisée."
        )

    txt += "\n" + interpretation + "\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    log.info("Rapport JSON : %s", json_path)
    log.info("Rapport TXT  : %s", txt_path)
    return json_path, txt_path


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def pipeline(
    input_dir:     Path = DEFAULT_INPUT,
    models_dir:    Path = DEFAULT_MODELS,
    splits_dir:    Path = DEFAULT_SPLITS,
    reports_dir:   Path = DEFAULT_REPORTS,
    model_base:    Optional[Path] = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lag:           int   = DEFAULT_LAG,
    min_epochs:    int   = DEFAULT_MIN_EPOCHS,
    max_epochs:    int   = DEFAULT_MAX_EPOCHS,
    batch_size:    int   = DEFAULT_BATCH_SIZE,
    device:        str   = DEFAULT_DEVICE,
    augment:       bool  = True,
    resize:        str   = "union",
    seed:          int   = 42,
    split_only:    bool  = False,
    eval_only:     bool  = False,
    eval_model:    Optional[Path] = None,
    dry_run:       bool  = False,
    force_download: bool = False,
    reset_split:    bool  = False,
) -> TrainingReport:
    """
    Pipeline d'entraînement complet :
        1. Vérification des prérequis (Kraken)
        2. Téléchargement du modèle de base CREMMA Generic
        3. Collecte et split des données
        4. Entraînement avec data augmentation
        5. Évaluation sur le jeu de test
        6. Génération du rapport

    Args :
        input_dir      : dossier contenant les données prétraitées (ALTO XML).
        models_dir     : dossier de sauvegarde des modèles.
        splits_dir     : dossier pour les fichiers train/dev/test.txt.
        reports_dir    : dossier pour les rapports d'entraînement.
        model_base     : modèle de base (téléchargé si None).
        learning_rate  : lr du fine-tuning (défaut : 1e-4).
        lag            : patience early stopping en epochs (défaut : 20).
        min_epochs     : nombre minimum d'epochs avant early stopping.
        max_epochs     : plafond d'epochs.
        batch_size     : taille des batchs.
        device         : "cpu" | "cuda" | "mps".
        augment        : activer la data augmentation Kraken.
        resize         : stratégie de resize du vocabulaire ("union" recommandé).
        seed           : graine aléatoire.
        split_only     : s'arrêter après la génération du split.
        eval_only      : évaluer uniquement (skip entraînement).
        eval_model     : modèle à évaluer en mode eval_only.
        dry_run        : afficher la config sans rien exécuter.
        force_download : re-télécharger le modèle de base même s'il existe.

    Returns:
        TrainingReport complet.
    """
    t_total = time.perf_counter()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    error = ""
    epochs_records: list[EpochRecord] = []
    test_cer = 1.0
    test_accuracy = 0.0
    best_epoch = 0
    best_val_cer = 1.0
    best_model_path = ""
    completed = False

    # ── Prérequis ─────────────────────────────────────────────────────────────
    log.info("Vérification de Kraken ...")
    if not check_kraken() and not dry_run:
        log.error(
            "Kraken introuvable. Installer avec : pip install kraken\n"
            "Documentation : https://kraken.re"
        )
        sys.exit(1)
    kraken_version = get_kraken_version()
    log.info("Kraken : %s", kraken_version)

    # ── Modèle de base ────────────────────────────────────────────────────────
    if model_base is None:
        if not dry_run:
            model_base = telecharger_modele_base(models_dir, force=force_download)
        else:
            model_base = models_dir / CREMMA_GENERIC_FILENAME
            log.info("(dry-run) Modèle de base : %s", model_base)
    else:
        if not model_base.exists() and not dry_run:
            log.error("Modèle de base introuvable : %s", model_base)
            sys.exit(1)
        log.info("Modèle de base fourni : %s", model_base)

    # ── Collecter les données ─────────────────────────────────────────────────
    log.info("Collecte des données dans %s ...", input_dir)
    if not input_dir.exists() and not dry_run:
        log.error(
            "Dossier d'entrée introuvable : %s\n"
            "Lancer d'abord : python pre_traitement.py", input_dir
        )
        sys.exit(1)

    manuscrits = collecter_manuscrits(input_dir) if not dry_run else []
    if not manuscrits and not dry_run:
        log.error("Aucun manuscrit trouvé dans %s", input_dir)
        sys.exit(1)

    total_files = sum(m["n_files"] for m in manuscrits)
    log.info(
        "  %d manuscrits trouvés, %d fichiers XML au total",
        len(manuscrits), total_files
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    # Charger le split verrouillé s'il existe — sinon en créer un nouveau.
    # Une fois créé, le lock est définitif : aucun relancement ne peut
    # réassigner des pages de test vers le train, sauf --reset-split explicite.
    split_paths: dict[str, Path] = {}
    stats: dict[str, SplitStats] = {}

    if not dry_run:
        if not reset_split:
            existing = charger_split_verrouille(splits_dir)
        else:
            existing = None
            lock_path = splits_dir / SPLIT_LOCK_FILENAME
            if lock_path.exists():
                lock_path.unlink()
                log.warning("--reset-split : ancien lock supprimé. Nouveau split créé.")

        if existing is not None:
            # Lock trouvé : utiliser le split existant, ignorer la graine
            split_paths = existing
            lock_data = json.load(open(splits_dir / SPLIT_LOCK_FILENAME, encoding="utf-8"))
            stats = {
                name: SplitStats(
                    name=name,
                    n_manuscripts=len(data["slugs"]),
                    n_files=data["n_files"],
                    langs={},
                    slugs=data["slugs"],
                )
                for name, data in lock_data["partitions"].items()
            }
            afficher_split(stats)
        else:
            # Premier lancement : construire et verrouiller
            split = construire_split(manuscrits, seed=seed) if manuscrits else {}
            stats = stats_split(split)
            afficher_split(stats)
            split_paths = sauvegarder_split_verrouille(split, splits_dir)
    else:
        # dry-run : simuler sans écrire
        split = construire_split(manuscrits, seed=seed) if manuscrits else {}
        stats = stats_split(split)
        afficher_split(stats)
        split_paths = {
            "train": splits_dir / "train.txt",
            "dev":   splits_dir / "dev.txt",
            "test":  splits_dir / "test.txt",
        }

    if split_only:
        log.info("--split-only : pipeline arrêtée après le split.")
        completed = True

    # ── Configuration ─────────────────────────────────────────────────────────
    config = TrainingConfig(
        input_dir=str(input_dir),
        model_base=str(model_base),
        output_dir=str(models_dir),
        splits_dir=str(splits_dir),
        learning_rate=learning_rate,
        lag=lag,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        augment=augment,
        resize=resize,
        seed=seed,
        timestamp=ts,
    )

    log.info("\nConfiguration d'entraînement :")
    log.info("  Learning rate  : %s", learning_rate)
    log.info("  Early stopping : lag=%d  min_epochs=%d  max_epochs=%d",
             lag, min_epochs, max_epochs)
    log.info("  Data augment.  : %s", "activée" if augment else "désactivée")
    log.info("  Resize mode    : %s  (union = intègre nouveaux caractères latins)", resize)
    log.info("  Device         : %s", device)

    if not split_only:
        if eval_only:
            # ── Mode évaluation seule ─────────────────────────────────────────
            model_to_eval = eval_model or trouver_meilleur_modele(models_dir)
            if model_to_eval is None:
                log.error("Aucun modèle trouvé pour l'évaluation dans %s", models_dir)
                sys.exit(1)
            log.info("Évaluation du modèle : %s", model_to_eval)
            if not dry_run:
                test_cer, test_accuracy = evaluer_modele(
                    model_to_eval, split_paths["test"], device=device
                )
            best_model_path = str(model_to_eval)
            completed = True

        else:
            # ── Entraînement ──────────────────────────────────────────────────
            try:
                epochs_records = lancer_entrainement(
                    config, split_paths, model_base, dry_run=dry_run
                )

                # Meilleure epoch (CER dev minimal)
                if epochs_records:
                    best = min(epochs_records, key=lambda r: r.val_cer)
                    best_epoch = best.epoch
                    best_val_cer = best.val_cer
                    log.info(
                        "Meilleure epoch : %d  (CER dev = %.2f%%)",
                        best_epoch, best_val_cer * 100
                    )

                # Trouver le meilleur modèle sauvegardé
                best_model = trouver_meilleur_modele(models_dir)
                if best_model:
                    best_model_path = str(best_model)
                    log.info("Meilleur modèle : %s", best_model)

                    # Évaluation finale sur le test
                    if not dry_run:
                        test_cer, test_accuracy = evaluer_modele(
                            best_model, split_paths["test"], device=device
                        )
                else:
                    log.warning("Aucun modèle trouvé après entraînement.")

                completed = True

            except KeyboardInterrupt:
                error = "Interrompu par l'utilisateur (KeyboardInterrupt)"
                log.warning(error)
            except Exception as e:
                error = str(e)
                log.error("Erreur pendant l'entraînement : %s", e)

    # ── Rapport ───────────────────────────────────────────────────────────────
    rapport = TrainingReport(
        config=config,
        split_stats={k: v for k, v in stats.items()},
        epochs=epochs_records,
        best_epoch=best_epoch,
        best_val_cer=best_val_cer,
        best_model_path=best_model_path,
        test_cer=test_cer,
        test_accuracy=test_accuracy,
        total_duration_s=time.perf_counter() - t_total,
        kraken_version=kraken_version,
        completed=completed,
        error=error,
    )

    if not dry_run:
        json_path, txt_path = generer_rapport(rapport, reports_dir)
        print(open(txt_path).read())

    return rapport


# ══════════════════════════════════════════════════════════════════════════════
# 8. INTERFACE EN LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement HTR XIIIe siècle — fine-tuning Kraken.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Chemins ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Dossier des données prétraitées (ALTO XML + images)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS,
        help="Dossier de sauvegarde des modèles",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Dossier pour les fichiers train/dev/test.txt",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS,
        help="Dossier pour les rapports d'entraînement",
    )
    parser.add_argument(
        "--model-base",
        type=Path,
        default=None,
        help="Modèle de base (téléchargé automatiquement si absent)",
    )

    # ── Hyperparamètres ───────────────────────────────────────────────────────
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        dest="learning_rate",
        help="Learning rate du fine-tuning",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=DEFAULT_LAG,
        help="Patience early stopping (epochs sans amélioration)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help="Nombre minimum d'epochs avant early stopping",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Plafond d'epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Taille des batchs",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=DEFAULT_DEVICE,
        help="Device d'entraînement",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Désactiver la data augmentation",
    )
    parser.add_argument(
        "--resize",
        choices=["union", "new", "fail"],
        default="union",
        help=(
            "Stratégie de resize du vocabulaire de sortie. "
            "'union' : intègre les nouveaux caractères latins (recommandé)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité",
    )

    # ── Modes ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Générer le split train/dev/test sans entraîner",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Évaluer uniquement (skip entraînement)",
    )
    parser.add_argument(
        "--eval-model",
        type=Path,
        default=None,
        help="Modèle .mlmodel à évaluer (avec --eval-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher la configuration sans rien exécuter",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-télécharger le modèle de base même s'il existe",
    )
    parser.add_argument(
        "--reset-split",
        action="store_true",
        help=(
            "Supprime le split_lock.json et régénère un nouveau split. "
            "ATTENTION : cela peut faire fuiter des pages de test dans le train "
            "si les résultats passés ne sont plus comparables."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Messages DEBUG",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline(
        input_dir=args.input,
        models_dir=args.models_dir,
        splits_dir=args.splits_dir,
        reports_dir=args.reports_dir,
        model_base=args.model_base,
        learning_rate=args.learning_rate,
        lag=args.lag,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        augment=not args.no_augment,
        resize=args.resize,
        seed=args.seed,
        split_only=args.split_only,
        eval_only=args.eval_only,
        eval_model=args.eval_model,
        dry_run=args.dry_run,
        force_download=args.force_download,
        reset_split=args.reset_split,
    )


if __name__ == "__main__":
    main()