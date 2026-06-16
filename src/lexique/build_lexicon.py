"""Construit un lexique (mots + fréquences) à partir d'une sortie HTR.

Lexique DESCRIPTIF : il liste les mots que le modèle produit dans les data
contracts JSON (data/predictions/). Utile pour inspecter le vocabulaire de la
sortie et repérer les mots aberrants. Ce n'est PAS un lexique de référence :
pour évaluer la qualité (token ratio), il faudra un lexique bâti sur le corpus
CREMMA annoté.

Méthode inspirée du dépôt Evaluation-HTR (Marine) :
  1. extraire le texte des lignes
  2. mettre en minuscules
  3. tokeniser avec une regex qui préserve les abréviations médiévales (⁊ ꝑ ~)
  4. écrire les mots uniques avec leur fréquence

Le CSV est *enrichi* : si le fichier existe déjà, les fréquences s'additionnent,
ce qui permet d'accumuler le lexique sur plusieurs contrats.

Colonnes : word, count

Exemple :
    python src/lexique/build_lexicon.py --input "data/predictions/btv1b55000507q_f325_20260616_114625.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htr_data_contract import iter_lines, load_json  # noqa: E402


DEFAULT_OUTPUT = "data/lexique/lexicon.csv"

# Regex du dépôt Evaluation-HTR : tout sauf espaces et ponctuation courante.
# Préserve les abréviations médiévales (⁊, ꝑ, ~) à l'intérieur des mots.
TOKEN_RE = re.compile(r"[^\s.,;:!?()\[\]{}\"']+")


def resolve_inputs(patterns: list[str]) -> list[Path]:
    """Developpe les chemins et motifs glob en liste de fichiers uniques, triee."""
    paths: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        matches = glob.glob(pat) or ([pat] if Path(pat).exists() else [])
        if not matches:
            print(f"[ATTENTION] aucun fichier pour : {pat}")
        for m in sorted(matches):
            rp = str(Path(m).resolve())
            if rp not in seen:
                seen.add(rp)
                paths.append(Path(m))
    return paths


def count_words(contract: dict) -> Counter:
    """Compte les mots (minuscules) du texte de toutes les lignes du contrat."""
    counter: Counter = Counter()
    for _, _, _, line in iter_lines(contract):
        text = line.get("text", "").lower()
        counter.update(TOKEN_RE.findall(text))
    return counter


def load_existing(path: Path) -> Counter:
    """Recharge un CSV lexique existant pour cumuler les fréquences."""
    counter: Counter = Counter()
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("word", "")
            count = int(row.get("count", 0))
            if word:
                counter[word] += count
    return counter


def write_csv(counter: Counter, path: Path) -> None:
    """Écrit le lexique trié par fréquence décroissante, puis alphabétiquement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])
        for word, count in rows:
            writer.writerow([word, count])


def main() -> int:
    parser = argparse.ArgumentParser(description="Construit un lexique d'une sortie HTR")
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Un ou plusieurs contracts JSON, ou motifs glob (ex: data/predictions/*.json)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"CSV de sortie (defaut: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    files = resolve_inputs(args.input)
    if not files:
        print("[ERREUR] aucun fichier a analyser.")
        return 1

    new_counts: Counter = Counter()
    for f in files:
        new_counts.update(count_words(load_json(str(f))))

    out_path = Path(args.output)
    merged = load_existing(out_path)
    merged.update(new_counts)
    write_csv(merged, out_path)

    total = sum(new_counts.values())
    print(f"Contrats analyses : {len(files)}")
    print(f"Mots ajoutes     : {total} ({len(new_counts)} uniques)")
    print(f"Lexique cumule   : {len(merged)} mots uniques")
    print(f"CSV mis a jour   : {out_path}")
    print("Top 10 (cumule)  :")
    enc = sys.stdout.encoding or "utf-8"
    for word, count in sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        safe = word.encode(enc, errors="replace").decode(enc)
        print(f"  {count:>4}  {safe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
