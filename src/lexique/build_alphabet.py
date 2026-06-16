"""Construit l'alphabet (caractères + fréquences) à partir d'une sortie HTR.

Lit un data contract JSON (produit par app.py, dans data/predictions/), compte
les caractères du texte BRUT (sans normalisation) et écrit le résultat dans un CSV.

Le CSV est *enrichi* : si le fichier de sortie existe déjà, les nouveaux comptes
sont fusionnés avec les anciens (les fréquences s'additionnent). On peut donc
accumuler l'alphabet de plusieurs contrats au fur et à mesure.

Colonnes du CSV : char, codepoint, count
  - char      : le caractère réel (un espace reste un espace ; voir codepoint pour lever l'ambiguïté)
  - codepoint : le point de code Unicode, ex. U+0061 — indispensable pour repérer
                les caractères invisibles ou parasites (mojibake, espaces spéciaux)
  - count     : nombre d'occurrences cumulées

Exemple :
    python src/lexique/build_alphabet.py --input "data/predictions/btv1b55000507q_f325_20260616_114625.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htr_data_contract import iter_lines, load_json  # noqa: E402


DEFAULT_OUTPUT = "data/lexique/alphabet.csv"


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


def count_chars(contract: dict) -> Counter:
    """Compte les caractères du texte brut de toutes les lignes du contrat."""
    counter: Counter = Counter()
    for _, _, _, line in iter_lines(contract):
        counter.update(line.get("text", ""))
    return counter


def load_existing(path: Path) -> Counter:
    """Recharge un CSV alphabet existant pour cumuler les fréquences."""
    counter: Counter = Counter()
    if not path.exists():
        return counter
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cp = row.get("codepoint", "")
            count = int(row.get("count", 0))
            if cp.startswith("U+"):
                char = chr(int(cp[2:], 16))
                counter[char] += count
    return counter


def write_csv(counter: Counter, path: Path) -> None:
    """Écrit le compteur trié par fréquence décroissante, puis par codepoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], ord(kv[0])))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["char", "codepoint", "count"])
        for char, count in rows:
            writer.writerow([char, f"U+{ord(char):04X}", count])


def main() -> int:
    parser = argparse.ArgumentParser(description="Construit l'alphabet d'une sortie HTR")
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
        new_counts.update(count_chars(load_json(str(f))))

    out_path = Path(args.output)
    merged = load_existing(out_path)
    merged.update(new_counts)
    write_csv(merged, out_path)

    total = sum(new_counts.values())
    print(f"Contrats analyses  : {len(files)}")
    print(f"Caracteres ajoutes : {total} ({len(new_counts)} uniques)")
    print(f"Alphabet cumule    : {len(merged)} caracteres uniques")
    print(f"CSV mis a jour     : {out_path}")
    print("Top 10 (cumule)    :")
    for char, count in sorted(merged.items(), key=lambda kv: (-kv[1], ord(kv[0])))[:10]:
        display = repr(char) if char.isspace() else char
        print(f"  {display:<6} U+{ord(char):04X}  {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
