"""Evaluate a trained HTR model against ground-truth ALTO/PageXML transcriptions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from cer_utils import cer, wer
from PIL import Image


@dataclass
class LineResult:
    doc: str
    line_id: str
    reference: str
    hypothesis: str
    cer: float
    wer: float
    correct: bool


def load_model(model_path: str | Path):
    from kraken.lib.models import TorchSeqRecognizer

    model_path = str(model_path)
    if model_path.endswith(".safetensors"):
        from kraken.models import load_safetensors

        vgsl_models = load_safetensors(model_path, tasks=["recognition"])
        return TorchSeqRecognizer(vgsl_models[0], device="cpu")

    from kraken.lib import vgsl

    net = vgsl.TorchVGSLModel.load_model(model_path)
    return TorchSeqRecognizer(net, device="cpu")


def find_pairs(dataset_dir: str | Path, xml_suffix: str = ".xml") -> list[tuple[Path, Path]]:
    """Find (xml, image) pairs in a dataset directory."""
    dataset_dir = Path(dataset_dir)
    pairs = []
    for xml_path in sorted(dataset_dir.glob(f"*{xml_suffix}")):
        if xml_path.name.endswith(".chocomufin.xml") and xml_suffix == ".xml":
            continue
        stem = xml_path.name[: -len(xml_suffix)]
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidate = dataset_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is not None:
            pairs.append((xml_path, image_path))
    return pairs


def evaluate_document(net, xml_path: Path, image_path: Path) -> list[LineResult]:
    from kraken.lib.xml import XMLPage
    from kraken import rpred

    page = XMLPage(xml_path, filetype="xml")
    segmentation = page.to_container()
    image = Image.open(image_path).convert("L")

    lines = segmentation.lines
    preds = list(rpred.rpred(net, image, segmentation))

    results = []
    for line, pred in zip(lines, preds):
        reference = (line.text or "").strip()
        hypothesis = (pred.prediction or "").strip()
        if not reference:
            continue
        line_cer = cer(reference, hypothesis)
        line_wer = wer(reference, hypothesis)
        results.append(LineResult(
            doc=xml_path.stem,
            line_id=line.id,
            reference=reference,
            hypothesis=hypothesis,
            cer=line_cer,
            wer=line_wer,
            correct=reference == hypothesis,
        ))
    return results


def evaluate_dataset(net, dataset_dir: str | Path) -> list[LineResult]:
    pairs = find_pairs(dataset_dir)
    if not pairs:
        raise ValueError(f"No XML/image pairs found in {dataset_dir}")

    all_results = []
    for xml_path, image_path in pairs:
        all_results.extend(evaluate_document(net, xml_path, image_path))
    return all_results


def summarize(results: list[LineResult]) -> dict:
    if not results:
        return {"num_lines": 0, "cer": None, "wer": None, "accuracy": None}

    total_chars = sum(max(1, len(r.reference)) for r in results)
    total_char_errors = sum(r.cer * max(1, len(r.reference)) for r in results)

    total_words = sum(max(1, len(r.reference.split())) for r in results)
    total_word_errors = sum(r.wer * max(1, len(r.reference.split())) for r in results)

    num_correct = sum(1 for r in results if r.correct)

    return {
        "num_lines": len(results),
        "cer": total_char_errors / total_chars,
        "wer": total_word_errors / total_words,
        "accuracy": num_correct / len(results),
        "mean_line_cer": sum(r.cer for r in results) / len(results),
        "mean_line_wer": sum(r.wer for r in results) / len(results),
    }


def write_csv(results: list[LineResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc", "line_id", "reference", "hypothesis", "cer", "wer", "correct"])
        for r in results:
            writer.writerow([r.doc, r.line_id, r.reference, r.hypothesis, f"{r.cer:.4f}", f"{r.wer:.4f}", r.correct])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to .safetensors or .mlmodel file")
    parser.add_argument("--dataset", required=True, help="Directory with ALTO/PageXML + image pairs")
    parser.add_argument("--csv-output", default=None, help="Optional path to write per-line results CSV")
    parser.add_argument("--json-output", default=None, help="Optional path to write summary JSON")
    args = parser.parse_args()

    net = load_model(args.model)
    results = evaluate_dataset(net, args.dataset)
    summary = summarize(results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.csv_output:
        write_csv(results, args.csv_output)
        print(f"Per-line results written to {args.csv_output}")

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
