"""Day-1 NLP utilities for HTR data contracts: validation, EDA, review queue, split sealing."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


ABBREVIATION_MARKERS = {"~", "ꝑ", "ꝗ", "ꝓ", "ꝙ"}


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_contract(contract: dict, schema: dict) -> list[str]:
    errors: list[str] = []

    validator = Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(contract), key=lambda e: e.path):
        loc = "/".join(str(x) for x in err.path)
        errors.append(f"schema:{loc}: {err.message}")

    for page in contract.get("pages", []):
        for line in page.get("lines", []):
            text = line.get("text", "")
            confs = line.get("char_confidences", [])
            if isinstance(text, str) and isinstance(confs, list):
                if len(confs) != len(text):
                    errors.append(
                        f"logic:{line.get('line_id', '<unknown>')}: char_confidences length ({len(confs)}) != len(text) ({len(text)})"
                    )

    return errors


def iter_lines(contract: dict):
    doc_id = contract.get("document_id", "")
    for page in contract.get("pages", []):
        page_id = page.get("page_id", "")
        image_path = page.get("image_path", "")
        for line in page.get("lines", []):
            yield doc_id, page_id, image_path, line


def compute_eda(contract: dict) -> dict:
    confidences: list[float] = []
    lengths: list[int] = []
    needs_review_count = 0
    abbr_count = 0
    n_lines = 0

    for _, _, _, line in iter_lines(contract):
        n_lines += 1
        conf = float(line.get("confidence", 0.0))
        confidences.append(conf)

        text = line.get("text", "")
        lengths.append(len(text))

        if bool(line.get("needs_review", False)):
            needs_review_count += 1

        for c in text:
            if c in ABBREVIATION_MARKERS:
                abbr_count += 1

    if n_lines == 0:
        return {
            "n_lines": 0,
            "mean_confidence": 0.0,
            "median_line_length": 0,
            "needs_review_rate": 0.0,
            "abbr_per_line": 0.0,
            "short_line_rate_lt_10": 0.0,
        }

    short_rate = sum(1 for x in lengths if x < 10) / n_lines
    return {
        "n_lines": n_lines,
        "mean_confidence": sum(confidences) / n_lines,
        "median_line_length": statistics.median(lengths),
        "needs_review_rate": needs_review_count / n_lines,
        "abbr_per_line": abbr_count / n_lines,
        "short_line_rate_lt_10": short_rate,
    }


def split_review_buckets(
    contract: dict,
    direct_threshold: float = 0.9,
    exclude_threshold: float = 0.6,
    char_std_threshold: float = 0.2,
) -> dict[str, list[dict]]:
    buckets = {"direct": [], "review": [], "exclude": []}

    for doc_id, page_id, image_path, line in iter_lines(contract):
        confidence = float(line.get("confidence", 0.0))
        char_confidences = line.get("char_confidences", [])
        char_std = statistics.pstdev(char_confidences) if len(char_confidences) > 1 else 0.0
        force_review = bool(line.get("needs_review", False)) or char_std > char_std_threshold

        row = {
            "document_id": doc_id,
            "page_id": page_id,
            "line_id": line.get("line_id", ""),
            "image_path": image_path,
            "confidence": confidence,
            "char_std": char_std,
            "needs_review": bool(line.get("needs_review", False)),
            "text": line.get("text", ""),
        }

        if confidence < exclude_threshold:
            buckets["exclude"].append(row)
        elif force_review or confidence < direct_threshold:
            buckets["review"].append(row)
        else:
            buckets["direct"].append(row)

    buckets["review"].sort(key=lambda x: x["confidence"])
    return buckets


def export_review_csv(review_rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "page_id",
                "line_id",
                "image_path",
                "confidence",
                "char_std",
                "needs_review",
                "proposed_text",
                "corrected_text",
            ],
        )
        writer.writeheader()
        for row in review_rows:
            writer.writerow({
                "document_id": row["document_id"],
                "page_id": row["page_id"],
                "line_id": row["line_id"],
                "image_path": row["image_path"],
                "confidence": row["confidence"],
                "char_std": row["char_std"],
                "needs_review": row["needs_review"],
                "proposed_text": row["text"],
                "corrected_text": "",
            })


def stratified_split_records(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 67,
) -> dict[str, list[dict]]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    random.seed(seed)
    by_strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        key = (str(rec.get("century_estimate", "")), str(rec.get("document_type", "")))
        by_strata[key].append(rec)

    split = {"train": [], "val": [], "test": []}
    for group in by_strata.values():
        random.shuffle(group)
        n = len(group)
        n_train = max(1, int(round(n * train_ratio))) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 1

        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        split["train"].extend(group[:n_train])
        split["val"].extend(group[n_train:n_train + n_val])
        split["test"].extend(group[n_train + n_val:])

    return split


def seal_test_set(test_records: list[dict], output_json: str | Path, output_sha: str | Path) -> str:
    payload = json.dumps(test_records, ensure_ascii=False, indent=2, sort_keys=True)
    out_json = Path(output_json)
    out_sha = Path(output_sha)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_sha.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    out_sha.write_text(digest + "\n", encoding="utf-8")
    return digest
