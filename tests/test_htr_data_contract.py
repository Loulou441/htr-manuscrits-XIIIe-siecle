"""Tests for HTR data contract validation and review buckets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from htr_data_contract import compute_eda, split_review_buckets, validate_contract


SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["document_id", "metadata", "pages"],
    "properties": {
        "document_id": {"type": "string"},
        "metadata": {"type": "object"},
        "pages": {"type": "array"},
    },
}


def _sample_contract():
    return {
        "document_id": "doc1",
        "metadata": {
            "source": "x",
            "century_estimate": "XIII",
            "document_type": "roman",
            "sha256": "abc123",
        },
        "pages": [
            {
                "page_id": "p1",
                "image_path": "p1.tif",
                "lines": [
                    {
                        "line_id": "l1",
                        "text": "d~e",
                        "confidence": 0.95,
                        "char_confidences": [0.95, 0.96, 0.95],
                        "needs_review": False,
                        "polygon": [[0, 0], [10, 0], [10, 1], [0, 1]],
                    },
                    {
                        "line_id": "l2",
                        "text": "abc",
                        "confidence": 0.65,
                        "char_confidences": [0.95, 0.2, 0.95],
                        "needs_review": False,
                        "polygon": [[0, 0], [3, 0], [3, 1], [0, 1]],
                    },
                ],
            }
        ],
    }


def test_validate_contract_ok():
    errors = validate_contract(_sample_contract(), SIMPLE_SCHEMA)
    assert errors == []


def test_validate_contract_rejects_missing_polygon():
    contract = _sample_contract()
    del contract["pages"][0]["lines"][0]["polygon"]
    errors = validate_contract(contract, SIMPLE_SCHEMA)
    assert any("missing polygon" in err for err in errors)


def test_eda_outputs_metrics():
    metrics = compute_eda(_sample_contract())
    assert metrics["n_lines"] == 2
    assert metrics["median_line_length"] == 3


def test_review_bucket_with_stddev_rule():
    buckets = split_review_buckets(_sample_contract(), direct_threshold=0.9, exclude_threshold=0.6, char_std_threshold=0.2)
    assert len(buckets["review"]) == 1
    assert buckets["review"][0]["line_id"] == "l2"
