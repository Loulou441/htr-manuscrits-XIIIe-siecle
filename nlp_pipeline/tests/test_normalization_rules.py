"""Tests for NLP rule-based normalization."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from normalization_rules import (
    MedievalFrenchNormalizer,
    NormalizerConfig,
    detect_normalization_candidates,
    find_lexical_errors,
)


def test_abbreviation_expansion():
    normalizer = MedievalFrenchNormalizer({"q~": "que", "d~e": "dame"})
    assert normalizer.normalize("q~ d~e") == "que dame"


def test_tilde_expansion():
    normalizer = MedievalFrenchNormalizer({})
    assert normalizer.normalize("a~ e~ o~") == "an en om"


def test_uv_rule_toggle():
    cfg = NormalizerConfig(normalize_uv=False)
    normalizer = MedievalFrenchNormalizer({}, cfg)
    assert normalizer.normalize("uero") == "uero"


def test_uv_rule_keeps_qu_gu_and_ui_digraphs():
    normalizer = MedievalFrenchNormalizer({})
    assert normalizer.normalize("que") == "que"
    assert normalizer.normalize("qui") == "qui"
    assert normalizer.normalize("guerre") == "guerre"
    assert normalizer.normalize("lui") == "lui"
    # genuine u->v transformations still happen elsewhere in the word
    assert normalizer.normalize("cheualier") == "chevalier"
    assert normalizer.normalize("auant") == "avant"


def test_ij_rule_keeps_ie_diphthong():
    normalizer = MedievalFrenchNormalizer({})
    assert normalizer.normalize("bien") == "bien"
    assert normalizer.normalize("mie") == "mie"
    assert normalizer.normalize("biel") == "biel"


def test_detect_normalization_candidates_from_lexicon(tmp_path):
    csv_path = tmp_path / "lexicon.csv"
    csv_path.write_text("word, count\nq̃, 10\na~, 5\nfoo, 1\n", encoding="utf-8")

    results = detect_normalization_candidates(lexicon_path=str(csv_path), abbreviations={})
    assert results["total_tokens"] == 3
    assert any(item["token"] == "q̃" and item["suggested"] == "que" for item in results["top_suspicious"])
    assert any(item["token"] == "a~" and item["suggested"] == "an" for item in results["top_suspicious"])


def test_detect_normalization_candidates_from_output_dir(tmp_path):
    output_dir = tmp_path / "nlp_output"
    output_dir.mkdir()
    contract = {
        "document_id": "doc1",
        "pages": [
            {
                "page_id": "p1",
                "image_path": "img",
                "lines": [
                    {"line_id": "l1", "text": "q̃ a~"},
                    {"line_id": "l2", "text": "foo"},
                ],
            }
        ],
    }
    json_path = output_dir / "sample.json"
    json_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

    results = detect_normalization_candidates(output_dir=str(output_dir), abbreviations={})
    assert results["total_tokens"] == 3
    assert any(item["token"] == "q̃" for item in results["top_suspicious"])


def test_find_lexical_errors(tmp_path):
    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(
        json.dumps({"foo": {"wiktionary_en": ["a word"], "cltk_fr": []}}),
        encoding="utf-8",
    )
    csv_path = tmp_path / "lexicon.csv"
    csv_path.write_text("word, count\nfoo, 10\nbarbarus, 5\n", encoding="utf-8")

    results = find_lexical_errors(dictionary_path=str(dictionary_path), lexicon_path=str(csv_path))
    assert results["total_tokens"] == 2
    assert results["dictionary_size"] == 1
    assert results["unknown_tokens"] == 1
    assert results["top_unknown"][0]["token"] == "barbarus"
