"""Tests for NLP rule-based normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from normalization_rules import MedievalFrenchNormalizer, NormalizerConfig


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
