"""Tests for NLP CLI utilities and scoring helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cer_utils import average_pairwise_cer


def test_average_pairwise_cer():
    variants = ["abc", "abd", "ab"]
    score = average_pairwise_cer(variants)
    assert score > 0
    assert score <= 1
