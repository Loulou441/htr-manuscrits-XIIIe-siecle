"""Utilities for Character Error Rate (CER) evaluation."""

from __future__ import annotations


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Compute CER as edit_distance / max(1, len(reference))."""
    denom = max(1, len(reference))
    return _levenshtein_distance(reference, hypothesis) / denom


def average_pairwise_cer(texts: list[str]) -> float:
    """Compute the average pairwise CER for a set of transcription variants."""
    if len(texts) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += cer(texts[i], texts[j])
            count += 1
    return total / count if count else 0.0
