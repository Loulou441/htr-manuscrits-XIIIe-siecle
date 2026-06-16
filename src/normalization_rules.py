"""Rule-based medieval text normalization used in  NLP workflow."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


VOWELS = set("aeiouy")


@dataclass
class NormalizerConfig:
    apply_nfc: bool = True
    lowercase: bool = True
    normalize_uv: bool = True
    normalize_ij: bool = True
    expand_tilde: bool = True
    expand_abbreviations: bool = True


class MedievalFrenchNormalizer:
    """Independent, toggleable rule normalizer with optional abbreviation table."""

    def __init__(self, abbreviations: dict[str, str] | None = None, config: NormalizerConfig | None = None):
        self.abbreviations = abbreviations or {}
        self.config = config or NormalizerConfig()

    @staticmethod
    def from_json(path: str | Path, config: NormalizerConfig | None = None) -> "MedievalFrenchNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Abbreviation table must be a JSON object")
        return MedievalFrenchNormalizer(abbreviations=data, config=config)

    def normalize(self, text: str) -> str:
        out = text
        if self.config.apply_nfc:
            out = unicodedata.normalize("NFC", out)
        if self.config.lowercase:
            out = out.lower()
        if self.config.normalize_uv:
            out = self._apply_uv_rule(out)
        if self.config.normalize_ij:
            out = self._apply_ij_rule(out)
        if self.config.expand_tilde:
            out = self._expand_tilde(out)
        if self.config.expand_abbreviations:
            out = self._apply_abbreviation_table(out)
        return out

    def _apply_abbreviation_table(self, text: str) -> str:
        # Longest keys first to avoid partial replacement collisions.
        pairs = sorted(self.abbreviations.items(), key=lambda kv: len(kv[0]), reverse=True)
        out = text
        for src, dst in pairs:
            out = out.replace(src, dst)
        return out

    def _apply_uv_rule(self, text: str) -> str:
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch not in {"u", "v"}:
                continue
            prev_c = chars[idx - 1] if idx > 0 else ""
            next_c = chars[idx + 1] if idx + 1 < len(chars) else ""
            at_word_start = idx == 0 or not prev_c.isalpha()
            before_vowel = next_c in VOWELS

            if ch == "u" and (at_word_start or before_vowel):
                chars[idx] = "v"
            elif ch == "v" and not (at_word_start or before_vowel):
                chars[idx] = "u"
        return "".join(chars)

    def _apply_ij_rule(self, text: str) -> str:
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch != "i":
                continue
            next_c = chars[idx + 1] if idx + 1 < len(chars) else ""
            if next_c in VOWELS:
                chars[idx] = "j"
        return "".join(chars)

    def _expand_tilde(self, text: str) -> str:
        out = re.sub(r"o~(?=$|[^a-z])", "om", text)
        out = re.sub(r"([ae])~", r"\1n", out)
        out = re.sub(r"o~", "on", out)
        return out
