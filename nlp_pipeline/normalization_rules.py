"""Rule-based medieval text normalization used in  NLP workflow."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[^\s.,;:!?()\[\]{}\"']+")
ABBREVIATION_MARKERS = {"~", "⁊", "ꝑ", "ꝗ", "ꝓ", "ꝙ", "ꝯ", "q̃", "qͥ", "q̄"}
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
        """Build a normalizer from a JSON abbreviation table (e.g. medieval_abbreviations.json)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Abbreviation table must be a JSON object")
        return MedievalFrenchNormalizer(abbreviations=data, config=config)

    def normalize(self, text: str) -> str:
        """Apply NFC, lowercase, u/v, i/j, tilde expansion and the abbreviation table, in order."""
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
        """Replace each known abbreviation by its expansion, longest keys first."""
        # Longest keys first to avoid partial replacement collisions.
        pairs = sorted(self.abbreviations.items(), key=lambda kv: len(kv[0]), reverse=True)
        out = text
        for src, dst in pairs:
            out = out.replace(src, dst)
        return out

    def _apply_uv_rule(self, text: str) -> str:
        """Resolve medieval u/v graphical variants into their consonantal/vocalic modern form."""
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch not in {"u", "v"}:
                continue
            prev_c = chars[idx - 1] if idx > 0 else ""
            next_c = chars[idx + 1] if idx + 1 < len(chars) else ""
            at_word_start = idx == 0 or not prev_c.isalpha()
            before_vowel = next_c in VOWELS

            if prev_c in {"q", "g"} or next_c == "i":
                # qu/gu digraphs (que, qui, guerre...) and u+i (lui, liu...)
                # keep their vocalic u: not a true consonantal context.
                continue

            if ch == "u" and (at_word_start or before_vowel):
                chars[idx] = "v"
            elif ch == "v" and not (at_word_start or before_vowel):
                chars[idx] = "u"
        return "".join(chars)

    def _apply_ij_rule(self, text: str) -> str:
        """Convert a vocalic i into its consonantal j form before a non-e vowel."""
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch != "i":
                continue
            next_c = chars[idx + 1] if idx + 1 < len(chars) else ""

            if next_c == "e":
                # "ie"/"ien"/"ier" diphthongs (bien, fier, mie...) keep their vocalic i.
                continue

            if next_c in VOWELS:
                chars[idx] = "j"
        return "".join(chars)

    def _expand_tilde(self, text: str) -> str:
        """Expand the medieval nasal tilde (precomposed or combining) into its vowel+n form."""
        out = text
        precomposed = {
            "ã": "an",
            "ẽ": "en",
            "ĩ": "in",
            "õ": "on",
            "ũ": "un",
        }
        for src, dst in precomposed.items():
            out = out.replace(src, dst)

        out = re.sub(r"o(?:~|\u0303)(?=$|[^a-z])", "om", out)
        out = re.sub(r"([aeiu])(?:~|\u0303)", r"\1n", out)
        out = re.sub(r"o(?:~|\u0303)", "on", out)
        return out


def _contains_combining_mark(text: str) -> bool:
    """Return True if text contains a Unicode combining mark (e.g. a floating tilde)."""
    return any(unicodedata.category(ch).startswith("M") for ch in text)


def _is_suspicious_token(token: str) -> bool:
    """Return True if token carries an abbreviation marker or a combining mark."""
    if any(ch in token for ch in ABBREVIATION_MARKERS):
        return True
    return _contains_combining_mark(token)


def _load_lexicon_csv(path: str | Path) -> list[tuple[str, int]]:
    """Parse a (word, count) lexicon CSV, skipping the header row."""
    rows: list[tuple[str, int]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            word = parts[0].strip()
            try:
                count = int(parts[1].strip())
            except ValueError:
                continue
            rows.append((word, count))
    return rows


def _build_word_counts_from_output_dir(output_dir: str | Path) -> list[tuple[str, int]]:
    """Tokenize every line.text across HTR contracts in output_dir and count word frequencies."""
    from htr_data_contract import iter_lines, load_json

    counts: Counter = Counter()
    output_path = Path(output_dir)
    for json_path in sorted(output_path.rglob("*.json")):
        try:
            contract = load_json(json_path)
        except Exception:
            continue
        for _, _, _, line in iter_lines(contract):
            text = str(line.get("text", ""))
            counts.update(TOKEN_RE.findall(text.lower()))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _propose_normalization_suggestion(token: str, abbreviations: dict[str, str]) -> str | None:
    """Suggest an expansion for a suspicious token not already in the abbreviation table."""
    if token in abbreviations:
        return None
    candidate = token
    if "q̃" in candidate:
        candidate = candidate.replace("q̃", "que")
    if "qͥ" in candidate:
        candidate = candidate.replace("qͥ", "que")
    if "ꝯ" in candidate:
        candidate = candidate.replace("ꝯ", "con")
    if "⁊̃" in candidate:
        candidate = candidate.replace("⁊̃", "et")

    candidate = re.sub(r"o(?:~|\u0303)(?=$|[^a-z])", "om", candidate)
    candidate = re.sub(r"([aeiu])(?:~|\u0303)", r"\1n", candidate)
    candidate = re.sub(r"o(?:~|\u0303)", "on", candidate)

    if candidate != token:
        return candidate
    return None


def _load_dictionary(path: str | Path) -> dict[str, dict]:
    """Load the Old French dictionary JSON (word -> {wiktionary_en, cltk_fr})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dictionary must be a JSON object")
    return data


def _is_known_word(token: str, dictionary: dict[str, dict]) -> bool:
    """Return True if token has at least one non-empty definition in the dictionary."""
    entry = dictionary.get(token.lower())
    if entry is None:
        return False
    return bool(entry.get("wiktionary_en")) or bool(entry.get("cltk_fr"))


def find_lexical_errors(
    dictionary_path: str | Path,
    lexicon_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 100,
) -> dict[str, object]:
    """Flag normalized tokens that are absent from the Old French dictionary."""
    if lexicon_path is not None and Path(lexicon_path).exists():
        rows = _load_lexicon_csv(lexicon_path)
    elif output_dir is not None and Path(output_dir).exists():
        rows = _build_word_counts_from_output_dir(output_dir)
    else:
        raise ValueError("Either lexicon_path or output_dir must point to an existing path")

    dictionary = _load_dictionary(dictionary_path)
    total_counts = sum(count for _, count in rows)
    unknown = [(word, count) for word, count in rows if word and not _is_known_word(word, dictionary)]
    top_unknown = sorted(unknown, key=lambda kv: (-kv[1], kv[0]))[:top_n]

    return {
        "total_tokens": len(rows),
        "total_counts": total_counts,
        "dictionary_size": len(dictionary),
        "unknown_tokens": len(unknown),
        "top_unknown": [{"token": word, "count": count} for word, count in top_unknown],
    }


def detect_normalization_candidates(
    lexicon_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 100,
    abbreviations: dict[str, str] | None = None,
) -> dict[str, object]:
    """Surface the most frequent tokens carrying abbreviation markers, with expansion suggestions."""
    if lexicon_path is not None and Path(lexicon_path).exists():
        rows = _load_lexicon_csv(lexicon_path)
    elif output_dir is not None and Path(output_dir).exists():
        rows = _build_word_counts_from_output_dir(output_dir)
    else:
        raise ValueError("Either lexicon_path or output_dir must point to an existing path")

    existing_abbr = abbreviations or {}
    total_counts = sum(count for _, count in rows)
    suspicious = [(word, count) for word, count in rows if _is_suspicious_token(word)]

    marker_counts: Counter = Counter()
    for word, count in rows:
        for marker in ABBREVIATION_MARKERS:
            if marker in word:
                marker_counts[marker] += count
        if _contains_combining_mark(word):
            marker_counts["combining_mark"] += count

    top_suspicious = sorted(suspicious, key=lambda kv: (-kv[1], kv[0]))[:top_n]
    suggestions: list[dict[str, object]] = []
    for word, count in top_suspicious:
        proposed = _propose_normalization_suggestion(word, existing_abbr)
        if proposed is not None:
            suggestions.append({"token": word, "count": count, "suggested": proposed})

    return {
        "total_tokens": len(rows),
        "total_counts": total_counts,
        "suspicious_tokens": len(suspicious),
        "marker_counts": dict(marker_counts),
        "top_suspicious": [
            {
                "token": word,
                "count": count,
                "suggested": _propose_normalization_suggestion(word, existing_abbr),
            }
            for word, count in top_suspicious
        ],
        "suggestions": suggestions,
    }
