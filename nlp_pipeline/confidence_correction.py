"""Confidence-guided contextual correction for ambiguous HTR characters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class VariantScorer(Protocol):
    def score(self, text: str, position: int) -> float:
        ...


class HeuristicVariantScorer:
    """Simple fallback scorer favoring letter continuity and vowel plausibility."""

    def score(self, text: str, position: int) -> float:
        score = 0.0
        c = text[position]
        if c.isalpha():
            score += 1.0
        if c in "aeiouy":
            score += 0.3

        left = text[position - 1] if position > 0 else " "
        right = text[position + 1] if position + 1 < len(text) else " "
        if left.isalpha() and c.isalpha() and right.isalpha():
            score += 0.4
        return score


class MaskedLMVariantScorer:
    """Masked language model scorer using an MLM to select the most probable token."""

    def __init__(self, model_name: str = "almanach/camembert-base", device: str | int | None = None):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for MaskedLMVariantScorer. "
                "Install it with `pip install transformers sentencepiece`."
            ) from exc

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for MaskedLMVariantScorer.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
        self.model.to(self.device)

    def score(self, text: str, position: int) -> float:
        if position < 0 or position >= len(text):
            return float("-inf")

        char = text[position]
        if not char:
            return float("-inf")

        mask_token = self.tokenizer.mask_token
        if mask_token is None:
            return float("-inf")

        masked_text = text[:position] + mask_token + text[position + 1 :]
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        mask_positions = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if mask_positions[0].numel() != 1:
            return float("-inf")

        mask_index = mask_positions[0].item()
        logits = outputs.logits[0, mask_index]
        token_ids = self.tokenizer.encode(char, add_special_tokens=False)
        if len(token_ids) != 1:
            return float("-inf")

        log_probs = self.torch.log_softmax(logits, dim=-1)
        return float(log_probs[token_ids[0]].item())


@dataclass
class CorrectionEvent:
    line_id: str
    position: int
    old_char: str
    new_char: str
    old_confidence: float


class ConfidenceGuidedCorrector:
    def __init__(
        self,
        threshold: float = 0.7,
        scorer: VariantScorer | None = None,
        mlm_model: str | None = None,
        device: str | int | None = None,
    ):
        self.threshold = threshold
        if mlm_model is not None:
            self.scorer = MaskedLMVariantScorer(model_name=mlm_model, device=device)
        else:
            self.scorer = scorer or HeuristicVariantScorer()

    def correct_line(self, line: dict) -> tuple[str, list[CorrectionEvent]]:
        text = line.get("text", "")
        confidences = line.get("char_confidences", [])
        candidates = line.get("candidates")

        if not text or not isinstance(confidences, list) or not candidates:
            return text, []

        if isinstance(candidates, dict):
            candidates_list = [candidates]
        else:
            candidates_list = [c for c in candidates if isinstance(c, dict)]

        corrected = text
        events: list[CorrectionEvent] = []

        for cand in candidates_list:
            pos = cand.get("position")
            options = cand.get("options", [])
            if not isinstance(pos, int) or pos < 0 or pos >= len(corrected):
                continue
            if pos >= len(confidences):
                continue
            conf = float(confidences[pos])
            if conf >= self.threshold:
                continue
            if not options:
                continue

            best = corrected[pos]
            best_score = float("-inf")
            for opt in options:
                if not opt:
                    continue
                candidate_char = str(opt)[0]
                variant = corrected[:pos] + candidate_char + corrected[pos + 1 :]
                score = self.scorer.score(variant, pos)
                if score > best_score:
                    best_score = score
                    best = candidate_char

            if best != corrected[pos]:
                events.append(
                    CorrectionEvent(
                        line_id=line.get("line_id", ""),
                        position=pos,
                        old_char=corrected[pos],
                        new_char=best,
                        old_confidence=conf,
                    )
                )
                corrected = corrected[:pos] + best + corrected[pos + 1 :]

        return corrected, events

    def apply_to_contract(self, contract: dict, log_path: str | Path | None = None) -> list[CorrectionEvent]:
        events: list[CorrectionEvent] = []
        for page in contract.get("pages", []):
            for line in page.get("lines", []):
                new_text, line_events = self.correct_line(line)
                if line_events:
                    line["text"] = new_text
                    events.extend(line_events)

        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for evt in events:
                    f.write(json.dumps(evt.__dict__, ensure_ascii=False) + "\n")
        return events
