"""Confidence-guided contextual correction for ambiguous HTR characters."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from cer_utils import cer


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


# Modèle MLM utilisé par défaut pour l'arbitrage contextuel (slide 54, étape 2).
# Activé par défaut : la correction guidée par confiance doit utiliser un vrai
# modèle de langue plutôt que l'heuristique de repli, conformément au principe
# du cours (CamemBERT en mode Masked Language Model).
DEFAULT_MLM_MODEL = "almanach/camembert-base"


@dataclass
class CorrectionEvent:
    line_id: str
    position: int
    old_char: str
    new_char: str
    old_confidence: float


@dataclass
class LineCorrectionResult:
    """Résultat de la correction d'une ligne, avec le nécessaire pour le CER pairwise."""
    line_id: str
    raw_text: str
    corrected_text: str
    events: list[CorrectionEvent]
    pairwise_cer: float  # CER(raw_text, corrected_text) — mesure l'ampleur du changement


class ConfidenceGuidedCorrector:
    def __init__(
        self,
        threshold: float = 0.7,
        scorer: VariantScorer | None = None,
        mlm_model: str | None = DEFAULT_MLM_MODEL,
        device: str | int | None = None,
        use_mlm: bool = True,
    ):
        """
        mlm_model : nom du modèle MLM HuggingFace à utiliser pour l'arbitrage contextuel.
            Par défaut almanach/camembert-base (cf. slide 54 / section 7 du cours).
        use_mlm : si False, force l'utilisation du HeuristicVariantScorer même si
            mlm_model est renseigné (utile pour les tests rapides ou les environnements
            sans GPU/sans poids téléchargés).
        scorer : permet d'injecter un scorer personnalisé ; prioritaire sur mlm_model.
        """
        self.threshold = threshold
        self.mlm_model_name = mlm_model if use_mlm else None
        if scorer is not None:
            self.scorer = scorer
        elif use_mlm and mlm_model is not None:
            self.scorer = MaskedLMVariantScorer(model_name=mlm_model, device=device)
        else:
            self.scorer = HeuristicVariantScorer()

    @property
    def using_mlm(self) -> bool:
        return isinstance(self.scorer, MaskedLMVariantScorer)

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

    def apply_to_contract(
        self,
        contract: dict,
        log_path: str | Path | None = None,
        update_needs_review: bool = True,
    ) -> dict:
        """Applique la correction guidée par confiance sur tout le contrat.

        En plus des étapes 1-3 de la slide 54 (identification, arbitrage MLM,
        application), cette méthode couvre la réinjection (étape 4) :
        - recalcule `needs_review` ligne par ligne une fois la correction appliquée
          (si `update_needs_review=True`), sans jamais l'aggraver : une ligne ne
          repasse à False que si toutes ses positions ambiguës connues ont été
          traitées et que sa confiance reste cohérente ;
        - calcule le CER pairwise (raw vs corrigé) pour CHAQUE ligne, et le CER
          pairwise moyen sur le contrat, afin que l'amplitude des changements
          introduits par la correction soit mesurée à cette étape, comme à
          chaque étape du pipeline (cf. CONVENTIONS_NLP.md section 3).

        Retourne un dict :
            {
                "events": list[CorrectionEvent],
                "line_results": list[LineCorrectionResult],
                "mean_pairwise_cer": float,
                "n_lines_corrected": int,
                "n_lines_total": int,
                "using_mlm": bool,
                "mlm_model": str | None,
            }
        """
        events: list[CorrectionEvent] = []
        line_results: list[LineCorrectionResult] = []

        for page in contract.get("pages", []):
            for line in page.get("lines", []):
                raw_text = str(line.get("text", ""))
                new_text, line_events = self.correct_line(line)
                line_pairwise_cer = cer(raw_text, new_text)

                if line_events:
                    line["text"] = new_text
                    events.extend(line_events)

                line_results.append(
                    LineCorrectionResult(
                        line_id=line.get("line_id", ""),
                        raw_text=raw_text,
                        corrected_text=new_text,
                        events=line_events,
                        pairwise_cer=line_pairwise_cer,
                    )
                )

                if update_needs_review:
                    line["needs_review"] = self._recompute_needs_review(line, line_events)

        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            run_at = datetime.now(timezone.utc).isoformat()
            with path.open("a", encoding="utf-8") as f:
                for evt in events:
                    record = {"run_at": run_at, **evt.__dict__}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        n_lines_total = len(line_results)
        n_lines_corrected = sum(1 for r in line_results if r.events)
        mean_pairwise_cer = (
            sum(r.pairwise_cer for r in line_results) / n_lines_total if n_lines_total else 0.0
        )

        return {
            "events": events,
            "line_results": line_results,
            "mean_pairwise_cer": mean_pairwise_cer,
            "n_lines_corrected": n_lines_corrected,
            "n_lines_total": n_lines_total,
            "using_mlm": self.using_mlm,
            "mlm_model": self.mlm_model_name,
        }

    @staticmethod
    def _recompute_needs_review(line: dict, line_events: list[CorrectionEvent]) -> bool:
        """Réinjection (slide 54, étape 4) : ne désactive needs_review que si la
        ligne n'a plus de signal d'ambiguïté connu après correction.

        Ne fait jamais passer une ligne de False à True (on ne dégrade pas une
        ligne jugée fiable), et ne fait jamais passer à False une ligne dont la
        confiance globale ou la dispersion des char_confidences reste mauvaise,
        même si une correction a été appliquée. Cela évite de masquer une ligne
        encore problématique simplement parce qu'un caractère a été corrigé.
        """
        previously_needed_review = bool(line.get("needs_review", False))
        if not previously_needed_review:
            return False

        confidence = float(line.get("confidence", 0.0))
        char_confidences = line.get("char_confidences", [])

        char_std = (
            statistics.pstdev(char_confidences) if len(char_confidences) > 1 else 0.0
        )

        candidates = line.get("candidates")
        if isinstance(candidates, dict):
            candidates_list = [candidates]
        elif isinstance(candidates, list):
            candidates_list = [c for c in candidates if isinstance(c, dict)]
        else:
            candidates_list = []

        corrected_positions = {evt.position for evt in line_events}
        all_candidate_positions_resolved = all(
            cand.get("position") in corrected_positions for cand in candidates_list
        )

        still_low_confidence = confidence < 0.9
        still_high_variance = char_std > 0.2

        if not candidates_list:
            # Aucune position ambiguë n'a jamais été identifiable (HTR sans
            # candidates) : la correction par confiance n'a rien pu faire ici,
            # donc on ne change pas le statut de review.
            return True

        if all_candidate_positions_resolved and not still_low_confidence and not still_high_variance:
            return False

        return True