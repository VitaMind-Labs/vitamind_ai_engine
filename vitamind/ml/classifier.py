"""Lazy adapter for the persisted sklearn ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionResult:
    routing_label: str
    pathway: str
    crisis_flag: bool
    top_score_internal: float
    second_label: str
    second_score_internal: float
    margin_internal: float
    class_scores_internal: dict[str, float]


class MiraMLClassifier:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "models" / "mira_v5_synthetic_ensemble.joblib"
        self._bundle = None

    def _load(self):
        if self._bundle is None:
            import joblib
            self._bundle = joblib.load(self.model_path)
        return self._bundle

    def predict(self, text: str) -> PredictionResult:
        bundle = self._load()
        routing = bundle["routing"]
        classes = list(routing["classes"])
        weights = routing["weights"]
        probabilities = [model.predict_proba([text])[0] for model in (routing["word_model"], routing["char_model"], routing["nb_model"])]
        scores = {label: float(sum(weight * values[index] for weight, values in zip(weights, probabilities))) for index, label in enumerate(classes)}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        second_label, second_score = ordered[1]
        safety_models = bundle["safety"]
        safety_scores = [model.predict_proba([text])[0][1] for model in (safety_models["word_model"], safety_models["char_model"])]
        crisis = sum(weight * score for weight, score in zip(safety_models["weights"], safety_scores)) >= safety_models["threshold"]
        return PredictionResult(top_label, bundle["pathway_map"].get(top_label, "clinical_review"), bool(crisis), top_score, second_label, second_score, top_score - second_score, scores)