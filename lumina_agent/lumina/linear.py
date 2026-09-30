"""Multi-head linear classifier trained from randomly-initialised weights.

One shared sparse feature space feeds several independent softmax heads, which is
the cheap version of the shared-encoder/multi-task idea in the spec (s122): the
heads share representation work while staying separately evaluable. There are no
pretrained weights, no downloaded checkpoints and no network access anywhere in
this module.

Every head carries a temperature (fitted on validation) and an abstention
threshold, so a low-confidence prediction becomes UNKNOWN instead of a confident
guess (spec s54, s124).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .features import vectorize
from .text import VERSION as NORMALIZATION_VERSION

FORMAT = "lumina-linear-v1"


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def build_heads(definitions):
    """Lay every head's labels out contiguously in one weight matrix."""
    heads, offset = {}, 0
    for name, labels in definitions.items():
        labels = list(labels)
        heads[name] = {"start": offset, "end": offset + len(labels), "labels": labels,
                       "temperature": 1.0, "abstain_below": 0.0, "abstain_label": None}
        offset += len(labels)
    return heads, offset


class MultiHeadLinear:
    """Inference side. Training lives in `train()` below."""

    def __init__(self, config, weights, bias):
        self.config, self.weights, self.bias = config, weights, bias
        self.heads = config["heads"]
        self.spec = config["features"]
        self.version = config.get("version")

    # -- persistence ------------------------------------------------------
    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != FORMAT:
            raise ValueError("unexpected model format " + repr(config.get("format")))
        if config.get("normalization") != NORMALIZATION_VERSION:
            raise ValueError("incompatible normalization version; retrain the model")
        raw = (folder / "weights.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != config["weights_sha256"]:
            raise ValueError("model checksum mismatch")
        with np.load(folder / "weights.npz", allow_pickle=False) as arrays:
            weights, bias = arrays["weights"], arrays["bias"]
        expected = len(config["features"]["vocabulary"])
        if weights.shape != (expected, len(bias)):
            raise ValueError("model weights do not match the stored feature space")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("model weights contain non-finite values")
        return cls(config, weights, bias)

    def save(self, folder, extra=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "weights.npz",
                            weights=self.weights.astype(np.float32),
                            bias=self.bias.astype(np.float32))
        digest = hashlib.sha256((folder / "weights.npz").read_bytes()).hexdigest()
        config = dict(self.config)
        config.update({"format": FORMAT, "normalization": NORMALIZATION_VERSION,
                       "weights_sha256": digest, "not_clinically_validated": True})
        if extra:
            config.update(extra)
        (folder / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
        self.config = config
        return folder

    # -- inference --------------------------------------------------------
    def logits(self, text):
        ids, values = vectorize(text, self.spec)
        if not ids:
            return self.bias.copy()
        return np.asarray(values) @ self.weights[ids] + self.bias

    def predict(self, text):
        """Return {head: {label, confidence, abstain, probabilities}}."""
        raw = self.logits(text)
        out = {}
        for name, head in self.heads.items():
            p = softmax(raw[head["start"]:head["end"]] / head.get("temperature", 1.0))
            labels = head["labels"]
            index = int(np.argmax(p))
            confidence = float(p[index])
            threshold = head.get("abstain_below", 0.0)
            abstain = confidence < threshold
            label = labels[index]
            if abstain and head.get("abstain_label") in labels:
                label = head["abstain_label"]
            out[name] = {"label": label, "predicted": labels[index],
                         "confidence": round(confidence, 5), "abstain": abstain,
                         "probabilities": {l: round(float(v), 5) for l, v in zip(labels, p)}}
        return out


def train(rows, head_definitions, label_of, feature_spec,
          text_of=lambda r: r["text"], seed=42, epochs=20, learning_rate=0.2,
          l2=1e-4, class_weight_cap=4.0, emphasis=None, log=print):
    """Fit weights from zero with per-sample SGD and inverse-frequency weighting.

    `emphasis` lets a caller raise the loss on classes that are expensive to miss
    (a missed crisis costs far more than a missed small-talk label), expressed as
    {head: {label: multiplier}}.
    """
    heads, width = build_heads(head_definitions)
    n_features = len(feature_spec["vocabulary"])
    weights = np.zeros((n_features, width), dtype=np.float64)
    bias = np.zeros(width, dtype=np.float64)

    vectors = []
    for row in rows:
        ids, values = vectorize(text_of(row), feature_spec)
        vectors.append((np.array(ids, dtype=np.int64), np.array(values, dtype=np.float64)))

    targets, class_weights = {}, {}
    for name, head in heads.items():
        labels = head["labels"]
        targets[name] = np.array([labels.index(label_of(r, name)) for r in rows])
        counts = np.bincount(targets[name], minlength=len(labels))
        w = np.sqrt(len(rows) / (len(labels) * np.maximum(counts, 1)))
        w = np.clip(w, 0.5, class_weight_cap)
        for label, multiplier in (emphasis or {}).get(name, {}).items():
            w[labels.index(label)] *= multiplier
        class_weights[name] = w

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        lr = learning_rate / math.sqrt(1 + epoch / 4)
        loss = 0.0
        for index in rng.permutation(len(rows)):
            ids, values = vectors[index]
            if ids.size == 0:
                continue
            raw = values @ weights[ids] + bias
            grad = np.zeros(width)
            for name, head in heads.items():
                start, end = head["start"], head["end"]
                p = softmax(raw[start:end])
                target = targets[name][index]
                loss -= math.log(max(p[target], 1e-12))
                p[target] -= 1.0
                grad[start:end] = p * class_weights[name][target]
            weights[ids] -= lr * (values[:, None] * grad[None, :] + l2 * weights[ids])
            bias -= lr * grad * 0.15
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log("  epoch {}/{}  mean head loss {:.4f}".format(
                epoch + 1, epochs, loss / (len(rows) * len(heads))))

    config = {"version": None, "heads": heads, "features": feature_spec,
              "training": {"seed": seed, "epochs": epochs, "rows": len(rows),
                           "learning_rate": learning_rate, "l2": l2,
                           "class_weight_cap": class_weight_cap,
                           "emphasis": emphasis or {}}}
    return MultiHeadLinear(config, weights, bias)


def calibrate(model, rows, label_of, text_of=lambda r: r["text"],
              grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)):
    """Fit one temperature per head by minimising validation NLL.

    Temperature scaling makes the reported confidence mean something, which is a
    precondition for the abstention thresholds below - and for the spec's rule
    that safety must never act on an uncalibrated 0.51.
    """
    cached = [model.logits(text_of(r)) for r in rows]
    for name, head in model.heads.items():
        labels = head["labels"]
        y = [labels.index(label_of(r, name)) for r in rows]
        start, end = head["start"], head["end"]

        def nll(t, start=start, end=end, y=y):
            return sum(-math.log(max(softmax(z[start:end] / t)[k], 1e-12))
                       for z, k in zip(cached, y)) / max(len(y), 1)

        head["temperature"] = float(min(grid, key=nll))
    return model


def fit_abstention(model, rows, label_of, abstain_labels,
                   text_of=lambda r: r["text"], target_precision=0.80,
                   grid=tuple(i / 100 for i in range(30, 96, 5))):
    """Pick the lowest threshold whose *accepted* predictions reach a precision.

    Returns the achieved precision/coverage per head so the choice is reported
    rather than assumed.
    """
    cached = [model.logits(text_of(r)) for r in rows]
    report = {}
    for name, head in model.heads.items():
        if name not in abstain_labels:
            continue
        labels = head["labels"]
        start, end = head["start"], head["end"]
        truth = [labels.index(label_of(r, name)) for r in rows]
        scored = []
        for z, k in zip(cached, truth):
            p = softmax(z[start:end] / head.get("temperature", 1.0))
            index = int(np.argmax(p))
            scored.append((float(p[index]), index == k))
        best = None
        for threshold in grid:
            kept = [ok for c, ok in scored if c >= threshold]
            if not kept:
                continue
            precision = sum(kept) / len(kept)
            coverage = len(kept) / len(scored)
            if precision >= target_precision:
                best = (threshold, precision, coverage)
                break
        if best is None:
            # Never reaches the target: keep the most selective threshold tried
            # and report the truth rather than pretending the target was met.
            threshold = grid[-1]
            kept = [ok for c, ok in scored if c >= threshold]
            precision = (sum(kept) / len(kept)) if kept else 0.0
            best = (threshold, precision, (len(kept) / len(scored)) if scored else 0.0)
        head["abstain_below"] = float(best[0])
        head["abstain_label"] = abstain_labels[name]
        report[name] = {"threshold": best[0], "abstain_label": abstain_labels[name],
                        "accepted_precision": round(best[1], 4),
                        "coverage": round(best[2], 4), "target_precision": target_precision,
                        "target_met": bool(best[1] >= target_precision)}
    return report
