"""Evaluation that answers the questions the spec actually asks (s76, s77).

Accuracy alone is useless on this data: a model that predicts NEUTRAL for every
utterance scores 84% on the dialogue set and is worth nothing. So the report is
built around macro-F1, per-class recall, calibration error, and - for safety -
the rate at which a severe level is scored as a mild one, which is the error that
hurts a patient.
"""
from __future__ import annotations

import collections
import math

import numpy as np

from lumina.linear import softmax


def confusion(truth, predicted, labels):
    index = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, predicted):
        matrix[index[t]][index[p]] += 1
    return matrix


def per_class(truth, predicted, labels):
    out = {}
    for label in labels:
        tp = sum(1 for t, p in zip(truth, predicted) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, predicted) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"support": tp + fn, "precision": round(precision, 4),
                      "recall": round(recall, 4), "f1": round(f1, 4)}
    return out


def expected_calibration_error(confidences, correct, bins=10):
    """How far the reported confidence is from the observed hit rate.

    A safety threshold is only meaningful if 0.8 really means 80%.
    """
    if not confidences:
        return 0.0
    total, error = len(confidences), 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in zip(confidences, correct)
                  if (c > low or b == 0) and c <= high]
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(ok for _, ok in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_confidence - accuracy)
    return round(error, 4)


def evaluate_head(model, rows, head, label_of, text_of=lambda r: r["text"]):
    spec = model.heads[head]
    labels = spec["labels"]
    start, end = spec["start"], spec["end"]
    temperature = spec.get("temperature", 1.0)
    threshold = spec.get("abstain_below", 0.0)
    abstain_label = spec.get("abstain_label")

    truth, predicted, confidences, accepted = [], [], [], []
    for row in rows:
        p = softmax(model.logits(text_of(row))[start:end] / temperature)
        index = int(np.argmax(p))
        truth.append(label_of(row, head))
        predicted.append(labels[index])
        confidences.append(float(p[index]))
        accepted.append(float(p[index]) >= threshold)

    correct = [t == p for t, p in zip(truth, predicted)]
    classes = per_class(truth, predicted, labels)
    # Macro-F1 over classes that actually occur, so absent classes do not drag
    # the headline number toward zero and hide a real change.
    present = [l for l in labels if classes[l]["support"] > 0]
    macro_f1 = sum(classes[l]["f1"] for l in present) / max(len(present), 1)

    kept = [(t, p) for t, p, a in zip(truth, predicted, accepted) if a]
    report = {
        "head": head,
        "rows": len(rows),
        "accuracy": round(sum(correct) / max(len(rows), 1), 4),
        "macro_f1": round(macro_f1, 4),
        "classes_present": len(present),
        "per_class": classes,
        "confusion": {"labels": labels, "matrix": confusion(truth, predicted, labels)},
        "calibration": {"temperature": temperature,
                        "expected_calibration_error": expected_calibration_error(confidences, correct),
                        "mean_confidence": round(sum(confidences) / max(len(confidences), 1), 4)},
        "abstention": {
            "threshold": threshold,
            "abstain_label": abstain_label,
            "coverage": round(sum(accepted) / max(len(rows), 1), 4),
            "accuracy_when_answering": round(
                sum(t == p for t, p in kept) / max(len(kept), 1), 4),
            "accuracy_when_abstaining": round(
                sum(t == p for t, p, a in zip(truth, predicted, accepted) if not a)
                / max(len(rows) - len(kept), 1), 4),
        },
    }
    return report, truth, predicted


def severity_report(truth, predicted, order):
    """Safety-specific: how often a severe level is called a milder one.

    Under-calling is the failure that reaches a patient, so it is reported on its
    own rather than being averaged into a single accuracy number.
    """
    under, over, exact = 0, 0, 0
    worst = collections.Counter()
    for t, p in zip(truth, predicted):
        dt, dp = order[t], order[p]
        if dp < dt:
            under += 1
            worst[f"{t}->{p}"] += 1
        elif dp > dt:
            over += 1
        else:
            exact += 1
    total = max(len(truth), 1)
    severe = [t for t in truth if order[t] >= order["HIGH"]]
    severe_caught = sum(1 for t, p in zip(truth, predicted)
                        if order[t] >= order["HIGH"] and order[p] >= order["HIGH"])
    return {
        "exact_rate": round(exact / total, 4),
        "under_called_rate": round(under / total, 4),
        "over_called_rate": round(over / total, 4),
        "severe_rows": len(severe),
        "severe_recall_at_or_above_HIGH": round(severe_caught / max(len(severe), 1), 4),
        "most_common_under_calls": dict(worst.most_common(6)),
    }
