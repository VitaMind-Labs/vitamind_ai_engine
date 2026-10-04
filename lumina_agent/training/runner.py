"""Shared training routine: fit -> calibrate -> set abstention -> evaluate -> register.

Every Lumina model goes through the same five steps in the same order, because
the order is what makes the numbers trustworthy:

* features and weights are fitted on train only;
* the temperature is fitted on val, never on test;
* the abstention threshold is chosen on val, never on test;
* test is touched exactly once, at the end, for the reported score.

The model card written alongside the weights records all of that, so a later
reader can tell what the metrics mean without rerunning anything.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from data_prep.common import DATASETS, read_dataset
from evaluation.metrics import evaluate_head, severity_report
from lumina.features import fit_features
from lumina.linear import calibrate, fit_abstention, train

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "models"


def merge_datasets(name, train_names, eval_names=None):
    """Build an in-memory merged dataset without mixing synthetic eval rows."""
    eval_names = eval_names or train_names
    return {
        "name": name,
        "splits": {
            "train": [row for dataset in train_names
                      for row in read_dataset(dataset, "train")],
            "val": [row for dataset in eval_names
                    for row in read_dataset(dataset, "val")],
            "test": [row for dataset in eval_names
                     for row in read_dataset(dataset, "test")],
        },
        "manifest": {"clinical_validity": "not_clinical"},
    }


def run(name, dataset, head_definitions, abstain_labels, *, seed=42, epochs=20,
        max_features=20000, emphasis=None, target_precision=0.80,
        min_document_frequency=2, learning_rate=0.2, l2=1e-4,
        class_weight_cap=4.0, use_weights=True, notes=None, log=print):
    if isinstance(dataset, dict):
        splits = dataset["splits"]
        dataset_name = dataset.get("name", name)
        dataset_manifest = dataset.get("manifest", {})
    else:
        splits = {s: read_dataset(dataset, s) for s in ("train", "val", "test")}
        dataset_name = dataset
        dataset_manifest = json.loads(
            (DATASETS / dataset / "manifest.json").read_text(encoding="utf-8"))

    def label_of(row, head):
        return row["labels"][head]

    train_rows = list(splits["train"])
    if not use_weights:
        # Used when a dataset's `weight` field is not meant to reach the loss.
        train_rows = [{**row, "weight": 1.0} for row in train_rows]
    total_weight = sum(float(r.get("weight", 1.0)) for r in train_rows)
    log(f"[{name}] train={len(train_rows)} rows (total weight {total_weight:.1f}) "
        f"val={len(splits['val'])} test={len(splits['test'])}")

    # Fit the feature space on the training partition only.
    spec = fit_features([r["text"] for r in splits["train"]],
                        max_features=max_features,
                        min_document_frequency=min_document_frequency)
    log(f"[{name}] features: {len(spec['vocabulary'])} of at most {max_features} "
        f"(min document frequency {min_document_frequency})")

    model = train(train_rows, head_definitions, label_of, spec, seed=seed,
                  epochs=epochs, learning_rate=learning_rate, l2=l2,
                  class_weight_cap=class_weight_cap, emphasis=emphasis, log=log)

    calibrate(model, splits["val"], label_of)
    log(f"[{name}] temperatures: " +
        ", ".join(f"{h}={d['temperature']}" for h, d in model.heads.items()))

    abstention = fit_abstention(model, splits["val"], label_of, abstain_labels,
                                target_precision=target_precision)
    for head, report in abstention.items():
        log(f"[{name}] abstain {head}: below {report['threshold']} -> "
            f"{report['abstain_label']} (val precision {report['accepted_precision']}, "
            f"coverage {report['coverage']})")

    evaluation = {}
    for split in ("val", "test"):
        evaluation[split] = {}
        for head in head_definitions:
            report, truth, predicted = evaluate_head(model, splits[split], head, label_of)
            if head in ("safety", "safety_signal"):
                from lumina.taxonomy import SAFETY_ORDER
                report["severity"] = severity_report(truth, predicted, SAFETY_ORDER)
            evaluation[split][head] = report
            log(f"[{name}] {split}/{head}: macro-F1 {report['macro_f1']} "
                f"accuracy {report['accuracy']} ECE "
                f"{report['calibration']['expected_calibration_error']}")

    version = f"{name}-seed{seed}-v1"
    model.config["version"] = version
    folder = ARTIFACTS / name
    model.save(folder, extra={
        "model_name": name,
        "version": version,
        "status": "candidate",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"name": dataset_name,
                    "rows": dataset_manifest.get("rows"),
                    "files_sha256": dataset_manifest.get("files_sha256"),
                    "clinical_validity": dataset_manifest.get("clinical_validity")},
        "row_weights_applied": bool(use_weights),
        "abstention": abstention,
        "evaluation": evaluation,
        "notes": notes or [],
        "no_pretrained_weights": True,
        "no_external_inference_api": True,
    })
    log(f"[{name}] saved {folder}")
    return model, evaluation, abstention
