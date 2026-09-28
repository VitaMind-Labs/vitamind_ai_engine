"""Collect trained model cards into one registry (spec s75).

A model is a `candidate` until someone promotes it. Promotion is a deliberate
act recorded here, never a side effect of training, so a retrain cannot quietly
replace what production serves.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
REGISTRY = ARTIFACTS / "registry.json"


def collect():
    entries = {}
    for config_path in sorted((ARTIFACTS / "models").glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        heads = config["heads"]
        evaluation = config.get("evaluation", {}).get("test", {})
        entries[config["model_name"]] = {
            "version": config["version"],
            "status": config.get("status", "candidate"),
            "created_at": config.get("created_at"),
            "architecture": "multi_head_sparse_linear_softmax",
            "trained_from": "random_initialisation",
            "parameters": sum(len(config["features"]["vocabulary"]) *
                              (h["end"] - h["start"]) for h in heads.values()),
            "features": len(config["features"]["vocabulary"]),
            "normalization": config["normalization"],
            "heads": {name: {"labels": h["labels"], "temperature": h["temperature"],
                             "abstain_below": h["abstain_below"]}
                      for name, h in heads.items()},
            "dataset": config.get("dataset"),
            "training": config.get("training"),
            "test_metrics": {head: {"macro_f1": report["macro_f1"],
                                    "accuracy": report["accuracy"],
                                    "expected_calibration_error":
                                        report["calibration"]["expected_calibration_error"]}
                             for head, report in evaluation.items()},
            "weights_sha256": config["weights_sha256"],
            "not_clinically_validated": config.get("not_clinically_validated", True),
            "no_pretrained_weights": config.get("no_pretrained_weights", True),
            "no_external_inference_api": config.get("no_external_inference_api", True),
            "notes": config.get("notes", []),
        }
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", nargs=2, metavar=("MODEL", "STATUS"),
                        help="record a promotion, e.g. --promote safety production")
    args = parser.parse_args()

    registry = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "models": collect()}
    if args.promote:
        name, status = args.promote
        if name not in registry["models"]:
            raise SystemExit(f"unknown model {name!r}")
        previous = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
        history = previous.get("promotions", [])
        history.append({"model": name, "status": status,
                        "version": registry["models"][name]["version"],
                        "at": registry["generated_at"]})
        registry["promotions"] = history
        registry["models"][name]["status"] = status
    elif REGISTRY.exists():
        registry["promotions"] = json.loads(
            REGISTRY.read_text(encoding="utf-8")).get("promotions", [])

    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, entry in registry["models"].items():
        print(f"{name:14s} {entry['version']:28s} {entry['status']:10s} "
              f"{entry['parameters']:>9,d} params  {entry['test_metrics']}")


if __name__ == "__main__":
    main()
