"""Small trained multi-head linear model. No remote service or pickle loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .features import vectorize
from .normalize import VERSION
from .schema import Context

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "journal-linear"

def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()

class LocalModel:
    def __init__(self, folder=DEFAULT_MODEL):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if self.config["normalization"] != VERSION:
            raise ValueError("incompatible normalization version")
        raw = (folder / "weights.npz").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.config["weights_sha256"]:
            raise ValueError("model checksum mismatch")
        with np.load(folder / "weights.npz", allow_pickle=False) as arrays:
            self.weights, self.bias = arrays["weights"], arrays["bias"]
        expected = len(self.config["features"]["vocabulary"])
        if self.weights.shape != (expected, len(self.bias)) or not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError("invalid model weights")
        self.version = self.config["version"]

    def predict(self, text):
        ids, values = vectorize(text, self.config["features"])
        logits = np.asarray(values) @ self.weights[ids] + self.bias if ids else self.bias.copy()
        out, probs = {}, {}
        for name, head in self.config["heads"].items():
            p = softmax(logits[head["start"]:head["end"]] / head.get("temperature", 1.0))
            labels = head["labels"]
            out[name] = labels[int(np.argmax(p))]
            probs[name] = {label:float(v) for label,v in zip(labels,p)}
        categories = [name.split(":",1)[1] for name in out if name.startswith("category:") and out[name] == "true"]
        return Context(tier=out["tier"], categories=categories, subject=out["subject"], temporal=out["temporal"], negated=out["negated"]=="true", is_idiom=out["is_idiom"]=="true", confidence=probs["tier"][out["tier"]], rationale="Learned word and character features; probabilities are not clinical risk estimates.", source="local_linear", probabilities=probs["tier"])
