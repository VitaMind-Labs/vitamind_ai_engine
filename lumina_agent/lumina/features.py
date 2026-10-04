"""Sparse TF-IDF features learned exclusively from the training partition.

Word unigrams and bigrams carry topic and phrasing; within-word character n-grams
absorb the misspellings, elongations and Arabic orthographic variation that real
patient text is full of. Nothing here is pretrained: the vocabulary and the IDF
weights are fitted on the training split of each dataset and stored in the model
config, so a model can never see validation or test vocabulary.
"""
import collections
import math
import re

from .text import feature_variants

WORD = re.compile(r"[\w']+", flags=re.UNICODE)


def tokens(text: str) -> list[str]:
    result = []
    for variant in feature_variants(text):
        words = WORD.findall(variant)
        result += ["w:" + w for w in words]
        result += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
        for word in words:
            padded = "^" + word + "$"
            for n in (2, 3, 4, 5):
                result += ["c:" + padded[i:i + n] for i in range(len(padded) - n + 1)]
    return result


def fit_features(texts, max_features=20000, min_document_frequency=2):
    """Fit vocabulary + IDF. Call with the TRAINING texts only."""
    df = collections.Counter()
    for text in texts:
        df.update(set(tokens(text)))
    chosen = sorted((k for k, n in df.items() if n >= min_document_frequency),
                    key=lambda k: (-df[k], k))[:max_features]
    return {"vocabulary": {k: i for i, k in enumerate(chosen)},
            "idf": [math.log((1 + len(texts)) / (1 + df[k])) + 1 for k in chosen],
            "max_features": max_features,
            "min_document_frequency": min_document_frequency}


def vectorize(text: str, spec):
    """Return (indices, values) of the L2-normalized sparse TF-IDF vector."""
    vocabulary, idf = spec["vocabulary"], spec["idf"]
    counts = collections.Counter(tokens(text))
    pairs = [(vocabulary[k], (1 + math.log(n)) * idf[vocabulary[k]])
             for k, n in counts.items() if k in vocabulary]
    norm = math.sqrt(sum(v * v for _, v in pairs)) or 1.0
    return [i for i, _ in pairs], [v / norm for _, v in pairs]
