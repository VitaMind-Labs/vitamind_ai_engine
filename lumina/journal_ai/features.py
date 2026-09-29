"""Sparse TF-IDF features, learned exclusively from the training partition."""
import collections
import math
import re
from .normalize import normalize

def tokens(text):
    words = re.findall(r"[\w']+", normalize(text), flags=re.UNICODE)
    result = ["w:" + w for w in words]
    result += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
    for word in words:
        padded = "^" + word + "$"
        for n in (3, 4, 5):
            result += ["c:" + padded[i:i+n] for i in range(len(padded)-n+1)]
    return result

def fit_features(texts, max_features=12000):
    df = collections.Counter()
    for text in texts:
        df.update(set(tokens(text)))
    chosen = sorted((k for k, n in df.items() if n >= 2), key=lambda k: (-df[k], k))[:max_features]
    return {"vocabulary": {k:i for i,k in enumerate(chosen)}, "idf": [math.log((1+len(texts))/(1+df[k]))+1 for k in chosen]}

def vectorize(text, spec):
    counts = collections.Counter(tokens(text))
    pairs = [(spec["vocabulary"][k], (1+math.log(n))*spec["idf"][spec["vocabulary"][k]]) for k,n in counts.items() if k in spec["vocabulary"]]
    norm = math.sqrt(sum(v*v for _,v in pairs)) or 1
    return [i for i,_ in pairs], [v/norm for _,v in pairs]
