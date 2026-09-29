"""Shared deterministic features for training and pickle-free local inference."""
import re
from .language import normalize

VERSION='word-char-1.0.0'

def feature_tokens(text):
    words=re.findall(r'\b\w+\b',normalize(text),flags=re.UNICODE)
    result=['w:'+word for word in words]
    result+=['b:'+a+' '+b for a,b in zip(words,words[1:])]
    for word in words:
        padded=' '+word+' '
        for size in (3,4,5):
            result+=['c:'+padded[start:start+size] for start in range(max(0,len(padded)-size+1))]
    return result
