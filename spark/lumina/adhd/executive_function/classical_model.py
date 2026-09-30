"""Inference for locally trained classical JSON artifacts, without pickle/sklearn."""
import collections,hashlib,json,math,re
from pathlib import Path
import numpy as np
from .language import normalize,languages
from .errors import AssistantError

class ClassicalModel:
    def __init__(self,task,folder=None):
        folder=Path(folder) if folder else Path(__file__).resolve().parents[3]/'models'/task
        self.artifact=None; self.coverage=[]; self.metadata={}
        if not (folder/'artifact.json').exists(): return
        try:
            raw=(folder/'artifact.json').read_bytes()
            metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            self.metadata=metadata
            if hashlib.sha256(raw).hexdigest()!=metadata['checksum']: raise ValueError('checksum mismatch')
            data=json.loads(raw)
            if data['format'] not in ('lumina-classical-v1','lumina-classical-v2') or data['normalizer']!='en-ar-1.0.0': raise ValueError('incompatible artifact')
            if data['format']=='lumina-classical-v2' and data.get('analyzer')!='word-char-1.0.0': raise ValueError('incompatible analyzer')
            if data['selected']=='rules': return
            self.coverage=metadata['languageCoverage']
            self.artifact=data
            self.heads=[(np.array(h['coef']),np.array(h['intercept']),h['classes']) for h in data['heads']]
            if any(not np.isfinite(w).all() or not np.isfinite(b).all() for w,b,_ in self.heads): raise ValueError('nonfinite weights')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise AssistantError('MODEL_OUTPUT_INVALID',f'Invalid local {task} artifact',503) from exc
    def supports(self,text):
        observed=languages(text); lang='MIXED' if len(observed)>1 else observed[0] if observed else 'UNKNOWN'
        if self.artifact is None or lang not in self.coverage: return False
        return any(t in self.artifact['vocabulary'] for t in self.features(text))
    def features(self,text):
        if self.artifact.get('format')=='lumina-classical-v2':
            from .text_features import feature_tokens
            return feature_tokens(text)
        words=re.findall(r'\b\w+\b',normalize(text),flags=re.UNICODE)
        return words+[' '.join(pair) for pair in zip(words,words[1:])]
    def predict(self,text):
        counts=collections.Counter(self.features(text))
        v=self.artifact['vocabulary']; idf=self.artifact['idf']
        pairs=[(v[t],(1+math.log(n))*idf[v[t]]) for t,n in counts.items() if t in v]
        length=math.sqrt(sum(value*value for _,value in pairs)) or 1
        ids=[i for i,_ in pairs]; values=np.array([value/length for _,value in pairs])
        output=[]
        for weights,bias,classes in self.heads:
            scores=weights[:,ids]@values+bias
            if len(classes)==2 and len(scores)==1:
                output.append(classes[int(scores[0]>0)])
            else: output.append(classes[int(np.argmax(scores))])
        return output
