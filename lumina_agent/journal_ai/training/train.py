"""Train a small multi-head classifier locally with NumPy; no key or GPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from journal_ai.features import fit_features, vectorize
from journal_ai.model import softmax, DEFAULT_MODEL
from journal_ai.normalize import VERSION
from journal_ai.schema import TRAIN_TIERS, CATEGORIES
from .prepare import ROOT, prepare

def make_heads():
    definitions = {"tier":list(TRAIN_TIERS),"subject":["self","other"],"temporal":["current","past"],"negated":["false","true"],"is_idiom":["false","true"]}
    definitions.update({"category:"+c:["false","true"] for c in CATEGORIES})
    heads, offset = {}, 0
    for name,labels in definitions.items():
        heads[name] = {"start":offset,"end":offset+len(labels),"labels":labels,"temperature":1.0}
        offset += len(labels)
    return heads, offset

def label(row, name):
    value = name.split(":")[1] in row["categories"] if name.startswith("category:") else row[name]
    return str(value).lower() if isinstance(value,bool) else value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--epochs",type=int,default=28)
    parser.add_argument("--max-features",type=int,default=12000)
    args=parser.parse_args()
    splits,audit=prepare(args.seed)
    train=splits["train"]
    spec=fit_features([r["text"] for r in train],args.max_features)
    heads,width=make_heads()
    weights=np.zeros((len(spec["vocabulary"]),width),dtype=np.float64)
    bias=np.zeros(width,dtype=np.float64)
    vectors=[(np.array(i,dtype=np.int64),np.array(v)) for i,v in (vectorize(r["text"],spec) for r in train)]
    targets={name:np.array([head["labels"].index(label(r,name)) for r in train]) for name,head in heads.items()}
    class_weights={}
    for name,h in heads.items():
        count=np.bincount(targets[name],minlength=len(h["labels"]))
        class_weights[name]=np.clip(np.sqrt(len(train)/(len(count)*np.maximum(count,1))),0.5,3)
    class_weights["tier"][3] *= 3
    rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        lr=0.18/math.sqrt(1+epoch/4)
        loss=0.0
        for index in rng.permutation(len(train)):
            ids,values=vectors[index]
            logits=values @ weights[ids] + bias
            grad=np.zeros(width)
            for name,h in heads.items():
                start,end=h["start"],h["end"]
                p=softmax(logits[start:end])
                target=targets[name][index]
                loss-=math.log(max(p[target],1e-12))
                p[target]-=1
                grad[start:end]=p*class_weights[name][target]
            weights[ids] -= lr * (values[:,None]*grad[None,:] + 0.0001*weights[ids])
            bias -= lr*grad*0.15
        if epoch==0 or (epoch+1)%7==0:
            print(f"epoch {epoch+1}/{args.epochs}: mean head loss {loss/(len(train)*len(heads)):.4f}",flush=True)
    # Separate grouped validation partition only. Temperature is not a clinical calibration.
    val_logits=[]
    for row in splits["val"]:
        ids,values=vectorize(row["text"],spec)
        val_logits.append(np.asarray(values) @ weights[ids]+bias)
    for name,h in heads.items():
        y=[h["labels"].index(label(r,name)) for r in splits["val"]]
        def nll(t):
            return sum(-math.log(max(softmax(z[h["start"]:h["end"]]/t)[k],1e-12)) for z,k in zip(val_logits,y))/len(y)
        h["temperature"]=float(min((0.5,0.75,1,1.25,1.5,2,3,4),key=nll))
    folder=DEFAULT_MODEL
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"weights.npz",weights=weights.astype(np.float32),bias=bias.astype(np.float32))
    config={"format":"journal-linear-v1","version":f"journal-linear-seed{args.seed}-v1","normalization":VERSION,"heads":heads,"features":spec,"weights_sha256":hashlib.sha256((folder/"weights.npz").read_bytes()).hexdigest(),"training":{"seed":args.seed,"epochs":args.epochs,"rows":len(train),"validation_rows":len(splits["val"]),"test_rows":len(splits["test"]),"high_tier_loss_multiplier":3,"source_hashes":audit["source_sha256"]},"not_clinically_validated":True}
    (folder/"config.json").write_text(json.dumps(config,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(f"Saved trained model: {folder}",flush=True)

if __name__=="__main__":
    main()
