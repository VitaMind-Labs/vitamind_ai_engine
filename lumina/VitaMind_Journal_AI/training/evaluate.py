"""Separate model-only and end-to-end scores. Never conceal the release gate."""
import json
import argparse
from collections import Counter
from .prepare import ROOT, read_jsonl
from journal_ai.model import LocalModel
from journal_ai.pipeline import JournalSentinel

def metrics(rows, predictions, expected_key="tier"):
    labels=("none","low","moderate","moderate_flagged","high")
    matrix={t:{p:0 for p in labels} for t in labels}
    for row,p in zip(rows,predictions):
        matrix[row[expected_key]][p]+=1
    recalls={t:(matrix[t][t]/sum(matrix[t].values()) if sum(matrix[t].values()) else None) for t in labels}
    f1s=[]
    for t in labels:
        if not sum(matrix[t].values()): continue
        tp=matrix[t][t]
        predicted=sum(matrix[y][t] for y in labels)
        actual=sum(matrix[t].values())
        f1s.append(2*tp/(predicted+actual) if predicted+actual else 0)
    false_high=sum(matrix[y]["high"] for y in labels if y!="high")
    actual_nonhigh=sum(sum(matrix[y].values()) for y in labels if y!="high")
    return {"count":len(rows),"accuracy":sum(matrix[t][t] for t in labels)/len(rows) if rows else None,"macro_f1_present_labels":sum(f1s)/len(f1s) if f1s else None,"high_recall":recalls["high"],"false_high_count":false_high,"false_high_rate":false_high/actual_nonhigh if actual_nonhigh else None,"recall_by_tier":recalls,"confusion_matrix":matrix}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--strict",action="store_true",help="Exit nonzero if any uploaded high-tier case is missed")
    args=parser.parse_args()
    model=LocalModel()
    agent=JournalSentinel(model=model)
    reports={}
    for name in ("val","test"):
        rows=read_jsonl(ROOT/"data"/"prepared"/f"{name}.jsonl")
        predictions=[model.predict(r["text"]) for r in rows]
        combined=[agent.analyze(r["text"])["tier"] for r in rows]
        reports[name]={"model_only":metrics(rows,[p.tier for p in predictions]),"pipeline":metrics(rows,combined),"model_by_language":{lang:metrics([r for r in rows if r["lang"]==lang],[p.tier for r,p in zip(rows,predictions) if r["lang"]==lang]) for lang in ("en","ar","mixed")},"context_accuracy":{key:sum(getattr(p,key)==r[key] for r,p in zip(rows,predictions))/len(rows) for key in ("subject","temporal","negated","is_idiom")}}
        reports[name]["pipeline_by_language"]={lang:metrics([r for r in rows if r["lang"]==lang],[p for r,p in zip(rows,combined) if r["lang"]==lang]) for lang in ("en","ar","mixed")}
    challenge=json.loads((ROOT/"data"/"source"/"journal-test-set.json").read_text(encoding="utf-8-sig"))
    outcomes=[agent.analyze(r["text"],include_audit=True) for r in challenge]
    reports["uploaded_challenge"]={"model_only":metrics(challenge,[model.predict(r["text"]).tier for r in challenge],"expectedRisk"),"pipeline":metrics(challenge,[o["tier"] for o in outcomes],"expectedRisk"),"results":[{"id":r["id"],"lang":r["lang"],"expected":r["expectedRisk"],"predicted":o["tier"],"reason":o["audit"]["fusion_reason"]} for r,o in zip(challenge,outcomes)]}
    high=[o["tier"]=="high" for r,o in zip(challenge,outcomes) if r["expectedRisk"]=="high"]
    reports["release_gate"]={"all_uploaded_high_examples_classified_high":bool(high) and all(high),"high_examples":len(high),"caught_high":sum(high),"clinical_use_approved":False,"explanation":"Synthetic regression gate only. Passing would not establish clinical reliability; failing means the draft specification's high-recall gate is unmet."}
    reports["evaluation_notes"]=["Weights and features fit training meanings only; temperatures fit validation meanings only.","Test-family and challenge labels were not used for learned parameter optimization.","The challenge set is visible to developers; support rules were refined after inspecting regression failures. Its score is not a blinded estimate.","Rules intentionally distinguish negation, third-person concern and past history; some dataset policy labels differ."]
    (ROOT/"reports"/"evaluation.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
    for name in ("val","test","uploaded_challenge"):
        print(name, "model:", round(reports[name]["model_only"]["accuracy"],3), "pipeline:",round(reports[name]["pipeline"]["accuracy"],3),"high recall:",reports[name]["pipeline"]["high_recall"])
    print("Release gate:", reports["release_gate"])
    if args.strict and not reports["release_gate"]["all_uploaded_high_examples_classified_high"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()
