"""Compare rules, logistic regression, LinearSVC and NB on future labeled data.

Run only after independent labels exist. No pretrained model or API is used.
Artifacts are promoted only if validation macro F1 exceeds the rule baseline.
"""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from typing import get_args
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,f1_score,classification_report
from lumina.adhd.executive_function.language import normalize
from lumina.adhd.executive_function.intent_classifier import rule_intent
from lumina.adhd.executive_function.friction_classifier import classify_friction,RULES
from lumina.adhd.executive_function.schemas import Intent,Friction

ROOT=Path(__file__).resolve().parents[1]

def load_labeled(path):
    if path.suffix.lower()!='.jsonl': raise ValueError('Use labeled planning JSONL. The supplied FAQ CSV has no intent/friction labels.')
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    required=('text','language','group_id','intent','friction')
    for i,r in enumerate(rows):
        if any(k not in r or r[k] is None for k in required): raise ValueError(f'Row {i+1} lacks reviewed planning labels: {required}')
        if r['language'] not in ('EN','AR','MIXED') or not isinstance(r['friction'],list) or not r['text'].strip(): raise ValueError(f'Invalid row {i+1}')
        if r['intent'] not in get_args(Intent) or any(f not in get_args(Friction) for f in r['friction']): raise ValueError(f'Unknown intent/friction label on row {i+1}')
        if r.get('reviewed') is not True: raise ValueError(f'Row {i+1} must be independently reviewed and marked reviewed: true.')
    if len(rows)<60 or len({r['group_id'] for r in rows})<15: raise ValueError('At least 60 labeled rows across 15 independent meaning groups are required for this comparison; this is an engineering minimum, not validation sufficiency.')
    # Refuse exact duplicates across claimed groups. All translations/context variants
    # must already share group_id; the human review contract is mandatory.
    seen={}
    for r in rows:
        key=normalize(r['text'])
        if key in seen and seen[key]!=r['group_id']: raise ValueError('Duplicate text crosses group IDs; repair the annotation grouping before training.')
        seen[key]=r['group_id']
    return rows

def export_head(model):
    if isinstance(model,MultinomialNB):
        return {'kind':'nb','coef':model.feature_log_prob_.tolist(),'intercept':model.class_log_prior_.tolist(),'classes':[str(x) for x in model.classes_]}
    return {'kind':'linear','coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':[str(x) for x in model.classes_]}

def train(path,kind):
    rows=load_labeled(path)
    groups=[r['group_id'] for r in rows]
    indices=np.arange(len(rows))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(indices,groups=groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=43).split(trainval,groups=[groups[i] for i in trainval]))
    tr,va=trainval[tr],trainval[va]
    texts=[r['text'] for r in rows]
    vec=TfidfVectorizer(preprocessor=normalize,lowercase=False,token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2),sublinear_tf=True,max_features=15000)
    xtr=vec.fit_transform([texts[i] for i in tr]); xva=vec.transform([texts[i] for i in va]); xte=vec.transform([texts[i] for i in test])
    if kind=='intent':
        y=np.array([r['intent'] for r in rows]); labels=sorted(set(y))
        if set(y[tr])!=set(y): raise ValueError('Training groups do not cover every label; supply more independent groups.')
        baseline=np.array([rule_intent(texts[i]) for i in va])
    else:
        mlb=MultiLabelBinarizer(classes=sorted(set(RULES)|{'UNKNOWN'}))
        y=mlb.fit_transform([r['friction'] or ['UNKNOWN'] for r in rows]); labels=list(mlb.classes_)
        if any(len(set(y[tr,j]))<2 for j in range(y.shape[1])): raise ValueError('Every friction label needs positive and negative training examples.')
        baseline=mlb.transform([classify_friction(texts[i]) for i in va])
    baseline_score=f1_score(y[va],baseline,average='macro',zero_division=0)
    candidates={'logistic_regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),'linear_svm':LinearSVC(class_weight='balanced',random_state=42),'naive_bayes':MultinomialNB()}
    fitted={}; metrics={'rules':{'validationMacroF1':float(baseline_score)}}
    for name,base in candidates.items():
        model=base if kind=='intent' else OneVsRestClassifier(base)
        model.fit(xtr,y[tr]); pred=model.predict(xva)
        fitted[name]=model
        metrics[name]={'validationMacroF1':float(f1_score(y[va],pred,average='macro',zero_division=0))}
    best=max(metrics,key=lambda name:metrics[name]['validationMacroF1'])
    coverage=[]; language_validation={}
    # A model selected overall must also beat rules on a language's validation
    # examples before that language can use it. Test labels never set coverage.
    if best!='rules':
        validation_pred=fitted[best].predict(xva)
        for language in ('EN','AR','MIXED'):
            mask=np.array([rows[i]['language']==language for i in va])
            if mask.any():
                rule_score=float(f1_score(y[va][mask],baseline[mask],average='macro',zero_division=0))
                model_score=float(f1_score(y[va][mask],validation_pred[mask],average='macro',zero_division=0))
                language_validation[language]={'count':int(mask.sum()),'rulesMacroF1':rule_score,'modelMacroF1':model_score}
                if model_score>rule_score and any(rows[i]['language']==language for i in tr): coverage.append(language)
    if best=='rules':
        pred=np.array([rule_intent(texts[i]) for i in test]) if kind=='intent' else mlb.transform([classify_friction(texts[i]) for i in test])
    else:
        pred=fitted[best].predict(xte)
        if kind=='intent': pred=pred.astype(object)
        for j,i in enumerate(test):
            if rows[i]['language'] not in coverage:
                pred[j]=rule_intent(texts[i]) if kind=='intent' else mlb.transform([classify_friction(texts[i])])[0]
    report={'selected':best,'validation':metrics,'validationByLanguage':language_validation,'testPolicy':'Selected model for validated languages, rule fallback elsewhere, matching runtime.','test':{'accuracy':float(accuracy_score(y[test],pred)),'macroF1':float(f1_score(y[test],pred,average='macro',zero_division=0)),'perClass':classification_report(y[test],pred,output_dict=True,zero_division=0,**({'target_names':labels} if kind=='friction' else {}))},'testByLanguage':{},'splitIds':{k:[rows[i].get('id',str(i)) for i in inds] for k,inds in [('train',tr),('validation',va),('test',test)]},'limitations':['No automatic proof of semantic independence: reviewed group_id values must include all paraphrases, translations and same-patient histories.','Test is evaluated only after selecting by validation. No thresholds are selected on test.','Per-language validation selection is an engineering rule, not proof of statistical or clinical reliability; inspect sample counts.']}
    if kind=='friction': report['test']['microF1']=float(f1_score(y[test],pred,average='micro',zero_division=0))
    for language in ('EN','AR','MIXED'):
        mask=np.array([rows[i]['language']==language for i in test])
        if mask.any(): report['testByLanguage'][language]={'count':int(mask.sum()),'macroF1':float(f1_score(y[test][mask],pred[mask],average='macro',zero_division=0))}
    folder=ROOT/'models'/kind; folder.mkdir(exist_ok=True,parents=True)
    artifact={'format':'lumina-classical-v1','normalizer':'en-ar-1.0.0','selected':best,'task':kind,'vocabulary':{k:int(v) for k,v in vec.vocabulary_.items()} if best!='rules' else {},'idf':vec.idf_.tolist() if best!='rules' else [],'labels':labels,'heads':[]}
    if best!='rules':
        artifact['heads']=[export_head(fitted[best])] if kind=='intent' else [export_head(m) for m in fitted[best].estimators_]
    raw=json.dumps(artifact,separators=(',',':'),ensure_ascii=False).encode()
    (folder/'artifact.json').write_bytes(raw)
    metadata={'modelVersion':kind+'-1.0.0','datasetVersion':hashlib.sha256(path.read_bytes()).hexdigest(),'trainingDate':datetime.now(timezone.utc).isoformat(),'languageCoverage':coverage,'languagesInDataset':sorted({r['language'] for r in rows}),'featureMethod':'TF-IDF word unigrams/bigrams; local classical models','metrics':report,'checksum':hashlib.sha256(raw).hexdigest(),'status':'RULE_BASELINE_SELECTED' if best=='rules' or not coverage else 'TRAINED_SELECTED_ON_VALIDATION'}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'selected':best,'test':report['test']},indent=2))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True); parser.add_argument('--task',choices=['intent','friction'],required=True)
    args=parser.parse_args()
    try: train(args.data,args.task)
    except (ValueError,KeyError,OSError) as exc: parser.exit(2,f'Cannot train: {exc}\n')

if __name__=='__main__': main()
