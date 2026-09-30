"""Train small local models from scratch. No model downloads, APIs or fine-tuning."""
import argparse,collections,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,f1_score,classification_report,precision_score
from lumina.adhd.executive_function.language import normalize
from lumina.adhd.executive_function.text_features import feature_tokens,VERSION
from lumina.adhd.executive_function.intent_classifier import rule_intent,protected_intent
from lumina.adhd.executive_function.friction_classifier import classify_friction,RULES
from .train_classifiers import export_head

ROOT=Path(__file__).resolve().parents[1]

def read_data(allow_synthetic=False):
    all_rows=[json.loads(line) for line in (ROOT/'data/planning/all.jsonl').read_text(encoding='utf-8').splitlines()]
    if not allow_synthetic and any(r['source']=='SYNTHETIC' for r in all_rows):
        raise ValueError('Pass --allow-synthetic to explicitly train on unreviewed synthetic labels.')
    families=collections.defaultdict(set); texts=collections.defaultdict(set)
    for row in all_rows:
        for family in row['family_ids']: families[family].add(row['split'])
        texts[normalize(row['text'])].add(row['split'])
    if any(len(x)>1 for x in families.values()): raise ValueError('Family overlap across splits')
    if any(len(x)>1 for x in texts.values()): raise ValueError('Exact text overlap across splits')
    return all_rows

def metrics(y,p,kind,labels):
    result={'count':len(y),'accuracy':float(accuracy_score(y,p)),'macroF1':float(f1_score(y,p,average='macro',zero_division=0)),'perClass':classification_report(y,p,output_dict=True,zero_division=0,**({'target_names':labels} if kind=='friction' else {}))}
    if kind=='friction': result['microF1']=float(f1_score(y,p,average='micro',zero_division=0))
    return result

def train(kind,all_rows):
    rows={split:[r for r in all_rows if r['split']==split and r[kind] is not None] for split in ('train','validation')}
    tr,va=rows['train'],rows['validation']
    texts=[r['text'] for r in tr]
    vectorizer=TfidfVectorizer(analyzer=feature_tokens,sublinear_tf=True,max_features=16000,min_df=2,dtype=np.float64)
    xtr=vectorizer.fit_transform(texts); xva=vectorizer.transform([r['text'] for r in va])
    if kind=='intent':
        ytr=np.array([r['intent'] for r in tr],dtype=object); yva=np.array([r['intent'] for r in va],dtype=object)
        labels=sorted(set(ytr)); baseline=np.array([protected_intent(r['text']) or rule_intent(r['text']) for r in va],dtype=object)
        if not set(yva)<=set(ytr): raise ValueError('Unseen label in validation')
    else:
        binarizer=MultiLabelBinarizer(classes=sorted(set(RULES)|{'UNKNOWN'}))
        ytr=binarizer.fit_transform([r['friction'] for r in tr]); yva=binarizer.transform([r['friction'] for r in va]); labels=list(binarizer.classes_)
        baseline=binarizer.transform([classify_friction(r['text']) for r in va])
        if any(len(set(ytr[:,i]))<2 for i in range(len(labels))): raise ValueError('Missing positive or negative training examples for a friction label')
    fitted={}; predictions={'rules':baseline}; thresholds={}
    comparison={'rules':metrics(yva,baseline,kind,labels)}
    candidates={'logistic_regression':LogisticRegression(C=4,max_iter=1200,class_weight='balanced',random_state=42),'linear_svm':LinearSVC(C=1,class_weight='balanced',random_state=42),'naive_bayes':MultinomialNB(alpha=.3)}
    for name,base in candidates.items():
        model=base if kind=='intent' else OneVsRestClassifier(base)
        model.fit(xtr,ytr)
        predicted=model.predict(xva)
        if kind=='intent':
            predicted=predicted.astype(object)
            for i,row in enumerate(va): predicted[i]=protected_intent(row['text']) or predicted[i]
        if kind=='friction':
            unknown=labels.index('UNKNOWN')
            scores=model.decision_function(xva) if name=='linear_svm' else model.predict_proba(xva)
            grid=[-.8,-.6,-.4,-.2,0,.2] if name=='linear_svm' else [.10,.20,.30,.40,.50,.60]
            options=[]
            for threshold in grid:
                p=(scores>threshold).astype(int)
                for i,row in enumerate(p):
                    if row.sum()>1: p[i,unknown]=0
                    if not p[i].any(): p[i,unknown]=1
                precision=precision_score(yva,p,average='micro',zero_division=0)
                # A single global validation threshold avoids fitting one threshold
                # to each tiny class. Do not choose a model predicting every label.
                if precision>=.45 and p.sum(axis=1).mean()<=3:
                    options.append((f1_score(yva,p,average='macro',zero_division=0),threshold,p))
            if options: _,threshold,predicted=max(options,key=lambda row:(row[0],row[1]))
            else:
                threshold=0 if name=='linear_svm' else .5
                predicted=(scores>threshold).astype(int)
                for i,row in enumerate(predicted):
                    if row.sum()>1: predicted[i,unknown]=0
                    if not predicted[i].any(): predicted[i,unknown]=1
            thresholds[name]=threshold
        fitted[name]=model; predictions[name]=predicted
        comparison[name]=metrics(yva,predicted,kind,labels)
    best=max(comparison,key=lambda name:comparison[name]['macroF1'])
    coverage=[]; language_validation={}
    for lang in ('EN','AR','MIXED'):
        mask=np.array([r['language']==lang for r in va]); train_count=sum(r['language']==lang for r in tr)
        if mask.any():
            language_validation[lang]={'trainCount':train_count,'validationCount':int(mask.sum()),'rulesMacroF1':float(f1_score(yva[mask],baseline[mask],average='macro',zero_division=0)),'candidateMacroF1':float(f1_score(yva[mask],predictions[best][mask],average='macro',zero_division=0))}
            if best!='rules' and train_count>=20 and mask.sum()>=20 and language_validation[lang]['candidateMacroF1']>language_validation[lang]['rulesMacroF1']:
                coverage.append(lang)
    if not coverage: best='rules'
    actual=predictions[best].copy()
    for i,row in enumerate(va):
        if row['language'] not in coverage: actual[i]=baseline[i]
    validation_by_source={}
    for source in sorted({r['source'] for r in va}):
        mask=np.array([r['source']==source for r in va])
        validation_by_source[source]=metrics(yva[mask],actual[mask],kind,labels)
    artifact={'format':'lumina-classical-v2','normalizer':'en-ar-1.0.0','analyzer':VERSION,'selected':best,'task':kind,'vocabulary':{key:int(value) for key,value in vectorizer.vocabulary_.items()} if best!='rules' else {},'idf':vectorizer.idf_.tolist() if best!='rules' else [],'labels':labels,'heads':[]}
    if best!='rules':
        artifact['heads']=[export_head(fitted[best])] if kind=='intent' else [export_head(m) for m in fitted[best].estimators_]
        if kind=='friction':
            t=thresholds[best]
            shift=t if best=='linear_svm' else float(np.log(t/(1-t)))
            for head in artifact['heads']:
                target=0 if len(head['intercept'])==1 else head['classes'].index('1')
                head['intercept'][target]-=shift
            artifact['validationThreshold']=t
    folder=ROOT/'models'/kind; folder.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(artifact,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    (folder/'artifact.json').write_bytes(raw)
    source_hash=hashlib.sha256((ROOT/'data/planning/all.jsonl').read_bytes()).hexdigest()
    report={'selected':best,'features':xtr.shape[1],'trainingRows':len(tr),'validationRows':len(va),'candidateValidation':comparison,'frictionGlobalThresholds':thresholds,'languageValidation':language_validation,'deployedValidation':metrics(yva,actual,kind,labels),'validationBySource':validation_by_source,'testStatus':'NOT_READ_DURING_SELECTION','selection':'Best validation macro F1; per-language deployment requires improvement over rules and at least 20 train/validation examples. Otherwise rule fallback. This is not statistical or clinical validation.'}
    metadata={'modelVersion':kind+'-planning-2.0.0','datasetVersion':source_hash,'trainingDate':datetime.now(timezone.utc).isoformat(),'languageCoverage':coverage,'languagesInDataset':sorted({r['language'] for r in tr}),'featureMethod':'TF-IDF word unigrams/bigrams and within-word character 3-5 grams','metrics':report,'checksum':hashlib.sha256(raw).hexdigest(),'status':'TRAINED_ON_PUBLIC_AND_SYNTHETIC' if kind=='intent' and best!='rules' else 'TRAINED_ON_SYNTHETIC' if best!='rules' else 'RULE_BASELINE_SELECTED','humanReviewed':False,'clinicalValidation':False,'testMetricsFile':'reports/planning-heldout.json'}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'task':kind,'selected':best,'coverage':coverage,'trainingRows':len(tr),'validationRows':len(va),'validationMacroF1':{name:round(value['macroF1'],4) for name,value in comparison.items()}},indent=2),flush=True)
    # Verify the exported artifact matches the selected estimator on validation.
    if best!='rules':
        from lumina.adhd.executive_function.classical_model import ClassicalModel
        runtime=ClassicalModel(kind,folder)
        checks=range(min(60,len(va)))
        for i in checks:
            output=runtime.predict(va[i]['text'])
            if kind=='intent': assert output[0]==fitted[best].predict(xva[i])[0]
            else:
                positive={label for label,value in zip(labels,output) if value=='1'}
                if len(positive)>1: positive.discard('UNKNOWN')
                assert (positive or {'UNKNOWN'})=={label for label,value in zip(labels,predictions[best][i]) if value==1}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--allow-synthetic',action='store_true'); parser.add_argument('--task',choices=['intent','friction','both'],default='both')
    args=parser.parse_args(); rows=read_data(args.allow_synthetic)
    for task in (('intent','friction') if args.task=='both' else (args.task,)): train(task,rows)

if __name__=='__main__': main()
