"""Score frozen local classifiers after validation selection. Never retrains."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from lumina.adhd.executive_function.intent_classifier import IntentClassifier,rule_intent,protected_intent
from lumina.adhd.executive_function.friction_classifier import FrictionClassifier,classify_friction,RULES

ROOT=Path(__file__).resolve().parents[1]

def summarize(y,p,kind,labels):
    result={'count':len(y),'accuracy':float(accuracy_score(y,p)),'macroF1':float(f1_score(y,p,average='macro',zero_division=0)),'perClass':classification_report(y,p,output_dict=True,zero_division=0,**({'target_names':labels} if kind=='friction' else {}))}
    if kind=='friction': result['microF1']=float(f1_score(y,p,average='micro',zero_division=0))
    return result

def main():
    test_path=ROOT/'data/planning/test.jsonl'
    all_rows=[json.loads(line) for line in test_path.read_text(encoding='utf-8').splitlines()]
    report={'benchmark':'GROUPED_PUBLIC_PLUS_SYNTHETIC_HOLDOUT','testDatasetSha256':hashlib.sha256(test_path.read_bytes()).hexdigest(),'testUsedForSelection':False,'syntheticTestIsIndependentHumanEvaluation':False,'clinicalValidation':False,'limitations':['MASSIVE evaluates three coarsely mapped calendar/nonplanning intents; it does not validate ADHD support.','Synthetic utterances and labels were authored by the same assistant that wrote the system, even though template families and translations are separated. These are not real patient outcomes.','This report evaluates intent/friction classification, not full extraction, calendar date correctness, free conversation or crisis reliability.','A small explicit intent guard and per-language rule fallback are included in deployed results.']}
    for kind,model in [('intent',IntentClassifier()),('friction',FrictionClassifier())]:
        rows=[r for r in all_rows if r[kind] is not None]
        labels=sorted(set(RULES)|{'UNKNOWN'}) if kind=='friction' else sorted({r['intent'] for r in rows})
        predictions=[model.predict(r['text'])[0] for r in rows]
        baseline=[(protected_intent(r['text']) or rule_intent(r['text'])) if kind=='intent' else classify_friction(r['text']) for r in rows]
        if kind=='friction':
            binarizer=MultiLabelBinarizer(classes=labels)
            y=binarizer.fit_transform([r['friction'] for r in rows]); p=binarizer.transform(predictions); b=binarizer.transform(baseline)
        else:
            y=np.array([r['intent'] for r in rows],dtype=object); p=np.array(predictions,dtype=object); b=np.array(baseline,dtype=object)
        item={'deployed':summarize(y,p,kind,labels),'rules':summarize(y,b,kind,labels),'bySource':{},'byLanguage':{},'predictions':[]}
        for field,target in [('source','bySource'),('language','byLanguage')]:
            for value in sorted({r[field] for r in rows}):
                mask=np.array([r[field]==value for r in rows])
                item[target][value]={'deployed':summarize(y[mask],p[mask],kind,labels),'rules':summarize(y[mask],b[mask],kind,labels)}
        for row,prediction,rule in zip(rows,predictions,baseline):
            item['predictions'].append({'id':row['id'],'source':row['source'],'language':row['language'],'expected':row[kind],'prediction':prediction,'rulesPrediction':rule})
        artifact=ROOT/f'models/{kind}/artifact.json'
        item['artifactSha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
        report[kind]=item
    (ROOT/'reports/planning-heldout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({kind:{'count':report[kind]['deployed']['count'],'accuracy':report[kind]['deployed']['accuracy'],'modelMacroF1':report[kind]['deployed']['macroF1'],'rulesMacroF1':report[kind]['rules']['macroF1'],'bySource':{source:{'count':stats['deployed']['count'],'modelMacroF1':stats['deployed']['macroF1'],'rulesMacroF1':stats['rules']['macroF1']} for source,stats in report[kind]['bySource'].items()}} for kind in ('intent','friction')},indent=2))

if __name__=='__main__': main()
