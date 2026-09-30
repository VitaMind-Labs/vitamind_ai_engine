"""Synthetic engineering fixtures verify export parity, not model quality."""
import hashlib,json
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from lumina.adhd.executive_function.language import normalize
from lumina.adhd.executive_function.classical_model import ClassicalModel
from training.train_classifiers import export_head
from lumina.adhd.executive_function.friction_classifier import RULES

@pytest.mark.parametrize('kind',['logistic_regression','linear_svm','naive_bayes'])
def test_local_json_inference_matches_sklearn(tmp_path,kind):
    texts=['add email James','add call Mary','اضف مكالمة البنك','اضف ارسال رسالة','plan my day','prioritize my tasks','رتب مهام اليوم','خطط ليومي']
    labels=['ADD_TASK']*4+['ORGANIZE_DAY']*4
    vec=TfidfVectorizer(preprocessor=normalize,lowercase=False,token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2),sublinear_tf=True)
    x=vec.fit_transform(texts)
    model={'logistic_regression':LogisticRegression(),'linear_svm':LinearSVC(),'naive_bayes':MultinomialNB()}[kind].fit(x,labels)
    artifact={'format':'lumina-classical-v1','normalizer':'en-ar-1.0.0','selected':kind,'task':'intent','vocabulary':vec.vocabulary_,'idf':vec.idf_.tolist(),'heads':[export_head(model)],'labels':list(model.classes_)}
    raw=json.dumps(artifact,ensure_ascii=False).encode('utf-8')
    (tmp_path/'artifact.json').write_bytes(raw)
    (tmp_path/'metadata.json').write_text(json.dumps({'checksum':hashlib.sha256(raw).hexdigest(),'languageCoverage':['EN','AR','MIXED']}),encoding='utf-8')
    runtime=ClassicalModel('intent',tmp_path)
    probes=texts+['email email email James','اضف رسالة جديدة','plan مهام اليوم','unseenword']
    expected=model.predict(vec.transform(probes)).tolist()
    assert [runtime.predict(text)[0] for text in probes]==expected

@pytest.mark.parametrize('task',['intent','friction'])
def test_future_training_pipeline_with_toy_labels(tmp_path,monkeypatch,task):
    from training import train_classifiers as trainer
    # Every independent toy group has every friction class. This exercises code;
    # it is NOT a reviewed ADHD dataset or evidence of real-world performance.
    rows=[]
    labels=sorted(set(RULES)|{'UNKNOWN'})
    for group in range(20):
        for index,label in enumerate(labels):
            rows.append({'id':f'{group}-{index}','text':f'signal{index} variant{group}','language':'EN','group_id':str(group),'intent':['ADD_TASK','ORGANIZE_DAY'][index%2],'friction':[label],'reviewed':True})
    path=tmp_path/'engineering-fixture.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    monkeypatch.setattr(trainer,'ROOT',tmp_path)
    trainer.train(path,task)
    folder=tmp_path/'models'/task
    runtime=ClassicalModel(task,folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert metadata['metrics']['test']['macroF1'] is not None
    assert runtime.supports('signal1')
    if task=='intent': assert runtime.predict('signal1')[0]=='ORGANIZE_DAY'
    else:
        predicted=[label for label,value in zip(runtime.artifact['labels'],runtime.predict('signal1')) if value=='1']
        assert predicted==[labels[1]]
