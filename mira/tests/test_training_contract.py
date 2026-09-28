import json
import pytest
from vitamind.ml.train import load_records,train_model,LABELS
from vitamind.ml.classifier import MiraMLClassifier

def records():
    # Deliberately tiny engineering fixture, not psychiatric training evidence.
    phrases={'ADHD':'distracted forgetful childhood attention','BIPOLAR':'energy episodes sleep spending',
             'PSYCHOSIS':'voices perceptions beliefs experiences','AMBIGUOUS':'unclear uncertain mixed incomplete',
             'OTHER':'context illness medicine alternative','HEALTHY':'no reported target difficulties'}
    return [{'text':f'{phrases[label]} fixture {split} {i}', 'label':label,'split':split,
             'group_id':f'{label}-{split}-{i}','input_type':'structured_evidence_projection'}
            for split in ['train','validation','test'] for label in sorted(LABELS) for i in range(2)]

def test_training_roundtrip_and_separate_test_report(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text('\n'.join(json.dumps(r) for r in records()),encoding='utf-8')
    out=tmp_path/'model.joblib';result=train_model(path,out)
    assert result['records']=={'train':12,'validation':12,'test':12}
    assert result['safety_model_trained'] is False
    prediction=MiraMLClassifier(out).predict('distracted forgetful childhood')
    assert prediction.routing_label in LABELS and prediction.crisis_flag is None

@pytest.mark.parametrize('kind',['group','duplicate','format'])
def test_training_rejects_leakage_and_wrong_input(tmp_path,kind):
    rows=records()
    if kind=='group': rows[-1]['group_id']=rows[0]['group_id']
    if kind=='duplicate': rows[-1]['text']=rows[0]['text']
    if kind=='format': rows[0]['input_type']='raw_mixed_chat'
    path=tmp_path/'data.jsonl';path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    with pytest.raises(ValueError): load_records(path)
