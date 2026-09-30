"""Artifact integration/regressions, separate from the frozen held-out benchmark."""
import hashlib,json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from lumina.api import create_app
from lumina.adhd.executive_function.intent_classifier import IntentClassifier,rule_intent
from lumina.adhd.executive_function.friction_classifier import FrictionClassifier
from lumina.adhd.executive_function.classical_model import ClassicalModel
from lumina.adhd.executive_function.errors import AssistantError

ROOT=Path(__file__).resolve().parents[1]

def test_actual_trained_artifacts_loaded():
    intent=IntentClassifier(); friction=FrictionClassifier()
    assert intent.model.artifact['selected']=='linear_svm'
    assert friction.model.artifact['selected']=='logistic_regression'
    assert intent.model.supports('Show my tasks') and intent.model.supports('اعرض مهامي')
    assert friction.model.supports('I am tired') and friction.model.supports('أنا متعب')

def test_learned_request_routes_beyond_old_rules():
    message='Give me an overview of my commitments today.'
    assert rule_intent(message)=='UNKNOWN'
    predicted,_,source=IntentClassifier().predict(message)
    assert predicted=='ORGANIZE_DAY' and source=='LOCAL_TRAINED_CLASSIFIER'

def test_learned_arabic_friction():
    labels,source=FrictionClassifier().predict('أشعر باستنزاف الطاقة وأحتاج مهمة أخف.')
    assert 'LOW_ENERGY' in labels and source=='LOCAL_TRAINED_CLASSIFIER'

def test_artifact_corruption_is_rejected(tmp_path):
    original=ROOT/'models/intent'
    (tmp_path/'artifact.json').write_bytes((original/'artifact.json').read_bytes()+b' ')
    (tmp_path/'metadata.json').write_bytes((original/'metadata.json').read_bytes())
    with pytest.raises(AssistantError,match='Invalid local'): ClassicalModel('intent',tmp_path)

def test_version_reports_active_models():
    with TestClient(create_app()) as client:
        ready=client.get('/ready').json(); version=client.get('/version').json()
        assert ready['intent']==ready['friction']=='LOCAL_TRAINED_CLASSIFIER'
        assert version['intentModel']=='intent-planning-2.0.0'
        assert version['frictionModel']=='friction-planning-2.0.0'

def test_dataset_preserves_provenance_and_group_boundaries():
    family_splits={}; texts={}
    from lumina.adhd.executive_function.language import normalize
    for row in [json.loads(line) for line in (ROOT/'data/planning/all.jsonl').read_text(encoding='utf-8').splitlines()]:
        assert row['source'] in ('SYNTHETIC','MASSIVE_1.1')
        assert row['reviewed'] is False
        if row['source']=='MASSIVE_1.1': assert row['friction'] is None
        for family in row['family_ids']: family_splits.setdefault(family,set()).add(row['split'])
        texts.setdefault(normalize(row['text']),set()).add(row['split'])
    assert all(len(s)==1 for s in family_splits.values())
    assert all(len(s)==1 for s in texts.values())
