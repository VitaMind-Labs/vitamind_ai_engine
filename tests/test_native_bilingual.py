import json
from pathlib import Path
import re
import pytest
from vitamind.ml.train import load_records, load_corpus
from vitamind.ml.classifier import MiraMLClassifier, PredictionResult
from vitamind.ml.integrated import IntegratedModel
from vitamind.ml.bilingual_text import INPUT_TYPE, build_input, text_key
from vitamind.clinical.patient_state import PatientState
from vitamind.mira.agent import MiraAgent

ROOT = Path(__file__).resolve().parents[1]


def test_delivered_model_is_supervised_in_both_languages():
    classifier = MiraMLClassifier()
    info = classifier.model_info()
    assert info['input_type'] == INPUT_TYPE
    assert set(info['languages']) == {'en', 'ar'}
    assert all(info['supervised_training_by_language'][lang] > 0 for lang in ('en', 'ar'))
    assert info['vocabulary']['word']['arabic_terms'] > 100
    assert info['vocabulary']['word']['latin_terms'] > 100
    assert info['arabic_corpus']['records'] > 0
    assert info['arabic_corpus']['supervised_labels_used'] is False
    for part in ('validation', 'test'):
        assert set(info['evaluation_by_language'][part]) == {'en', 'ar'}


@pytest.mark.parametrize('language,text',[
    ('ar','أفكاري تتدفق بسرعة وأسمع أصواتا لا يسمعها الآخرون'),
    ('en','My attention wanders and I keep misplacing my belongings'),
])
def test_raw_text_reaches_learned_features_without_rule_extraction(language,text):
    state = PatientState.new('adult', language=language)
    classifier = MiraMLClassifier()
    result = IntegratedModel(classifier).assess(state,text=text)
    assert result['status'] == 'available'
    assert result['native_arabic_model'] is True
    assert result['input_type'] == INPUT_TYPE
    assert not result['features_used']
    assert not state.observations
    vectors = classifier._load()['routing']['word_model'].steps[0][1].transform([result['model_input']])
    assert vectors.nnz > 0
    if language == 'ar':
        assert re.search(r'[\u0621-\u064a]', result['model_input'])
        assert not re.search(r'[a-zA-Z]', result['model_input'])


def test_a_learned_suggestion_asks_for_confirmation_without_inventing_evidence():
    class Learned:
        def model_info(self): return {'input_type': INPUT_TYPE, 'languages':['en','ar']}
        def predict(self,text):
            return PredictionResult('PSYCHOSIS','clinical_review',None,.91,'OTHER',.06,.85,{'PSYCHOSIS':.91})
    agent = MiraAgent(model=IntegratedModel(Learned()))
    session,_ = agent.create_session(language='ar')
    reply = agent.respond(session,'تحدث أمور غريبة يصعب علي وصفها')
    assert reply.model_assessment['unconfirmed_followup_only']
    assert session.pending.key == 'voices'
    assert not session.state.observations
    agent.respond(session,'لا')
    result = agent.respond(session,'تقرير').result
    assert result['candidate_patterns'] == []


def test_other_person_raw_text_is_not_a_patient_model_input():
    state = PatientState.new('adult')
    text,features = build_input(state,'My sister is hearing things when alone.')
    assert text == '' and features == []


def test_absent_and_conflicting_evidence_are_not_native_anchors():
    state = PatientState.new('adult',language='ar')
    state.add_observation('attention','distractibility','absent',1,'synthetic')
    for value in ('present','absent'):
        state.add_observation('psychosis','persecutory_ideas',value,1,'synthetic')
    assert build_input(state) == ('',[])


def test_bilingual_pairs_and_normalized_texts_do_not_cross_splits():
    rows = load_records(ROOT/'data/mira_bilingual.jsonl')
    paired = {}
    for row in rows:
        assert row['synthetic'] and not row['clinician_reviewed']
        assert paired.setdefault(row['paired_state_id'],row['split']) == row['split']
    assert len({r['language'] for r in rows}) == 2


@pytest.mark.parametrize('bad_field,bad_value',[('split','test'),('label','5'),('input_type','mixed_chat')])
def test_vocabulary_import_does_not_accept_source_labels_or_evaluation_rows(tmp_path,bad_field,bad_value):
    row={'text':'أشعر بالقلق عندما أخرج من المنزل','split':'train','language':'ar','input_type':'unlabeled_patient_opening'}
    row[bad_field]=bad_value
    path=tmp_path/'corpus.jsonl';path.write_text(json.dumps(row),encoding='utf-8')
    with pytest.raises(ValueError): load_corpus(path,[])


def test_arabic_normalization_deduplicates_diacritics_and_punctuation():
    assert text_key('أَنا، مُتعب.') == text_key('انا متعب')


@pytest.mark.parametrize('language,text',[('ar','مرحبا'),('ar','ماذا يمكنك أن تفعل'),('en','hello'),('en','what can you do')])
def test_greeting_and_capabilities_do_not_consume_assessment_answers(language,text):
    agent=MiraAgent();session,_=agent.create_session(language=language)
    old=session.pending
    reply=agent.respond(session,text)
    assert session.pending == old and session.messages == 0
    assert reply.text
