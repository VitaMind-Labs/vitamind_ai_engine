"""Verify the supplied synthetic file against the trainer AND live projection."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vitamind.clinical.feature_extractor import RULES
from vitamind.clinical.patient_state import PatientState
from vitamind.ml.classifier import PredictionResult
from vitamind.ml.integrated import IntegratedModel
from vitamind.ml.train import load_records, LABELS
from generate_synthetic_projections import assign_label, group_and_split


class CaptureClassifier:
    """Capture the real model input without depending on sklearn or weights."""
    input_type = 'structured_evidence_projection'

    def predict(self, text):
        self.seen = text
        return PredictionResult('AMBIGUOUS', 'qualified_clinical_assessment', None,
                                .4, 'OTHER', .3, .1, {'AMBIGUOUS': .4, 'OTHER': .3})


def validate(path):
    records = load_records(path)
    card = json.loads(Path(path).with_suffix('.data_card.json').read_text(encoding='utf-8'))
    assert sha256(Path(path).read_bytes()).hexdigest() == card['sha256'], 'Dataset checksum mismatch'
    assert len(records) == card['records'], 'Record count mismatch'
    capture = CaptureClassifier()
    model = IntegratedModel(capture)
    by_feature = {rule.feature: rule for rule in RULES}
    ids = set()
    for row in records:
        assert row['record_id'] not in ids, 'Duplicate record ID'
        ids.add(row['record_id'])
        assert row['synthetic'] is True and row['clinician_reviewed'] is False
        assert row['clinical_validation'] is False
        features = set(row['present_features'])
        assert len(features) == len(row['present_features'])
        assert features <= by_feature.keys(), 'Unknown feature'
        assert row['label'] == assign_label(features), 'Fixture label policy mismatch'
        group, split = group_and_split(features, card['seed'])
        assert (row['group_id'], row['split']) == (group, split)
        # Call the real live projection in both supported session languages.
        # These are state-to-model checks, not natural-language comprehension tests.
        for language in ('en', 'ar'):
            state = PatientState.new('adult', language=language)
            for feature in reversed(row['present_features']):
                rule = by_feature[feature]
                state.add_observation(rule.domain, rule.feature, 'present', 1, 'synthetic fixture')
            unused = [rule for rule in RULES if rule.feature not in features]
            # Denials and conflicting evidence must never enter the model text.
            if unused:
                rule = unused[0]
                state.add_observation(rule.domain, rule.feature, 'absent', 1, 'synthetic denial')
            if len(unused) > 1:
                rule = unused[1]
                for polarity in ('present', 'absent'):
                    state.add_observation(rule.domain, rule.feature, polarity, 1, 'synthetic conflict')
            result = model.assess(state)
            assert result['status'] == 'available'
            assert capture.seen == row['text'] == result['model_input'], 'Live input mismatch'
            assert result['features_used'] == row['present_features']
    for split in ('train', 'validation', 'test'):
        assert {r['label'] for r in records if r['split'] == split} == LABELS
    empty = model.assess(PatientState.new('adult'))
    assert empty['status'] == 'insufficient_evidence'
    return {'passed': True, 'records': len(records), 'live_projection_checks': 2 * len(records),
            'splits': dict(Counter(r['split'] for r in records)),
            'duplicate_texts': 0, 'groups_crossing_partitions': 0,
            'clinician_reviewed': False, 'clinical_validation': False,
            'checks': ['trainer schema', 'record IDs', 'checksum', 'fixture-label consistency',
                       'group isolation including secondary-feature variations',
                       'all six labels in every split', 'exact live en/ar state-to-model projection',
                       'absent/conflicting evidence exclusion', 'empty-evidence bypass'],
            'does_not_test': ['real language understanding', 'clinical validity', 'model training quality']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/reviewed_projections.jsonl')
    args = parser.parse_args()
    print(json.dumps(validate(args.data), indent=2))
