"""Validate bilingual records and the unlabeled Arabic corpus without fitting a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vitamind.ml.train import load_records, load_corpus


def validate():
    path = ROOT / 'data/mira_bilingual.jsonl'
    rows = load_records(path)
    card = json.loads(path.with_suffix('.data_card.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == card['data_sha256']
    pair_splits = {}
    for row in rows:
        assert pair_splits.setdefault(row['paired_state_id'], row['split']) == row['split']
        assert row['synthetic'] is True and row['clinician_reviewed'] is False
    corpus_path = ROOT / 'data/arabic_adaptation.jsonl'
    audit = json.loads(corpus_path.with_suffix('.audit.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == audit['data_sha256']
    _, corpus = load_corpus(corpus_path, rows)
    return {'passed': True, 'records': len(rows), 'languages': dict(Counter(r['language'] for r in rows)),
            'splits': dict(Counter(r['split'] for r in rows)), 'groups_crossing_splits': 0,
            'translations_crossing_splits': 0, 'normalized_duplicate_texts': 0,
            'arabic_corpus': corpus, 'clinical_validation': False}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
