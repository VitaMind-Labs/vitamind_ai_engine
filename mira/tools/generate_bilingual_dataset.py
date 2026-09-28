"""Generate native English/Arabic routing fixtures with paired split isolation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vitamind.ml.bilingual_text import FEATURE_TEXT, INPUT_TYPE, text_key


def generate(source, output, seed=42):
    rng = random.Random(seed)
    states = [json.loads(line) for line in Path(source).read_text(encoding='utf-8').splitlines() if line.strip()]
    records, seen = [], {}
    for state in states:
        for language in ('en', 'ar'):
            for view in range(2):
                sentences = [rng.choice(FEATURE_TEXT[feature][language]) for feature in state['present_features']]
                rng.shuffle(sentences)
                text = '. '.join(sentences) + '.'
                key = text_key(text)
                if key in seen:
                    assert seen[key] == (state['label'], state['split']), 'Conflicting/overlapping paired fixture'
                    continue
                seen[key] = state['label'], state['split']
                records.append({'text': text, 'label': state['label'], 'language': language,
                                'split': state['split'], 'group_id': state['group_id'],
                                'input_type': INPUT_TYPE, 'synthetic': True, 'clinician_reviewed': False,
                                'label_source': 'author_defined_fixture_policy_v1',
                                'record_id': hashlib.sha256((language + key).encode()).hexdigest()[:20],
                                'paired_state_id': state['record_id']})
    rng.shuffle(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    card = {'input_type': INPUT_TYPE, 'languages': ['en', 'ar'], 'seed': seed,
            'records': len(records), 'synthetic': True, 'clinician_reviewed': False,
            'source_projection_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'data_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'counts': {s: {lang: dict(Counter(r['label'] for r in records if r['split'] == s and r['language'] == lang))
                           for lang in ('en', 'ar')} for s in ('train', 'validation', 'test')},
            'split_policy': 'All English/Arabic views inherit their source scenario-family partition.',
            'clinical_validation': False,
            'limitations': ['Author-defined synthetic routing policy, not reviewed diagnostic ground truth.',
                            'The legacy HEALTHY label means sparse evidence, not a confirmed healthy patient.',
                            'Template vocabulary is shared across partitions; metrics are not real-world language validation.',
                            'These examples train classification, not response generation or crisis recognition.']}
    output.with_suffix('.data_card.json').write_text(json.dumps(card, indent=2) + '\n', encoding='utf-8')
    return card


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/reviewed_projections.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/mira_bilingual.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.output, args.seed), indent=2))
