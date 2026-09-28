"""Generate engineering fixtures matching Mira's live evidence projection.

No real patients, confirmed diagnoses, clinician review or clinical accuracy
are represented by this data. See docs/TRAINING.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vitamind.clinical.feature_extractor import RULES
from vitamind.ml.train import load_records

ATTENTION = {'distractibility', 'forgetfulness', 'procrastination', 'hyperactivity_impulsivity'}
ACTIVATION = {'grandiosity', 'racing_thoughts', 'pressured_speech', 'flight_of_ideas',
              'impulsive_spending', 'increased_goal_directed_activity'}
PERCEPTUAL = {'auditory_perceptual_experience', 'persecutory_ideas'}
SPARSE = ATTENTION | {'reduced_sleep', 'racing_thoughts', 'social_withdrawal'}
# Variations in these secondary features share a group, including when their
# synthetic routing label changes. They cannot leak across partitions.
SECONDARY = {'procrastination', 'reduced_sleep', 'racing_thoughts',
             'pressured_speech', 'social_withdrawal'}
MEANINGS = {
    'ADHD': 'synthetic_attention_history_followup',
    'BIPOLAR': 'synthetic_episodic_activation_followup',
    'PSYCHOSIS': 'synthetic_perceptual_or_belief_followup',
    'AMBIGUOUS': 'overlapping_or_incomplete_synthetic_pattern',
    'OTHER': 'nonspecific_features_without_selected_target_pattern',
    'HEALTHY': 'legacy_compatibility_only_sparse_evidence_not_confirmed_health',
}
SPLITS = ('train', 'validation', 'test')


def assign_label(features):
    """Transparent author-defined fixture policy, NOT diagnostic criteria."""
    attention = 'childhood_onset' in features and len(features & ATTENTION) >= 2
    activation = ({'decreased_need_for_sleep', 'elevated_mood', 'episodic_pattern'} <= features
                  and bool(features & ACTIVATION))
    perceptual = bool(features & PERCEPTUAL)
    branches = [label for label, active in [('ADHD', attention), ('BIPOLAR', activation),
                                          ('PSYCHOSIS', perceptual)] if active]
    if len(branches) > 1:
        return 'AMBIGUOUS'
    if branches:
        return branches[0]
    if len(features) <= 2 and features <= SPARSE:
        return 'HEALTHY'
    if (len(features & ATTENTION) >= 2 or len(features & ACTIVATION) >= 2
            or 'decreased_need_for_sleep' in features):
        return 'AMBIGUOUS'
    return 'OTHER'


def group_and_split(features, seed):
    signature = '|'.join(sorted(features - SECONDARY)) or 'secondary-features-only'
    digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()
    split_number = int(hashlib.sha256(f'{seed}|{signature}'.encode()).hexdigest()[:8], 16) % 100
    split = 'train' if split_number < 70 else 'validation' if split_number < 85 else 'test'
    # Preserve two entire sparse-evidence families for evaluation (5 records
    # each), because the legacy HEALTHY category has only 28 unique states.
    # Every other label sharing one of these signatures follows the same split.
    sparse_core = features - SECONDARY
    if sparse_core <= (SPARSE - SECONDARY):
        split = ('validation' if sparse_core == {'distractibility'} else
                 'test' if sparse_core == {'forgetfulness'} else 'train')
    return f'synthetic-family-{digest[:16]}', split


def generate(output, seed=42, per_label=400):
    if per_label < 20:
        raise ValueError('--per-label must be at least 20')
    features_ordered = [rule.feature for rule in RULES]
    pools = defaultdict(list)
    # Enumerate distinct feature states, not paraphrases or randomly assigned
    # diagnoses. Cap the fixture at eight present features per state.
    for size in range(1, 9):
        for selection in combinations(range(len(RULES)), size):
            features = {features_ordered[i] for i in selection}
            if 'decreased_need_for_sleep' in features and 'reduced_sleep' not in features:
                continue
            if 'childhood_onset' in features and not (features & ATTENTION):
                continue
            label = assign_label(features)
            group, split = group_and_split(features, seed)
            pools[label, split].append((selection, group))
    rng = random.Random(seed)
    counts = {'train': round(per_label * .7), 'validation': round(per_label * .15)}
    counts['test'] = per_label - sum(counts.values())
    records = []
    for label in sorted(MEANINGS):
        for split in SPLITS:
            available = pools[label, split]
            rng.shuffle(available)
            # The current projection cannot represent "no symptoms". Retain
            # only naturally available sparse fixtures for the legacy label;
            # never inflate this category with off-format reassuring prose.
            selected = available if label == 'HEALTHY' else available[:counts[split]]
            if not selected or (label != 'HEALTHY' and len(selected) != counts[split]):
                raise ValueError(f'Not enough {label}/{split} fixtures; lower --per-label or change seed')
            for selection, group in selected:
                text = ' '.join('I ' + RULES[i].present[0] + '.' for i in selection)
                records.append({
                    'text': text, 'label': label, 'split': split, 'group_id': group,
                    'input_type': 'structured_evidence_projection',
                    'record_id': 'synthetic-' + hashlib.sha256(text.encode()).hexdigest()[:16],
                    'present_features': [RULES[i].feature for i in selection],
                    'synthetic': True, 'clinician_reviewed': False,
                    'label_source': 'author_defined_fixture_policy_v1',
                    'label_meaning': MEANINGS[label],
                    'clinical_validation': False,
                })
    rng.shuffle(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
    load_records(output)
    card = {
        'dataset': 'Mira synthetic evidence-projection training fixtures v1',
        'seed': seed, 'records': len(records), 'synthetic': True,
        'clinician_reviewed': False, 'clinical_validation': False,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'feature_inventory_sha256': hashlib.sha256(json.dumps(
            [(r.domain, r.feature, r.present[0]) for r in RULES], ensure_ascii=False).encode()).hexdigest(),
        'splits': dict(Counter(r['split'] for r in records)),
        'labels': dict(Counter(r['label'] for r in records)),
        'labels_by_split': {s: dict(Counter(r['label'] for r in records if r['split'] == s)) for s in SPLITS},
        'groups_by_split': {s: len({r['group_id'] for r in records if r['split'] == s}) for s in SPLITS},
        'split_policy': 'Hash core signature 70/15/15, with whole sparse families allocated 18/5/5 for legacy HEALTHY coverage; cap other categories per split.',
        'label_meanings': MEANINGS,
        'model_text_language': 'fixed English evidence projection used by both en/ar sessions',
        'safety_examples_included': False,
        'limitations': [
            'Synthetic routing fixtures, not clinical histories or diagnostic ground truth.',
            'Only text and label enter model fitting. Metadata is not a model feature.',
            'Train/validation/test share a generator and label policy; scores measure fixture-policy fit only.',
            'No empty projection is representable in the current trainer. Mira bypasses inference when no positive evidence exists.',
            'HEALTHY is a required legacy compatibility category for sparse evidence, never proof of health.',
            'OTHER does not identify an alternative diagnosis.',
            'Duration, impairment, absent/conflicting evidence and context are not passed to this classifier.',
            'These fixtures do not teach Arabic extraction, negation, crisis recognition, conversation or report writing.',
            'The same underlying signature with secondary-feature variations stays in one partition.',
            'The word reviewed in the requested filename does not mean clinician review has occurred.',
        ],
    }
    output.with_suffix('.data_card.json').write_text(json.dumps(card, indent=2) + '\n', encoding='utf-8')
    return card


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/reviewed_projections.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per-label', type=int, default=400,
                        help='Total per routing category, except the deliberately small legacy HEALTHY category')
    args = parser.parse_args()
    result = generate(args.output, args.seed, args.per_label)
    print(json.dumps({key: result[key] for key in ['records', 'splits', 'labels', 'sha256']}, indent=2))
