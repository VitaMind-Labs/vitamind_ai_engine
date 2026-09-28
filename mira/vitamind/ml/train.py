"""Local bilingual routing training; Arabic source label meanings are not guessed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from .bilingual_text import INPUT_TYPE, text_key
from ..clinical.language import normalize

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'ADHD', 'BIPOLAR', 'PSYCHOSIS', 'AMBIGUOUS', 'OTHER', 'HEALTHY'}
LEGACY_INPUT = 'structured_evidence_projection'


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    groups, texts, modes = {}, set(), set()
    for row in records:
        if not {'text', 'label', 'split', 'group_id', 'input_type'} <= row.keys():
            raise ValueError('Each record needs text, label, split, group_id and input_type.')
        if row['label'] not in LABELS or row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid routing label or split.')
        if row['input_type'] not in {INPUT_TYPE, LEGACY_INPUT}:
            raise ValueError('Use patient_text_bilingual_v1, not mixed patient/assistant transcripts.')
        if not isinstance(row['text'], str) or not row['text'].strip() or not row['group_id']:
            raise ValueError('Empty model input or group ID.')
        modes.add(row['input_type'])
        if row['input_type'] == INPUT_TYPE:
            if row.get('language') not in {'en', 'ar'}:
                raise ValueError('Native bilingual records need language en or ar.')
            pattern = r'[\u0621-\u064a]' if row['language'] == 'ar' else r'[a-zA-Z]'
            if not re.search(pattern, row['text']):
                raise ValueError('Language tag does not match text script.')
        key = text_key(row['text'])
        if row['group_id'] in groups and groups[row['group_id']] != row['split']:
            raise ValueError('A patient/scenario/translation group crosses partitions.')
        if key in texts:
            raise ValueError('Duplicate normalized model text; deduplicate without crossing partitions.')
        groups[row['group_id']] = row['split']
        texts.add(key)
    if len(modes) != 1:
        raise ValueError('Do not mix legacy projections with native-language records.')
    if {r['split'] for r in records} != {'train', 'validation', 'test'}:
        raise ValueError('Provide nonempty train, validation and test partitions.')
    if {r['label'] for r in records if r['split'] == 'train'} != LABELS:
        raise ValueError('The training partition must contain all six routing labels.')
    if modes == {INPUT_TYPE}:
        for language in ('en', 'ar'):
            if {r['label'] for r in records if r['split'] == 'train' and r['language'] == language} != LABELS:
                raise ValueError('All six labels must be represented in training for each language.')
            if not all(any(r['split'] == split and r['language'] == language for r in records)
                       for split in ('validation', 'test')):
                raise ValueError('Both languages need validation and test records.')
    return records


def load_corpus(path, records):
    if path is None:
        return [], {'records': 0, 'usage': 'none'}
    heldout = {text_key(r['text']) for r in records if r['split'] != 'train'}
    seen, texts, excluded = set(), [], 0
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get('split') != 'train' or row.get('language') != 'ar'
                or row.get('input_type') != 'unlabeled_patient_opening' or 'label' in row):
            raise ValueError('Arabic vocabulary corpus must contain unlabeled train-only Arabic openings.')
        text = row.get('text')
        if not isinstance(text, str) or not re.search(r'[\u0621-\u064a]', text):
            raise ValueError('Invalid Arabic corpus text.')
        key = text_key(text)
        if key in heldout or key in seen:
            excluded += 1
            continue
        seen.add(key)
        texts.append(text)
    return texts, {'records': len(texts), 'excluded_duplicate_or_heldout': excluded,
                   'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   'usage': 'unsupervised vocabulary and IDF fitting only', 'supervised_labels_used': False}


def train_model(data, output, arabic_corpus=None):
    import joblib
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.metrics import classification_report

    records = load_records(data)
    training = [r for r in records if r['split'] == 'train']
    x, y = [r['text'] for r in training], [r['label'] for r in training]
    corpus, corpus_info = load_corpus(arabic_corpus, records)
    if corpus and records[0]['input_type'] != INPUT_TYPE:
        raise ValueError('Arabic adaptation requires native bilingual records, not English projections.')
    word = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), max_features=40000)
    char = TfidfVectorizer(preprocessor=normalize, analyzer='char_wb', ngram_range=(3, 5), max_features=60000)
    word.fit(x + corpus)
    char.fit(x + corpus)
    word_x, char_x = word.transform(x), char.transform(x)
    word_lr = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42).fit(word_x, y)
    char_lr = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42).fit(char_x, y)
    nb = ComplementNB().fit(word_x, y)
    models = [Pipeline([('tfidf', word), ('classifier', word_lr)]),
              Pipeline([('tfidf', char), ('classifier', char_lr)]),
              Pipeline([('tfidf', word), ('classifier', nb)])]
    classes = list(word_lr.classes_)
    assert all(list(model.classes_) == classes for model in models)
    weights = [.5, .35, .15]
    reports, language_reports = {}, {}

    def evaluate(rows):
        probs = sum(w * m.predict_proba([r['text'] for r in rows]) for w, m in zip(weights, models))
        predicted = [classes[i] for i in np.argmax(probs, axis=1)]
        return classification_report([r['label'] for r in rows], predicted, labels=classes,
                                     output_dict=True, zero_division=0)

    for split in ('validation', 'test'):
        rows = [r for r in records if r['split'] == split]
        reports[split] = evaluate(rows)
        language_reports[split] = {language: evaluate([r for r in rows if r.get('language', 'en') == language])
                                   for language in ('en', 'ar') if any(r.get('language', 'en') == language for r in rows)}
    metadata = {
        'clinical_validation': False, 'input_type': records[0]['input_type'], 'seed': 42,
        'languages': sorted({r.get('language', 'en') for r in training}), 'model_version': 'mira-native-bilingual-v1',
        'dataset_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'records': {s: sum(r['split'] == s for r in records) for s in ('train', 'validation', 'test')},
        'supervised_training_by_language': dict(Counter(r.get('language', 'en') for r in training)),
        'supervised_training_label_sources': dict(Counter(r.get('label_source', 'not_supplied') for r in training)),
        'arabic_corpus': corpus_info,
        'vocabulary': {name: {'total': len(vec.vocabulary_),
                              'arabic_terms': sum(bool(re.search(r'[\u0621-\u064a]', t)) for t in vec.vocabulary_),
                              'latin_terms': sum(bool(re.search(r'[a-zA-Z]', t)) for t in vec.vocabulary_)}
                       for name, vec in [('word', word), ('char', char)]},
        'split_policy': 'Disjoint supplied scenario/translation groups; normalized text deduplication; vocabulary fits train only.',
        'safety_model_trained': False, 'evaluation': reports, 'evaluation_by_language': language_reports,
    }
    if records[0]['input_type'] == INPUT_TYPE:
        assert metadata['vocabulary']['word']['arabic_terms'] > 0
        assert metadata['vocabulary']['word']['latin_terms'] > 0
    bundle = {'metadata': metadata,
              'routing': dict(zip(['word_model', 'char_model', 'nb_model'], models), classes=classes, weights=weights),
              'pathway_map': {label: 'qualified_clinical_assessment' for label in classes}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(output)
    output.with_suffix('.evaluation.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata


def train():
    parser = argparse.ArgumentParser(description='Train the local English/Arabic assessment routing model.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/mira_bilingual.jsonl')
    parser.add_argument('--arabic-corpus', type=Path, default=ROOT / 'data/arabic_adaptation.jsonl')
    parser.add_argument('--without-arabic-corpus', action='store_true', help='Fit only the supervised dataset.')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/mira_bilingual.joblib')
    args = parser.parse_args()
    try:
        result = train_model(args.data, args.output, None if args.without_arabic_corpus else args.arabic_corpus)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output': str(args.output), 'records': result['records'],
                      'languages': result['languages'], 'arabic_corpus': result['arabic_corpus'],
                      'vocabulary': result['vocabulary'], 'clinical_validation': False}, indent=2))
