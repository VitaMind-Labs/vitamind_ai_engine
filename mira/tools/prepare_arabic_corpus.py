"""Import Arabic openings for vocabulary fitting only; never guess label meanings."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vitamind.ml.bilingual_text import text_key

PUBLISHED_MD5 = {'train.csv': '548e0f29ad32c05548b5d50631bdcdc1',
                 'validation.csv': '0c2674177c22043defe6c51f4ef8b061',
                 'test.csv': 'a710c2db8b2b7e8cc817113bd1c025c3'}


def opening(row):
    if None in row or any(row.get(k) is None for k in ('text', 'dialect', 'chatbot_related_questions', 'Label')):
        return None, 'malformed_row'
    if row['dialect'].strip().strip('"') not in {'MSA', 'GLF', 'EGY', 'LEV'}:
        return None, 'invalid_dialect_or_embedded_header'
    if row['chatbot_related_questions'].lower() != 'false':
        return None, 'chatbot_question_or_invalid_flag'
    if row['Label'] not in {'0', '1', '2', '3', '4', '5'}:
        return None, 'invalid_source_label'
    # Roles are not explicitly annotated in this source. Use only the opening,
    # assuming it is user-first; never learn or replay the following bot answers.
    text = row['text'].split('|', 1)[0].strip().strip('"').strip()
    if not 3 <= len(text.split()) <= 250 or not re.search(r'[\u0621-\u064a]', text):
        return None, 'empty_short_long_or_non_arabic'
    if re.search(r'https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\d{7,}', text, re.I):
        return None, 'contact_identifier_pattern'
    return text, None


def prepare(source, output):
    split_rows, rejected, counts, checksums = {}, {}, {}, {}
    with ZipFile(source) as archive:
        for split in ('train', 'validation', 'test'):
            name = split + '.csv'
            raw = archive.read(name)
            checksums[name] = hashlib.md5(raw).hexdigest()
            reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
            accepted, failures, total = [], Counter(), 0
            for row in reader:
                total += 1
                text, reason = opening(row)
                if reason:
                    failures[reason] += 1
                else:
                    accepted.append(text)
            split_rows[split] = accepted
            rejected[split] = dict(failures)
            counts[split] = total
    heldout = {text_key(t) for s in ('validation', 'test') for t in split_rows[s]}
    seen, rows, duplicate, overlap = set(), [], 0, 0
    for text in split_rows['train']:
        key = text_key(text)
        if key in heldout:
            overlap += 1
            continue
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        rows.append({'text': text, 'language': 'ar', 'split': 'train',
                     'input_type': 'unlabeled_patient_opening',
                     'source': 'zenodo:20568736',
                     'source_id': hashlib.sha256(key.encode()).hexdigest()[:20],
                     'label_mapping_verified': False})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    report = {'source_url': 'https://zenodo.org/records/20568736',
              'source_archive_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'csv_md5': checksums, 'matches_published_files': checksums == PUBLISHED_MD5,
              'source_rows': counts, 'rejected': rejected, 'duplicate_train_openings_removed': duplicate,
              'train_openings_overlapping_heldout_removed': overlap, 'vocabulary_training_openings': len(rows),
              'fit_partition': 'train only; validation/test consulted only for overlap exclusion',
              'supervised_source_labels_used': False, 'numeric_label_mapping': 'unverified',
              'speaker_roles': 'first turn assumed patient; source has no explicit role annotations',
              'assistant_responses_used': False, 'data_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'limitations': ['Source categories do not supply verified ADHD/bipolar/psychosis targets.',
                              'Only vocabulary/IDF adaptation, not supervised diagnostic learning from this source.',
                              'Duplicate filtering is exact normalized opening matching, not a patient-level or near-duplicate guarantee.',
                              'Simple identifier-pattern filtering is not certified anonymization.']}
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/arabic_adaptation.jsonl')
    args = parser.parse_args()
    print(json.dumps(prepare(args.zip, args.output), indent=2))
