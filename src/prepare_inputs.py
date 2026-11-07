"""Rebase previously audited splits onto the desktop dataset without re-splitting."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def relative(path):
    parts = Path(path).parts
    index = parts.index('fer_data')
    return Path(*parts[index:])

def save(path, records, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

def main():
    source = ROOT / 'data' / 'original_splits'
    inventory = {str(relative(r['path'])).lower(): r['sha256'] for r in rows(source / 'image_inventory.csv')}
    manifest = rows(source / 'manifest.csv')
    inference = rows(source / 'inference.csv')
    originals = {'manifest.csv': sha(source / 'manifest.csv'), 'inference.csv': sha(source / 'inference.csv')}
    groups = {key: set() for key in ['train', 'val', 'test', 'unlabeled']}
    image_hashes = []
    for index, r in enumerate(manifest + inference):
        rel = relative(r['path'])
        dest = ROOT / 'data' / 'raw' / rel
        digest = sha(dest)
        if digest != inventory[str(rel).lower()]:
            raise ValueError(f'Image content changed: {dest}')
        r['path'] = str(dest)
        group = r.get('split', 'unlabeled')
        if group != 'unlabeled' and digest in groups[group]:
            raise ValueError('Unexpected duplicate within labeled partition')
        groups[group].add(digest)
        image_hashes.append(digest)
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if groups[a] & groups[b]:
                raise ValueError(f'Byte-identical image leakage: {a}/{b}')
    template = rows(ROOT / 'data' / 'raw' / 'submission.csv')
    if [r['ID'] for r in template] != [r['ID'] for r in inference]:
        raise ValueError('Submission order mismatch')
    if set(r['label'] for r in manifest) != set(CLASSES):
        raise ValueError('Unexpected class mapping')
    features = ROOT / 'features'
    save(features / 'manifest.csv', manifest, ['path', 'label', 'subject_id', 'split', 'fold'])
    save(features / 'unlabeled.csv', inference, ['ID', 'path'])
    report = {'classes': CLASSES, 'labeled_samples': len(manifest), 'inference_samples': len(inference),
              'splits': dict(Counter(r['split'] for r in manifest)), 'class_counts': dict(Counter(r['label'] for r in manifest)),
              'image_content_sha256': hashlib.sha256(''.join(image_hashes).encode()).hexdigest(),
              'original_csv_sha256': originals, 'byte_leakage_checks_passed': True,
              'limitations': ['No subject IDs: person independence is unknown.', 'Recompressed/near duplicates not detected.'],
              'manifest_sha256': sha(features / 'manifest.csv'), 'inference_sha256': sha(features / 'unlabeled.csv')}
    (ROOT / 'data' / 'split_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
