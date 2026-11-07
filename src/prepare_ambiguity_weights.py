"""Train-only, grouped out-of-fold uncertainty weights and copied review buckets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
POLICY = {'teachers': ['baseline', 'cosine'], 'entropy_threshold': .60,
          'margin_threshold': .25, 'view_entropy_threshold': .55, 'view_margin_threshold': .20,
          'severe_entropy_threshold': .80, 'severe_margin_threshold': .10,
          'max_downweighted_fraction_per_class': .25, 'max_severe_fraction_per_class': .10,
          'raw_weights': {'normal': 1., 'uncertain': .75, 'very_uncertain': .5},
          'score': '.6*normalized_entropy + .3*(1-margin) + .1*four_view_JS/log(4)',
          'normalization': 'divide raw weights by their mean within each training class',
          'loss': 'mean(training_weight * per_image_cross_entropy); uniform sampling',
          'validation_weight': 1., 'labels_changed': False, 'deleted_images': 0}
PRESERVED_SUBMISSION = 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def image_key(path):
    parts = str(path).replace('\\', '/').split('/')
    return '/'.join(parts[parts.index('fer_data'):]).casefold()


def probability(value, count):
    p = np.asarray(value, dtype=np.float64)
    if p.shape != (count, 6) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError('Invalid official six-class probability array')
    return p/p.sum(1, keepdims=True)


def entropy(p):
    return -(p*np.log(np.clip(p, 1e-12, 1))).sum(-1)


def assign_weights(targets, views):
    """Views are four OOF predictions: two classifiers, each original and flip."""
    n = len(targets)
    views = np.stack([probability(view, n) for view in views])
    average = views.mean(0)
    teacher_a, teacher_b = views[:2].mean(0), views[2:].mean(0)
    normalized_entropy = entropy(average)/math.log(6)
    ordered = np.sort(average, axis=1)
    margin = ordered[:, -1]-ordered[:, -2]
    disagree = teacher_a.argmax(1) != teacher_b.argmax(1)
    view_disagree = (views[0].argmax(1) != views[1].argmax(1)) | (views[2].argmax(1) != views[3].argmax(1))
    js = np.maximum(0., entropy(average)-entropy(views).mean(0))/math.log(4)
    score = .6*normalized_entropy+.3*(1-margin)+.1*js
    eligible = ((normalized_entropy>=.60)&(margin<=.25)) | (disagree&(margin<=.25)) | (view_disagree&(normalized_entropy>=.55)&(margin<=.20))
    severe_eligible = eligible&(normalized_entropy>=.80)&(margin<=.10)
    raw = np.ones(n, dtype=np.float64)
    bucket = np.full(n, 'normal', dtype='<U16')
    for target in range(6):
        members = np.flatnonzero(targets == target)
        if not len(members):
            raise ValueError('All six training classes are required')
        candidates = members[eligible[members]]
        order = np.lexsort((candidates, -score[candidates]))
        selected = candidates[order[:math.floor(.25*len(members))]]
        raw[selected], bucket[selected] = .75, 'uncertain'
        strongest = selected[severe_eligible[selected]]
        order = np.lexsort((strongest, -score[strongest]))
        strongest = strongest[order[:math.floor(.10*len(members))]]
        raw[strongest], bucket[strongest] = .5, 'very_uncertain'
    weights = raw.copy()
    for target in range(6):
        selected = targets == target
        weights[selected] /= raw[selected].mean()
    if not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Invalid positive sample weights')
    return {'probability': average, 'entropy': normalized_entropy, 'margin': margin,
            'teacher_disagree': disagree, 'view_disagree': view_disagree, 'js': js,
            'score': score, 'eligible': eligible, 'bucket': bucket, 'raw': raw, 'weights': weights}


def verified_oof(root, train_rows):
    source = root/'experiments'/'weighted_heads'
    summary_path = root/'weighted_results'/'summary.json'
    summary, config = read(summary_path), read(source/'run_config.json')
    if summary.get('state') != 'complete' or summary.get('classes') != CLASSES or summary.get('train_rows') != len(train_rows) or summary.get('run_signature') != signature(config):
        raise ValueError('OOF run is incomplete or source/config/membership changed')
    if config.get('old_validation_test_unlabeled_read') is not False or summary.get('external_answers_used') is not False:
        raise ValueError('Only train-only grouped OOF predictions are allowed')
    pins = {str(summary_path): sha(summary_path)}
    def check(path):
        expected = summary['artifact_sha256'].get(str(path))
        if not expected or sha(path) != expected:
            raise ValueError(f'OOF artifact checksum changed: {path}')
        pins[str(path)] = expected
    check(source/'run_config.json')
    for name, expected in config['source_sha256'].items():
        path = root/'src'/name
        if sha(path) != expected:
            raise ValueError(f'OOF source changed: {name}')
        pins[str(path)] = expected
    views, common_rows = [], None
    for teacher in POLICY['teachers']:
        folder = source/teacher
        for name in ['oof_predictions.csv', 'oof_probabilities.npy', 'summary.json']:
            check(folder/name)
        rows = csv_rows(folder/'oof_predictions.csv')
        if len(rows) != len(train_rows):
            raise ValueError('OOF count differs from original train membership')
        for i, (row, train) in enumerate(zip(rows, train_rows)):
            if int(row['cache_row']) != i or int(row['manifest_index']) != train['manifest_index'] or int(row['target']) != train['target'] or image_key(row['path']) != image_key(train['path']):
                raise ValueError('OOF row order, target or image differs from train manifest')
        metadata = [(int(row['manifest_index']), int(row['target']), int(row['fold']), row['group']) for row in rows]
        if common_rows is not None and metadata != common_rows:
            raise ValueError('Teacher fold/group/label memberships differ')
        common_rows = metadata
        folds = np.array([int(row['fold']) for row in rows])
        groups = np.array([row['group'] for row in rows])
        original, flipped, coverage = np.empty((len(rows), 6)), np.empty((len(rows), 6)), np.zeros(len(rows), dtype=np.int64)
        for fold in range(3):
            fold_folder = folder/f'fold_{fold}'
            for name in ['summary.json', 'fold_contract.json', 'model.pt', 'validation_original.npy', 'validation_flip.npy', 'validation_probabilities.npy']:
                check(fold_folder/name)
            receipt, contract = read(fold_folder/'summary.json'), read(fold_folder/'fold_contract.json')
            train_indices = np.array(contract['train_cache_indices'], dtype=np.int64)
            heldout = np.array(contract['heldout_cache_indices'], dtype=np.int64)
            if (receipt['state'] != 'complete' or receipt['epochs_completed'] != 12 or receipt['contract'] != contract or contract['run_signature'] != summary['run_signature']
                or not np.array_equal(train_indices, np.flatnonzero(folds!=fold)) or not np.array_equal(heldout, np.flatnonzero(folds==fold))
                or set(groups[train_indices]) & set(groups[heldout])
                or contract['heldout_manifest_indices'] != [train_rows[i]['manifest_index'] for i in heldout]
                or contract['train_manifest_indices'] != [train_rows[i]['manifest_index'] for i in train_indices]):
                raise ValueError('A teacher saw its held-out image/group or has a different contract')
            original[heldout] = probability(np.load(fold_folder/'validation_original.npy', allow_pickle=False), len(heldout))
            flipped[heldout] = probability(np.load(fold_folder/'validation_flip.npy', allow_pickle=False), len(heldout))
            coverage[heldout] += 1
        stored = probability(np.load(folder/'oof_probabilities.npy', allow_pickle=False), len(rows))
        csv_probability = np.array([[float(row['p_'+name]) for name in CLASSES] for row in rows])
        if not (coverage==1).all() or not np.allclose((original+flipped)/2, stored, atol=1e-12, rtol=0) or not np.allclose(stored, csv_probability, atol=1e-12, rtol=0):
            raise ValueError('OOF outputs differ from their verified held-out predictions')
        views += [original, flipped]
    return views, common_rows, pins


def build(root, copy_images=False):
    folder = root/'experiments'/'hse_ambiguity_v1'/'weights'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = root/'features'/'manifest.csv'
    rows = []
    for index, row in enumerate(csv_rows(manifest)):
        if row['split'] == 'train':
            if row['label'] not in CLASSES:
                raise ValueError('Unexpected training class')
            rows.append({'manifest_index': index, 'path': row['path'], 'label': row['label'], 'target': CLASSES.index(row['label'])})
    if len(rows) != 21248 or sha(manifest) != read(root/'data'/'split_audit.json')['manifest_sha256']:
        raise ValueError('Original train membership changed')
    if sha(root/'hse_results'/'submission.csv') != PRESERVED_SUBMISSION:
        raise ValueError('The latest best submission changed')
    views, membership, pins = verified_oof(root, rows)
    pins[str(manifest)], pins[str(Path(__file__))] = sha(manifest), sha(__file__)
    binding = {'policy': POLICY, 'classes': CLASSES, 'train_membership_sha256': signature(rows), 'source_sha256': pins}
    summary_path = folder/'summary.json'
    if summary_path.exists():
        existing = read(summary_path)
        if existing.get('binding') != binding:
            raise ValueError('Existing weight experiment sources or policy changed')
        for name, expected in existing['file_sha256'].items():
            if sha(folder/name) != expected:
                raise ValueError('Existing weight artifact changed')
        return existing
    targets = np.array([row['target'] for row in rows])
    values = assign_weights(targets, views)
    fields = ['manifest_index', 'path', 'label', 'target', 'fold', 'group', 'difficulty', 'raw_weight', 'training_weight', 'entropy', 'margin', 'teacher_disagree', 'view_disagree', 'four_view_js', 'uncertainty_score', 'oof_predicted_label', 'oof_target_probability']
    entries = []
    for i, row in enumerate(rows):
        entries.append({**row, 'fold': membership[i][2], 'group': membership[i][3], 'difficulty': str(values['bucket'][i]),
            'raw_weight': float(values['raw'][i]), 'training_weight': float(values['weights'][i]),
            'entropy': float(values['entropy'][i]), 'margin': float(values['margin'][i]),
            'teacher_disagree': bool(values['teacher_disagree'][i]), 'view_disagree': bool(values['view_disagree'][i]),
            'four_view_js': float(values['js'][i]), 'uncertainty_score': float(values['score'][i]),
            'oof_predicted_label': CLASSES[int(values['probability'][i].argmax())], 'oof_target_probability': float(values['probability'][i, row['target']])})
    artifact_names = []
    for name, subset in [('train_weights.csv', entries)] + [(bucket+'.csv', [entry for entry in entries if entry['difficulty']==bucket]) for bucket in POLICY['raw_weights']]:
        with (folder/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(subset)
        artifact_names.append(name)
    np.save(folder/'training_weights.npy', values['weights'], allow_pickle=False)
    artifact_names.append('training_weights.npy')
    copied = 0
    if copy_images:
        review = folder.parent/'review_images'
        started = time.monotonic()
        for entry in entries:
            target = review/entry['difficulty']/entry['label']/f"{entry['manifest_index']:05d}_{Path(entry['path']).name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(review.resolve()):
                raise ValueError('Review image target escapes experiment directory')
            if not target.exists():
                shutil.copyfile(entry['path'], target)
            if sha(target) != sha(entry['path']):
                raise ValueError('Review copy differs from original image')
            copied += 1
            if copied%4000 == 0:
                print(json.dumps({'stage': 'copy_review_images', 'copied': copied, 'total': len(entries), 'seconds': time.monotonic()-started}), flush=True)
    per_class = {}
    for target, name in enumerate(CLASSES):
        selected = targets == target
        counts = {bucket: int(np.sum(selected&(values['bucket']==bucket))) for bucket in POLICY['raw_weights']}
        if counts['uncertain']+counts['very_uncertain'] > math.floor(.25*selected.sum()) or not np.isclose(values['weights'][selected].mean(), 1., atol=1e-12):
            raise ValueError('Class proportion or equal class mean protection failed')
        per_class[name] = {'images': int(selected.sum()), 'counts': counts, 'mean_training_weight': float(values['weights'][selected].mean()), 'min_weight': float(values['weights'][selected].min()), 'max_weight': float(values['weights'][selected].max())}
    summary = {'state': 'complete', 'classes': CLASSES, 'binding': binding, 'train_rows': len(rows),
        'counts': {bucket: int(np.sum(values['bucket']==bucket)) for bucket in POLICY['raw_weights']},
        'per_class': per_class, 'training_weight_mean': float(values['weights'].mean()),
        'min_training_weight': float(values['weights'].min()), 'max_training_weight': float(values['weights'].max()),
        'review_images_copied': copied, 'weight_labels_changed': False, 'validation_weights_used': False,
        'internal_test_read': False, 'external_answers_used': False, 'preserved_submission_sha256': PRESERVED_SUBMISSION,
        'uncertainty_caveat': 'OOF model uncertainty is a proxy for ambiguity, not a verified incorrect label or human judgment.',
        'file_sha256': {name: sha(folder/name) for name in artifact_names}}
    write_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in ['state', 'train_rows', 'counts', 'min_training_weight', 'max_training_weight', 'review_images_copied']}, ensure_ascii=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--copy-images', action='store_true')
    args = parser.parse_args()
    build(args.project_root.resolve(), args.copy_images)


if __name__ == '__main__':
    main()
