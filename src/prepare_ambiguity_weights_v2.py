"""Stronger weights for the unchanged, verified train-only uncertainty groups."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import prepare_ambiguity_weights as previous

ROOT = Path(__file__).resolve().parents[1]
MIN_WEIGHT = .2
PRESERVED_SUBMISSION = previous.PRESERVED_SUBMISSION
RAW_WEIGHTS = {'normal': 1., 'uncertain': .5, 'very_uncertain': .2}
POLICY = {**previous.POLICY, 'version': 2, 'raw_weights': RAW_WEIGHTS,
          'grouping': 'reuse verified v1 grouped OOF uncertainty categories; no threshold refitting',
          'sampling': 'uniform shuffle; reduce loss weights, not physical sample counts',
          'reported_share': 'sum of sample weights / total sample weights, not observed loss share'}


def build(root):
    old_folder = root/'experiments'/'hse_ambiguity_v1'/'weights'
    output = root/'experiments'/'hse_ambiguity_v2'/'weights'
    output.mkdir(parents=True, exist_ok=True)
    old = previous.read(old_folder/'summary.json')
    if old.get('state')!='complete' or old.get('classes')!=previous.CLASSES or old.get('train_rows')!=21248:
        raise ValueError('Previous uncertainty groups are incomplete or have different membership')
    pins = dict(old['binding']['source_sha256'])
    pins[str(old_folder/'summary.json')] = previous.sha(old_folder/'summary.json')
    for name, expected in old['file_sha256'].items():
        pins[str(old_folder/name)] = expected
    pins[str(Path(__file__))] = previous.sha(__file__)
    for path, expected in pins.items():
        if previous.sha(path)!=expected:
            raise ValueError(f'Previous grouping/source changed: {path}')
    if previous.sha(root/'hse_results'/'submission.csv')!=PRESERVED_SUBMISSION:
        raise ValueError('The latest best submission changed')
    binding = {'policy': POLICY, 'classes': previous.CLASSES,
               'train_membership_sha256': old['binding']['train_membership_sha256'], 'source_sha256': pins}
    summary_path = output/'summary.json'
    if summary_path.exists():
        existing = previous.read(summary_path)
        if existing.get('binding')!=binding:
            raise ValueError('Existing strong-weight experiment has different sources or policy')
        for name, expected in existing['file_sha256'].items():
            if previous.sha(output/name)!=expected:
                raise ValueError('Existing strong-weight artifact changed')
        return existing
    rows = previous.csv_rows(old_folder/'train_weights.csv')
    old_weights = np.load(old_folder/'training_weights.npy', allow_pickle=False)
    if len(rows)!=21248 or old_weights.shape!=(21248,):
        raise ValueError('Previous weight count changed')
    targets = np.array([int(row['target']) for row in rows])
    buckets = np.array([row['difficulty'] for row in rows])
    if set(buckets)!=set(RAW_WEIGHTS) or not np.array_equal(old_weights, np.array([float(row['training_weight']) for row in rows])):
        raise ValueError('Previous weight CSV differs from its array or grouping')
    raw = np.array([RAW_WEIGHTS[bucket] for bucket in buckets])
    weights = raw.copy()
    per_class = {}
    for target, label in enumerate(previous.CLASSES):
        selected = targets==target
        if not selected.any():
            raise ValueError('Every training class must be present')
        weights[selected] /= raw[selected].mean()
        if not np.isclose(weights[selected].mean(), 1., atol=1e-12):
            raise ValueError('Per-class weight total changed')
        per_class[label] = {'images': int(selected.sum()),
            'counts': {bucket: int(np.sum(selected&(buckets==bucket))) for bucket in RAW_WEIGHTS},
            'mean_training_weight': float(weights[selected].mean()),
            'min_weight': float(weights[selected].min()), 'max_weight': float(weights[selected].max())}
    hard = buckets!='normal'
    if not np.isfinite(weights).all() or (weights<MIN_WEIGHT).any() or not (weights[hard]<old_weights[hard]).all():
        raise ValueError('New weights are invalid or fail to reduce every uncertain image contribution')
    for i, row in enumerate(rows):
        row['previous_training_weight'] = row['training_weight']
        row['raw_weight'] = float(raw[i]); row['training_weight'] = float(weights[i])
    fields = list(rows[0])
    files = []
    for name, subset in [('train_weights.csv', rows)] + [(bucket+'.csv', [row for row in rows if row['difficulty']==bucket]) for bucket in RAW_WEIGHTS]:
        with (output/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(subset)
        files.append(name)
    np.save(output/'training_weights.npy', weights, allow_pickle=False); files.append('training_weights.npy')
    shares = {bucket: {'images': int(np.sum(buckets==bucket)),
        'physical_image_share': float(np.mean(buckets==bucket)),
        'previous_weight_share': float(old_weights[buckets==bucket].sum()/old_weights.sum()),
        'new_weight_share': float(weights[buckets==bucket].sum()/weights.sum()),
        'min_training_weight': float(weights[buckets==bucket].min()),
        'max_training_weight': float(weights[buckets==bucket].max())} for bucket in RAW_WEIGHTS}
    summary = {'state': 'complete', 'classes': previous.CLASSES, 'binding': binding, 'train_rows': len(rows),
        'counts': {bucket: values['images'] for bucket, values in shares.items()}, 'per_class': per_class,
        'group_weight_shares': shares, 'previous_uncertain_weight_share': float(old_weights[hard].sum()/old_weights.sum()),
        'uncertain_weight_share': float(weights[hard].sum()/weights.sum()),
        'training_weight_mean': float(weights.mean()), 'min_training_weight': float(weights.min()),
        'max_training_weight': float(weights.max()), 'groups_changed': False, 'labels_changed': False,
        'validation_weights_used': False, 'internal_test_read': False, 'external_answers_used': False,
        'review_images_directory': str(root/'experiments'/'hse_ambiguity_v1'/'review_images'),
        'preserved_submission_sha256': PRESERVED_SUBMISSION,
        'file_sha256': {name: previous.sha(output/name) for name in files}}
    previous.write_json(summary_path, summary)
    print(previous.json.dumps({key: summary[key] for key in ['state','counts','previous_uncertain_weight_share','uncertain_weight_share','min_training_weight','max_training_weight']}, ensure_ascii=True), flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args=parser.parse_args();build(args.project_root.resolve())


if __name__=='__main__':
    main()
