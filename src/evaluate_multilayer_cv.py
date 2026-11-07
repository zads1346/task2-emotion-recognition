"""Fixed three-fold grouped CV using only frozen official train-image features.

Four prespecified heads train for a fixed epoch budget. Fold holdout labels are
used once after the final epoch; no fold selects an epoch or early-stopping rule.
This script never reads the old validation/test/unlabeled sets or emits a submission.
"""
from __future__ import annotations

import os
for _variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_variable] = '4'

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CANDIDATES = [('last_layer_linear', 2304, 3840, 'linear'), ('last4_linear', 0, 3840, 'linear'),
              ('last_layer_mlp', 2304, 3840, 'mlp'), ('last4_mlp', 0, 3840, 'mlp')]
FOLD_ARTIFACTS = ['model.pt', 'validation_original.npy', 'validation_flip.npy', 'validation_probabilities.npy',
                  'validation_predictions.csv', 'history.csv', 'fold_contract.json']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, records, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing required columns in {path}')
        records = list(reader)
    if not records:
        raise ValueError(f'Empty input: {path}')
    return records


def verify_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Input/artifact checksum mismatch: {path}')


def canonical(path):
    value = os.fspath(path)
    if not os.path.isabs(value):
        raise ValueError('Producer image paths must already be absolute')
    # Inputs are audited absolute paths; lexical normalization avoids tens of
    # thousands of Windows filesystem round-trips from Path.resolve().
    return os.path.normcase(os.path.normpath(value))


def load_training_inputs(root):
    """Read train-only producer artifacts, never open the original full manifest."""
    features_dir = root/'features_multilayer_train'
    groups_dir = root/'experiments'/'multilayer_groups'
    metadata_path, groups_metadata_path = features_dir/'metadata.json', groups_dir/'groups_summary.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    groups_metadata = json.loads(groups_metadata_path.read_text(encoding='utf-8'))
    if (metadata.get('format') != 'dinov2_multilayer_train_v1' or metadata.get('complete') is not True or
            metadata.get('partition') != 'train' or metadata.get('classes') != CLASSES or
            metadata.get('frozen') is not True or metadata.get('trainable_parameters') != 0 or
            metadata.get('model') != 'dinov2_vitb14' or metadata.get('layer_indices') != [8,9,10,11] or
            metadata.get('feature_dim') != 3840 or metadata.get('last_layer_feature_slice') != [2304, 3840]):
        raise ValueError('Require completed official frozen 3840-D train-only multilayer feature cache')
    if groups_metadata.get('complete') is not True or groups_metadata.get('no_group_cross_fold') is not True:
        raise ValueError('Grouping producer is incomplete or reports fold leakage')
    if metadata['manifest_sha256'] != groups_metadata['source_manifest_sha256']:
        raise ValueError('Feature/group producers reference different source manifests')
    if metadata['train_image_content_sha256'] != groups_metadata['train_image_content_sha256']:
        raise ValueError('Feature/group producers reference different ordered train image bytes')
    verify_hash(root/'src'/'extract_multilayer_features.py', metadata.get('extractor_sha256'))
    verify_hash(root/'src'/'group_training_images.py', groups_metadata.get('grouping_script_sha256'))
    expected_layout = [{'name':f'cls_block_{layer}','start':i*768,'end':(i+1)*768}
                       for i,layer in enumerate([8,9,10,11])]
    expected_layout.append({'name':'mean_patch_block_11','start':3072,'end':3840})
    if metadata.get('readout_layout') != expected_layout:
        raise ValueError('Multilayer feature layout differs from the prespecified slice comparison')
    names = ['train_rows.csv', 'train_original.npy', 'train_flip.npy']
    data_files = {}
    for name in names:
        path = features_dir/name
        verify_hash(path, metadata['file_sha256'].get(name))
        data_files[str(path)] = metadata['file_sha256'][name]
    group_path = groups_dir/'train_groups.csv'
    verify_hash(group_path, groups_metadata['train_groups_sha256'])
    data_files[str(group_path)] = groups_metadata['train_groups_sha256']
    for path in [metadata_path, groups_metadata_path, root/'src'/'extract_multilayer_features.py', root/'src'/'group_training_images.py']:
        data_files[str(path)] = sha256(path)
    rows = read_csv(features_dir/'train_rows.csv', ['manifest_index', 'path', 'label', 'target'])
    grouped = read_csv(group_path, ['manifest_index', 'path', 'label', 'target', 'group', 'fold'])
    if len(rows) != metadata['num_rows'] or len(grouped) != len(rows):
        raise ValueError('Train-only feature/group row count differs')
    manifest_indices = [int(row['manifest_index']) for row in rows]
    if len(set(manifest_indices)) != len(rows) or manifest_indices != sorted(manifest_indices):
        raise ValueError('Feature rows must have unique increasing original manifest indices')
    if len({canonical(row['path']) for row in rows}) != len(rows):
        raise ValueError('Duplicate train image paths')
    group_lookup = {int(row['manifest_index']): row for row in grouped}
    if len(group_lookup) != len(grouped) or set(group_lookup) != set(manifest_indices):
        raise ValueError('Group rows differ from the feature training membership')
    folds, groups, targets = [], [], []
    group_folds = {}
    for row in rows:
        assigned = group_lookup[int(row['manifest_index'])]
        target = int(row['target'])
        if not 0 <= target < 6 or row['label'] != CLASSES[target]:
            raise ValueError('Feature labels are not in official class order')
        if (canonical(row['path']) != canonical(assigned['path']) or row['label'] != assigned['label'] or
                target != int(assigned['target'])):
            raise ValueError('Group/feature row identity or label mismatch')
        group, fold = assigned['group'].strip(), int(assigned['fold'])
        if not group or fold not in [0, 1, 2]:
            raise ValueError('Invalid group or fold ID')
        group_folds.setdefault(group, set()).add(fold)
        groups.append(group); folds.append(fold); targets.append(target)
    if any(len(values) != 1 for values in group_folds.values()):
        raise ValueError('A near-duplicate group crosses CV folds')
    folds, groups, targets = np.asarray(folds), np.asarray(groups), np.asarray(targets, dtype=np.int64)
    for fold in range(3):
        if set(targets[folds == fold]) != set(range(6)) or set(targets[folds != fold]) != set(range(6)):
            raise ValueError('Each CV training/heldout partition must contain all six classes')
    arrays = {}
    for view in ['original', 'flip']:
        array = np.load(features_dir/f'train_{view}.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != (len(rows), 3840) or array.dtype != np.float32:
            raise ValueError('Invalid multilayer feature dimensions/dtype')
        for start in range(0, len(rows), 1024):
            if not np.isfinite(array[start:start+1024]).all():
                raise ValueError('Feature cache contains NaN/Inf')
        arrays[view] = array
    receipt = {'data_file_sha256': data_files, 'feature_metadata': metadata, 'grouping_metadata': groups_metadata,
               'row_order': 'train_rows.csv order; original manifest indices retained', 'train_rows': len(rows),
               'groups': len(group_folds), 'no_group_cross_fold_verified': True}
    return rows, targets, folds, groups, arrays, receipt


class FrozenFeatureHead(nn.Module):
    def __init__(self, dimension, kind, mean=None, scale=None):
        super().__init__()
        self.dimension, self.kind = int(dimension), kind
        self.register_buffer('feature_mean', torch.zeros(dimension) if mean is None else torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('feature_scale', torch.ones(dimension) if scale is None else torch.tensor(scale, dtype=torch.float32))
        if kind == 'linear':
            self.head = nn.Linear(dimension, 6)
        elif kind == 'mlp':
            self.head = nn.Sequential(nn.Linear(dimension, 256), nn.LayerNorm(256), nn.GELU(),
                                      nn.Dropout(.3), nn.Linear(256, 6))
        else:
            raise ValueError('Unknown head kind')

    def forward(self, raw):
        return self.head((raw.float()-self.feature_mean)/self.feature_scale)


def fit_scaler(original, train_indices, first, last):
    scaler = StandardScaler()
    for start in range(0, len(train_indices), 1024):
        scaler.partial_fit(original[train_indices[start:start+1024], first:last])
    return scaler


def normalized(p):
    result = np.asarray(p, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 6 or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(result.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError('Probabilities do not sum to one')
    return result/result.sum(1, keepdims=True)


def score(targets, probabilities):
    probabilities = normalized(probabilities)
    predictions = probabilities.argmax(1)
    return {'accuracy': float(accuracy_score(targets, predictions)),
            'macro_f1': float(f1_score(targets, predictions, labels=list(range(6)), average='macro', zero_division=0)),
            'log_loss': float(log_loss(targets, probabilities, labels=list(range(6))))}


@torch.inference_mode()
def predict_head(model, array, indices, first, last, batch_size):
    model.eval()
    device = next(model.parameters()).device
    probabilities = np.empty((len(indices), 6), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start+batch_size]
        x = torch.from_numpy(np.array(array[chosen, first:last], copy=True)).to(device)
        probabilities[start:start+len(chosen)] = model(x).softmax(1).cpu().numpy()
    return normalized(probabilities)


def write_predictions(path, rows, indices, targets, probabilities, groups, folds):
    records = []
    for index, probability in zip(indices, probabilities):
        records.append({'cache_row': int(index), 'manifest_index': int(rows[index]['manifest_index']),
                        'path': rows[index]['path'], 'group': str(groups[index]), 'fold': int(folds[index]),
                        'target': int(targets[index]), 'prediction': int(probability.argmax()),
                        **{f'p_{name}': float(probability[j]) for j, name in enumerate(CLASSES)}})
    write_csv(path, records, ['cache_row','manifest_index','path','group','fold','target','prediction']+
              [f'p_{name}' for name in CLASSES])


def train_fold(folder, candidate, fold, rows, targets, folds, groups, arrays, config, run_signature):
    name, first, last, kind = candidate
    train_indices, heldout_indices = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if set(groups[train_indices]) & set(groups[heldout_indices]):
        raise ValueError('Group leakage at training boundary')
    contract = {'run_signature': run_signature, 'candidate': name, 'feature_slice': [first,last], 'kind': kind,
                'fold': fold, 'seed': config['seed']+fold, 'epochs': config['epochs'],
                'train_cache_indices': train_indices.tolist(), 'heldout_cache_indices': heldout_indices.tolist(),
                'train_manifest_indices': [int(rows[i]['manifest_index']) for i in train_indices],
                'heldout_manifest_indices': [int(rows[i]['manifest_index']) for i in heldout_indices]}
    fold_signature = signature(contract)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('state') != 'complete' or summary.get('fold_signature') != fold_signature:
            raise ValueError('Completed fold belongs to different source/config/data')
        for artifact in FOLD_ARTIFACTS:
            verify_hash(folder/artifact, summary['artifact_sha256'].get(artifact))
        if summary['epochs_completed'] != config['epochs'] or summary['contract'] != contract:
            raise ValueError('Completed fold epoch/membership contract differs')
        cached = normalized(np.load(folder/'validation_probabilities.npy', allow_pickle=False))
        expected_metrics = score(targets[heldout_indices], cached)
        if any(not np.isclose(expected_metrics[key], summary['metrics'][key], atol=1e-12, rtol=1e-12)
               for key in expected_metrics):
            raise ValueError('Cached fold metrics differ from its checksummed heldout probabilities')
        print(f'Reused verified completed fold: {name} / {fold}', flush=True)
        return summary, cached
    config_path = folder/'fold_contract.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != contract:
        raise ValueError('Partial fold source/config/data changed')
    write_json(config_path, contract)
    seed = config['seed']+fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    scaler = fit_scaler(arrays['original'], train_indices, first, last)
    model = FrozenFeatureHead(last-first, kind, scaler.mean_, scaler.scale_).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_function = nn.CrossEntropyLoss()
    history, started = [], time.perf_counter()
    if str(config['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config['epochs']+1):
        model.train()
        order = rng.permutation(train_indices)
        view_digest = hashlib.sha256()
        loss_sum, correct, seen = 0.0, 0, 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), config['batch_size']):
            indices = order[start:start+config['batch_size']]
            x_array = np.array(arrays['original'][indices, first:last], copy=True)
            flip = rng.random(len(indices)) < .5
            view_digest.update(flip.tobytes())
            x_array[flip] = arrays['flip'][indices[flip], first:last]
            x = torch.from_numpy(x_array).to(config['device'])
            target = torch.from_numpy(targets[indices].copy()).to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_function(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite head training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
            correct += int((logits.detach().argmax(1) == target).sum())
            seen += len(indices)
            del x, target, logits, loss
        entry = {'epoch': epoch, 'train_loss': loss_sum/seen, 'train_accuracy': correct/seen,
                 'epoch_seconds': time.perf_counter()-epoch_start,
                 'training_order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),
                 'flip_assignment_sha256':view_digest.hexdigest()}
        history.append(entry)
        write_csv(folder/'history.csv', history, ['epoch','train_loss','train_accuracy','epoch_seconds',
                                                'training_order_sha256','flip_assignment_sha256'])
        print(json.dumps({'candidate': name, 'fold': fold, **entry}), flush=True)
    # Exactly one heldout evaluation after the prespecified final epoch.
    original_p = predict_head(model, arrays['original'], heldout_indices, first, last, config['batch_size'])
    flip_p = predict_head(model, arrays['flip'], heldout_indices, first, last, config['batch_size'])
    probabilities = normalized((original_p+flip_p)/2)
    fold_metrics = score(targets[heldout_indices], probabilities)
    checkpoint = {'format': 'frozen_multilayer_cv_head_v1', 'classes': CLASSES, 'contract': contract,
                  'config': config, 'fold_signature': fold_signature,
                  'model_state': {key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                  'normalization_fit': 'fold training original features only',
                  'train_paths': [rows[i]['path'] for i in train_indices],
                  'heldout_paths': [rows[i]['path'] for i in heldout_indices],
                  'train_groups': sorted(set(groups[train_indices].tolist())),
                  'heldout_groups': sorted(set(groups[heldout_indices].tolist())),
                  'epochs_completed': config['epochs'], 'epoch_selection': False, 'tta_flip': True}
    temporary_model = folder/'model.pt.tmp'
    torch.save(checkpoint, temporary_model); os.replace(temporary_model, folder/'model.pt')
    for artifact, array in [('validation_original.npy',original_p),('validation_flip.npy',flip_p),
                            ('validation_probabilities.npy',probabilities)]:
        np.save(folder/artifact, array, allow_pickle=False)
    write_predictions(folder/'validation_predictions.csv', rows, heldout_indices, targets, probabilities, groups, folds)
    peak = float(torch.cuda.max_memory_allocated()/1024**3) if str(config['device']).startswith('cuda') else None
    summary = {'state': 'complete', 'candidate': name, 'fold': fold, 'fold_signature': fold_signature,
               'contract': contract, 'epochs_completed': config['epochs'], 'metrics': fold_metrics,
               'train_rows': len(train_indices), 'heldout_rows': len(heldout_indices),
               'train_group_count': len(set(groups[train_indices])), 'heldout_group_count': len(set(groups[heldout_indices])),
               'peak_allocated_gpu_gib': peak, 'elapsed_seconds': time.perf_counter()-started,
               'artifact_sha256': {artifact:sha256(folder/artifact) for artifact in FOLD_ARTIFACTS}}
    write_json(summary_path, summary)
    del optimizer, model
    if str(config['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    print(json.dumps({'candidate': name, 'fold': fold, 'heldout': fold_metrics}), flush=True)
    return summary, probabilities


def paired_counts(targets, probability_a, probability_b):
    a, b = probability_a.argmax(1) == targets, probability_b.argmax(1) == targets
    return {'both_correct': int((a&b).sum()), 'a_only_correct': int((a&~b).sum()),
            'b_only_correct': int((~a&b).sum()), 'both_wrong': int((~a&~b).sum()),
            'net_correct_b_minus_a': int(b.sum()-a.sum()),
            'prediction_disagreements': int((probability_a.argmax(1) != probability_b.argmax(1)).sum())}


def run(args):
    root = Path(args.project_root).resolve()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('epochs, batch-size and lr must be positive')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; CPU requires explicit --device cpu')
    torch.set_num_threads(4)
    rows, targets, folds, groups, arrays, data_receipt = load_training_inputs(root)
    experiment = root/'experiments'/'multilayer_cv'
    report_folder = root/'multilayer_results'
    experiment.mkdir(parents=True, exist_ok=True); report_folder.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'classes': CLASSES, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'lr': args.lr, 'weight_decay': .001, 'dropout_mlp': .3, 'hidden_dim': 256,
              'seed': args.seed, 'device': str(args.device), 'dtype': 'float32', 'optimizer': 'AdamW',
              'normalization': 'fit each fold training originals only; same scaler for original and flip',
              'augmentation': 'one randomly chosen original/flip feature per training image per epoch',
              'validation': 'fixed epoch, mean original/flip softmax, no early stop or epoch selection',
              'folds': 3, 'candidate_definitions': CANDIDATES, 'source_sha256': sha256(__file__),
              'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
              'data_receipt': data_receipt, 'old_validation_test_unlabeled_read': False,
              'backbones': 'only frozen official pretrained DINOv2; no dataset-fitted CNN/LoRA features'}
    # JSON roundtrip makes tuple/list equality stable across restarted processes.
    config = json.loads(json.dumps(config))
    run_signature = signature(config)
    config_path = experiment/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing CV experiment has different data/source/config; preserve it and use another project copy')
    write_json(config_path, config)
    previous_report_path = report_folder/'summary.json'
    if previous_report_path.exists():
        previous_report = json.loads(previous_report_path.read_text(encoding='utf-8'))
        if previous_report.get('run_signature') != run_signature:
            raise ValueError('Previous multilayer report belongs to a different run')
        for path, digest in previous_report['artifact_sha256'].items():
            verify_hash(path, digest)
        print('Verified completed multilayer CV report; no training or fold reevaluation repeated.', flush=True)
        return previous_report
    candidates, oof_arrays, all_artifacts = [], {}, {}
    for candidate in CANDIDATES:
        name = candidate[0]
        folder = experiment/name
        oof = np.empty((len(rows), 6), dtype=np.float64)
        visited = np.zeros(len(rows), dtype=np.int64)
        fold_summaries = []
        for fold in range(3):
            fold_folder = folder/f'fold_{fold}'
            summary, probabilities = train_fold(fold_folder, candidate, fold, rows, targets, folds, groups,
                                                 arrays, config, run_signature)
            selected = np.flatnonzero(folds == fold)
            if probabilities.shape != (len(selected), 6):
                raise ValueError('Fold probabilities do not match heldout row order/count')
            oof[selected] = probabilities; visited[selected] += 1
            fold_summaries.append(summary)
            for artifact in FOLD_ARTIFACTS+['summary.json','fold_contract.json']:
                all_artifacts[str(fold_folder/artifact)] = sha256(fold_folder/artifact)
        if not np.all(visited == 1):
            raise ValueError('OOF rows must be predicted exactly once by their heldout fold')
        oof_arrays[name] = oof
        np.save(folder/'oof_probabilities.npy', oof, allow_pickle=False)
        write_predictions(folder/'oof_predictions.csv', rows, np.arange(len(rows)), targets, oof, groups, folds)
        fold_metrics = [summary['metrics'] for summary in fold_summaries]
        pooled = score(targets, oof)
        result = {'candidate': name, 'feature_dim': candidate[2]-candidate[1], 'kind': candidate[3],
                  'per_fold': [{'fold': i, 'heldout_rows': summary['heldout_rows'], **summary['metrics']}
                               for i,summary in enumerate(fold_summaries)],
                  'mean_accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
                  'mean_macro_f1': float(np.mean([m['macro_f1'] for m in fold_metrics])),
                  'mean_log_loss': float(np.mean([m['log_loss'] for m in fold_metrics])),
                  'std_accuracy': float(np.std([m['accuracy'] for m in fold_metrics],ddof=1)),
                  'pooled_oof': pooled, 'confusion_matrix': confusion_matrix(targets,oof.argmax(1),labels=list(range(6))).tolist()}
        write_json(folder/'summary.json', result)
        for artifact in ['oof_probabilities.npy','oof_predictions.csv','summary.json']:
            all_artifacts[str(folder/artifact)] = sha256(folder/artifact)
        candidates.append(result)
    paired = []
    names = list(oof_arrays)
    for index,a in enumerate(names):
        for b in names[index+1:]:
            paired.append({'a': a, 'b': b, **paired_counts(targets,oof_arrays[a],oof_arrays[b]),
                           'per_fold': [{'fold': fold, **paired_counts(targets[folds==fold],oof_arrays[a][folds==fold],
                                                                     oof_arrays[b][folds==fold])} for fold in range(3)]})
    winner = max(candidates,key=lambda row:(row['mean_accuracy'],row['mean_macro_f1'],-row['mean_log_loss']))
    comparisons=[]
    for kind in ['linear','mlp']:
        a,b = f'last_layer_{kind}',f'last4_{kind}'
        matching = next(row for row in paired if row['a']==a and row['b']==b)
        comparisons.append({'head':kind,'baseline':a,'multilayer':b,
                            'fold_net_correct': [row['net_correct_b_minus_a'] for row in matching['per_fold']],
                            'fold_accuracy_difference': [float((oof_arrays[b][folds==fold].argmax(1)==targets[folds==fold]).mean()-
                                                                (oof_arrays[a][folds==fold].argmax(1)==targets[folds==fold]).mean())
                                                         for fold in range(3)],
                            'pooled_net_correct': matching['net_correct_b_minus_a'],
                            'last4_corrected_baseline_errors':matching['b_only_correct'],
                            'last4_new_errors':matching['a_only_correct'],
                            'strictly_better_on_all_3_folds':all(row['net_correct_b_minus_a']>0 for row in matching['per_fold'])})
    comparison_fields=['candidate','feature_dim','kind','mean_accuracy','mean_macro_f1','mean_log_loss','std_accuracy']
    write_csv(report_folder/'candidate_comparison.csv', [{key:row[key] for key in comparison_fields} for row in candidates],comparison_fields)
    paired_fields=['a','b','both_correct','a_only_correct','b_only_correct','both_wrong','net_correct_b_minus_a','prediction_disagreements']
    write_csv(report_folder/'paired_corrections.csv',[{key:row[key] for key in paired_fields} for row in paired],paired_fields)
    for path in [config_path,report_folder/'candidate_comparison.csv',report_folder/'paired_corrections.csv']:
        all_artifacts[str(path)]=sha256(path)
    summary={'version':VERSION,'state':'complete','run_signature':run_signature,'classes':CLASSES,'train_rows':len(rows),
             'groups':len(set(groups)),'folds':3,'epochs_per_fold':args.epochs,'candidates':candidates,
             'selected':winner,'selection_rule':'mean fold accuracy, then mean macro F1, then lower mean log loss',
             'paired_corrections':paired,'multilayer_vs_last_layer':comparisons,
             'old_validation_test_unlabeled_read':False,'external_answers_used':False,'submission_generated':False,
             'selection_caveat':'OOF estimates use group-separated fixed-epoch folds. The winner is selected on these same CV scores; '
                                'its selected score is not an unbiased final model-selection estimate. Three folds do not establish '
                                'statistical significance or guarantee unseen-data improvement. Fold standard deviation is descriptive, '
                                'not a confidence interval.',
             'data_receipt':data_receipt,'source_sha256':config['source_sha256'],'artifact_sha256':all_artifacts}
    write_json(previous_report_path,summary)
    print(json.dumps({'state':'complete','selected':winner,'multilayer_vs_last_layer':comparisons,
                      'report':str(previous_report_path),'selection_caveat':summary['selection_caveat']},ensure_ascii=False,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())


if __name__=='__main__':
    main()
