"""Refit the already selected cosine MLP on all original training images.

The architecture and twelve-epoch schedule are fixed before this run. This module
reads only train-specific frozen feature/group artifacts and the completed CV
selection receipts. It does not open any validation/test/competition images or
labels, choose an epoch, or produce predictions. An interrupted run restarts at
epoch one; a completed run is reused only after checking its inputs and outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate_multilayer_cv as base
import evaluate_weighted_heads as weighted

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
EXPECTED_TRAIN_ROWS = 21248
CLASSES = base.CLASSES
ARTIFACTS = ['model.pt', 'history.csv', 'train_rows.csv', 'config.json']
HISTORY_FIELDS = ['epoch', 'lr', 'train_loss', 'train_accuracy', 'epoch_seconds',
                  'training_order_sha256', 'flip_assignment_sha256']
FIXED = {'epochs': 12, 'batch_size': 256, 'lr': .001, 'min_lr': 1e-5, 'weight_decay': .001,
         'gradient_clip': 5., 'seed': 42, 'feature_dim': 3840, 'hidden_dim': 256, 'dropout': .3,
         'optimizer': 'AdamW', 'label_smoothing': 0., 'mixup': False, 'ema': False,
         'sample_weighting': False, 'epoch_selection': False}


def source_receipt(root):
    modules = [base, weighted, weighted.weight_builder]
    sources = {}
    for module in modules:
        name = Path(module.__file__).name
        sources[name] = base.sha256(module.__file__)
        base.verify_hash(root/'src'/name, sources[name])
    sources[Path(__file__).name] = base.sha256(__file__)
    base.verify_hash(root/'src'/Path(__file__).name, sources[Path(__file__).name])
    return sources


def selected_cosine_receipt(root, data_receipt, sources, row_count):
    """Accept the saved choice without re-ranking or re-evaluating any CV fold."""
    summary_path = root/'weighted_results'/'summary.json'
    config_path = root/'experiments'/'weighted_heads'/'run_config.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (summary.get('state') != 'complete' or summary.get('classes') != CLASSES or
            summary.get('train_rows') != row_count or summary.get('folds') != 3 or
            summary.get('epochs_per_fold') != 12 or summary.get('selected', {}).get('candidate') != 'cosine'):
        raise ValueError('Require completed twelve-epoch grouped CV with saved cosine selection')
    base.verify_hash(config_path, summary.get('artifact_sha256', {}).get(str(config_path)))
    if summary.get('run_signature') != base.signature(config):
        raise ValueError('Weighted CV run signature differs from its configuration')
    expected_sources = {key: sources[key] for key in ['evaluate_multilayer_cv.py', 'evaluate_weighted_heads.py',
                                                     'image_sample_weights.py']}
    if summary.get('source_sha256') != expected_sources or config.get('source_sha256') != expected_sources:
        raise ValueError('Selected CV helper source hashes differ from the currently imported helpers')
    if summary.get('data_receipt') != data_receipt or config.get('data_receipt') != data_receipt:
        raise ValueError('Selected CV and full-training refit reference different train caches/groups')
    if config.get('candidate_definitions') != weighted.CANDIDATES:
        raise ValueError('Selected CV candidate definitions differ')
    required = {'epochs': 12, 'batch_size': 256, 'lr': .001, 'weight_decay': .001, 'seed': 42,
                'hidden_dim': 256, 'dropout_mlp': .3, 'feature_slice': [0, 3840], 'kind': 'mlp',
                'optimizer': 'AdamW', 'gradient_clip': 5., 'folds': 3, 'dtype': 'float32', 'classes': CLASSES}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Selected CV hyperparameters differ from this fixed refit protocol')
    candidates = summary.get('candidates', [])
    if [item.get('candidate') for item in candidates] != [item['name'] for item in weighted.CANDIDATES]:
        raise ValueError('Completed CV must contain the seven prespecified candidate receipts')
    for item, definition in zip(candidates, weighted.CANDIDATES):
        if item.get('definition') != definition or item.get('feature_dim') != 3840 or item.get('kind') != 'mlp':
            raise ValueError('Completed candidate receipt differs from its prespecified definition')
    cosine = next(item for item in candidates if item['candidate'] == 'cosine')
    if summary['selected'] != cosine:
        raise ValueError('Saved selected candidate is not the cosine receipt in the candidate list')
    return {'selected_candidate': 'cosine', 'selection_recomputed': False,
            'weighted_summary_sha256': base.sha256(summary_path), 'weighted_config_sha256': base.sha256(config_path),
            'weighted_run_signature': summary['run_signature'],
            'selection_caveat': 'Cosine was selected on reused grouped CV; this refit does not provide an independent score.'}


def load_inputs_and_config(root):
    root = Path(root).resolve()
    sources = source_receipt(root)
    rows, targets, folds, groups, arrays, data_receipt = base.load_training_inputs(root)
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f'Require exactly {EXPECTED_TRAIN_ROWS} original training rows')
    selection = selected_cosine_receipt(root, data_receipt, sources, len(rows))
    config = {'format': 'cosine_full_train_config_v1', 'version': VERSION, 'classes': CLASSES,
              'fixed_training': FIXED, 'train_rows': len(rows), 'source_sha256': sources,
              'data_receipt': data_receipt, 'selection_receipt': selection,
              'normalization': 'StandardScaler fitted on every original training feature row only, in chunks',
              'sampling': 'uniform permutation each epoch, independently seeded numpy RNG chooses original/flip with p=0.5',
              'validation_used': False, 'test_used': False, 'unlabeled_used': False,
              'restart_strategy': 'incomplete run restarts from epoch 1 using fixed seed; completed outputs require hash verification',
              'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)}
    return rows, targets, arrays, json.loads(json.dumps(config))


def buffer_receipt(state):
    result = {}
    for name in ['feature_mean', 'feature_scale']:
        value = state[name].detach().cpu().numpy()
        if value.shape != (3840,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid saved standardization buffer')
        if name == 'feature_scale' and (value <= 0).any():
            raise ValueError('Saved standardization scale must be positive')
        result[name+'_sha256'] = hashlib.sha256(value.tobytes()).hexdigest()
    return result


def verify_completed(root, rows, config):
    folder = root/'experiments'/'cosine_full_train'
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('format') != 'cosine_full_train_receipt_v1' or summary.get('state') != 'complete' or
            summary.get('classes') != CLASSES or summary.get('train_rows') != len(rows) or
            summary.get('epochs_completed') != 12 or summary.get('config') != config or
            summary.get('run_signature') != base.signature(config) or
            summary.get('source_sha256') != config['source_sha256'] or summary.get('epoch_selection') is not False):
        raise ValueError('Completed full-training head has a different or incomplete contract')
    if set(summary.get('artifact_sha256', {})) != set(ARTIFACTS):
        raise ValueError('Completed refit artifact list differs')
    for name in ARTIFACTS:
        base.verify_hash(folder/name, summary['artifact_sha256'][name])
    if summary.get('model_sha256') != summary['artifact_sha256']['model.pt']:
        raise ValueError('Model weight hash differs from completion receipt')
    if json.loads((folder/'config.json').read_text(encoding='utf-8')) != config:
        raise ValueError('Saved refit configuration differs')
    saved_rows = base.read_csv(folder/'train_rows.csv', ['manifest_index', 'path', 'label', 'target'])
    if saved_rows != rows:
        raise ValueError('Saved training membership/order differs from the verified cache')
    history = base.read_csv(folder/'history.csv', HISTORY_FIELDS)
    if ([int(item['epoch']) for item in history] != list(range(1,13)) or
            any(float(item['lr']) != weighted.cosine_lr(i, 12, .001, 1e-5) for i, item in enumerate(history, 1))):
        raise ValueError('Completed refit history does not contain the fixed twelve-epoch cosine schedule')
    return summary


def load_trained_head(root, device='cpu'):
    """Return an eval-mode raw-3840-feature model and verified completion receipt.

    Logit columns follow [Angry,Fear,Happy,Sad,Surprise,Neutral]. This verifies
    sources, training caches, selection receipt, saved membership, and model
    checksum. It neither fits a scaler nor reads any validation/test inputs.
    """
    root = Path(root).resolve()
    rows, targets, arrays, config = load_inputs_and_config(root)
    summary = verify_completed(root, rows, config)
    checkpoint = torch.load(root/'experiments'/'cosine_full_train'/'model.pt', map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'cosine_full_train_head_v1' or checkpoint.get('classes') != CLASSES or
            checkpoint.get('config') != config or checkpoint.get('run_signature') != base.signature(config) or
            checkpoint.get('epochs_completed') != 12 or checkpoint.get('epoch_selection') is not False or
            checkpoint.get('train_manifest_indices') != [int(row['manifest_index']) for row in rows] or
            checkpoint.get('train_paths') != [row['path'] for row in rows] or
            checkpoint.get('normalizer') != summary['normalizer']):
        raise ValueError('Model checkpoint metadata/membership differs from its verified completion receipt')
    state = checkpoint['model_state']
    if buffer_receipt(state) != summary['normalizer'] or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Saved model state or standardization buffers are invalid')
    model = base.FrozenFeatureHead(3840, 'mlp')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, summary


def run(args):
    root = Path(args.project_root).resolve()
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use explicit --device cpu for CPU execution')
    torch.set_num_threads(4)
    rows, targets, arrays, config = load_inputs_and_config(root)
    folder = root/'experiments'/'cosine_full_train'
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Existing full-training run belongs to different inputs/source/config')
    else:
        if any(folder.iterdir()):
            raise ValueError('Nonempty output folder has no fixed refit contract')
        base.write_json(config_path, config)
    if (folder/'summary.json').exists():
        summary = verify_completed(root, rows, config)
        print('Reused verified completed full-training cosine head; no training or prediction repeated.', flush=True)
        return summary
    base.write_csv(folder/'train_rows.csv', rows, ['manifest_index', 'path', 'label', 'target'])
    seed = FIXED['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rows), dtype=np.int64)
    scaler = base.fit_scaler(arrays['original'], indices, 0, 3840)
    model = base.FrozenFeatureHead(3840, 'mlp', scaler.mean_, scaler.scale_).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    loss_function = nn.CrossEntropyLoss()
    started, history = time.perf_counter(), []
    for epoch in range(1,13):
        lr = weighted.cosine_lr(epoch, 12, .001, 1e-5)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        order = rng.permutation(indices)
        flip_digest = hashlib.sha256()
        loss_sum, correct, seen = 0., 0, 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), 256):
            chosen = order[start:start+256]
            data = np.array(arrays['original'][chosen, :3840], copy=True)
            flip = rng.random(len(chosen)) < .5
            flip_digest.update(flip.tobytes())
            data[flip] = arrays['flip'][chosen[flip], :3840]
            x = torch.from_numpy(data).to(args.device)
            target = torch.from_numpy(targets[chosen].copy()).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_function(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full-training head loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())*len(chosen)
            correct += int((logits.detach().argmax(1) == target).sum())
            seen += len(chosen)
            del x, target, logits, loss
        entry = {'epoch': epoch, 'lr': lr, 'train_loss': loss_sum/seen, 'train_accuracy': correct/seen,
                 'epoch_seconds': time.perf_counter()-epoch_start,
                 'training_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                 'flip_assignment_sha256': flip_digest.hexdigest()}
        history.append(entry); base.write_csv(folder/'history.csv', history, HISTORY_FIELDS)
        print(json.dumps({'stage': 'cosine_full_train', **entry}), flush=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    normalizer = buffer_receipt(state)
    checkpoint = {'format': 'cosine_full_train_head_v1', 'classes': CLASSES, 'config': config,
                  'run_signature': base.signature(config), 'epochs_completed': 12, 'epoch_selection': False,
                  'train_manifest_indices': [int(row['manifest_index']) for row in rows],
                  'train_paths': [row['path'] for row in rows], 'normalizer': normalizer, 'model_state': state}
    torch.save(checkpoint, folder/'model.pt.tmp'); os.replace(folder/'model.pt.tmp', folder/'model.pt')
    summary = {'format': 'cosine_full_train_receipt_v1', 'state': 'complete', 'classes': CLASSES,
               'train_rows': len(rows), 'epochs_completed': 12, 'epoch_selection': False,
               'run_signature': base.signature(config), 'config': config,
               'source_sha256': config['source_sha256'], 'data_receipt': config['data_receipt'],
               'normalizer': normalizer, 'model_sha256': base.sha256(folder/'model.pt'),
               'training_device': str(args.device), 'elapsed_seconds': time.perf_counter()-started,
               'train_metrics_purpose': 'online dropout/random-view diagnostic; no validation score in this stage',
               'validation_used': False, 'test_used': False, 'unlabeled_used': False,
               'artifact_sha256': {name: base.sha256(folder/name) for name in ARTIFACTS}}
    base.write_json(folder/'summary.json', summary)
    del optimizer, model, checkpoint, state
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    print(json.dumps({'stage': 'cosine_full_train', 'state': 'complete', 'model': str(folder/'model.pt')}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
