"""A separate four-epoch HSE continuation with train-only OOF sample weights.

--stage smoke checks a real weighted GPU update without saving model weights.
--stage train --mode downweighted trains with the uncertainty weights.
--stage train --mode uniform runs a matched unweighted continuation control.
The original 12-epoch experiment and its submission are never modified.
"""
from __future__ import annotations

import os
for _thread_option in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_thread_option] = '4'

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_hse_specialist as base
import prepare_ambiguity_weights as weights_api

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {'epochs': 4, 'batch_size': 8, 'accumulation': 4, 'effective_batch_size': 32,
            'head_warmup_epochs': 0, 'backbone_lr': 5e-6, 'head_lr': 5e-5,
            'min_lr_ratio': .1, 'weight_decay': .01, 'label_smoothing': .05,
            'seed': 42, 'workers': 0, 'cpu_threads': 4,
            'loss': 'mean(weight * per_image_CE), with training-class mean weight=1',
            'sampling': 'uniform shuffle; no weighted sampler',
            'validation': 'unweighted original-view metrics; original+flip stored separately',
            'initial_checkpoint': 'original HSE best epoch 7; also included as epoch-0 candidate'}


def weighted_loss(logits, targets, weights):
    if weights.shape != targets.shape or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Sample weights must be positive and aligned with targets')
    losses = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=.05)
    return (losses*weights).mean(), losses.mean()


def load_weights(root, rows):
    folder = root/'experiments'/'hse_ambiguity_v1'/'weights'
    summary = base.json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if summary.get('state') != 'complete' or summary.get('classes') != base.CLASSES or summary.get('train_rows') != len(rows):
        raise ValueError('Uncertainty weights are incomplete or use different classes/count')
    for path, expected in summary['binding']['source_sha256'].items():
        base.check_hash(path, expected)
    for name, expected in summary['file_sha256'].items():
        base.check_hash(folder/name, expected)
    with (folder/'train_weights.csv').open(encoding='utf-8-sig', newline='') as stream:
        entries = list(csv.DictReader(stream))
    weights = np.load(folder/'training_weights.npy', allow_pickle=False)
    if len(entries) != len(rows) or weights.shape != (len(rows),):
        raise ValueError('Weight row count differs from original training membership')
    for i, (row, entry) in enumerate(zip(rows, entries)):
        if (int(entry['manifest_index']) != row['manifest_index'] or entry['path'] != row['path']
            or entry['label'] != row['label'] or int(entry['target']) != row['target']
            or float(entry['training_weight']) != weights[i]):
            raise ValueError('Weight image/target/row differs from original training row')
    targets = np.array([row['target'] for row in rows])
    if not np.isfinite(weights).all() or (weights<.5).any():
        raise ValueError('Invalid or overly aggressive sample weights')
    for target in range(6):
        if not np.isclose(weights[targets==target].mean(), 1., atol=1e-12):
            raise ValueError('Training class total weight changed')
    return weights.astype(np.float32), summary, entries


def configuration(root, mode):
    rows, image_receipt = base.load_trainval(root)
    weights, weight_summary, entries = load_weights(root, rows['train'])
    original_folder = root/'experiments'/'hse_specialist_v1'
    initial_receipt = json.loads((original_folder/'summary.json').read_text(encoding='utf-8'))
    if initial_receipt.get('state') != 'complete' or initial_receipt.get('best_epoch') != 7 or initial_receipt.get('input_sha256') != image_receipt:
        raise ValueError('The initial HSE checkpoint has different training/validation membership')
    for name in ['best.pt', 'val_original.npy', 'val_flip.npy']:
        base.check_hash(original_folder/name, initial_receipt['artifact_sha256'][name])
    base.check_hash(root/'hse_results'/'submission.csv', weights_api.PRESERVED_SUBMISSION)
    config = {'format': 'hse_ambiguity_continuation_v1', 'mode': mode, 'settings': SETTINGS,
              'classes': base.CLASSES, 'model_provenance': base.model_api().source_receipt(root),
              'input_receipt': image_receipt, 'initial_checkpoint_sha256': initial_receipt['best_checkpoint_sha256'],
              'weight_summary_sha256': base.sha256(root/'experiments'/'hse_ambiguity_v1'/'weights'/'summary.json'),
              'train_indices': [row['manifest_index'] for row in rows['train']], 'val_indices': [row['manifest_index'] for row in rows['val']],
              'source_sha256': {name: base.sha256(root/'src'/name) for name in ['train_ambiguity_hse.py', 'prepare_ambiguity_weights.py', 'train_hse_specialist.py', 'hse_model.py']},
              'baseline_submission_sha256': weights_api.PRESERVED_SUBMISSION,
              'weight_selection_used_validation': False, 'internal_test_read': False, 'external_answers_used': False}
    if mode == 'uniform':
        weights = np.ones(len(weights), dtype=np.float32)
    return rows, weights, entries, config, initial_receipt


class WeightedImages(Dataset):
    def __init__(self, rows, weights):
        self.images = base.ExpressionImages(rows, True)
        self.weights = weights
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image, target = self.images[index]
        return image, target, float(self.weights[index])


def set_training(model):
    model.requires_grad_(True); model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_epoch(model, rows, weights, optimizer, scaler, epoch, device):
    set_training(model)
    factor = .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/3))
    for group in optimizer.param_groups:
        group['lr'] = SETTINGS[group['name']+'_lr']*factor
    generator = torch.Generator().manual_seed(42+1009*epoch)
    loader = DataLoader(WeightedImages(rows, weights), batch_size=8, shuffle=True, num_workers=0,
                        pin_memory=str(device).startswith('cuda'), generator=generator)
    optimizer.zero_grad(set_to_none=True)
    seen, correct, weighted_sum, plain_sum, steps, skipped = 0, 0, 0., 0., 0, 0
    started = time.monotonic()
    for index, (images, targets, weight) in enumerate(loader):
        images, targets, weight = images.to(device), targets.to(device), weight.to(device, dtype=torch.float32)
        group_samples = min(32, len(rows)-(index//4)*32)
        with base.autocast_for(device):
            logits = model(images)
            loss, plain = weighted_loss(logits, targets, weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite weighted training loss')
        scaler.scale(loss*(len(targets)/group_samples)).backward()
        weighted_sum += float(loss.detach())*len(targets)
        plain_sum += float(plain.detach())*len(targets)
        correct += int((logits.detach().argmax(1)==targets).sum()); seen += len(targets)
        if (index+1)%4 == 0 or index+1 == len(loader):
            _, did_skip = base.optimizer_step(model, optimizer, scaler)
            skipped += int(did_skip); steps += int(not did_skip)
        if (index+1)%50 == 0 or index+1 == len(loader):
            print(json.dumps({'stage': 'train', 'epoch': epoch, 'batch': index+1, 'batches': len(loader),
                'train_loss': weighted_sum/seen, 'unweighted_train_loss': plain_sum/seen,
                'successful_optimizer_steps': steps, 'amp_skipped_steps': skipped,
                'elapsed_seconds': time.monotonic()-started}), flush=True)
    if steps == 0:
        raise RuntimeError('No successful weighted optimizer update')
    return {'train_loss': weighted_sum/seen, 'unweighted_train_loss': plain_sum/seen,
            'train_accuracy': correct/seen, 'successful_optimizer_steps': steps, 'amp_skipped_steps': skipped,
            'lr_backbone': optimizer.param_groups[0]['lr'], 'lr_head': optimizer.param_groups[1]['lr'],
            'train_seconds': time.monotonic()-started}


def materialize(folder, saved, config):
    base.save_torch(folder/'best.pt', {'classes': base.CLASSES, 'model_provenance': config['model_provenance'],
        'model_state': saved['best_model_state'], 'best_epoch': saved['best_epoch'],
        'best_validation': saved['best_validation'], 'config': config, 'run_signature': base.signature(config)})
    base.save_array(folder/'val_original.npy', saved['best_original'].numpy())
    base.save_array(folder/'val_flip.npy', saved['best_flip'].numpy())
    if saved['history']:
        base.write_history(folder, saved['history'])


def smoke(root, mode, device):
    rows, weights, entries, config, _ = configuration(root, mode)
    base.seed_everything(42)
    indices = []
    for bucket, count in [('normal', 4), ('uncertain', 2), ('very_uncertain', 2)]:
        indices += [i for i, entry in enumerate(entries) if entry['difficulty']==bucket][:count]
    if len(indices) != 8:
        raise ValueError('Smoke requires examples from all three weight buckets')
    chosen = [rows['train'][i] for i in indices]
    loader = DataLoader(WeightedImages(chosen, weights[indices]), batch_size=8, num_workers=0)
    model = base.model_api().load_trained_model(root, root/'experiments'/'hse_specialist_v1'/'best.pt', device)
    set_training(model)
    before_bn = {name: value.detach().clone() for name, value in model.named_buffers() if 'running_' in name or 'num_batches_tracked' in name}
    parameter = next(value for name,value in model.named_parameters() if 'classifier' not in name and value.requires_grad)
    before_parameter = parameter.detach().clone()
    optimizer, scaler = base.optimizer_for(model), base.make_scaler(device)
    images, targets, w = next(iter(loader))
    images, targets, w = images.to(device), targets.to(device), w.to(device, dtype=torch.float32)
    with base.autocast_for(device):
        logits = model(images)
        loss, plain = weighted_loss(logits, targets, w)
    scaler.scale(loss).backward()
    norm, skipped = base.optimizer_step(model, optimizer, scaler)
    if skipped or norm is None or norm<=0 or torch.equal(before_parameter, parameter):
        raise RuntimeError('Weighted smoke did not perform a finite backbone update')
    if any(not torch.equal(value, dict(model.named_buffers())[name]) for name,value in before_bn.items()):
        raise RuntimeError('BatchNorm running statistics changed')
    receipt = {'state': 'passed', 'stage': 'smoke', 'mode': mode, 'sample_weights': w.cpu().tolist(),
        'weighted_loss': float(loss.detach()), 'unweighted_loss': float(plain.detach()), 'gradient_norm': norm,
        'backbone_updated': True, 'bn_running_stats_unchanged': True, 'amp_skipped_steps': int(skipped),
        'config_signature': base.signature(config), 'model_weights_saved': False,
        'old_submission_preserved': base.sha256(root/'hse_results'/'submission.csv')==weights_api.PRESERVED_SUBMISSION}
    folder = root/'experiments'/'hse_ambiguity_v1'/mode; folder.mkdir(parents=True, exist_ok=True)
    base.write_json(folder/'smoke.json', receipt)
    print(json.dumps(receipt, ensure_ascii=True), flush=True)


def train(root, mode, device):
    rows, weights, _, config, initial = configuration(root, mode)
    folder = root/'experiments'/'hse_ambiguity_v1'/mode; folder.mkdir(parents=True, exist_ok=True)
    config_path = folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing weighted continuation source/data/configuration changed')
    base.write_json(config_path, config)
    if (folder/'summary.json').exists():
        completed = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if completed['state'] != 'complete' or completed['run_signature'] != base.signature(config):
            raise ValueError('Completed continuation has a different configuration')
        for name, expected in completed['file_sha256'].items():
            base.check_hash(folder/name, expected)
        return completed
    base.seed_everything(42)
    model = base.model_api().load_trained_model(root, root/'experiments'/'hse_specialist_v1'/'best.pt', device)
    optimizer, scaler = base.optimizer_for(model), base.make_scaler(device)
    initial_folder = root/'experiments'/'hse_specialist_v1'
    original = base.probability(np.load(initial_folder/'val_original.npy', allow_pickle=False))
    flipped = base.probability(np.load(initial_folder/'val_flip.npy', allow_pickle=False))
    initial_metrics = base.metrics(rows['val'], original)
    if any(not np.isclose(initial_metrics[key], initial['best_validation'][key], atol=1e-12) for key in initial_metrics):
        raise ValueError('Initial validation cache differs from the original selected checkpoint')
    saved = {'epoch': 0, 'history': [], 'best_epoch': 0, 'best_validation': initial_metrics,
             'best_model_state': base.cpu_tree(model.state_dict()), 'best_original': torch.from_numpy(original),
             'best_flip': torch.from_numpy(flipped)}
    if (folder/'last.json').exists():
        pointer = json.loads((folder/'last.json').read_text(encoding='utf-8'))
        base.check_hash(folder/pointer['checkpoint'], pointer['sha256'])
        saved = torch.load(folder/pointer['checkpoint'], map_location='cpu', weights_only=True)
        if saved['config'] != config or saved['epoch'] != pointer['epoch'] or pointer['run_signature'] != base.signature(config):
            raise ValueError('Continuation checkpoint belongs to a different run')
        model.load_state_dict(saved['model_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state']); scaler.load_state_dict(saved['scaler_state'])
        base.restore_rng(saved['rng_state']); materialize(folder, saved, config)
        print(json.dumps({'stage': 'resume', 'committed_epoch': saved['epoch']}), flush=True)
    (folder/'epochs').mkdir(exist_ok=True)
    for epoch in range(saved['epoch']+1, 5):
        record = train_epoch(model, rows['train'], weights, optimizer, scaler, epoch, device)
        original, flipped = base.predict_validation(model, rows['val'], epoch, device)
        metrics = base.metrics(rows['val'], original)
        if base.metric_key(metrics)>base.metric_key(saved['best_validation']):
            saved.update(best_epoch=epoch, best_validation=metrics, best_model_state=base.cpu_tree(model.state_dict()),
                         best_original=torch.from_numpy(original), best_flip=torch.from_numpy(flipped))
        saved['history'].append({'epoch': epoch, **record, **{'val_'+key: value for key,value in metrics.items()}, 'best_epoch': saved['best_epoch']})
        saved.update(epoch=epoch, config=config, model_state=base.cpu_tree(model.state_dict()),
                     optimizer_state=base.cpu_tree(optimizer.state_dict()), scaler_state=scaler.state_dict(), rng_state=base.rng_state())
        path = folder/'epochs'/f'epoch_{epoch:02d}.pt'; base.save_torch(path, saved)
        base.write_json(folder/'last.json', {'epoch': epoch, 'checkpoint': str(path.relative_to(folder)),
                                          'sha256': base.sha256(path), 'run_signature': base.signature(config)})
        materialize(folder, saved, config)
        print(json.dumps({'stage': 'epoch_complete', 'epoch': epoch, **metrics, 'best_epoch': saved['best_epoch']}), flush=True)
    names = ['config.json', 'best.pt', 'val_original.npy', 'val_flip.npy', 'history.csv', 'last.json']
    summary = {'state': 'complete', 'mode': mode, 'epochs_completed': 4, 'classes': base.CLASSES,
        'best_epoch': saved['best_epoch'], 'best_validation': saved['best_validation'], 'initial_validation': initial_metrics,
        'config': config, 'run_signature': base.signature(config), 'validation_weights_used': False,
        'internal_test_read': False, 'external_answers_used': False,
        'preserved_submission_sha256': base.sha256(root/'hse_results'/'submission.csv'),
        'file_sha256': {name: base.sha256(folder/name) for name in names}}
    if summary['preserved_submission_sha256'] != weights_api.PRESERVED_SUBMISSION:
        raise ValueError('Original latest best submission changed')
    base.write_json(folder/'summary.json', summary)
    print(json.dumps({'stage': 'train', 'state': 'complete', 'mode': mode, 'best_epoch': saved['best_epoch'], 'best_validation': saved['best_validation']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--stage', required=True, choices=['smoke', 'train'])
    parser.add_argument('--mode', choices=['uniform', 'downweighted'], default='downweighted')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    (smoke if args.stage=='smoke' else train)(args.project_root.resolve(), args.mode, args.device)


if __name__ == '__main__':
    main()
