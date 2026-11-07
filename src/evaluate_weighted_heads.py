"""Seven prespecified single-factor experiments on frozen train-only features.

Uniform sampling is unchanged. Sample weights multiply per-image cross entropy;
their mean over this fold's training images is one, and a batch loss is
mean(weight * cross_entropy), never divided by the random batch weight sum.
Each heldout fold is evaluated only after the fixed final epoch, with unweighted
hard-label metrics and original/flip probability averaging. No old holdout,
competition image, answer file, or submission is read or created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import evaluate_multilayer_cv as base
import image_sample_weights as weight_builder

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
CLASSES = base.CLASSES
CANDIDATES = [
    {'name': 'baseline', 'weight_mode': 'uniform'},
    {'name': 'label_smoothing', 'weight_mode': 'uniform', 'label_smoothing': .05},
    {'name': 'cosine', 'weight_mode': 'uniform', 'min_lr': 1e-5},
    {'name': 'feature_mixup', 'weight_mode': 'uniform', 'mixup_alpha': .2, 'mixup_probability': .5},
    {'name': 'class_weight', 'weight_mode': 'class'},
    {'name': 'duplicate_weight', 'weight_mode': 'duplicate'},
    {'name': 'class_duplicate_weight', 'weight_mode': 'class_duplicate'},
]
FOLD_ARTIFACTS = base.FOLD_ARTIFACTS + ['image_weights.csv', 'weight_receipt.json']
HISTORY_FIELDS = ['epoch', 'lr', 'train_loss', 'train_accuracy', 'train_metrics_purpose',
                  'mixup_batches', 'epoch_seconds', 'training_order_sha256', 'flip_assignment_sha256']
REPORT_FILES = ['comparison.csv', 'paired_corrections.csv', 'class_recall.csv']


def cosine_lr(epoch, epochs, maximum, minimum=1e-5):
    """Epoch 1 uses maximum and the fixed last epoch uses minimum."""
    if not 1 <= epoch <= epochs or not 0 < minimum <= maximum:
        raise ValueError('Invalid epoch or cosine learning-rate endpoints')
    if epochs == 1:
        return float(maximum)
    return float(minimum + .5 * (maximum-minimum) * (1+math.cos(math.pi*(epoch-1)/(epochs-1))))


def weighted_ce(logits, targets, weights):
    """Weights are normalized over the complete training fold, not each batch."""
    if weights.ndim != 1 or weights.shape != targets.shape:
        raise ValueError('Per-image weights must align with batch targets')
    return (weights * F.cross_entropy(logits, targets, reduction='none')).mean()


def mixup_ce(logits, target_a, target_b, lam):
    if not 0 <= lam <= 1:
        raise ValueError('MixUp lambda must lie in [0,1]')
    return lam*F.cross_entropy(logits, target_a)+(1-lam)*F.cross_entropy(logits, target_b)


def get_weights(targets, groups, train_indices, mode):
    weights, receipt = weight_builder.build_sample_weights(targets, groups, train_indices, mode)
    weights = np.asarray(weights, dtype=np.float32)
    if (weights.shape != (len(train_indices),) or not np.isfinite(weights).all() or
            (weights <= 0).any() or not np.isclose(weights.mean(dtype=np.float64), 1., atol=1e-6, rtol=0)):
        raise ValueError('Training weights must be positive, aligned, and full-fold mean one')
    if mode == 'uniform' and not np.array_equal(weights, np.ones(len(train_indices), dtype=np.float32)):
        raise ValueError('Uniform baseline weights must be exactly one')
    receipt = json.loads(json.dumps(receipt))
    return weights, {'builder_receipt': receipt, 'mode': mode,
                     'effective_dtype': 'float32', 'effective_mean': float(weights.mean(dtype=np.float64)),
                     'effective_weight_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
                     'aligned_training_cache_indices_sha256': hashlib.sha256(train_indices.tobytes()).hexdigest(),
                     'loss_formula': 'mean_i_in_batch(w_i * CE_i); mean_i_in_fold_train(w_i) = 1',
                     'sampling': 'uniform permutation without replacement; no weighted sampler',
                     'view_policy': 'original and flip share the same image weight'}


def recall_report(targets, probabilities):
    cm = base.confusion_matrix(targets, probabilities.argmax(1), labels=list(range(6)))
    return [{'target': i, 'class': name, 'support': int(cm[i].sum()), 'correct': int(cm[i, i]),
             'recall': float(cm[i, i]/cm[i].sum()) if cm[i].sum() else 0.}
            for i, name in enumerate(CLASSES)]


def train_fold(folder, candidate, fold, rows, targets, folds, groups, arrays, config, run_signature):
    name = candidate['name']
    train_indices, heldout_indices = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if set(groups[train_indices]) & set(groups[heldout_indices]):
        raise ValueError('Group leakage at training boundary')
    weights, weight_receipt = get_weights(targets, groups, train_indices, candidate['weight_mode'])
    contract = {'run_signature': run_signature, 'candidate': candidate, 'feature_slice': [0, 3840], 'kind': 'mlp',
                'fold': fold, 'seed': config['seed']+fold, 'epochs': config['epochs'],
                'train_cache_indices': train_indices.tolist(), 'heldout_cache_indices': heldout_indices.tolist(),
                'train_manifest_indices': [int(rows[i]['manifest_index']) for i in train_indices],
                'heldout_manifest_indices': [int(rows[i]['manifest_index']) for i in heldout_indices],
                'weight_receipt': weight_receipt}
    fold_signature = base.signature(contract)
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary.get('state') != 'complete' or summary.get('fold_signature') != fold_signature or
                summary.get('epochs_completed') != config['epochs'] or summary.get('contract') != contract):
            raise ValueError('Completed fold source/config/data/weight contract differs')
        if set(summary.get('artifact_sha256', {})) != set(FOLD_ARTIFACTS):
            raise ValueError('Completed fold artifact list differs')
        for artifact in FOLD_ARTIFACTS:
            base.verify_hash(folder/artifact, summary['artifact_sha256'][artifact])
        if json.loads((folder/'fold_contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Saved fold contract differs')
        if json.loads((folder/'weight_receipt.json').read_text(encoding='utf-8')) != weight_receipt:
            raise ValueError('Saved training weight receipt differs')
        cached = base.normalized(np.load(folder/'validation_probabilities.npy', allow_pickle=False))
        original_p = base.normalized(np.load(folder/'validation_original.npy', allow_pickle=False))
        flip_p = base.normalized(np.load(folder/'validation_flip.npy', allow_pickle=False))
        if (cached.shape != (len(heldout_indices), 6) or original_p.shape != cached.shape or
                flip_p.shape != cached.shape or not np.allclose(cached, (original_p+flip_p)/2, atol=1e-12, rtol=0)):
            raise ValueError('Cached fold TTA probabilities/order/count differ')
        expected = base.score(targets[heldout_indices], cached)
        if any(not np.isclose(expected[k], summary['metrics'][k], atol=1e-12, rtol=1e-12) for k in expected):
            raise ValueError('Cached fold metrics disagree with checksummed probabilities')
        print(f'Reused verified completed fold: {name} / {fold}', flush=True)
        return summary, cached
    contract_path = folder/'fold_contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8')) != contract:
        raise ValueError('Partial fold source/config/data/weights changed')
    if not contract_path.exists() and any(folder.iterdir()):
        raise ValueError('Nonempty fold directory lacks its experiment contract')
    # A partial fold starts again from epoch 1 using its fixed seed. No epoch resume is claimed.
    base.write_json(contract_path, contract)
    base.write_json(folder/'weight_receipt.json', weight_receipt)
    base.write_csv(folder/'image_weights.csv', [
        {'cache_row': int(i), 'manifest_index': int(rows[i]['manifest_index']), 'path': rows[i]['path'],
         'target': int(targets[i]), 'label': CLASSES[int(targets[i])], 'group': str(groups[i]), 'weight': float(w)}
        for i, w in zip(train_indices, weights)],
        ['cache_row', 'manifest_index', 'path', 'target', 'label', 'group', 'weight'])
    full_weights = np.zeros(len(rows), dtype=np.float32)
    full_weights[train_indices] = weights
    seed = config['seed']+fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    mixup_rng = np.random.default_rng(seed+1000003)
    scaler = base.fit_scaler(arrays['original'], train_indices, 0, 3840)
    model = base.FrozenFeatureHead(3840, 'mlp', scaler.mean_, scaler.scale_).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    loss_function = nn.CrossEntropyLoss(label_smoothing=candidate.get('label_smoothing', 0.))
    history, started = [], time.perf_counter()
    if str(config['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config['epochs']+1):
        lr = cosine_lr(epoch, config['epochs'], config['lr'], candidate['min_lr']) if name == 'cosine' else config['lr']
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = lr
        model.train()
        order = rng.permutation(train_indices)
        view_digest = hashlib.sha256()
        loss_sum, correct, seen, mixup_batches = 0., 0, 0, 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), config['batch_size']):
            indices = order[start:start+config['batch_size']]
            x_array = np.array(arrays['original'][indices, :3840], copy=True)
            flip = rng.random(len(indices)) < .5
            view_digest.update(flip.tobytes())
            x_array[flip] = arrays['flip'][indices[flip], :3840]
            x = torch.from_numpy(x_array).to(config['device'])
            target = torch.from_numpy(targets[indices].copy()).to(config['device'])
            optimizer.zero_grad(set_to_none=True)
            if name == 'feature_mixup' and mixup_rng.random() < candidate['mixup_probability']:
                lam = float(mixup_rng.beta(candidate['mixup_alpha'], candidate['mixup_alpha']))
                permutation = torch.from_numpy(mixup_rng.permutation(len(indices))).to(config['device'])
                x = lam*x+(1-lam)*x[permutation]
                logits = model(x)
                loss = mixup_ce(logits, target, target[permutation], lam)
                mixup_batches += 1
                del permutation
            else:
                logits = model(x)
                if candidate['weight_mode'] != 'uniform':
                    batch_weights = torch.from_numpy(full_weights[indices].copy()).to(config['device'])
                    loss = weighted_ce(logits, target, batch_weights)
                    del batch_weights
                else:
                    loss = loss_function(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite head training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
            if name != 'feature_mixup':
                correct += int((logits.detach().argmax(1) == target).sum())
            seen += len(indices)
            del x, target, logits, loss
        entry = {'epoch': epoch, 'lr': lr, 'train_loss': loss_sum/seen,
                 'train_accuracy': None if name == 'feature_mixup' else correct/seen,
                 'train_metrics_purpose': 'accuracy omitted: mixed feature/soft target batches' if name == 'feature_mixup'
                 else 'online dropout/random view diagnostic; train loss follows candidate objective',
                 'mixup_batches': mixup_batches, 'epoch_seconds': time.perf_counter()-epoch_start,
                 'training_order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                 'flip_assignment_sha256': view_digest.hexdigest()}
        history.append(entry)
        base.write_csv(folder/'history.csv', history, HISTORY_FIELDS)
        print(json.dumps({'candidate': name, 'fold': fold, **entry}), flush=True)
    # No heldout labels are used in model/scaler/weight fitting or epoch selection.
    original_p = base.predict_head(model, arrays['original'], heldout_indices, 0, 3840, config['batch_size'])
    flip_p = base.predict_head(model, arrays['flip'], heldout_indices, 0, 3840, config['batch_size'])
    probabilities = base.normalized((original_p+flip_p)/2)
    fold_metrics = base.score(targets[heldout_indices], probabilities)
    checkpoint = {'format': 'weighted_frozen_head_cv_v1', 'classes': CLASSES, 'contract': contract,
                  'config': config, 'fold_signature': fold_signature,
                  'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  'normalization_fit': 'fold training original features only', 'weight_receipt': weight_receipt,
                  'train_paths': [rows[i]['path'] for i in train_indices],
                  'heldout_paths': [rows[i]['path'] for i in heldout_indices],
                  'train_groups': sorted(set(groups[train_indices].tolist())),
                  'heldout_groups': sorted(set(groups[heldout_indices].tolist())),
                  'epochs_completed': config['epochs'], 'epoch_selection': False, 'tta_flip': True}
    torch.save(checkpoint, folder/'model.pt.tmp'); os.replace(folder/'model.pt.tmp', folder/'model.pt')
    for artifact, array in [('validation_original.npy', original_p), ('validation_flip.npy', flip_p),
                            ('validation_probabilities.npy', probabilities)]:
        np.save(folder/artifact, array, allow_pickle=False)
    base.write_predictions(folder/'validation_predictions.csv', rows, heldout_indices, targets, probabilities, groups, folds)
    peak = float(torch.cuda.max_memory_allocated()/1024**3) if str(config['device']).startswith('cuda') else None
    summary = {'state': 'complete', 'candidate': name, 'fold': fold, 'fold_signature': fold_signature,
               'contract': contract, 'epochs_completed': config['epochs'], 'metrics': fold_metrics,
               'class_recall': recall_report(targets[heldout_indices], probabilities),
               'train_rows': len(train_indices), 'heldout_rows': len(heldout_indices),
               'train_group_count': len(set(groups[train_indices])), 'heldout_group_count': len(set(groups[heldout_indices])),
               'peak_allocated_gpu_gib': peak, 'elapsed_seconds': time.perf_counter()-started,
               'artifact_sha256': {artifact: base.sha256(folder/artifact) for artifact in FOLD_ARTIFACTS}}
    base.write_json(summary_path, summary)
    del optimizer, model, checkpoint
    if str(config['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    print(json.dumps({'candidate': name, 'fold': fold, 'heldout': fold_metrics}), flush=True)
    return summary, probabilities


def expected_artifacts(experiment, report_folder):
    paths = [experiment/'run_config.json']+[report_folder/name for name in REPORT_FILES]
    for candidate in CANDIDATES:
        folder = experiment/candidate['name']
        paths += [folder/name for name in ['oof_probabilities.npy', 'oof_predictions.csv', 'summary.json']]
        for fold in range(3):
            paths += [folder/f'fold_{fold}'/name for name in FOLD_ARTIFACTS+['summary.json']]
    return paths


def compare_with_baseline(targets, folds, baseline, candidate, name):
    def one(mask):
        a, b, y = baseline[mask], candidate[mask], targets[mask]
        a_metrics, b_metrics = base.score(y, a), base.score(y, b)
        paired = base.paired_counts(y, a, b)
        return {**paired, 'corrected_baseline_errors': paired['b_only_correct'],
                'new_errors': paired['a_only_correct'],
                **{f'delta_{key}': b_metrics[key]-a_metrics[key] for key in a_metrics}}
    return {'baseline': 'baseline', 'candidate': name,
            'pooled': one(np.ones(len(targets), dtype=bool)),
            'per_fold': [{'fold': fold, **one(folds == fold)} for fold in range(3)]}


def run(args):
    root = Path(args.project_root).resolve()
    if args.epochs < 1 or args.batch_size < 1 or args.lr < 1e-5:
        raise ValueError('Require positive epochs/batch size and initial lr >= cosine endpoint 1e-5')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU requires explicit --device cpu')
    torch.set_num_threads(4)
    source_hashes = {}
    for module in [base, weight_builder]:
        filename = Path(module.__file__).name
        digest = base.sha256(module.__file__)
        base.verify_hash(root/'src'/filename, digest)
        source_hashes[filename] = digest
    source_hashes[Path(__file__).name] = base.sha256(__file__)
    base.verify_hash(root/'src'/Path(__file__).name, source_hashes[Path(__file__).name])
    rows, targets, folds, groups, arrays, data_receipt = base.load_training_inputs(root)
    experiment, report_folder = root/'experiments'/'weighted_heads', root/'weighted_results'
    experiment.mkdir(parents=True, exist_ok=True); report_folder.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'classes': CLASSES, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'lr': args.lr, 'weight_decay': .001, 'dropout_mlp': .3, 'hidden_dim': 256,
              'feature_slice': [0, 3840], 'kind': 'mlp', 'seed': args.seed, 'device': str(args.device),
              'dtype': 'float32', 'optimizer': 'AdamW', 'gradient_clip': 5.,
              'normalization': 'fit each fold training originals only; same scaler for both views',
              'sampling': 'uniform permutation each epoch, image random original/flip probability 0.5',
              'rng': 'data np.default_rng(seed+fold); independent MixUp np.default_rng(seed+fold+1000003)',
              'validation': 'final fixed epoch only; original/flip mean softmax; unweighted hard-label metrics',
              'loss_formula': 'weighted candidates: mean(w_i * CE_i); training-fold mean weight=1; uniform sampling',
              'cosine_formula': 'min_lr + .5*(lr-min_lr)*(1+cos(pi*(epoch-1)/(epochs-1)))',
              'mixup_formula': 'lambda*CE(logits,y_a)+(1-lambda)*CE(logits,y_b); no smoothing or sample weights',
              'folds': 3, 'candidate_definitions': CANDIDATES, 'source_sha256': source_hashes,
              'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
              'data_receipt': data_receipt, 'old_validation_test_unlabeled_read': False,
              'backbones': 'only frozen official pretrained DINOv2 train cache; no dataset-fitted CNN/LoRA features'}
    config = json.loads(json.dumps(config))
    run_signature = base.signature(config)
    config_path = experiment/'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Existing weighted experiment has different data/source/config')
    else:
        if any(experiment.iterdir()) or any(report_folder.iterdir()):
            raise ValueError('Existing output artifacts lack their run configuration')
        base.write_json(config_path, config)
    report_path = report_folder/'summary.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if (previous.get('state') != 'complete' or previous.get('run_signature') != run_signature or
                previous.get('classes') != CLASSES or previous.get('epochs_per_fold') != args.epochs):
            raise ValueError('Completed weighted report belongs to a different or incomplete run')
        required = {str(path) for path in expected_artifacts(experiment, report_folder)}
        if set(previous.get('artifact_sha256', {})) != required:
            raise ValueError('Completed weighted report artifact set differs')
        for path in required:
            base.verify_hash(path, previous['artifact_sha256'][path])
        print('Verified completed weighted CV report; no training/inference repeated.', flush=True)
        return previous
    candidates, oof_arrays = [], {}
    for candidate in CANDIDATES:
        name, folder = candidate['name'], experiment/candidate['name']
        oof = np.empty((len(rows), 6), dtype=np.float64)
        visited = np.zeros(len(rows), dtype=np.int64)
        summaries = []
        for fold in range(3):
            summary, probabilities = train_fold(folder/f'fold_{fold}', candidate, fold, rows, targets, folds,
                                                 groups, arrays, config, run_signature)
            indices = np.flatnonzero(folds == fold)
            if probabilities.shape != (len(indices), 6):
                raise ValueError('Fold prediction row count differs')
            oof[indices] = probabilities; visited[indices] += 1; summaries.append(summary)
        if not np.all(visited == 1):
            raise ValueError('Every OOF row must be predicted exactly once by its heldout fold')
        oof_arrays[name] = oof
        np.save(folder/'oof_probabilities.npy', oof, allow_pickle=False)
        base.write_predictions(folder/'oof_predictions.csv', rows, np.arange(len(rows)), targets, oof, groups, folds)
        metrics = [item['metrics'] for item in summaries]
        result = {'candidate': name, 'definition': candidate, 'feature_dim': 3840, 'kind': 'mlp',
                  'per_fold': [{'fold': i, 'heldout_rows': item['heldout_rows'], **item['metrics'],
                                'class_recall': item['class_recall']} for i, item in enumerate(summaries)],
                  **{f'mean_{key}': float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                  'std_accuracy': float(np.std([m['accuracy'] for m in metrics], ddof=1)),
                  'pooled_oof': base.score(targets, oof), 'class_recall': recall_report(targets, oof),
                  'confusion_matrix': base.confusion_matrix(targets, oof.argmax(1), labels=list(range(6))).tolist()}
        base.write_json(folder/'summary.json', result)
        candidates.append(result)
    paired = [compare_with_baseline(targets, folds, oof_arrays['baseline'], oof_arrays[item['name']], item['name'])
              for item in CANDIDATES]
    comparison_fields = ['candidate', 'feature_dim', 'kind', 'mean_accuracy', 'mean_macro_f1', 'mean_log_loss', 'std_accuracy']
    base.write_csv(report_folder/'comparison.csv', [{key: item[key] for key in comparison_fields} for item in candidates], comparison_fields)
    paired_fields = ['baseline', 'candidate', 'partition', 'both_correct', 'a_only_correct', 'b_only_correct', 'both_wrong',
                     'net_correct_b_minus_a', 'prediction_disagreements', 'corrected_baseline_errors', 'new_errors',
                     'delta_accuracy', 'delta_macro_f1', 'delta_log_loss']
    paired_rows = []
    for item in paired:
        for partition, values in [('pooled', item['pooled'])]+[(f'fold_{row["fold"]}', row) for row in item['per_fold']]:
            record = {'baseline': 'baseline', 'candidate': item['candidate'], 'partition': partition, **values}
            paired_rows.append({key: record[key] for key in paired_fields})
    base.write_csv(report_folder/'paired_corrections.csv', paired_rows, paired_fields)
    recall_rows = []
    for item in candidates:
        for partition, recalls in [('pooled', item['class_recall'])]+[(f'fold_{row["fold"]}', row['class_recall']) for row in item['per_fold']]:
            recall_rows.extend({'candidate': item['candidate'], 'partition': partition, **row} for row in recalls)
    base.write_csv(report_folder/'class_recall.csv', recall_rows, ['candidate', 'partition', 'target', 'class', 'support', 'correct', 'recall'])
    winner = max(candidates, key=lambda item: (item['mean_accuracy'], item['mean_macro_f1'], -item['mean_log_loss']))
    summary = {'version': VERSION, 'state': 'complete', 'run_signature': run_signature, 'classes': CLASSES,
               'train_rows': len(rows), 'groups': len(set(groups)), 'folds': 3, 'epochs_per_fold': args.epochs,
               'candidates': candidates, 'paired_vs_baseline': paired, 'selected': winner,
               'selection_rule': 'descriptive ranking: mean fold accuracy, then macro F1, then lower log loss; no automatic combination',
               'old_validation_test_unlabeled_read': False, 'external_answers_used': False, 'submission_generated': False,
               'selection_caveat': 'These existing grouped CV folds are reused for seven prespecified comparisons. '
                                  'No fold selects an epoch. Choosing the winner on these same scores makes its selected score '
                                  'an optimistic model-selection estimate, not independent confirmation. Three-fold standard '
                                  'deviation is descriptive, not a confidence interval. No automatic combination or final refit is performed.',
               'data_receipt': data_receipt, 'source_sha256': source_hashes,
               'artifact_sha256': {str(path): base.sha256(path) for path in expected_artifacts(experiment, report_folder)}}
    base.write_json(report_path, summary)
    print(json.dumps({'state': 'complete', 'selected': winner['candidate'], 'report': str(report_path),
                      'selection_caveat': summary['selection_caveat']}, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
