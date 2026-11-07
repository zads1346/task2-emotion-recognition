"""Lock a matched 0/.15/1 LoRA continuation and test real weighted gradients."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['XFORMERS_DISABLED'] = '1'
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_hse_specialist as h
import train_lora as l
import train_task_lora_v4 as v4
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = 'cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd'
PRESERVED = {'hse_results/submission.csv': 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7',
             'fusion_results/submission.csv': '2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad',
             'lora_task_v4_results/submission.csv': BASELINE_SHA}
SETTINGS = {'epochs': 4, 'seed': 42, 'augmentation_epoch_offset': 40,
            'label_smoothing': .02, 'weight_decay': .001, 'lr_new': 1e-5,
            'lr_inherited': 2.5e-6, 'lr_head': 1e-5, 'warmup_successful_steps': 50,
            'min_lr_ratio': .1, 'ema_decay': .995, 'batch_size': 8, 'accumulation': 4,
            'effective_batch_size': 32, 'workers': 0, 'cpu_threads': 4, 'image_size': 224,
            'amp_initial_scale': 1024., 'gradient_clip': 1., 'validation_batch_size': 4,
            'rank': 8, 'alpha': 16., 'lora_dropout': .05, 'last_blocks': 8,
            'sample_weights': {'normal': 1., 'uncertain': .15, 'very_uncertain': 0.},
            'class_normalization': False, 'sampler': 'uniform, same ordered rows in both arms',
            'loss_normalization': 'sum(w_i*CE_i)/sum(w_i) over full effective group of32',
            'modes': ['weighted', 'uniform_control'], 'mixup': False, 'cutmix': False}
SOURCES = ['prepare_zero_weight_v5.py', 'prepare_zero_weights_v5.py', 'train_zero_weight_lora_v5.py',
           'finalize_zero_weight_v5.py', 'run_zero_weight_v5.py', 'zero_weight_v5_worker.py']

def verify_plan(root=ROOT):
    plan = c.read_json(root/'logs/zero_weight_v5_plan.json')
    if plan['classes'] != h.CLASSES or plan['settings'] != SETTINGS:
        raise ValueError('Locked classes/settings differ')
    c.verify_pins(root, plan['source_pins'])
    for path, digest in PRESERVED.items(): h.check_hash(root/path, digest)
    h.check_hash(plan['parent_checkpoint'], plan['parent_checkpoint_sha256'])
    h.check_hash(root/'experiments/lora_zero_weight_v5/weights/summary.json', plan['weight_summary_sha256'])
    if l.repository_info(root) != plan['dino_repository'] or l.verify_official_weights(root) != plan['official_dino_sha256']:
        raise ValueError('Official pretrained backbone/vendor differs')
    return plan

def cpu_tests():
    import train_zero_weight_lora_v5 as t
    torch.manual_seed(11)
    logits = torch.randn(32, 6, requires_grad=True)
    targets = torch.arange(32) % 6
    weights = torch.tensor(([1.]*21)+([.15]*7)+([0.]*4))
    denominator = float(weights.double().sum())
    direct = t.weighted_loss(logits, targets, weights, denominator)
    gradient = torch.autograd.grad(direct, logits)[0]
    if torch.count_nonzero(gradient[-4:]).item() != 0:
        raise ValueError('Zero-weight images still contribute label gradient')
    clone = logits.detach().clone().requires_grad_()
    total = sum(t.weighted_loss(clone[i:i+8], targets[i:i+8], weights[i:i+8], denominator)
                for i in range(0, 32, 8))
    group_gradient = torch.autograd.grad(total, clone)[0]
    if not torch.allclose(direct, total, atol=1e-6, rtol=1e-6) or not torch.allclose(gradient, group_gradient, atol=1e-7, rtol=1e-6):
        raise ValueError('Microbatch accumulation differs from full-group weighted objective')
    altered = targets.clone(); altered[-4:] = (altered[-4:]+3)%6
    if not torch.equal(t.weighted_loss(logits, altered, weights, denominator), direct):
        raise ValueError('Zero-weight labels alter the objective')
    if t.should_update(0.) or not t.should_update(.15):
        raise ValueError('Zero group update decision is incorrect')
    zero_model = torch.nn.Linear(3, 6)
    original = {key: value.detach().clone() for key, value in zero_model.state_dict().items()}
    def forbidden_forward(*args, **kwargs): raise AssertionError('Zero group called model.forward')
    zero_model.forward = forbidden_forward
    optimizer = torch.optim.AdamW(zero_model.parameters(), lr=.1, weight_decay=.5)
    for parameter in zero_model.parameters(): parameter.grad = torch.ones_like(parameter)
    class Forbidden:
        def __getattr__(self, key): raise AssertionError('Zero group touched scaler/EMA: '+key)
    empty = t.train_group(zero_model, optimizer, Forbidden(), Forbidden(), torch.zeros(8, 3),
                          torch.zeros(8, dtype=torch.long), torch.zeros(8), device='cpu', batch_size=8)
    if empty['success'] is not None or not empty['zero_weight_group'] or empty['microbatches'] != 0:
        raise ValueError('Empty group was treated as an optimizer update')
    if optimizer.state or any(not torch.equal(original[key], value) for key, value in zero_model.state_dict().items()):
        raise ValueError('Empty group changed weights/AdamW state')
    if any(parameter.grad is not None for parameter in zero_model.parameters()):
        raise ValueError('Empty group did not clear stale gradients')
    return {'passed': True, 'zero_label_gradient_exactly_zero': True,
            'zero_labels_do_not_change_loss': True, 'microbatch_gradient_matches_full_group': True,
            'zero_group_should_update': False, 'zero_group_no_forward_scaler_ema_or_weight_decay': True}

def release():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def gpu_smoke(root, rows):
    import train_zero_weight_lora_v5 as t
    wrows = c.csv_rows(root/'experiments/lora_zero_weight_v5/weights/train_weights.csv',
                       {'manifest_index', 'difficulty', 'training_weight'})
    selected = []
    for bucket, count in [('normal', 24), ('uncertain', 7), ('very_uncertain', 1)]:
        selected.extend([i for i, row in enumerate(wrows) if row['difficulty'] == bucket][:count])
    if len(selected) != 32: raise ValueError('Smoke lacks one of the three difficulty groups')
    records = [rows['train'][i] for i in selected]
    images, targets = next(iter(l.make_loader(records, 32, 0, 42, training=False)))
    weights = torch.tensor([float(wrows[i]['training_weight']) for i in selected], dtype=torch.float64)
    denominator = float(weights.sum())
    l.seed_everything(42); torch.cuda.reset_peak_memory_stats()
    parent = root/'experiments/lora_task_v4/candidates/epoch_05_ema.pt'
    model, metadata = v4.load_checkpoint(root, parent, 'cuda')
    v4._attach_checkpointing(model, True)
    original = l.adapter_state(model)
    payload = torch.load(parent, map_location='cpu', weights_only=True)
    if set(original) != v4.ALL_NAMES or any(not torch.equal(original[name], payload['adapter_state'][name]) for name in original):
        raise ValueError('Full trained last8 parent adapters/head were not inherited exactly')
    del payload
    if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError('BatchNorm would couple zero-weight images through running statistics')
    model.eval()
    with torch.inference_mode(), h.autocast_for('cuda'):
        before = model(images[:8].cuda()).float().cpu()
    model.train()
    optimizer = v4.make_optimizer(model, SETTINGS)
    scaler = h.make_scaler('cuda')
    ema = v4.AdapterEMA(model, SETTINGS['ema_decay'])
    frozen = {name: param.detach().clone() for name, param in model.named_parameters()
              if not param.requires_grad and name.endswith('qkv.base.weight')}
    for step in range(2):
        v4.set_learning_rate(optimizer, step, 664*4, SETTINGS)
        update = t.train_group(model, optimizer, scaler, ema, images, targets, weights, device='cuda', batch_size=8)
        if not update['success'] or update['zero_weight_group'] or update['microbatches'] != 4:
            raise ValueError('Weighted GPU probe failed to update')
    after = l.adapter_state(model)
    for names in [v4.NEW_NAMES, v4.INHERITED_NAMES-{'head.weight','head.bias'}, {'head.weight','head.bias'}]:
        if not any(not torch.equal(original[name], after[name]) for name in names):
            raise ValueError('One trainable parameter group did not actually update')
    if any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items()):
        raise ValueError('Frozen official qkv parameters changed')
    peak = torch.cuda.max_memory_allocated()/2**20
    if peak > 4000: raise RuntimeError('GPU probe exceeded allocated-memory budget')
    receipt = {'passed': True, 'successful_optimizer_steps': 2, 'amp_skipped_steps': 0,
               'all_18_parent_tensors_inherited_exactly': True, 'all_trainable_groups_updated': True,
               'frozen_qkv_unchanged': True, 'zero_weight_group_tested_in_cpu': True,
               'group_size': 32, 'microbatch': 8, 'weights_counts': {'1': 24, '.15': 7, '0': 1},
               'group_weight_sum': denominator, 'peak_allocated_mib': peak, 'probe_weights_saved': False}
    del model, optimizer, scaler, ema, original, after, frozen, images, targets, before
    release()
    return receipt

def prepare(root=ROOT):
    root = Path(root).resolve(); plan_path = root/'logs/zero_weight_v5_plan.json'
    if plan_path.exists(): return verify_plan(root)
    import prepare_zero_weights_v5 as weights_api
    weights_summary = weights_api.build(root)
    rows, input_receipt = h.load_trainval(root)
    parent = root/'experiments/lora_task_v4/candidates/epoch_05_ema.pt'
    for path, digest in PRESERVED.items(): h.check_hash(root/path, digest)
    previous = root/'lora_task_v4_results'
    summary = c.read_json(previous/'summary.json'); recipe = c.read_json(previous/'selected_recipe.json')
    if summary['state'] != 'complete' or summary['selected'] != 'epoch_05_ema':
        raise ValueError('Latest verified best V4 changed')
    baseline_files = {}
    for path, digest in summary['file_sha256'].items():
        h.check_hash(previous/path, digest); baseline_files[str((previous/path).relative_to(root))] = digest
    baseline_files['lora_task_v4_results/summary.json'] = h.sha256(previous/'summary.json')
    if recipe['selected']['artifact_sha256']['model'] != h.sha256(parent):
        raise ValueError('Parent is not the current best fusion LoRA branch')
    val_rows = c.val_rows(root)
    if [int(row['manifest_index']) for row in val_rows] != [row['manifest_index'] for row in rows['val']]:
        raise ValueError('Validation member order differs')
    targets = np.asarray([int(row['target']) for row in val_rows])
    current = c.check_probability(np.load(previous/'selected_val_probabilities.npy', allow_pickle=False), 2656)
    current_metrics = c.metrics(targets, current)
    if current_metrics['correct_count'] != 1915 or current_metrics != summary['selected_metrics']:
        raise ValueError('Latest development baseline does not reproduce1915/2656')
    parent_views = []
    for view in ['original', 'flip']:
        path = parent.parent/f'epoch_05_ema_{view}.npy'
        h.check_hash(path, recipe['selected']['artifact_sha256'][view])
        baseline_files[str(path.relative_to(root))] = h.sha256(path)
        parent_views.append(c.check_probability(np.load(path, allow_pickle=False), 2656))
    baseline_files[str(parent.relative_to(root))] = h.sha256(parent)
    parent_json = parent.with_suffix('.json')
    if c.read_json(parent_json) != recipe['selected']: raise ValueError('Parent candidate binding differs')
    baseline_files[str(parent_json.relative_to(root))] = h.sha256(parent_json)
    v4config = c.read_json(root/'experiments/lora_task_v4/config.json')
    for split in ('train', 'val'):
        if v4config[split+'_indices'] != [row['manifest_index'] for row in rows[split]]:
            raise ValueError('Parent trained on different members')
    if any(input_receipt[key] != v4config['input_receipt'][key] for key in input_receipt):
        raise ValueError('Training/validation images differ from parent model inputs')
    pins = {**recipe['producer_files'], **v4config['source_pins']}
    for path, digest in baseline_files.items(): pins[str(root/path)] = digest
    for name in SOURCES: pins[str(root/'src'/name)] = h.sha256(root/'src'/name)
    for name in ['start_zero_weight_v5.ps1', 'run_zero_weight_v5_worker.ps1']:
        pins[str(root/name)] = h.sha256(root/name)
    for path in [root/'logs/lora_task_v4_plan.json', root/'experiments/lora_task_v4/config.json',
                 root/'experiments/lora_task_v4/summary.json', root/'experiments/lora_zero_weight_v5/weights/summary.json']:
        pins[str(path)] = h.sha256(path)
    weights_dir = root/'experiments/lora_zero_weight_v5/weights'
    for path in weights_dir.iterdir():
        if path.is_file(): pins[str(path)] = h.sha256(path)
    # The prepared weight summary also freezes both prior OOF and weighting chains.
    pins.update(weights_summary['binding']['source_sha256'])
    c.verify_pins(root, pins)
    tests = cpu_tests()
    probe = gpu_smoke(root, rows)
    plan = {'format': 'zero_weight_lora_v5_locked', 'classes': h.CLASSES, 'settings': SETTINGS,
            'source_pins': pins, 'input_receipt': input_receipt,
            'train_indices': [r['manifest_index'] for r in rows['train']],
            'val_indices': [r['manifest_index'] for r in rows['val']],
            'weight_summary_sha256': h.sha256(weights_dir/'summary.json'),
            'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': h.sha256(parent),
            'parent_candidate_source': 'epoch_05_ema', 'baseline_files': baseline_files,
            'baseline_submission_sha256': BASELINE_SHA, 'baseline_correct_count': 1915,
            'baseline_metrics': current_metrics, 'parent_lora_tta_metrics': c.metrics(targets, (parent_views[0]+parent_views[1])*.5),
            'dino_repository': l.repository_info(root), 'official_dino_sha256': l.verify_official_weights(root),
            'competition_image_content_sha256': summary['competition_image_content_sha256'],
            'competition_row_order_sha256': summary['competition_row_order_sha256'],
            'selection_order': ['incumbent', 'uniform_control', 'weighted'],
            'selection': 'strict fused correct-count increase; ties retain earlier candidate; candidates epoch0, then each epoch raw/EMA',
            'fusion_weights': {'hse': .5, 'lora': .2, 'cnn': .2, 'mlp': .1},
            'cpu_tests': tests, 'gpu_smoke': probe, 'internal_test_read': False, 'external_answers_used': False,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    c.verify_pins(root, pins)
    h.write_json(plan_path, plan)
    print(json.dumps({'stage': 'prepare', 'state': 'locked', 'weights': SETTINGS['sample_weights'],
                      'cpu_tests': tests, 'gpu_smoke': probe, 'baseline_correct_count': 1915}, ensure_ascii=False), flush=True)
    return plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Local CUDA is unavailable')
    prepare(args.project_root)
