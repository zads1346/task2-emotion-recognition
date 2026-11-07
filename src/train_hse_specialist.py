"""Fixed HSE emotion-specialist fine-tuning on train/val only.

Stage smoke tests one fully unfrozen batch without saving training weights.
Stage train commits whole epochs; optimizer, scaler and all RNG states resume
from the last committed epoch. Only original-view validation accuracy selects
the checkpoint; horizontal-flip probabilities are saved separately for fusion.
"""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_name] = '4'

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COUNTS = {'train': 21248, 'val': 2656}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
VERSION = '1.0.0'
FIXED = {'epochs': 12, 'batch_size': 8, 'accumulation': 4, 'effective_batch_size': 32,
         'workers': 0, 'cpu_threads': 4, 'seed': 42, 'head_warmup_epochs': 1,
         'backbone_lr': 5e-5, 'head_lr': 5e-4, 'min_lr_ratio': .1,
         'weight_decay': .01, 'label_smoothing': .05, 'gradient_clip': 1.,
         'amp_dtype': 'float16', 'initial_grad_scale': 1024., 'batchnorm_running_stats': 'frozen_eval',
         'optimizer': 'AdamW', 'checkpoint_selection_view': 'original',
         'selection_rule': 'original validation accuracy, then macro_f1, then lower unsmoothed log_loss',
         'preprocessing': 'EXIF transpose, RGB, bilinear resize 224x224, ImageNet mean/std',
         'augmentation': 'horizontal flip p=.5; affine degrees6 translate(.03,.03) scale(.95,1.05); brightness/contrast jitter .1',
         'validation_flip': 'normalized deterministic original tensor.flip(-1)',
         'resume': 'complete epoch only; restore optimizer/scaler/Python/NumPy/Torch/CUDA RNG; deterministic epoch loader seed'}
OUTPUT_FILES = ['config.json', 'history.csv', 'best.pt', 'val_original.npy', 'val_flip.npy', 'last.json']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); os.replace(temporary, path)


def save_array(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'File checksum differs: {path}')


def image_key(path):
    parts = str(path).replace('\\', '/').split('/')
    if 'fer_data' not in parts or any(part in ['.', '..'] for part in parts):
        raise ValueError('Image path is outside the audited FER archive layout')
    return '/'.join(parts[parts.index('fer_data'):]).casefold()


def load_trainval(root):
    """Filter split before inspecting labels; never open an internal-test image."""
    manifest, audit_path = root/'features'/'manifest.csv', root/'data'/'split_audit.json'
    inventory_path = root/'data'/'original_splits'/'image_inventory.csv'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('classes') != CLASSES:
        raise ValueError('Split audit class order differs')
    check_hash(manifest, audit.get('manifest_sha256'))
    rows = {'train': [], 'val': []}
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'label', 'split'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest must contain path, label, split')
        for index, row in enumerate(reader):
            split = row.get('split')
            if split not in rows:
                continue
            if row.get('label') not in CLASSES or not row.get('path') or not os.path.isabs(row['path']):
                raise ValueError('Invalid train/val image or label')
            rows[split].append({'manifest_index': index, 'path': row['path'], 'label': row['label'],
                                'target': CLASSES.index(row['label'])})
    selected = rows['train']+rows['val']
    if any(len(rows[split]) != COUNTS[split] or {row['target'] for row in rows[split]} != set(range(6)) for split in rows):
        raise ValueError('Training/validation membership count or six-class coverage differs')
    if len({os.path.normcase(os.path.normpath(row['path'])) for row in selected}) != len(selected):
        raise ValueError('Train/val image paths overlap or repeat')
    wanted = {image_key(row['path']) for row in selected}
    inventory = {}
    with inventory_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'path', 'sha256'}.issubset(reader.fieldnames or []):
            raise ValueError('Original image inventory lacks path/hash')
        for row in reader:
            key = image_key(row['path'])
            if key in wanted:
                if key in inventory:
                    raise ValueError('Duplicate original image inventory key')
                inventory[key] = row['sha256']
    content = {split: hashlib.sha256() for split in rows}
    byte_sets = {split: set() for split in rows}
    checked, start = 0, time.perf_counter()
    for split in ['train', 'val']:
        for row in rows[split]:
            expected = inventory.get(image_key(row['path']))
            check_hash(row['path'], expected)
            content[split].update(expected.encode('ascii')); byte_sets[split].add(expected)
            checked += 1
            if checked % 4000 == 0:
                print(json.dumps({'stage': 'verify_trainval', 'verified_images': checked,
                                  'elapsed_seconds': time.perf_counter()-start}), flush=True)
    if byte_sets['train'] & byte_sets['val']:
        raise ValueError('Identical image bytes cross train/validation boundary')
    receipt = {'manifest_sha256': sha256(manifest), 'split_audit_sha256': sha256(audit_path),
               'image_inventory_sha256': sha256(inventory_path),
               'train_image_content_sha256': content['train'].hexdigest(),
               'val_image_content_sha256': content['val'].hexdigest(),
               'train_rows_sha256': signature(rows['train']), 'val_rows_sha256': signature(rows['val']),
               'verified_images': len(selected), 'internal_test_images_read': 0, 'internal_test_labels_parsed': False}
    return rows, receipt


def model_api():
    return importlib.import_module('hse_model')


def prepare_config(root):
    rows, input_receipt = load_trainval(root)
    api = model_api()
    provenance = api.source_receipt(root)
    source_hashes = {Path(__file__).name: sha256(__file__), 'hse_model.py': sha256(api.__file__)}
    for name, digest in source_hashes.items():
        check_hash(root/'src'/name, digest)
    config = {'format': 'hse_specialist_config_v1', 'version': VERSION, 'classes': CLASSES,
              'fixed_training': FIXED, 'source_sha256': source_hashes, 'input_sha256': input_receipt,
              'model_provenance': provenance, 'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__),
              'train_indices': [row['manifest_index'] for row in rows['train']],
              'val_indices': [row['manifest_index'] for row in rows['val']],
              'train_paths': [row['path'] for row in rows['train']], 'val_paths': [row['path'] for row in rows['val']],
              'checkpoint_selection_view': 'original', 'external_answers_used': False, 'internal_test_read': False}
    return rows, json.loads(json.dumps(config))


def make_eval_transform():
    return transforms.Compose([transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_train_transform():
    return transforms.Compose([transforms.Resize((224,224), interpolation=InterpolationMode.BILINEAR),
                               transforms.RandomHorizontalFlip(.5),
                               transforms.RandomAffine(6, translate=(.03,.03), scale=(.95,1.05),
                                                       interpolation=InterpolationMode.BILINEAR),
                               transforms.ColorJitter(brightness=.1, contrast=.1),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ExpressionImages(Dataset):
    def __init__(self, rows, training):
        self.rows = rows
        self.transform = make_train_transform() if training else make_eval_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['path']) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return tensor, int(row['target'])


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('Cannot resume a CUDA RNG state on CPU')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    return value


def classifier_parameters(model):
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Linear) or classifier.out_features != 6:
        raise ValueError('HSE model must expose its six-class Linear classifier')
    return list(classifier.parameters())


def set_training_stage(model, epoch):
    model.requires_grad_(epoch > FIXED['head_warmup_epochs'])
    for parameter in classifier_parameters(model): parameter.requires_grad_(True)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()


def optimizer_for(model):
    head = classifier_parameters(model); head_ids = {id(value) for value in head}
    backbone = [value for value in model.parameters() if id(value) not in head_ids]
    if not backbone: raise ValueError('Model has no backbone parameters')
    return torch.optim.AdamW([{'params': backbone, 'lr': FIXED['backbone_lr'], 'name': 'backbone'},
                             {'params': head, 'lr': FIXED['head_lr'], 'name': 'head'}], weight_decay=FIXED['weight_decay'])


def learning_rates(epoch):
    if not 1 <= epoch <= FIXED['epochs']: raise ValueError('Epoch outside fixed training schedule')
    factor = FIXED['min_lr_ratio']+(1-FIXED['min_lr_ratio'])*.5*(1+math.cos(math.pi*(epoch-1)/(FIXED['epochs']-1)))
    return {'backbone': FIXED['backbone_lr']*factor, 'head': FIXED['head_lr']*factor}


def loader_for(rows, training, epoch, device):
    # A separate generator prevents loader seed/shuffle from consuming the model's RNG.
    generator = torch.Generator().manual_seed(FIXED['seed']+1009*epoch+(0 if training else 1000003))
    return DataLoader(ExpressionImages(rows, training), batch_size=FIXED['batch_size'], shuffle=training,
                      num_workers=0, pin_memory=str(device).startswith('cuda'), generator=generator, drop_last=False)


def autocast_for(device):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                          enabled=torch.device(device).type == 'cuda')


def make_scaler(device):
    return torch.amp.GradScaler('cuda', enabled=torch.device(device).type == 'cuda', init_scale=FIXED['initial_grad_scale'])


def optimizer_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], FIXED['gradient_clip'],
                                   error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
    if not skipped and not torch.isfinite(norm):
        raise RuntimeError('Nonfinite gradients without AMP skip')
    optimizer.zero_grad(set_to_none=True)
    return (None if not torch.isfinite(norm) else float(norm)), bool(skipped)


def train_epoch(model, rows, optimizer, scaler, epoch, device):
    set_training_stage(model, epoch)
    rates = learning_rates(epoch)
    for group in optimizer.param_groups: group['lr'] = rates[group['name']]
    loader = loader_for(rows, True, epoch, device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=FIXED['label_smoothing'])
    optimizer.zero_grad(set_to_none=True)
    loss_sum, correct, seen, successes, skipped = 0., 0, 0, 0, 0
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        group_first_sample = (batch_index//FIXED['accumulation'])*FIXED['effective_batch_size']
        group_samples = min(FIXED['effective_batch_size'], len(rows)-group_first_sample)
        with autocast_for(device):
            logits = model(images)
            if logits.shape != (len(targets),6): raise ValueError('Expected six logits per image')
            loss = loss_function(logits, targets)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        # Correct denominator also for a partial final accumulation group.
        scaler.scale(loss * (len(targets)/group_samples)).backward()
        loss_sum += float(loss.detach())*len(targets)
        correct += int((logits.detach().argmax(1)==targets).sum()); seen += len(targets)
        if (batch_index+1)%FIXED['accumulation'] == 0 or batch_index+1 == len(loader):
            _, did_skip = optimizer_step(model, optimizer, scaler)
            skipped += int(did_skip); successes += int(not did_skip)
        if (batch_index+1)%50 == 0 or batch_index+1 == len(loader):
            print(json.dumps({'stage':'train', 'epoch':epoch, 'batch':batch_index+1, 'batches':len(loader),
                              'seen':seen, 'train_loss':loss_sum/seen, 'elapsed_seconds':time.perf_counter()-started,
                              'successful_optimizer_steps':successes, 'amp_skipped_steps':skipped}), flush=True)
        del images, targets, logits, loss
    if successes == 0: raise RuntimeError('No successful optimizer step in this epoch')
    return {'train_loss':loss_sum/seen, 'train_accuracy':correct/seen, 'backbone_frozen':epoch == 1,
            'lr_backbone':rates['backbone'], 'lr_head':rates['head'], 'successful_optimizer_steps':successes,
            'amp_skipped_steps':skipped, 'amp_scale':float(scaler.get_scale()), 'train_seconds':time.perf_counter()-started}


def probability(value):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid official six-class probabilities')
    return p/p.sum(1,keepdims=True)


def metrics(rows, probabilities):
    targets = np.asarray([row['target'] for row in rows]); p = probability(probabilities)
    if len(p) != len(targets): raise ValueError('Validation probability row count differs')
    return {'accuracy':float(accuracy_score(targets,p.argmax(1))),
            'macro_f1':float(f1_score(targets,p.argmax(1),labels=list(range(6)),average='macro',zero_division=0)),
            'log_loss':float(log_loss(targets,p,labels=list(range(6))))}


@torch.inference_mode()
def predict_validation(model, rows, epoch, device):
    model.eval(); original, flip = [], []
    for images, _ in loader_for(rows, False, epoch, device):
        images = images.to(device)
        with autocast_for(device):
            a, b = model(images), model(images.flip(-1))
        original.append(a.float().softmax(1).cpu().numpy()); flip.append(b.float().softmax(1).cpu().numpy())
    return probability(np.concatenate(original)), probability(np.concatenate(flip))


def metric_key(value):
    return value['accuracy'], value['macro_f1'], -value['log_loss']


def write_history(folder, history):
    path = folder/'history.csv'; temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    os.replace(temporary,path)


def membership_metadata(config):
    return {key:config[key] for key in ['classes','train_indices','val_indices','train_paths','val_paths','source_sha256','model_provenance']}


def materialize_best(folder, checkpoint, config):
    best = {'format':'hse_specialist_best_v1', **membership_metadata(config), 'config':config,
            'manifest_sha256':config['input_sha256']['manifest_sha256'], 'input_sha256':config['input_sha256'],
            'run_signature':signature(config), 'best_epoch':checkpoint['best_epoch'],
            'best_validation':checkpoint['best_validation'], 'checkpoint_selection_view':'original',
            'model_state':checkpoint['best_model_state']}
    save_torch(folder/'best.pt',best)
    save_array(folder/'val_original.npy',probability(checkpoint['best_val_original'].numpy()))
    save_array(folder/'val_flip.npy',probability(checkpoint['best_val_flip'].numpy()))
    write_history(folder,checkpoint['history'])


def load_committed(folder, config, device):
    pointer = json.loads((folder/'last.json').read_text(encoding='utf-8'))
    epoch = pointer.get('epoch')
    if type(epoch) != int or not 1 <= epoch <= FIXED['epochs'] or pointer.get('run_signature') != signature(config):
        raise ValueError('Committed epoch pointer differs from this run')
    expected_name = f'epochs/epoch_{epoch:02d}.pt'
    if pointer.get('checkpoint') != expected_name: raise ValueError('Unexpected epoch checkpoint path')
    path = folder/expected_name; check_hash(path,pointer.get('sha256'))
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if (checkpoint.get('format') != 'hse_specialist_epoch_v1' or checkpoint.get('config') != config or
            checkpoint.get('run_signature') != signature(config) or checkpoint.get('epoch') != epoch or
            checkpoint.get('training_device') != str(device) or len(checkpoint.get('history',[])) != epoch or
            [entry.get('epoch') for entry in checkpoint['history']] != list(range(1,epoch+1)) or
            not 1 <= checkpoint.get('best_epoch',0) <= epoch):
        raise ValueError('Committed epoch checkpoint metadata differs')
    if metric_key(checkpoint['best_validation']) != max(metric_key({key:entry['val_'+key] for key in ['accuracy','macro_f1','log_loss']}) for entry in checkpoint['history']):
        raise ValueError('Committed best model does not follow original-view selection')
    return checkpoint, pointer


def load_receipt(root):
    root = Path(root).resolve(); _, config = prepare_config(root)
    folder = root/'experiments'/'hse_specialist_v1'
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('state') != 'complete' or summary.get('epochs_completed') != 12 or
            summary.get('config') != config or summary.get('run_signature') != signature(config) or
            summary.get('classes') != CLASSES or summary.get('checkpoint_selection_view') != 'original'):
        raise ValueError('Completed HSE training receipt differs from its inputs/source/protocol')
    if (any(summary.get(key) != value for key,value in membership_metadata(config).items()) or
            summary.get('manifest_sha256') != config['input_sha256']['manifest_sha256'] or
            summary.get('input_sha256') != config['input_sha256'] or
            summary.get('train_rows') != COUNTS['train'] or summary.get('val_rows') != COUNTS['val'] or
            not 1 <= summary.get('best_epoch',0) <= 12):
        raise ValueError('Completed HSE top-level membership or provenance aliases differ')
    if set(summary.get('artifact_sha256',{})) != set(OUTPUT_FILES):
        raise ValueError('Completed HSE output artifact list differs')
    for name in OUTPUT_FILES: check_hash(folder/name,summary['artifact_sha256'][name])
    for field,name in [('best_checkpoint_sha256','best.pt'),('val_original_sha256','val_original.npy'),('val_flip_sha256','val_flip.npy')]:
        if summary.get(field) != summary['artifact_sha256'][name]: raise ValueError('HSE model/probability receipt aliases differ')
    pointer = json.loads((folder/'last.json').read_text(encoding='utf-8'))
    if pointer.get('epoch') != 12 or pointer.get('run_signature') != signature(config):
        raise ValueError('Completed HSE last epoch pointer differs')
    if pointer.get('checkpoint') != 'epochs/epoch_12.pt': raise ValueError('Unexpected final epoch checkpoint path')
    check_hash(folder/pointer['checkpoint'],pointer.get('sha256'))
    return summary


def run_smoke(root, device):
    rows, config = prepare_config(root)
    seed_everything(42)
    model = model_api().load_initial_model(root,device)
    set_training_stage(model,2)
    optimizer, scaler = optimizer_for(model), make_scaler(device)
    head_ids = {id(parameter) for parameter in classifier_parameters(model)}
    backbone_parameter = next(parameter for parameter in model.parameters() if id(parameter) not in head_ids)
    backbone_before = backbone_parameter.detach().cpu().clone()
    before = {name:value.detach().cpu().clone() for name,value in model.named_buffers()
              if name.endswith(('running_mean','running_var','num_batches_tracked'))}
    if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats()
    images, targets = next(iter(loader_for(rows['train'],True,1,device)))
    images, targets = images.to(device), targets.to(device)
    with autocast_for(device):
        logits = model(images)
        loss = nn.functional.cross_entropy(logits,targets,label_smoothing=.05)
    if logits.shape != (len(targets),6) or not torch.isfinite(loss): raise RuntimeError('Smoke forward failed')
    scaler.scale(loss).backward()
    backbone_gradient_nonzero = (backbone_parameter.grad is not None and
                                 bool(torch.isfinite(backbone_parameter.grad).all()) and
                                 bool(torch.count_nonzero(backbone_parameter.grad)))
    norm, skipped = optimizer_step(model,optimizer,scaler)
    if skipped or norm is None: raise RuntimeError('Smoke AMP gradients overflowed; training was not started')
    backbone_updated = not torch.equal(backbone_before,backbone_parameter.detach().cpu())
    if not backbone_gradient_nonzero or not backbone_updated:
        raise RuntimeError('Smoke backbone gradient/update was not verified')
    frozen_bn = all(not module.training for module in model.modules() if isinstance(module,nn.modules.batchnorm._BatchNorm))
    buffers_same = all(torch.equal(before[name],dict(model.named_buffers())[name].detach().cpu()) for name in before)
    if not frozen_bn or not buffers_same: raise RuntimeError('Smoke changed BatchNorm running statistics')
    receipt = {'state':'passed','stage':'smoke','classes':CLASSES,'batch_size':len(targets),'loss':float(loss.detach()),
               'finite_gradient_norm_before_clip':norm,'bn_running_stats_unchanged':buffers_same,'all_bn_eval':frozen_bn,
               'backbone_trainable':True,'backbone_gradient_nonzero':backbone_gradient_nonzero,
               'backbone_parameter_updated':backbone_updated,'successful_optimizer_steps':1,'amp_skipped_steps':0,
               'amp_scale':float(scaler.get_scale()),'device':str(device),'source_sha256':config['source_sha256'],
               'config_signature':signature(config),'model_provenance':config['model_provenance'],
               'peak_allocated_gpu_gib':float(torch.cuda.max_memory_allocated()/1024**3) if torch.device(device).type=='cuda' else None,
               'formal_training_weights_saved':False}
    folder = root/'experiments'/'hse_specialist_v1'; folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'smoke.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt


def run_train(root, device):
    rows, config = prepare_config(root)
    folder = root/'experiments'/'hse_specialist_v1'; folder.mkdir(parents=True,exist_ok=True)
    config_path = folder/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Existing HSE run source/config/data changed; preserve it and use a new experiment')
    else:
        if any(path.name not in ['smoke.json','failure.json'] for path in folder.iterdir()):
            raise ValueError('Unbound training artifacts exist without their configuration')
        write_json(config_path,config)
    if (folder/'summary.json').exists():
        receipt = load_receipt(root)
        print('Verified completed HSE run; no training/inference repeated.',flush=True)
        return receipt
    seed_everything(42)
    model = model_api().load_initial_model(root,device)
    optimizer, scaler = optimizer_for(model), make_scaler(device)
    best_state, best_p, best_flip, best_validation, best_epoch, history, first_epoch = None,None,None,None,0,[],1
    if (folder/'last.json').exists():
        saved, _ = load_committed(folder,config,device)
        model.load_state_dict(saved['model_state'],strict=True)
        optimizer.load_state_dict(saved['optimizer_state']); scaler.load_state_dict(saved['scaler_state'])
        restore_rng(saved['rng_state'])
        best_state,best_p,best_flip = saved['best_model_state'],saved['best_val_original'],saved['best_val_flip']
        best_validation,best_epoch,history = saved['best_validation'],saved['best_epoch'],saved['history']
        first_epoch = saved['epoch']+1
        materialize_best(folder,saved,config)
        print(json.dumps({'stage':'resume','committed_epoch':saved['epoch'],'next_epoch':first_epoch}),flush=True)
        del saved
    (folder/'epochs').mkdir(exist_ok=True)
    for epoch in range(first_epoch,13):
        entry = train_epoch(model,rows['train'],optimizer,scaler,epoch,device)
        original,flip = predict_validation(model,rows['val'],epoch,device)
        validation = metrics(rows['val'],original)  # flip scores never choose the checkpoint.
        state = cpu_tree(model.state_dict())
        if best_validation is None or metric_key(validation)>metric_key(best_validation):
            best_validation,best_epoch,best_state = validation,epoch,state
            best_p,best_flip = torch.from_numpy(original.copy()),torch.from_numpy(flip.copy())
        history.append({'epoch':epoch,**entry,**{'val_'+key:value for key,value in validation.items()},
                        'checkpoint_selection_view':'original','best_epoch':best_epoch})
        committed = {'format':'hse_specialist_epoch_v1','config':config,'run_signature':signature(config),
                     'epoch':epoch,'training_device':str(device),'model_state':state,
                     'optimizer_state':cpu_tree(optimizer.state_dict()),'scaler_state':scaler.state_dict(),
                     'rng_state':rng_state(),'history':history,'best_epoch':best_epoch,'best_validation':best_validation,
                     'best_model_state':best_state,'best_val_original':best_p,'best_val_flip':best_flip}
        epoch_path = folder/'epochs'/f'epoch_{epoch:02d}.pt'
        save_torch(epoch_path,committed)
        write_json(folder/'last.json',{'epoch':epoch,'checkpoint':f'epochs/epoch_{epoch:02d}.pt',
                                      'sha256':sha256(epoch_path),'run_signature':signature(config)})
        materialize_best(folder,committed,config)
        print(json.dumps({'stage':'epoch_complete','epoch':epoch,**validation,'best_epoch':best_epoch}),flush=True)
        del committed,state,original,flip
    summary = {'format':'hse_specialist_receipt_v1','state':'complete','classes':CLASSES,
               'epochs_completed':12,'train_rows':len(rows['train']),'val_rows':len(rows['val']),
               'best_epoch':best_epoch,'best_validation':best_validation,'checkpoint_selection_view':'original',
               'config':config,'run_signature':signature(config),**membership_metadata(config),
               'manifest_sha256':config['input_sha256']['manifest_sha256'],'input_sha256':config['input_sha256'],
               'best_checkpoint_sha256':sha256(folder/'best.pt'),'val_original_sha256':sha256(folder/'val_original.npy'),
               'val_flip_sha256':sha256(folder/'val_flip.npy'),'internal_test_read':False,'external_answers_used':False,
               'train_metrics_purpose':'online augmentation diagnostics; train loss uses smoothing, validation log_loss does not',
               'artifact_sha256':{name:sha256(folder/name) for name in OUTPUT_FILES}}
    write_json(folder/'summary.json',summary)
    print(json.dumps({'stage':'train','state':'complete','best_epoch':best_epoch,'best_validation':best_validation}),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['smoke','train'],required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args(); root = args.project_root.resolve()
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    try:
        (run_smoke if args.stage=='smoke' else run_train)(root,args.device)
    except BaseException as error:
        folder = root/'experiments'/'hse_specialist_v1'; folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'failure.json',{'stage':args.stage,'state':'failed','error_type':type(error).__name__,
                                        'message':str(error),'source_sha256':sha256(__file__),
                                        'note':'No completion is claimed; only committed whole epochs may resume.'})
        raise


if __name__=='__main__':
    main()
