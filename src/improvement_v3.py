"""Prespecified sequential experiments. Train/val only; preserve all old outputs.

Stage 1: evaluate CURRENT states of the eight saved weighted epochs in fixed fusion.
Stage 2: matched uniform HSE continuation, full vs final stage, four epochs each.
Stage 3: shared train-only cached head warmup, rank-8 qkv LoRA last 4 vs 8 blocks.
Every epoch is committed atomically. Selection uses a strict gain in correct count;
ties retain the incumbent. No fusion-weight search, calibration or answer files.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['XFORMERS_DISABLED'] = '1'
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'improvement_v3_results'
EXP = ROOT/'experiments'/'improvement_v3'
PLAN = ROOT/'logs'/'improvement_v3_plan.json'
CLASSES = h.CLASSES
PRESERVED = 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7'
PROTOCOL = {
    'classes': CLASSES, 'seed': 42, 'cpu_threads': 4, 'workers': 0,
    'diagnostic': {'versions': [1, 2], 'epochs': [1, 2, 3, 4], 'use': 'current model_state, never best_model_state'},
    'hse': {'modes': ['full', 'last_stage'], 'epochs': 4, 'batch_size': 8, 'accumulation': 4,
            'backbone_lr': 5e-6, 'head_lr': 5e-5, 'weight_decay': .01, 'min_lr_ratio': .1,
            'label_smoothing': .05, 'sample_weights': 'uniform', 'batchnorm': 'running statistics frozen',
            'last_stage_prefixes': ['blocks.6.', 'conv_head.', 'bn2.', 'classifier.'],
            'initial': 'original HSE best epoch 7'},
    'head': {'feature_dim': 1536, 'epochs': 5, 'lr': .001, 'batch_size': 256,
             'weight_decay': .01, 'label_smoothing': .05, 'validation_selection': False,
             'views': 'one random original/flip view per training row per epoch'},
    'lora': {'last_blocks': [4, 8], 'rank': 8, 'alpha': 16., 'lora_dropout': .05,
             'epochs': 4, 'effective_batch_size': 32, 'lr_lora': 2e-5, 'lr_head': .0001,
             'weight_decay': .001, 'min_lr_ratio': .1, 'label_smoothing': .05,
             'activation_checkpoint': 'adapted blocks only, use_reentrant=False', 'ema': False,
             'initial': 'same official DINOv2 + same fixed train-only warm head'},
    'fusion': {'hse': .5, 'lora': .2, 'cnn': .2, 'mlp': .1, 'views': 'mean original/flip'},
    'selection': 'strict correct-count increase; ties keep incumbent; checkpoint candidates fixed in advance',
    'validation_caveat': 'Repeated development validation, not independent accuracy or Kaggle score.',
    'internal_test_read': False, 'external_answers_used': False,
}

def emit(stage, **values):
    print(json.dumps({'stage': stage, **values}, ensure_ascii=False), flush=True)

def clean_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def checked_plan():
    plan = c.read_json(PLAN)
    if plan['protocol'] != PROTOCOL: raise ValueError('Locked protocol changed')
    c.verify_pins(ROOT, plan['pins'])
    h.check_hash(ROOT/'hse_results'/'submission.csv', PRESERVED)
    return plan

def bind_config(folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'config.json'
    if path.exists() and c.read_json(path) != config: raise ValueError('Experiment config changed')
    h.write_json(path, config)

def members():
    return h.load_trainval(ROOT)

def previous(stage):
    if stage == 'diagnostic':
        folder = ROOT/'hse_results'
        return {'name': 'previous_best', 'kind': 'hse',
                'model': str(ROOT/'experiments'/'hse_specialist_v1'/'best.pt'),
                'original': str(ROOT/'experiments'/'hse_specialist_v1'/'val_original.npy'),
                'flip': str(ROOT/'experiments'/'hse_specialist_v1'/'val_flip.npy'),
                'hse_model': str(ROOT/'experiments'/'hse_specialist_v1'/'best.pt'),
                'probability': str(folder/'selected_val_probabilities.npy'),
                'export_folder': str(folder)}
    parent = 'diagnostic' if stage == 'hse' else 'hse'
    folder = OUT/parent
    summary = c.read_json(folder/'summary.json')
    if summary['state'] != 'complete': raise ValueError('Previous export is not complete')
    for name, digest in summary['file_sha256'].items(): h.check_hash(folder/name, digest)
    recipe = c.read_json(folder/'selected_recipe.json')
    c.verify_pins(ROOT, recipe['producer_files'])
    return {**recipe['selected'], 'export_folder': str(folder)}

def fixed_hse_fusion(original, flipped):
    p0 = h.probability(np.load(ROOT/'fusion_validation_results'/'selected_val_probabilities.npy', allow_pickle=False))
    return h.probability(.5*p0+.25*h.probability(original)+.25*h.probability(flipped))

def add_candidate(folder, name, model_path, original, flipped, rows, kind='hse', extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    for view, p in [('original', original), ('flip', flipped)]:
        h.save_array(folder/f'{view}.npy', h.probability(p))
    info = {'name': name, 'kind': kind, 'model': str(model_path),
            'original': str(folder/'original.npy'), 'flip': str(folder/'flip.npy'),
            'single_original': c.metrics(np.array([r['target'] for r in rows]), original),
            'single_tta': c.metrics(np.array([r['target'] for r in rows]), h.probability((original+flipped)/2)),
            'classes': CLASSES, 'val_indices': [r['manifest_index'] for r in rows],
            'artifact_sha256': {str(path): h.sha256(path) for path in [model_path, folder/'original.npy', folder/'flip.npy']},
            **(extra or {})}
    h.write_json(folder/'candidate.json', info)
    return info

def diagnostic():
    plan = checked_plan(); rows, receipt = members(); h.seed_everything(42)
    if receipt != plan['input_receipt']: raise ValueError('Image membership changed')
    for version in [1, 2]:
        origin = ROOT/'experiments'/f'hse_ambiguity_v{version}'/'downweighted'
        config = c.read_json(origin/'config.json')
        final = torch.load(origin/'epochs'/'epoch_04.pt', map_location='cpu', weights_only=True)
        for epoch in [1, 2, 3, 4]:
            name = f'weighted_v{version}_epoch_{epoch:02d}'; folder = EXP/'diagnostic'/name
            if (folder/'candidate.json').exists():
                candidate = c.read_json(folder/'candidate.json')
                c.verify_pins(ROOT, candidate['artifact_sha256']); continue
            state = torch.load(origin/'epochs'/f'epoch_{epoch:02d}.pt', map_location='cpu', weights_only=True)
            if (state['config'] != config or state['epoch'] != epoch or state['history'] != final['history'][:epoch]
                or config['train_indices'] != [r['manifest_index'] for r in rows['train']]
                or config['val_indices'] != [r['manifest_index'] for r in rows['val']]):
                raise ValueError('Saved CURRENT epoch state is not bound to the original run')
            payload = {'classes': CLASSES, 'model_provenance': config['model_provenance'],
                       'model_state': state['model_state'], 'epoch': epoch, 'source_checkpoint': str(origin/'epochs'/f'epoch_{epoch:02d}.pt')}
            folder.mkdir(parents=True, exist_ok=True); model_path = folder/'model.pt'; h.save_torch(model_path, payload)
            del state; clean_gpu()
            model = h.model_api().load_trained_model(ROOT, model_path, 'cuda')
            original, flip = h.predict_validation(model, rows['val'], epoch, 'cuda')
            score = h.metrics(rows['val'], original)
            for key in score:
                if not np.isclose(score[key], final['history'][epoch-1]['val_'+key], atol=1e-10, rtol=0):
                    raise ValueError('Recomputed CURRENT checkpoint validation differs from recorded metrics')
            add_candidate(folder, name, model_path, original, flip, rows['val'], extra={'original_epoch': epoch, 'weight_version': version})
            fused = c.metrics(np.array([r['target'] for r in rows['val']]), fixed_hse_fusion(original, flip))
            emit('diagnostic_candidate', name=name, single_accuracy=score['accuracy'], fused_accuracy=fused['accuracy'], correct_count=fused['correct_count'])
            del model; clean_gpu()
        del final; clean_gpu()

def hse_scope(model, mode):
    for name, p in model.named_parameters():
        p.requires_grad_(mode == 'full' or name.startswith(tuple(PROTOCOL['hse']['last_stage_prefixes'])))
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
    if not any(p.requires_grad for name, p in model.named_parameters() if not name.startswith('classifier.')):
        raise ValueError('No backbone parameters selected')

def hse_optimizer(model):
    backbone = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')]
    return torch.optim.AdamW([{'params': backbone, 'lr': 5e-6, 'base_lr': 5e-6},
                             {'params': model.classifier.parameters(), 'lr': 5e-5, 'base_lr': 5e-5}], weight_decay=.01)

def cosine_epoch(optimizer, epoch):
    factor = .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/3))
    for group in optimizer.param_groups: group['lr'] = group['base_lr']*factor

def smoke_hse(rows=None):
    if rows is None: rows, _ = members()
    h.seed_everything(42)
    model = h.model_api().load_trained_model(ROOT, ROOT/'experiments'/'hse_specialist_v1'/'best.pt', 'cuda')
    hse_scope(model, 'last_stage')
    frozen = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    bn = {n:p.detach().cpu().clone() for n,p in model.named_buffers()}
    before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer, scaler = hse_optimizer(model), h.make_scaler('cuda')
    images, targets = next(iter(h.loader_for(rows['train'][:8], True, 1, 'cuda')))
    with h.autocast_for('cuda'): loss = nn.functional.cross_entropy(model(images.cuda()), targets.cuda(), label_smoothing=.05)
    scaler.scale(loss).backward(); norm, skipped = h.optimizer_step(model, optimizer, scaler)
    named = dict(model.named_parameters())
    if skipped or any(not torch.equal(p, named[n].detach().cpu()) for n,p in frozen.items()): raise ValueError('Frozen parameters changed')
    if any(not torch.equal(p, dict(model.named_buffers())[n].detach().cpu()) for n,p in bn.items()): raise ValueError('BN buffers changed')
    updated = [n for n,p in before.items() if not torch.equal(p, named[n])]
    if not any(n.startswith('blocks.6.') for n in updated) or not any(n.startswith('classifier.') for n in updated):
        raise ValueError('Last stage/head did not receive a real update')
    receipt = {'passed': True, 'frozen_unchanged': True, 'bn_buffers_unchanged': True, 'updated_names': updated,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'gradient_norm': norm,
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'model_weights_saved': False}
    del model, optimizer, scaler, before; clean_gpu(); return receipt

def hse_train(mode):
    plan = checked_plan(); rows, receipt = members()
    if receipt != plan['input_receipt']: raise ValueError('Image membership changed')
    folder = EXP/f'hse_{mode}'
    config = {'format': 'improvement_v3_hse', 'mode': mode, 'protocol': PROTOCOL['hse'],
              'classes': CLASSES, 'plan_sha256': h.sha256(PLAN), 'input_receipt': receipt}
    bind_config(folder, config)
    if (folder/'summary.json').exists():
        summary = c.read_json(folder/'summary.json'); c.verify_pins(ROOT, summary['artifact_sha256']); return
    h.seed_everything(42)
    model = h.model_api().load_trained_model(ROOT, ROOT/'experiments'/'hse_specialist_v1'/'best.pt', 'cuda')
    hse_scope(model, mode); optimizer, scaler = hse_optimizer(model), h.make_scaler('cuda')
    named = dict(model.named_parameters()); initial_frozen = {n:p.detach().cpu().clone() for n,p in named.items() if not p.requires_grad}
    initial_buffers = {n:p.detach().cpu().clone() for n,p in model.named_buffers()}
    done, history = 0, []
    if (folder/'last.json').exists():
        pointer = c.read_json(folder/'last.json'); h.check_hash(folder/pointer['checkpoint'], pointer['sha256'])
        state = torch.load(folder/pointer['checkpoint'], map_location='cpu', weights_only=True)
        if state['config'] != config or state['epoch'] != pointer['epoch']: raise ValueError('Resume binding changed')
        model.load_state_dict(state['model_state'], strict=True); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        h.restore_rng(state['rng']); done, history = state['epoch'], state['history']; del state
    for epoch in range(done+1, 5):
        hse_scope(model, mode); cosine_epoch(optimizer, epoch)
        loader = h.loader_for(rows['train'], True, epoch, 'cuda'); optimizer.zero_grad(set_to_none=True)
        seen, total, correct, steps, skipped = 0, 0., 0, 0, 0; start = time.monotonic()
        for i,(images,targets) in enumerate(loader):
            images,targets = images.cuda(),targets.cuda()
            group_samples = min(32, len(rows['train'])-(i//4)*32)
            with h.autocast_for('cuda'):
                logits = model(images); loss = nn.functional.cross_entropy(logits,targets,label_smoothing=.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite HSE loss')
            scaler.scale(loss*len(targets)/group_samples).backward()
            total += float(loss.detach())*len(targets); seen += len(targets); correct += int((logits.detach().argmax(1)==targets).sum())
            if (i+1)%4 == 0 or i+1 == len(loader):
                _,skip = h.optimizer_step(model,optimizer,scaler); steps += int(not skip); skipped += int(skip)
            if (i+1)%50 == 0 or i+1 == len(loader): emit('train', experiment=f'hse_{mode}', epoch=epoch, batch=i+1,batches=len(loader),train_loss=total/seen,elapsed_seconds=time.monotonic()-start)
        if steps == 0: raise RuntimeError('No optimizer steps')
        if any(not torch.equal(p, named[n].detach().cpu()) for n,p in initial_frozen.items()): raise ValueError('Frozen HSE parameters changed')
        if any(not torch.equal(p, dict(model.named_buffers())[n].detach().cpu()) for n,p in initial_buffers.items()): raise ValueError('Frozen HSE buffers changed')
        original,flip = h.predict_validation(model, rows['val'], epoch, 'cuda')
        history.append({'epoch':epoch,'train_loss':total/seen,'train_accuracy':correct/seen,'successful_optimizer_steps':steps,'amp_skipped_steps':skipped,'train_seconds':time.monotonic()-start,**h.metrics(rows['val'],original)})
        epoch_folder = folder/f'epoch_{epoch:02d}'; epoch_folder.mkdir(exist_ok=True); model_path = epoch_folder/'model.pt'
        h.save_torch(model_path, {'classes': CLASSES,'model_provenance':h.model_api().source_receipt(ROOT),'model_state':h.cpu_tree(model.state_dict()),'epoch':epoch,'config':config})
        add_candidate(epoch_folder, f'hse_{mode}_epoch_{epoch:02d}', model_path, original,flip,rows['val'])
        state_path = folder/'last.pt'; h.save_torch(state_path, {'epoch':epoch,'config':config,'model_state':h.cpu_tree(model.state_dict()),'optimizer':h.cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict(),'rng':h.rng_state(),'history':history})
        h.write_json(folder/'last.json', {'epoch':epoch,'checkpoint':'last.pt','sha256':h.sha256(state_path)})
        h.write_history(folder,history); emit('epoch_complete',experiment=f'hse_{mode}',**history[-1])
    pins = {str(path):h.sha256(path) for path in folder.rglob('*') if path.is_file() and path.name!='summary.json'}
    h.write_json(folder/'summary.json', {'state':'complete','epochs_completed':4,'config':config,'history':history,'artifact_sha256':pins,'validation_weighted':False})
    del model,optimizer,scaler; clean_gpu()

def lora_model(blocks, checkpointed=True, device='cuda'):
    import train_lora as l
    from torch.utils.checkpoint import checkpoint
    model = l.make_model(ROOT, {'last_blocks':blocks,'rank':8,'alpha':16.,'lora_dropout':.05}, device)
    # Match the official forward exactly, while checkpointing only adapted blocks.
    def forward(images):
        x = model.backbone.prepare_tokens_with_masks(images, None)
        for index, block in enumerate(model.backbone.blocks):
            if model.training and checkpointed and index in model.adapted_blocks:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            else: x = block(x)
        x = model.backbone.norm(x)
        features = torch.cat((x[:,0],x[:,model.backbone.num_register_tokens+1:].mean(1)),dim=1)
        return model.head(features)
    model.forward = forward
    return model

def smoke_lora(rows=None):
    import train_lora as l
    if rows is None: rows,_ = members()
    l.seed_everything(42); meta = c.read_json(ROOT/'features'/'metadata.json')
    if meta['classes'] != CLASSES or meta['feature_dim'] != 1536 or not meta['frozen']: raise ValueError('Warmup feature format changed')
    for name in ['labeled_original.npy','labeled_flip.npy']: h.check_hash(ROOT/'features'/name, meta['file_sha256'][name])
    attempts=[]
    for batch in [4,2,1]:
        model=None
        try:
            torch.cuda.reset_peak_memory_stats(); l.seed_everything(42); model=lora_model(8)
            images, targets=next(iter(l.make_loader(rows['train'][:batch],batch,0,42,training=False)))
            images,targets=images.cuda(),targets.cuda(); model.eval()
            with torch.inference_mode(),h.autocast_for('cuda'):
                official=model.backbone.forward_features(images)
                fresh=torch.cat((official['x_norm_clstoken'],official['x_norm_patchtokens'].mean(1)),1)
                a=model(images); b=model.head(fresh)
                if not torch.allclose(a,b,atol=1e-4,rtol=1e-3): raise ValueError('Checkpoint forward differs from official')
            cached=np.load(ROOT/'features'/'labeled_original.npy',mmap_mode='r',allow_pickle=False)
            expected=np.asarray(cached[[r['manifest_index'] for r in rows['train'][:batch]]],dtype=np.float32)
            actual=fresh.float().cpu().numpy(); error=float(np.max(np.abs(expected-actual)))
            cosine=float(((actual*expected).sum(1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))).min())
            # AMP GEMM kernels differ with batch shape. The original cache used
            # batch 32; verify that shape directly, rather than rejecting small
            # numerical differences around zero at the training batch size.
            reference_input=torch.stack([l.FaceImages(rows['train'][:32],training=False)[i][0] for i in range(32)]).cuda()
            reference_error=0.
            for view in ['original','flip']:
                with torch.inference_mode(),h.autocast_for('cuda'):
                    z=model.backbone.forward_features(reference_input if view=='original' else reference_input.flip(-1))
                    z=torch.cat((z['x_norm_clstoken'],z['x_norm_patchtokens'].mean(1)),1).float().cpu().numpy()
                reference_cache=np.load(ROOT/'features'/f'labeled_{view}.npy',mmap_mode='r',allow_pickle=False)
                reference=np.asarray(reference_cache[[r['manifest_index'] for r in rows['train'][:32]]],dtype=np.float32)
                reference_error=max(reference_error,float(np.abs(z-reference).max()))
                if not np.allclose(z,reference,atol=1e-4,rtol=1e-4): raise ValueError('Cached batch-32 features differ from the official live model')
            if cosine<.99998 or error>.03: raise ValueError('Small-batch AMP feature difference exceeds the verified numerical budget')
            del reference_input,z,reference,reference_cache
            model.train(); optimizer=l.make_optimizer(model, {'lr_lora':2e-5,'lr_head':.0001,'weight_decay':.001}); scaler=h.make_scaler('cuda')
            frozen_sample={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad and n.endswith('base.weight')}
            before=l.adapter_state(model)
            for _ in range(2):
                with h.autocast_for('cuda'): loss=nn.functional.cross_entropy(model(images),targets,label_smoothing=.05)
                scaler.scale(loss).backward(); _,skip=h.optimizer_step(model,optimizer,scaler)
                if skip: raise RuntimeError('AMP skipped LoRA smoke update')
            updated=[n for n,p in l.adapter_state(model).items() if not torch.equal(p,before[n])]
            if not any(n.endswith('lora_A') for n in updated) or not any(n.endswith('lora_B') for n in updated): raise ValueError('Checkpointed LoRA gradients missing')
            if any(not torch.equal(p,dict(model.named_parameters())[n]) for n,p in frozen_sample.items()): raise ValueError('Frozen qkv base weights changed')
            peak=torch.cuda.max_memory_allocated()/2**20
            if peak>5000: raise torch.cuda.OutOfMemoryError('Probe exceeds reserved 5GB allocated budget')
            receipt={'passed':True,'batch_size':batch,'accumulation':32//batch,'successful_steps':2,'small_batch_feature_max_error':error,'small_batch_feature_cosine_min':cosine,'cache_reference_batch32_max_error_both_views':reference_error,'checkpoint_official_forward_equivalent':True,'lora_A_and_B_updated':True,'frozen_qkv_unchanged':True,'peak_allocated_mib':peak,'attempts':attempts,'weights_saved':False}
            del model,optimizer,scaler,images,targets,fresh,official,a,b,before,frozen_sample; clean_gpu(); return receipt
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({'batch_size':batch,'error':str(error)}); del model; clean_gpu()
    raise RuntimeError('Cannot fit a real LoRA update on local GPU')

def prepare():
    if PLAN.exists(): checked_plan(); return
    EXP.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
    rows,receipt=members()
    h.check_hash(ROOT/'hse_results'/'submission.csv',PRESERVED)
    import finalize_hse_specialist as old
    _,old_recipe,_=old.verify_locked(ROOT)
    _,targets,baseline,baseline_receipt=old.baseline_inputs(ROOT)
    pins=dict(old_recipe['producer_files']); pins.update(baseline_receipt['producer_files'])
    for relative in ['hse_results/selected_recipe.json','hse_results/summary.json','hse_results/submission.csv','hse_results/probabilities.npy','hse_results/selected_val_probabilities.npy','features/metadata.json','features/labeled_original.npy','features/labeled_flip.npy','features/unlabeled.csv','data/raw/submission.csv','data/split_audit.json','data/original_splits/image_inventory.csv','models/dinov2_vitb14_pretrain.pth','models/dinov2_vitb14_pretrain.json']:
        pins[str(ROOT/relative)]=h.sha256(ROOT/relative)
    cache=ROOT/'fusion_results'/'prediction_cache'/'refined_lora'
    cache_meta=c.read_json(cache/'metadata.json'); legacy_export=c.read_json(ROOT/'fusion_results'/'summary.json')
    h.check_hash(cache/'metadata.json',legacy_export['prediction_cache_receipts']['refined_lora']['metadata_sha256'])
    if cache_meta['binding']['classes']!=CLASSES or cache_meta['binding']['checkpoint_sha256']!=h.sha256(ROOT/'experiments'/'lora_refinement'/'best.pt'):
        raise ValueError('Old LoRA competition view cache differs from the trained model')
    for name in ['metadata.json','original.npy','flip.npy']:
        if name!='metadata.json': h.check_hash(cache/name,cache_meta['file_sha256'][name])
        pins[str(cache/name)]=h.sha256(cache/name)
    # These arrays are only hashed here; competition pixels are used after locking
    # the selected validation recipe, and never for fitting or model selection.
    for version in [1,2]:
        folder=ROOT/'experiments'/f'hse_ambiguity_v{version}'/'downweighted'
        summary=c.read_json(folder/'summary.json')
        if summary['state']!='complete' or summary['epochs_completed']!=4: raise ValueError('Original weighting run incomplete')
        for name,digest in summary['file_sha256'].items(): h.check_hash(folder/name,digest)
        for name,digest in summary['config']['source_sha256'].items(): h.check_hash(ROOT/'src'/name,digest)
        pointer=c.read_json(folder/'last.json'); h.check_hash(folder/pointer['checkpoint'],pointer['sha256'])
        for path in [folder/'summary.json',folder/'config.json',folder/'last.json',*[folder/'epochs'/f'epoch_{i:02d}.pt' for i in range(1,5)]]: pins[str(path)]=h.sha256(path)
    for name in ['improvement_v3.py','run_improvement_v3.py','improvement_v3_worker.py','train_hse_specialist.py','hse_model.py','train_lora.py','refine_lora.py','finalize_hse_specialist.py','validate_cosine_fusion.py']:
        pins[str(ROOT/'src'/name)]=h.sha256(ROOT/'src'/name)
    import train_lora as l
    official=l.verify_official_weights(ROOT); repo=l.repository_info(ROOT)
    smoke={'hse':smoke_hse(rows),'lora':smoke_lora(rows)}
    plan={'format':'improvement_v3_locked','protocol':PROTOCOL,'pins':pins,'input_receipt':receipt,
          'train_indices':[r['manifest_index'] for r in rows['train']], 'val_indices':[r['manifest_index'] for r in rows['val']],
          'dino_repository':repo,'official_dino_sha256':official,'smoke':smoke,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    current=h.probability(np.load(ROOT/'hse_results'/'selected_val_probabilities.npy',allow_pickle=False))
    original=h.probability(np.load(ROOT/'experiments'/'hse_specialist_v1'/'val_original.npy',allow_pickle=False)); flipped=h.probability(np.load(ROOT/'experiments'/'hse_specialist_v1'/'val_flip.npy',allow_pickle=False))
    if not np.allclose(current,.5*baseline+.25*original+.25*flipped,atol=1e-12,rtol=0) or c.metrics(targets,current)['correct_count']!=1913: raise ValueError('Baseline reconstruction changed')
    c.verify_pins(ROOT,pins); h.write_json(PLAN,plan); emit('prepare',state='locked',smoke=smoke)

def head_warmup():
    import train_lora as l
    plan=checked_plan(); rows,receipt=members(); folder=EXP/'head_warmup'
    config={'protocol':PROTOCOL['head'],'classes':CLASSES,'plan_sha256':h.sha256(PLAN),'input_receipt':receipt}
    bind_config(folder,config)
    if (folder/'summary.json').exists():
        c.verify_pins(ROOT,c.read_json(folder/'summary.json')['artifact_sha256']); return
    h.seed_everything(42); head=nn.Linear(1536,6); optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    arrays=[np.load(ROOT/'features'/f'labeled_{view}.npy',mmap_mode='r',allow_pickle=False) for view in ['original','flip']]
    if any(a.shape!=(26560,1536) for a in arrays): raise ValueError('Unexpected frozen cache shape')
    indices=np.array([r['manifest_index'] for r in rows['train']]); targets=torch.tensor([r['target'] for r in rows['train']],dtype=torch.long)
    done=0; history=[]
    if (folder/'last.json').exists():
        pointer=c.read_json(folder/'last.json'); h.check_hash(folder/'last.pt',pointer['sha256']); saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=config: raise ValueError('Warmup resume differs')
        head.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer']); h.restore_rng(saved['rng']); done=saved['epoch']; history=saved['history']
    for epoch in range(done+1,6):
        generator=np.random.default_rng(42+epoch); order=generator.permutation(len(indices)); total=0.; seen=0
        for start in range(0,len(order),256):
            chosen=order[start:start+256]; mask=generator.integers(0,2,size=len(chosen))
            x=np.empty((len(chosen),1536),dtype=np.float32)
            for view in [0,1]:
                selected=mask==view; x[selected]=arrays[view][indices[chosen[selected]]]
            logits=head(torch.from_numpy(x)); loss=nn.functional.cross_entropy(logits,targets[chosen],label_smoothing=.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite head warmup loss')
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            total+=float(loss.detach())*len(chosen); seen+=len(chosen)
        history.append({'epoch':epoch,'train_loss':total/seen}); emit('head_warmup',**history[-1])
        h.save_torch(folder/'last.pt',{'config':config,'epoch':epoch,'head':h.cpu_tree(head.state_dict()),'optimizer':h.cpu_tree(optimizer.state_dict()),'rng':h.rng_state(),'history':history})
        h.write_json(folder/'last.json',{'epoch':epoch,'sha256':h.sha256(folder/'last.pt')})
    h.save_torch(folder/'head.pt',{'config':config,'epoch':5,'head':h.cpu_tree(head.state_dict()),'train_indices':indices.tolist(),'classes':CLASSES})
    h.write_json(folder/'summary.json',{'state':'complete','epochs_completed':5,'validation_used':False,'history':history,'artifact_sha256':{str(folder/'head.pt'):h.sha256(folder/'head.pt'),str(folder/'last.pt'):h.sha256(folder/'last.pt')}})
    del head,optimizer,arrays; clean_gpu()

def lora_train(blocks):
    import train_lora as l
    import refine_lora as ref
    plan=checked_plan(); rows,receipt=members(); head_path=EXP/'head_warmup'/'head.pt'
    c.verify_pins(ROOT,c.read_json(EXP/'head_warmup'/'summary.json')['artifact_sha256'])
    folder=EXP/f'lora_last{blocks}'; batch=plan['smoke']['lora']['batch_size']; accumulation=32//batch
    config={'format':'improvement_v3_lora','classes':CLASSES,'last_blocks':blocks,'protocol':PROTOCOL['lora'],
            'batch_size':batch,'accumulation':accumulation,'plan_sha256':h.sha256(PLAN),'head_sha256':h.sha256(head_path),'input_receipt':receipt}
    bind_config(folder,config)
    if (folder/'summary.json').exists(): c.verify_pins(ROOT,c.read_json(folder/'summary.json')['artifact_sha256']); return
    h.seed_everything(42); model=lora_model(blocks); warm=torch.load(head_path,map_location='cpu',weights_only=True)
    if warm['classes']!=CLASSES or warm['train_indices']!=plan['train_indices']: raise ValueError('Warm head trained on other rows/classes')
    model.head.load_state_dict(warm['head'],strict=True)
    optimizer=l.make_optimizer(model,PROTOCOL['lora']); scaler=h.make_scaler('cuda'); done=0; history=[]
    loader_val=l.make_loader(rows['val'],batch,0,42,training=False)
    def save_candidate(epoch,original,flipped):
        ef=folder/f'epoch_{epoch:02d}'; ef.mkdir(exist_ok=True); path=ef/'model.pt'
        h.save_torch(path,{'format':'improvement_v3_lora','classes':CLASSES,'config':config,'epoch':epoch,'adapter_state':l.adapter_state(model),
                          'official_weight_sha256':plan['official_dino_sha256'],'dino_repository':plan['dino_repository']})
        return add_candidate(ef,f'lora_last{blocks}_epoch_{epoch:02d}',path,original,flipped,rows['val'],kind='lora')
    if (folder/'last.json').exists():
        pointer=c.read_json(folder/'last.json'); h.check_hash(folder/'last.pt',pointer['sha256']); saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=config or saved['epoch']!=pointer['epoch']: raise ValueError('LoRA resume binding changed')
        l.restore_adapter(model,saved['adapter_state']); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); h.restore_rng(saved['rng']); done=saved['epoch']; history=saved['history']; del saved
    if done==0 and not (folder/'epoch_00'/'candidate.json').exists():
        _,a,b=ref.validate_views(model,loader_val,'cuda'); save_candidate(0,a,b)
    for epoch in range(done+1,5):
        model.train(); factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/3))
        for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
        loader=l.make_loader(rows['train'],batch,0,42,epoch=epoch,training=True)
        optimizer.zero_grad(set_to_none=True); total=0.; seen=0; correct=0; steps=0; skipped=0; start=time.monotonic()
        for i,(images,targets) in enumerate(loader):
            images,targets=images.cuda(),targets.cuda(); group_samples=min(32,len(rows['train'])-(i//accumulation)*32)
            with h.autocast_for('cuda'): logits=model(images); loss=nn.functional.cross_entropy(logits,targets,label_smoothing=.05)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite LoRA training loss')
            scaler.scale(loss*len(targets)/group_samples).backward(); total+=float(loss.detach())*len(targets); seen+=len(targets); correct+=int((logits.detach().argmax(1)==targets).sum())
            if (i+1)%accumulation==0 or i+1==len(loader):
                _,skip=h.optimizer_step(model,optimizer,scaler); steps+=int(not skip); skipped+=int(skip)
            if (i+1)%100==0 or i+1==len(loader): emit('train',experiment=f'lora_last{blocks}',epoch=epoch,batch=i+1,batches=len(loader),train_loss=total/seen,successful_optimizer_steps=steps,amp_skipped_steps=skipped,elapsed_seconds=time.monotonic()-start)
        if steps==0: raise RuntimeError('No successful LoRA updates')
        _,original,flipped=ref.validate_views(model,loader_val,'cuda'); save_candidate(epoch,original,flipped)
        history.append({'epoch':epoch,'train_loss':total/seen,'train_accuracy':correct/seen,'successful_optimizer_steps':steps,'amp_skipped_steps':skipped,'train_seconds':time.monotonic()-start,**h.metrics(rows['val'],h.probability((original+flipped)/2))})
        h.save_torch(folder/'last.pt',{'epoch':epoch,'config':config,'adapter_state':l.adapter_state(model),'optimizer':h.cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict(),'rng':h.rng_state(),'history':history})
        h.write_json(folder/'last.json',{'epoch':epoch,'sha256':h.sha256(folder/'last.pt')}); h.write_history(folder,history)
        emit('epoch_complete',experiment=f'lora_last{blocks}',**history[-1])
    pins={str(p):h.sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='summary.json'}
    h.write_json(folder/'summary.json',{'state':'complete','epochs_completed':4,'config':config,'history':history,'artifact_sha256':pins})
    del model,optimizer,scaler; clean_gpu()

def load_new_lora(path,device='cuda'):
    import train_lora as l
    plan=checked_plan(); state=torch.load(path,map_location='cpu',weights_only=True)
    if state['format']!='improvement_v3_lora' or state['classes']!=CLASSES or state['config']['plan_sha256']!=h.sha256(PLAN): raise ValueError('LoRA model binding changed')
    if state['official_weight_sha256']!=l.verify_official_weights(ROOT) or state['dino_repository']!=l.repository_info(ROOT): raise ValueError('LoRA backbone differs')
    model=lora_model(state['config']['last_blocks'],False,device); l.restore_adapter(model,state['adapter_state']); return model.eval()

def select(stage):
    plan=checked_plan(); rows,receipt=members(); targets=np.array([r['target'] for r in rows['val']]); incumbent=previous(stage)
    folder=OUT/stage; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'selected_recipe.json').exists():
        recipe=c.read_json(folder/'selected_recipe.json'); c.verify_pins(ROOT,recipe['producer_files']); return recipe
    incumbent_p=h.probability(np.load(incumbent['probability'],allow_pickle=False)); score=c.metrics(targets,incumbent_p)
    selected=dict(incumbent); best=score; best_p=incumbent_p; entries=[{'name':incumbent['name'],**score}]; pins=dict(plan['pins'])
    if stage!='diagnostic':
        parent_recipe=c.read_json(Path(incumbent['export_folder'])/'selected_recipe.json')
        c.verify_pins(ROOT,parent_recipe['producer_files']); pins.update(parent_recipe['producer_files'])
        pins[str(Path(incumbent['export_folder'])/'selected_recipe.json')]=h.sha256(Path(incumbent['export_folder'])/'selected_recipe.json')
    pins[str(Path(incumbent['model']))]=h.sha256(incumbent['model'])
    pins[str(Path(incumbent['hse_model']))]=h.sha256(incumbent['hse_model'])
    candidate_paths=[]
    if stage=='diagnostic': candidate_paths=sorted((EXP/'diagnostic').glob('*/candidate.json'))
    elif stage=='hse':
        for mode in ['full','last_stage']:
            summary=c.read_json(EXP/f'hse_{mode}'/'summary.json'); c.verify_pins(ROOT,summary['artifact_sha256']); pins.update(summary['artifact_sha256'])
            candidate_paths+=sorted((EXP/f'hse_{mode}').glob('epoch_*/candidate.json'))
    else:
        for blocks in [4,8]:
            summary=c.read_json(EXP/f'lora_last{blocks}'/'summary.json'); c.verify_pins(ROOT,summary['artifact_sha256']); pins.update(summary['artifact_sha256'])
            candidate_paths+=sorted((EXP/f'lora_last{blocks}').glob('epoch_*/candidate.json'))
    expected={'diagnostic':8,'hse':8,'lora':10}[stage]
    if len(candidate_paths)!=expected: raise ValueError('Fixed candidate set incomplete')
    old_lora=(h.probability(np.load(ROOT/'experiments'/'lora_refinement'/'val_original.npy',allow_pickle=False))+h.probability(np.load(ROOT/'experiments'/'lora_refinement'/'val_flip.npy',allow_pickle=False)))/2
    for path in candidate_paths:
        item=c.read_json(path); c.verify_pins(ROOT,item['artifact_sha256']); pins.update(item['artifact_sha256']); pins[str(path)]=h.sha256(path)
        if item['classes']!=CLASSES or item['val_indices']!=plan['val_indices']: raise ValueError('Candidate class/validation row order differs')
        a=h.probability(np.load(item['original'],allow_pickle=False)); b=h.probability(np.load(item['flip'],allow_pickle=False))
        if item['kind']=='hse': p=fixed_hse_fusion(a,b)
        else:
            raw=incumbent_p-.2*old_lora+.1*a+.1*b
            if raw.min() < -1e-10: raise ValueError('LoRA branch replacement produced negative probabilities')
            p=h.probability(np.maximum(raw,0))
        current=c.metrics(targets,p); entries.append({'name':item['name'],'single_accuracy':item['single_original']['accuracy'],'single_tta_accuracy':item['single_tta']['accuracy'],**current})
        if current['correct_count']>best['correct_count']:
            best,best_p=current,p
            selected={**item,'hse_model':item['model'] if item['kind']=='hse' else incumbent['hse_model']}
    for p in [Path(incumbent['probability']),Path(incumbent['export_folder'])/'summary.json',PLAN]: pins[str(p)]=h.sha256(p)
    selected['probability']=str(folder/'selected_val_probabilities.npy')
    recipe={'state':'locked','stage':stage,'protocol':PROTOCOL,'selected':selected,'metrics':best,'incumbent_metrics':score,
            'improved':best['correct_count']>score['correct_count'],'incumbent_export_folder':incumbent['export_folder'],
            'producer_files':pins,'val_indices':plan['val_indices'],'val_rows_sha256':receipt['val_rows_sha256'],
            'validation_candidates':entries,'plan_sha256':h.sha256(PLAN)}
    h.save_array(folder/'selected_val_probabilities.npy',best_p); recipe['selected_val_sha256']=h.sha256(folder/'selected_val_probabilities.npy')
    h.write_json(folder/'selected_recipe.json',recipe)
    fields=['name','single_accuracy','single_tta_accuracy','accuracy','macro_f1','log_loss','correct_count']
    with (folder/'candidate_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(entries)
    emit('select',experiment=stage,selected=selected['name'],accuracy=best['accuracy'],correct_count=best['correct_count'],improved=recipe['improved'])
    return recipe

def export(stage):
    import finalize_hse_specialist as old
    recipe=select(stage); folder=OUT/stage; recipe_sha=h.sha256(folder/'selected_recipe.json')
    if (folder/'summary.json').exists():
        summary=c.read_json(folder/'summary.json'); c.verify_pins(ROOT,{str(folder/n):v for n,v in summary['file_sha256'].items()}); return
    c.verify_pins(ROOT,recipe['producer_files']); h.check_hash(folder/'selected_val_probabilities.npy',recipe['selected_val_sha256'])
    rows=c.unlabeled_rows(ROOT); images=c.image_receipt(ROOT,rows,verify_inventory=False)
    def lock_check():
        h.check_hash(folder/'selected_recipe.json',recipe_sha)
        h.check_hash(Path(recipe['selected']['model']), recipe['producer_files'][recipe['selected']['model']])
    if not recipe['improved']:
        parent=Path(recipe['incumbent_export_folder']); p=h.probability(np.load(parent/'probabilities.npy',allow_pickle=False))
    elif recipe['selected']['kind']=='hse':
        a,b=old.predict_hse_views(ROOT,Path(recipe['selected']['model']),rows,True,'cuda',16,lock_check)
        for view,array in [('original',a),('flip',b)]: h.save_array(folder/f'hse_competition_{view}.npy',array)
        p=h.probability(.5*h.probability(np.load(ROOT/'fusion_results'/'probabilities.npy',allow_pickle=False))+.25*a+.25*b)
    else:
        import train_lora as l
        model=load_new_lora(recipe['selected']['model']); loader=DataLoader(l.FaceImages([{**r,'target':0} for r in rows],training=False),batch_size=16,num_workers=0,shuffle=False)
        outputs=[[],[]]
        with torch.inference_mode():
            for index,(x,_) in enumerate(loader):
                lock_check(); x=x.cuda()
                with h.autocast_for('cuda'): a=model(x); b=model(x.flip(-1))
                for view,logits in enumerate([a,b]): outputs[view].append(logits.float().softmax(1).cpu().numpy())
                if (index+1)%50==0: emit('export',experiment=stage,seen=min((index+1)*16,7065),total=7065)
        a,b=[h.probability(np.concatenate(v)) for v in outputs]; del model; clean_gpu()
        for view,array in [('original',a),('flip',b)]: h.save_array(folder/f'lora_competition_{view}.npy',array)
        parent=Path(recipe['incumbent_export_folder']); existing=h.probability(np.load(parent/'probabilities.npy',allow_pickle=False))
        cache=ROOT/'fusion_results'/'prediction_cache'/'refined_lora'; meta=c.read_json(cache/'metadata.json')
        old_views=[]
        for view in ['original','flip']:
            h.check_hash(cache/f'{view}.npy',meta['file_sha256'][f'{view}.npy']); old_views.append(h.probability(np.load(cache/f'{view}.npy',allow_pickle=False)))
        if meta['binding']['classes']!=CLASSES or meta['binding']['row_order_sha256']!=c.signature(rows): raise ValueError('Old LoRA competition cache row order differs')
        raw=existing-.1*old_views[0]-.1*old_views[1]+.1*a+.1*b
        if raw.min()<-1e-10: raise ValueError('Invalid competition branch replacement')
        p=h.probability(np.maximum(raw,0))
    h.save_array(folder/'probabilities.npy',p)
    temp=folder/'submission.csv.tmp'
    with temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['ID','Emotion']); writer.writerows((r['ID'],int(label)) for r,label in zip(rows,p.argmax(1)))
    temp.replace(folder/'submission.csv'); c.submission_check(folder/'submission.csv',rows,p)
    with (folder/'submission.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['ID','Emotion'] or len(list(reader))!=7065: raise ValueError('Submission format/count differs')
    c.verify_pins(ROOT,recipe['producer_files']); checked_plan()
    # Rehash competition image bytes after inference to bind order/content to outputs.
    if c.image_receipt(ROOT,rows,verify_inventory=False)['image_content_sha256']!=images['image_content_sha256']: raise ValueError('Competition image bytes changed during inference')
    names=['submission.csv','probabilities.npy','selected_val_probabilities.npy','selected_recipe.json','candidate_metrics.csv']+[p.name for p in folder.glob('*_competition_*.npy')]
    summary={'state':'complete','classes':CLASSES,'stage':stage,'selected':recipe['selected']['name'],'improved':recipe['improved'],
             'metrics':recipe['metrics'],'incumbent_metrics':recipe['incumbent_metrics'],'submission_rows':7065,'recipe_sha256':recipe_sha,
             'official_label_order_verified':True,'template_ID_order_verified':True,'probability_argmax_verified':True,
             'competition_image_content_sha256':images['image_content_sha256'],'competition_row_order_sha256':c.signature(rows),
             'preserved_old_submission_sha256':h.sha256(ROOT/'hse_results'/'submission.csv'),'internal_test_read':False,'external_answers_used':False,
             'validation_caveat':PROTOCOL['validation_caveat'],'file_sha256':{n:h.sha256(folder/n) for n in names}}
    h.write_json(folder/'summary.json',summary); emit('export_complete',experiment=stage,selected=summary['selected'],accuracy=summary['metrics']['accuracy'],submission=str(folder/'submission.csv'))
    if stage=='lora':
        # Top-level delivery is a verified copy, with old folders always preserved.
        for name in ['submission.csv','probabilities.npy','selected_val_probabilities.npy','selected_recipe.json']: shutil.copyfile(folder/name,OUT/name)
        h.write_json(OUT/'summary.json',{**summary,'final_stage_folder':str(folder),'delivery_sha256':{n:h.sha256(OUT/n) for n in ['submission.csv','probabilities.npy','selected_val_probabilities.npy','selected_recipe.json']}})

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',required=True,choices=['prepare','diagnostic','hse','head','lora','select','export']); parser.add_argument('--mode',choices=['full','last_stage']); parser.add_argument('--blocks',type=int,choices=[4,8]); parser.add_argument('--family',choices=['diagnostic','hse','lora'])
    args=parser.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no silent CPU training')
    if args.stage=='prepare': prepare()
    elif args.stage=='diagnostic': diagnostic()
    elif args.stage=='hse':
        if args.mode is None: parser.error('--mode required')
        hse_train(args.mode)
    elif args.stage=='head': head_warmup()
    elif args.stage=='lora':
        if args.blocks is None: parser.error('--blocks required')
        lora_train(args.blocks)
    else:
        if args.family is None: parser.error('--family required')
        (select if args.stage=='select' else export)(args.family)

if __name__=='__main__': main()
