"""Lock an inherited, expanded task LoRA and verify real GPU gradients."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[name]='4'
os.environ['XFORMERS_DISABLED']='1'
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import train_hse_specialist as h
import train_lora as l
import refine_lora as ref
import validate_cosine_fusion as c
import finalize_hse_specialist as old

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'logs'/'lora_task_v4_plan.json'
PARENT=ROOT/'experiments'/'lora_refinement'/'best.pt'
PRESERVED='f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7'
SETTINGS={'epochs':6,'seed':42,'augmentation_epoch_offset':20,'label_smoothing':.02,
          'weight_decay':.001,'lr_new':2e-5,'lr_inherited':5e-6,'lr_head':2e-5,
          'warmup_successful_steps':100,'min_lr_ratio':.1,'ema_decay':.995,
          'rank':8,'alpha':16.,'lora_dropout':.05,'last_blocks':8,
          'effective_batch_size':32,'workers':0,'cpu_threads':4,'image_size':224,
          'amp_initial_scale':1024.,'gradient_clip':1.,'validation_batch_size':4,
          'sample_weights':'uniform','selection':'fixed .5 HSE + .2 LoRA + .2 CNN + .1 MLP; strict correct-count increase; ties retain incumbent',
          'checkpoint':'adapted blocks only; use_reentrant=False; preserve_rng_state=True'}

def verify_plan(root=ROOT):
    import train_task_lora_v4 as trainer
    plan=c.read_json(root/'logs'/'lora_task_v4_plan.json')
    if plan['classes']!=h.CLASSES: raise ValueError('Official class order changed')
    for key,value in SETTINGS.items():
        if plan['settings'].get(key)!=value: raise ValueError('Locked LoRA settings changed: '+key)
    if plan['settings']['batch_size']*plan['settings']['accumulation']!=32: raise ValueError('Effective batch changed')
    c.verify_pins(root,plan['source_pins'])
    h.check_hash(PARENT,plan['parent_checkpoint_sha256']); h.check_hash(root/'hse_results'/'submission.csv',PRESERVED)
    if l.repository_info(root)!=plan['dino_repository'] or l.verify_official_weights(root)!=plan['official_dino_sha256']:
        raise ValueError('Official DINO runtime differs')
    return plan

def release():
    gc.collect(); torch.cuda.empty_cache()

def _smoke_once(rows, batch):
    import train_task_lora_v4 as trainer
    l.seed_everything(42); torch.cuda.reset_peak_memory_stats()
    parent,meta=ref.load_refinement_checkpoint(PARENT,ROOT,'cuda')
    model,inheritance=trainer.build_model(ROOT,PARENT,'cuda',True)
    if set(l.adapter_state(parent))-set(l.adapter_state(model)): raise ValueError('Inherited parameters missing')
    for name,value in l.adapter_state(parent).items():
        if not torch.equal(value,l.adapter_state(model)[name]): raise ValueError('Inherited tensor changed')
    images,targets=next(iter(l.make_loader(rows['train'][:batch],batch,0,42,training=False)))
    images,targets=images.cuda(),targets.cuda(); parent.eval(); model.eval()
    with torch.inference_mode(),h.autocast_for('cuda'):
        before=parent(images).float(); expanded=model(images).float()
    delta=float((before-expanded).abs().max())
    if not torch.allclose(before,expanded,atol=1e-5,rtol=1e-5): raise ValueError('Expanded LoRA does not preserve the parent function')
    del parent,meta,before,expanded; release()
    model.train(); settings={**SETTINGS,'batch_size':batch,'accumulation':32//batch}
    optimizer=trainer.make_optimizer(model,settings); scaler=h.make_scaler('cuda')
    original=l.adapter_state(model)
    frozen={name:parameter.detach().clone() for name,parameter in model.named_parameters()
            if not parameter.requires_grad and name.endswith('qkv.base.weight')}
    new_A_second_gradient=0.; skipped=0
    for step in range(2):
        trainer.set_learning_rate(optimizer,step,664*6,settings)
        with h.autocast_for('cuda'):
            logits=model(images); loss=nn.functional.cross_entropy(logits,targets,label_smoothing=.02)
        if not torch.isfinite(loss): raise RuntimeError('Smoke has nonfinite loss')
        scaler.scale(loss).backward()
        if step==1:
            new_A_second_gradient=sum(float(p.grad.detach().float().abs().sum()) for name,p in model.named_parameters()
                if name.endswith('lora_A') and name in trainer.new_adapter_names(model) and p.grad is not None)
        norm,did_skip=h.optimizer_step(model,optimizer,scaler); skipped+=int(did_skip)
        if did_skip or norm is None: raise RuntimeError('No successful finite LoRA smoke update')
    after=l.adapter_state(model); new_names=trainer.new_adapter_names(model)
    if new_A_second_gradient<=0 or not any(not torch.equal(original[n],after[n]) for n in new_names if n.endswith('lora_B')):
        raise ValueError('Added LoRA adapters have no real gradient/update')
    if not any(not torch.equal(original[n],after[n]) for n in original if n.startswith('head.')): raise ValueError('Head did not update')
    if any(not torch.equal(value,dict(model.named_parameters())[name]) for name,value in frozen.items()): raise ValueError('Frozen pretrained qkv changed')
    peak=torch.cuda.max_memory_allocated()/2**20
    if peak>4000: raise torch.cuda.OutOfMemoryError('Probe exceeds 4GB allocated budget')
    receipt={'passed':True,'batch_size':batch,'accumulation':32//batch,'successful_optimizer_steps':2,'amp_skipped_steps':skipped,
             'initial_same_batch_logits_max_error':delta,'inherited_parameters_exact':True,
             'new_A_second_step_gradient_positive':True,'new_B_and_head_updated':True,'frozen_qkv_unchanged':True,
             'peak_allocated_mib':peak,'inheritance':inheritance,'smoke_model_saved':False}
    del model,optimizer,scaler,original,after,frozen,images,targets,logits,loss; release(); return receipt

def smoke(rows):
    attempts=[]
    for batch in [8,4,2]:
        try:
            receipt=_smoke_once(rows,batch)
            receipt['attempts']=attempts
            return receipt
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({'batch_size':batch,'error':str(error)})
        # The failed probe frame and its traceback have left scope before GC.
        release()
    raise RuntimeError('A successful LoRA update did not fit the local GPU')

def prepare():
    if PLAN.exists(): return verify_plan()
    import train_task_lora_v4 as trainer
    ROOT.joinpath('logs').mkdir(exist_ok=True); ROOT.joinpath('experiments','lora_task_v4').mkdir(exist_ok=True)
    rows,receipt=h.load_trainval(ROOT)
    h.check_hash(ROOT/'hse_results'/'submission.csv',PRESERVED)
    _,previous_recipe,_=old.verify_locked(ROOT)
    val_rows,targets,three,baseline_receipt=old.baseline_inputs(ROOT)
    pins={**previous_recipe['producer_files'],**baseline_receipt['producer_files']}
    if [int(r['manifest_index']) for r in val_rows]!=[r['manifest_index'] for r in rows['val']]: raise ValueError('Validation order differs from the true best recipe')
    parent=torch.load(PARENT,map_location='cpu',weights_only=True)
    for split in ['train','val']:
        if parent[split+'_indices']!=[r['manifest_index'] for r in rows[split]] or parent[split+'_paths']!=[r['path'] for r in rows[split]]:
            raise ValueError('Parent trained on different membership')
    parent_summary=c.read_json(PARENT.parent/'summary.json')
    if parent_summary['state']!='complete' or parent_summary['best_checkpoint_sha256']!=h.sha256(PARENT): raise ValueError('Parent is not the verified best LoRA')
    for view in ['original','flip']: h.check_hash(PARENT.parent/f'val_{view}.npy',parent_summary[f'val_{view}_sha256'])
    old_lora=(h.probability(np.load(PARENT.parent/'val_original.npy',allow_pickle=False))+h.probability(np.load(PARENT.parent/'val_flip.npy',allow_pickle=False)))/2
    current=h.probability(np.load(ROOT/'hse_results'/'selected_val_probabilities.npy',allow_pickle=False))
    hse_views=[h.probability(np.load(ROOT/'experiments'/'hse_specialist_v1'/f'val_{view}.npy',allow_pickle=False)) for view in ['original','flip']]
    if not np.allclose(current,.5*three+.25*hse_views[0]+.25*hse_views[1],atol=1e-12,rtol=0): raise ValueError('Current fusion reconstruction changed')
    current_metrics=c.metrics(targets,current); parent_metrics=c.metrics(targets,old_lora)
    if current_metrics['correct_count']!=1913 or not np.isclose(parent_metrics['accuracy'],parent_summary['best_validation']['accuracy'],atol=1e-12): raise ValueError('Current/parent validation baseline changed')
    exports=c.read_json(ROOT/'hse_results'/'summary.json')
    for name,digest in exports['file_sha256'].items(): h.check_hash(ROOT/'hse_results'/name,digest)
    cache=ROOT/'fusion_results'/'prediction_cache'/'refined_lora'; meta=c.read_json(cache/'metadata.json')
    legacy=c.read_json(ROOT/'fusion_results'/'summary.json')
    h.check_hash(cache/'metadata.json',legacy['prediction_cache_receipts']['refined_lora']['metadata_sha256'])
    if meta['binding']['classes']!=h.CLASSES or meta['binding']['checkpoint_sha256']!=h.sha256(PARENT): raise ValueError('Existing LoRA competition views use another checkpoint')
    for name,digest in meta['file_sha256'].items(): h.check_hash(cache/name,digest)
    for path in [PARENT,PARENT.parent/'config.json',PARENT.parent/'summary.json',PARENT.parent/'val_original.npy',PARENT.parent/'val_flip.npy',
                 ROOT/'features'/'manifest.csv',ROOT/'features'/'unlabeled.csv',ROOT/'data'/'raw'/'submission.csv',
                 ROOT/'hse_results'/'selected_recipe.json',ROOT/'hse_results'/'summary.json',ROOT/'hse_results'/'submission.csv',ROOT/'hse_results'/'probabilities.npy',ROOT/'hse_results'/'selected_val_probabilities.npy',
                 cache/'metadata.json',cache/'original.npy',cache/'flip.npy',ROOT/'models'/'dinov2_vitb14_pretrain.pth',ROOT/'models'/'dinov2_vitb14_pretrain.json']:
        pins[str(path)]=h.sha256(path)
    for name in ['train_task_lora_v4.py','prepare_task_lora_v4.py','finalize_task_lora_v4.py','run_lora_task_v4.py','lora_task_v4_worker.py','train_lora.py','refine_lora.py','train_hse_specialist.py','hse_model.py','validate_cosine_fusion.py','finalize_hse_specialist.py']:
        pins[str(ROOT/'src'/name)]=h.sha256(ROOT/'src'/name)
    # Pin the vendor implementation too, instead of only storing its fingerprint.
    for path in (ROOT/'vendor'/'dinov2'/'dinov2').rglob('*.py'): pins[str(path)]=h.sha256(path)
    del parent; release()
    probe=smoke(rows)
    settings={**SETTINGS,'batch_size':probe['batch_size'],'accumulation':probe['accumulation']}
    plan={'format':'task_lora_v4_locked','classes':h.CLASSES,'settings':settings,'parent_checkpoint':str(PARENT),
          'parent_checkpoint_sha256':h.sha256(PARENT),'source_pins':pins,'input_receipt':receipt,
          'train_indices':[r['manifest_index'] for r in rows['train']],'val_indices':[r['manifest_index'] for r in rows['val']],
          'dino_repository':l.repository_info(ROOT),'official_dino_sha256':l.verify_official_weights(ROOT),'smoke':probe,
          'baseline_metrics':current_metrics,'parent_lora_tta_metrics':parent_metrics,
          'fusion_weights':{'hse':.5,'lora':.2,'cnn':.2,'mlp':.1},'selection':'strict fused correct count; retain incumbent on ties',
          'internal_test_read':False,'external_answers_used':False,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    c.verify_pins(ROOT,pins); h.write_json(PLAN,plan)
    print(json.dumps({'stage':'prepare','state':'locked','settings':settings,'smoke':probe,'baseline_accuracy':current_metrics['accuracy']},ensure_ascii=False),flush=True)
    return plan

if __name__=='__main__':
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Local CUDA unavailable')
    prepare()
