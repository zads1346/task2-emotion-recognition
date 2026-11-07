"""Lock the audited, matched CE versus CE+flip-consistency experiment."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
os.environ['XFORMERS_DISABLED']='1'
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
import train_hse_specialist as h
import train_lora as l
import train_task_lora_v4 as v4
import validate_cosine_fusion as c

ROOT=Path(__file__).resolve().parents[1]
BASELINE_SHA='cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd'
PRESERVED={'lora_task_v4_results/submission.csv':BASELINE_SHA,
 'lora_zero_weight_v5_results/submission.csv':BASELINE_SHA,
 'hse_results/submission.csv':'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7',
 'fusion_results/submission.csv':'2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad'}
MODE_SETTINGS={'ce_control':{'consistency_lambda':0.},'flip_consistency':{'consistency_lambda':.1}}
SETTINGS={'epochs':4,'seed':42,'augmentation_epoch_offset':60,'label_smoothing':0.,
 'weight_decay':.001,'lr_new':1e-5,'lr_inherited':2.5e-6,'lr_head':1e-5,
 'warmup_successful_steps':50,'min_lr_ratio':.1,'ema_decay':.995,
 'batch_size':4,'accumulation':8,'effective_batch_size':32,'workers':0,'cpu_threads':4,
 'image_size':224,'amp_initial_scale':1024.,'gradient_clip':1.,'validation_batch_size':4,
 'rank':8,'alpha':16.,'lora_dropout':.05,'last_blocks':8,'sample_weights':'uniform1',
 'physical_microbatch_size':8,'views_per_image':2,'temperature':1.,
 'ce_definition':'mean of standard CE on original and horizontally flipped views',
 'consistency_definition':'symmetric KL in FP32, both branches differentiable, temperature1',
 'mixup':False,'cutmix':False,'sampling':'uniform full-member shuffle; identical paired augmentation and order'}
SOURCES=['prepare_sample_audit_v6.py','prepare_flip_consistency_v6.py','train_flip_consistency_v6.py',
 'finalize_flip_consistency_v6.py','run_flip_consistency_v6.py','flip_consistency_v6_worker.py']

def verify_plan(root=ROOT):
    plan=c.read_json(root/'logs/flip_consistency_v6_plan.json')
    if plan['classes']!=h.CLASSES or plan['settings']!=SETTINGS or plan['mode_settings']!=MODE_SETTINGS:
        raise ValueError('Locked V6 classes/settings differ')
    c.verify_pins(root,plan['source_pins'])
    for name,digest in PRESERVED.items(): h.check_hash(root/name,digest)
    h.check_hash(root/'experiments/lora_flip_consistency_v6/audit/summary.json',plan['audit_summary_sha256'])
    return plan

def cpu_tests():
    import train_flip_consistency_v6 as t
    torch.manual_seed(41)
    results=[]
    for n in [30,32]:
        x=torch.randn(n,6,requires_grad=True); y=torch.randn(n,6,requires_grad=True); target=torch.arange(n)%6
        losses=t.paired_loss(x,y,target,.1)
        logp=F.log_softmax(x.float(),dim=1); logq=F.log_softmax(y.float(),dim=1)
        reference=.5*(F.kl_div(logq,logp,reduction='batchmean',log_target=True)+F.kl_div(logp,logq,reduction='batchmean',log_target=True))
        ce=.5*(F.cross_entropy(x,target)+F.cross_entropy(y,target))
        if not torch.allclose(losses['consistency'],reference,atol=1e-6,rtol=1e-6) or not torch.allclose(losses['total'],ce+.1*reference,atol=1e-6,rtol=1e-6):
            raise ValueError('Pair loss differs from independent CE/symmetric-KL reference')
        zero=t.paired_loss(x,y,target,0.)
        if not torch.equal(zero['total'],zero['ce']): raise ValueError('CE control contains another objective')
        swap=t.paired_loss(y,x,target,.1)
        if not torch.allclose(swap['consistency'],losses['consistency'],atol=1e-6): raise ValueError('KL is not symmetric')
        if abs(float(t.paired_loss(x,x,target,.1)['consistency'].detach()))>1e-6: raise ValueError('Identical views do not have zero KL')
        grads=torch.autograd.grad(losses['total'],(x,y))
        copies=[z.detach().clone().requires_grad_() for z in (x,y)]
        accumulated=sum(t.paired_loss(copies[0][i:i+4],copies[1][i:i+4],target[i:i+4],.1)['total']*len(target[i:i+4])/n for i in range(0,n,4))
        split_grads=torch.autograd.grad(accumulated,copies)
        if not all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads): raise ValueError('A branch has no finite gradient')
        if not all(torch.allclose(a,b,atol=1e-7,rtol=1e-5) for a,b in zip(grads,split_grads)): raise ValueError('Partial group accumulation gradient differs')
        results.append({'logical_group_size':n,'microbatch':4,'physical_max_batch':8,'paired_gradient_test_passed':True})
    return {'passed':True,'independent_symmetric_KL_and_CE':True,'both_branches_have_gradients':True,
            'lambda0_is_standard_CE':True,'identical_views_zero_KL':True,'groups':results}

def release():
    gc.collect(); torch.cuda.empty_cache()

def gpu_smoke(root,rows):
    import train_flip_consistency_v6 as t
    parent=root/'experiments/lora_task_v4/candidates/epoch_05_ema.pt'
    l.seed_everything(42); torch.cuda.reset_peak_memory_stats()
    model,_=v4.load_checkpoint(root,parent,'cuda'); v4._attach_checkpointing(model,True)
    original=l.adapter_state(model)
    saved=torch.load(parent,map_location='cpu',weights_only=True)['adapter_state']
    if set(original)!=v4.ALL_NAMES or any(not torch.equal(original[key],saved[key]) for key in original):
        raise ValueError('The complete trained last8 parent was not inherited exactly')
    if any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()): raise ValueError('Unexpected cross-image BatchNorm')
    images,targets=next(iter(l.make_loader(rows['train'][:32],32,0,42,training=False)))
    frozen={name:p._version for name,p in model.named_parameters() if not p.requires_grad}
    modes={}
    for mode,params in MODE_SETTINGS.items():
        v4._strict_state(model,original,v4.ALL_NAMES); l.seed_everything(42); model.train()
        optimizer=v4.make_optimizer(model,SETTINGS); scaler=h.make_scaler('cuda'); ema=v4.AdapterEMA(model,.995)
        diagnostics=[]
        for step in range(2):
            v4.set_learning_rate(optimizer,step,664*4,SETTINGS)
            result=t.train_group(model,optimizer,scaler,ema,images,targets,device='cuda',batch_size=4,
                                 consistency_lambda=params['consistency_lambda'])
            if not result['success'] or result['seen']!=32 or result['microbatches']!=8:
                raise ValueError('Real paired-view GPU probe did not update correctly')
            diagnostics.append({key:result[key] for key in ('ce_sum','consistency_sum','loss_sum')})
        after=l.adapter_state(model)
        if not all(any(not torch.equal(original[name],after[name]) for name in names)
                   for names in [v4.NEW_NAMES,v4.INHERITED_NAMES-{'head.weight','head.bias'},{'head.weight','head.bias'}]):
            raise ValueError('A LoRA/head parameter group failed to update')
        modes[mode]={'successful_optimizer_steps':2,'amp_skipped_steps':0,'diagnostics':diagnostics}
        del optimizer,scaler,ema,after
    if any(p._version!=frozen[name] or p.grad is not None for name,p in model.named_parameters() if not p.requires_grad):
        raise ValueError('Frozen pretrained base was modified')
    peak=torch.cuda.max_memory_allocated()/2**20
    if peak>4000: raise RuntimeError('Paired-view training exceeds allocated-memory budget')
    result={'passed':True,'modes':modes,'all_18_parent_tensors_exact':True,'frozen_base_unchanged':True,
            'logical_microbatch':4,'physical_microbatch':8,'effective_logical_images':32,'peak_allocated_mib':peak,'probe_weights_saved':False}
    del model,original,saved,images,targets; release(); return result

def prepare(root=ROOT):
    root=Path(root).resolve(); plan_path=root/'logs/flip_consistency_v6_plan.json'
    if plan_path.exists(): return verify_plan(root)
    audit_path=root/'experiments/lora_flip_consistency_v6/audit/summary.json'; audit=c.read_json(audit_path)
    if audit['state']!='complete' or audit['sample_rows']!=192 or audit['excluded_train_indices']!=[7727,10448]: raise ValueError('Visual audit is incomplete')
    if audit['labels_changed'] or audit['images_changed'] or audit['weights_changed'] or not audit['training_membership_changed']: raise ValueError('Unexpected audit policy')
    raw,input_receipt=h.load_trainval(root)
    original_indices=[r['manifest_index'] for r in raw['train']]
    if original_indices!=audit['original_train_indices'] or input_receipt['train_rows_sha256']!=audit['train_membership_sha256']: raise ValueError('Original audit membership differs')
    rows={'train':[r for r in raw['train'] if r['manifest_index'] not in audit['excluded_train_indices']],'val':raw['val']}
    filtered=h.signature(rows['train'])
    if filtered!=audit['filtered_membership_sha256'] or len(rows['train'])!=21246: raise ValueError('Shared filtered membership differs')
    if [r['manifest_index'] for r in rows['train']]!=audit['train_indices'] or [r['manifest_index'] for r in rows['val']]!=audit['val_indices']: raise ValueError('Audit row order differs')
    parent=root/'experiments/lora_task_v4/candidates/epoch_05_ema.pt'; base=root/'lora_task_v4_results'
    for name,digest in PRESERVED.items(): h.check_hash(root/name,digest)
    previous=c.read_json(base/'summary.json'); recipe=c.read_json(base/'selected_recipe.json')
    if previous['state']!='complete' or previous['selected']!='epoch_05_ema': raise ValueError('Current best recipe changed')
    baseline_files={}
    for name,digest in previous['file_sha256'].items():
        h.check_hash(base/name,digest); baseline_files[str((base/name).relative_to(root))]=digest
    baseline_files['lora_task_v4_results/summary.json']=h.sha256(base/'summary.json')
    val=c.val_rows(root); targets=np.asarray([int(r['target']) for r in val])
    if [int(r['manifest_index']) for r in val]!=audit['val_indices']: raise ValueError('Validation order differs')
    current=c.check_probability(np.load(base/'selected_val_probabilities.npy',allow_pickle=False),2656)
    baseline=c.metrics(targets,current)
    if baseline!=previous['selected_metrics'] or baseline['correct_count']!=1915: raise ValueError('Current validation baseline differs')
    for path in [parent,parent.with_suffix('.json'),*[parent.parent/f'epoch_05_ema_{v}.npy' for v in ('original','flip')]]:
        baseline_files[str(path.relative_to(root))]=h.sha256(path)
    if c.read_json(parent.with_suffix('.json'))!=recipe['selected']: raise ValueError('Parent candidate differs')
    if h.sha256(parent)!=recipe['selected']['artifact_sha256']['model']: raise ValueError('Parent model differs')
    parent_config=c.read_json(root/'experiments/lora_task_v4/config.json')
    if parent_config['train_indices']!=original_indices or parent_config['val_indices']!=audit['val_indices'] or parent_config['input_receipt']!=input_receipt:
        raise ValueError('Parent original training/validation inputs differ')
    pins={**recipe['producer_files'],**parent_config['source_pins'],**audit['source_pins']}
    for name,digest in baseline_files.items(): pins[str(root/name)]=digest
    for name in SOURCES: pins[str(root/'src'/name)]=h.sha256(root/'src'/name)
    for name in ['start_flip_consistency_v6.ps1','run_flip_consistency_v6_worker.ps1']: pins[str(root/name)]=h.sha256(root/name)
    pins[str(audit_path)]=h.sha256(audit_path)
    for name,digest in audit['file_sha256'].items(): pins[str(audit_path.parent/name)]=digest
    for path in [root/'logs/lora_task_v4_plan.json',root/'experiments/lora_task_v4/config.json',root/'experiments/lora_task_v4/summary.json',root/'lora_zero_weight_v5_results/summary.json']:
        pins[str(path)]=h.sha256(path)
    c.verify_pins(root,pins); tests=cpu_tests(); smoke=gpu_smoke(root,rows)
    plan={'format':'flip_consistency_lora_v6_locked','classes':h.CLASSES,'settings':SETTINGS,'mode_settings':MODE_SETTINGS,
          'source_pins':pins,'input_receipt':input_receipt,'original_train_indices':original_indices,
          'excluded_train_indices':audit['excluded_train_indices'],'train_indices':audit['train_indices'],'val_indices':audit['val_indices'],
          'filtered_membership_sha256':filtered,'audit_summary_sha256':h.sha256(audit_path),
          'parent_checkpoint':str(parent),'parent_checkpoint_sha256':h.sha256(parent),'parent_candidate_source':'epoch_05_ema',
          'baseline_files':baseline_files,'baseline_submission_sha256':BASELINE_SHA,'baseline_correct_count':1915,'baseline_metrics':baseline,
          'dino_repository':l.repository_info(root),'official_dino_sha256':l.verify_official_weights(root),
          'competition_image_content_sha256':previous['competition_image_content_sha256'],'competition_row_order_sha256':previous['competition_row_order_sha256'],
          'fusion_weights':{'hse':.5,'lora':.2,'cnn':.2,'mlp':.1},'selection_order':['incumbent','ce_control','flip_consistency'],
          'selection':'strict fused correct_count increase; all ties retain earlier candidate; each arm epoch0 then per-epoch raw/EMA',
          'cpu_tests':tests,'gpu_smoke':smoke,'internal_test_read':False,'external_answers_used':False,
          'comparison_caveat':'Paired control isolates the added consistency term. Versus the older incumbent, both new arms also use two-view CE, zero smoothing, fresh continuation and two invalid-image exclusions. Existing validation is reused.',
          'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    c.verify_pins(root,pins); h.write_json(plan_path,plan)
    print(json.dumps({'stage':'prepare','state':'locked','train_rows':21246,'excluded_train_indices':audit['excluded_train_indices'],'settings':SETTINGS,'gpu_smoke':smoke,'cpu_tests':tests},ensure_ascii=False),flush=True)
    return plan

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project-root',type=Path,default=ROOT); args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Local CUDA unavailable')
    prepare(args.project_root)
