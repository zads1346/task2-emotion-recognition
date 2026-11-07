"""Fixed 20% LoRA branch replacement; lock validation, then export once."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[name]='4'
os.environ['XFORMERS_DISABLED']='1'
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import train_hse_specialist as h
import train_lora as l
import validate_cosine_fusion as c
import prepare_task_lora_v4 as prep

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'experiments'/'lora_task_v4'
OUT=ROOT/'lora_task_v4_results'

def candidate_files(item):
    return {key:TRAIN/item[key] for key in ['model','original','flip']}

def verify_training(plan):
    import train_task_lora_v4 as trainer
    summary=c.read_json(TRAIN/'summary.json'); config=c.read_json(TRAIN/'config.json')
    if summary['state']!='complete' or summary['epochs_completed']!=6 or summary['classes']!=h.CLASSES:
        raise ValueError('Six LoRA epochs are not complete')
    if summary['plan_sha256']!=h.sha256(prep.PLAN) or config['plan_sha256']!=h.sha256(prep.PLAN): raise ValueError('LoRA training plan binding differs')
    if config['source_pins']!=plan['source_pins'] or config['train_indices']!=plan['train_indices'] or config['val_indices']!=plan['val_indices']:
        raise ValueError('LoRA source/membership binding differs')
    for relative,digest in summary['artifact_sha256'].items(): h.check_hash(TRAIN/relative,digest)
    pointer=c.read_json(TRAIN/'last.json'); path=TRAIN/pointer['checkpoint']; h.check_hash(path,pointer['sha256'])
    final=torch.load(path,map_location='cpu',weights_only=True)
    if pointer['epoch']!=6 or final['epoch']!=6 or final['config']!=config: raise ValueError('Final committed epoch differs')
    if len(final['history'])!=6 or [row['epoch'] for row in final['history']]!=list(range(1,7)):
        raise ValueError('Final checkpoint does not include six complete rounds')
    candidates=c.read_json(TRAIN/'candidates.json')
    expected={(0,'baseline')}|{(epoch,view) for epoch in range(1,7) for view in ['raw','ema']}
    if len(candidates)!=13 or {(item['epoch'],item['weight_source']) for item in candidates}!=expected:
        raise ValueError('Fixed raw/EMA candidate set differs')
    pins={**plan['source_pins'],str(TRAIN/'summary.json'):h.sha256(TRAIN/'summary.json')}
    for relative,digest in summary['artifact_sha256'].items(): pins[str(TRAIN/relative)]=digest
    for item in candidates:
        if item['classes']!=h.CLASSES or item['val_indices']!=plan['val_indices'] or item['plan_sha256']!=h.sha256(prep.PLAN):
            raise ValueError('Candidate classes/order/source changed')
        files=candidate_files(item)
        for key,path in files.items(): h.check_hash(path,item['artifact_sha256'][key]); pins[str(path)]=item['artifact_sha256'][key]
        state=torch.load(files['model'],map_location='cpu',weights_only=True)
        if state['config']!=config or state['classes']!=h.CLASSES or state['epoch']!=item['epoch'] or state['weight_source']!=item['weight_source']:
            raise ValueError('Candidate checkpoint is bound to another epoch/run')
        if state['source_pins']!=plan['source_pins'] or state['parent_sha']!=plan['parent_checkpoint_sha256'] or set(state['adapter_state'])!=trainer.ALL_NAMES:
            raise ValueError('Candidate is not the complete inherited+expanded LoRA')
        if any(not torch.isfinite(value).all() for value in state['adapter_state'].values()): raise ValueError('Nonfinite adapter state')
        if item['epoch']==6:
            expected_state=final['adapter_state'] if item['weight_source']=='raw' else final['ema_state']['shadow']
            if set(state['adapter_state'])!=set(expected_state) or any(not torch.equal(value,expected_state[name]) for name,value in state['adapter_state'].items()):
                raise ValueError('Last round candidate differs from the committed raw/EMA state')
        del state
    return candidates,pins,summary

def select():
    plan=prep.verify_plan(); OUT.mkdir(exist_ok=True)
    path=OUT/'selected_recipe.json'
    if path.exists():
        recipe=c.read_json(path); c.verify_pins(ROOT,recipe['producer_files'])
        h.check_hash(OUT/'selected_val_probabilities.npy',recipe['selected_val_sha256']); return recipe
    rows,receipt=h.load_trainval(ROOT)
    if receipt!=plan['input_receipt']: raise ValueError('Image bytes/membership changed after training')
    targets=np.array([r['target'] for r in rows['val']])
    incumbent=h.probability(np.load(ROOT/'hse_results'/'selected_val_probabilities.npy',allow_pickle=False)); incumbent_score=c.metrics(targets,incumbent)
    if incumbent_score['correct_count']!=1913: raise ValueError('Incumbent score changed')
    old_views=[h.probability(np.load(prep.PARENT.parent/f'val_{view}.npy',allow_pickle=False)) for view in ['original','flip']]
    items,pins,training=verify_training(plan); best_p=incumbent; best=incumbent_score; selected=None
    entries=[{'name':'previous_best',**incumbent_score}]
    for item in sorted(items,key=lambda item:(item['epoch'],{'baseline':0,'raw':1,'ema':2}[item['weight_source']])):
        files=candidate_files(item); a=h.probability(np.load(files['original'],allow_pickle=False)); b=h.probability(np.load(files['flip'],allow_pickle=False))
        single=c.metrics(targets,(a+b)/2)
        if not np.isclose(single['accuracy'],item['single_tta_metrics']['accuracy'],atol=1e-12): raise ValueError('Candidate standalone score differs from saved views')
        if item['epoch']==0:
            if not np.allclose(a,old_views[0],atol=1e-5,rtol=1e-5) or not np.allclose(b,old_views[1],atol=1e-5,rtol=1e-5): raise ValueError('Expanded epoch-zero probabilities differ from inherited parent')
            p=incumbent.copy()
        else:
            raw=incumbent-.1*old_views[0]-.1*old_views[1]+.1*a+.1*b
            if raw.min()<-1e-10: raise ValueError('Negative branch-replacement probability')
            p=h.probability(np.maximum(raw,0))
        score=c.metrics(targets,p)
        entries.append({'name':item['name'],'epoch':item['epoch'],'weight_source':item['weight_source'],'single_accuracy':single['accuracy'],'single_macro_f1':single['macro_f1'],**score})
        if score['correct_count']>best['correct_count']:
            selected=item; best=score; best_p=p
    h.save_array(OUT/'selected_val_probabilities.npy',best_p)
    pins[str(prep.PLAN)]=h.sha256(prep.PLAN)
    recipe={'state':'locked','classes':h.CLASSES,'selected':selected,'improved':selected is not None,
            'baseline_metrics':incumbent_score,'selected_metrics':best,'validation_candidates':entries,'producer_files':pins,
            'fusion_weights':plan['fusion_weights'],'candidate_count':13,'plan_sha256':h.sha256(prep.PLAN),
            'selected_val_sha256':h.sha256(OUT/'selected_val_probabilities.npy'),'val_indices':plan['val_indices'],
            'val_rows_sha256':receipt['val_rows_sha256'],'finalizer_sha256':h.sha256(__file__),
            'internal_test_read':False,'external_answers_used':False,
            'validation_caveat':'Repeated development validation; not Kaggle or an independent accuracy estimate. No new unexpanded continuation control.'}
    h.write_json(path,recipe)
    fields=['name','epoch','weight_source','single_accuracy','single_macro_f1','accuracy','macro_f1','log_loss','correct_count']
    with (OUT/'candidate_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(entries)
    print(json.dumps({'stage':'selection','improved':recipe['improved'],'selected':selected['name'] if selected else 'previous_best','accuracy':best['accuracy'],'correct_count':best['correct_count']}),flush=True)
    return recipe

def export():
    import train_task_lora_v4 as trainer
    recipe=select(); path=OUT/'selected_recipe.json'; recipe_sha=h.sha256(path)
    if (OUT/'summary.json').exists():
        summary=c.read_json(OUT/'summary.json')
        if summary['recipe_sha256']!=recipe_sha: raise ValueError('Completed export uses another recipe')
        for name,digest in summary['file_sha256'].items(): h.check_hash(OUT/name,digest)
        return summary
    c.verify_pins(ROOT,recipe['producer_files']); rows=c.unlabeled_rows(ROOT)
    image_receipt=c.image_receipt(ROOT,rows,verify_inventory=False)
    legacy_hse=c.read_json(ROOT/'hse_results'/'summary.json')
    legacy_lora=c.read_json(ROOT/'fusion_results'/'prediction_cache'/'refined_lora'/'metadata.json')
    if (image_receipt['image_content_sha256']!=legacy_hse['unlabeled_image_content_sha256']
        or image_receipt['image_content_sha256']!=legacy_lora['binding']['image_content_sha256']
        or c.signature(rows)!=legacy_lora['binding']['row_order_sha256']):
        raise ValueError('Current competition images/IDs differ from the reused old model probabilities')
    if recipe['improved']:
        files=candidate_files(recipe['selected']); model,_=trainer.load_checkpoint(ROOT,files['model'],'cuda')
        loader=DataLoader(l.FaceImages([{**r,'target':0} for r in rows],training=False),batch_size=16,num_workers=0,shuffle=False,pin_memory=True)
        outputs=[[],[]]; started=last=time.monotonic()
        with torch.inference_mode():
            for index,(x,_) in enumerate(loader):
                x=x.cuda()
                with h.autocast_for('cuda'): a=model(x); b=model(x.flip(-1))
                for view,logits in enumerate([a,b]): outputs[view].append(logits.float().softmax(1).cpu().numpy())
                if time.monotonic()-last>=30 or index+1==len(loader):
                    h.check_hash(path,recipe_sha); h.check_hash(files['model'],recipe['selected']['artifact_sha256']['model'])
                    print(json.dumps({'stage':'export','seen':min((index+1)*16,7065),'total':7065,'elapsed_seconds':time.monotonic()-started}),flush=True); last=time.monotonic()
        a,b=[h.probability(np.concatenate(v)) for v in outputs]
        for view,array in [('original',a),('flip',b)]: h.save_array(OUT/f'lora_competition_{view}.npy',array)
        baseline=h.probability(np.load(ROOT/'hse_results'/'probabilities.npy',allow_pickle=False))
        cache=ROOT/'fusion_results'/'prediction_cache'/'refined_lora'
        meta=c.read_json(cache/'metadata.json')
        if meta['binding']['row_order_sha256']!=c.signature(rows) or meta['binding']['classes']!=h.CLASSES: raise ValueError('Cached LoRA competition ID order differs')
        old_views=[h.probability(np.load(cache/f'{view}.npy',allow_pickle=False)) for view in ['original','flip']]
        raw=baseline-.1*old_views[0]-.1*old_views[1]+.1*a+.1*b
        if raw.min()<-1e-10: raise ValueError('Invalid competition probability replacement')
        probability=h.probability(np.maximum(raw,0)); del model; torch.cuda.empty_cache()
    else:
        probability=h.probability(np.load(ROOT/'hse_results'/'probabilities.npy',allow_pickle=False))
    h.save_array(OUT/'probabilities.npy',probability)
    temporary=OUT/'submission.csv.tmp'
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['ID','Emotion']); writer.writerows((row['ID'],int(label)) for row,label in zip(rows,probability.argmax(1)))
    temporary.replace(OUT/'submission.csv'); c.submission_check(OUT/'submission.csv',rows,probability)
    with (OUT/'submission.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['ID','Emotion'] or len(list(reader))!=7065: raise ValueError('Submission count/columns differ')
    h.check_hash(path,recipe_sha); c.verify_pins(ROOT,recipe['producer_files']); prep.verify_plan()
    if c.image_receipt(ROOT,rows,verify_inventory=False)['image_content_sha256']!=image_receipt['image_content_sha256']: raise ValueError('Competition images changed during export')
    names=['submission.csv','probabilities.npy','selected_val_probabilities.npy','selected_recipe.json','candidate_metrics.csv']+[p.name for p in OUT.glob('lora_competition_*.npy')]
    summary={'state':'complete','classes':h.CLASSES,'submission_rows':7065,'improved':recipe['improved'],
             'selected':recipe['selected']['name'] if recipe['selected'] else 'previous_best',
             'baseline_metrics':recipe['baseline_metrics'],'selected_metrics':recipe['selected_metrics'],'recipe_sha256':recipe_sha,
             'official_label_order_verified':True,'ID_template_order_verified':True,'probability_argmax_verified':True,
             'competition_row_order_sha256':c.signature(rows),'competition_image_content_sha256':image_receipt['image_content_sha256'],
             'old_submission_sha256':h.sha256(ROOT/'hse_results'/'submission.csv'),'internal_test_read':False,'external_answers_used':False,
             'validation_caveat':recipe['validation_caveat'],'file_sha256':{name:h.sha256(OUT/name) for name in names}}
    h.write_json(OUT/'summary.json',summary)
    print(json.dumps({'stage':'complete','selected':summary['selected'],'accuracy':summary['selected_metrics']['accuracy'],'improved':summary['improved'],'submission':str(OUT/'submission.csv')}),flush=True)
    return summary

if __name__=='__main__':
    torch.set_num_threads(4); export()
