"""Real GPU smoke on32 authorized training images; no checkpoint is retained."""
import csv
import gc
import json
from pathlib import Path
import time
import torch
import train_hse_sam_v7 as v
import train_hse_specialist as h

ROOT = Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the real training smoke')
    parent = ROOT/'experiments/hse_specialist_v1/best.pt'
    rows=[]
    with (ROOT/'features/manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            if row['split']!='train' or index in (7727,10448): continue
            rows.append({'manifest_index':index,'path':row['path'],'label':row['label'],'target':v.CLASSES.index(row['label'])})
            if len(rows)==32: break
    assert len(rows)==32
    h.seed_everything(42)
    dataset=h.ExpressionImages(rows,True)
    examples=[dataset[i] for i in range(32)]
    x=torch.stack([e[0] for e in examples]);y=torch.tensor([e[1] for e in examples])
    results={}
    for mode,rho in [('adamw_control',0.),('sam',.05)]:
        h.seed_everything(42)
        model=h.model_api().load_trained_model(ROOT,parent,'cuda').train().requires_grad_(True)
        v.freeze_batchnorm(model)
        head=h.classifier_parameters(model);ids={id(p) for p in head}
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in ids],'lr':5e-6},
                                    {'params':head,'lr':5e-5}],weight_decay=.01)
        ema=v.FullModelEMA(model,.995);scaler=v.ManualGradScaler(True,1024.,2000)
        before={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
        frozen={k:z.detach().clone() for k,z in model.named_buffers()}
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        steps=[v.group_step(model,optimizer,scaler,ema,x,y,device='cuda',batch_size=16,rho=rho) for _ in range(2)]
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        assert all(s['success'] for s in steps) and ema.updates==2
        changed=sum(not torch.equal(before[k],p.detach().cpu()) for k,p in model.named_parameters())
        assert changed>0
        assert all(torch.equal(z,frozen[k]) for k,z in model.named_buffers())
        with ema.apply(model),torch.inference_mode():
            output=model(x[:2].cuda())
            assert output.shape==(2,6) and torch.isfinite(output).all()
        results[mode]={'successful_steps':2,'ema_updates':ema.updates,'changed_parameter_tensors':changed,
                       'seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                       'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'steps':steps,
                       'batchnorm_buffers_bitwise_frozen':True}
        del model,optimizer,ema,scaler,output,before,frozen
        gc.collect();torch.cuda.empty_cache()
    report={'state':'passed','gpu':torch.cuda.get_device_name(0),'microbatch':16,'effective_batch':32,
            'parent_sha256':h.sha256(parent),'trainer_sha256':h.sha256(Path(v.__file__)),
            'smoke_source_sha256':h.sha256(__file__),'train_manifest_indices':[r['manifest_index'] for r in rows],
            'training_image_sha256':[h.sha256(r['path']) for r in rows],'arms':results,
            'internal_test_read':False,'checkpoint_retained':False}
    h.write_json(ROOT/'logs/hse_sam_v7_gpu_smoke.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
