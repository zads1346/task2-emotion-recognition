"""Train-only investigation of cached versus live AMP DINO features."""
import csv,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import improvement_v3 as v
import train_lora as l
torch.set_num_threads(4); l.seed_everything(42)
rows=[]
with (ROOT/'features'/'manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
    for index,row in enumerate(csv.DictReader(stream)):
        if row['split']!='train': continue
        rows.append({'path':row['path'],'target':l.CLASSES.index(row['label']),'manifest_index':index})
        if len(rows)==32: break
cache=np.load(ROOT/'features'/'labeled_original.npy',mmap_mode='r',allow_pickle=False)
expected=np.asarray(cache[[r['manifest_index'] for r in rows]],dtype=np.float32)
model=v.lora_model(8).eval()
inputs=torch.stack([l.FaceImages(rows)[i][0] for i in range(32)]).cuda()
def check(name,x):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        z=model.backbone.forward_features(x)
        actual=torch.cat((z['x_norm_clstoken'],z['x_norm_patchtokens'].mean(1)),1).float().cpu().numpy()
    e=expected[:len(x)]; d=actual-e
    print(json.dumps({'name':name,'shape':actual.shape,'max_error':float(np.abs(d).max()),'mean_error':float(np.abs(d).mean()),'rmse':float(np.sqrt((d*d).mean())),
        'cosine_min':float(((actual*e).sum(1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(e,axis=1))).min()),'first_cached':e[0,:8].tolist(),'first_live':actual[0,:8].tolist()}),flush=True)
check('wrapped_batch4',inputs[:4]); check('wrapped_batch32',inputs)
for i in model.adapted_blocks:
    model.backbone.blocks[i].attn.qkv=model.backbone.blocks[i].attn.qkv.base
check('official_batch4',inputs[:4]); check('official_batch32',inputs)
