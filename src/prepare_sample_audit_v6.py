"""Deterministic train-only visual sample audit; no relabeling/filtering/weighting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
QUOTAS = {'very_uncertain': 16, 'uncertain': 8, 'normal': 8}
SEED = 20261015

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(path)

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))

def prepare(root=ROOT):
    root = Path(root).resolve(); out = root/'experiments/lora_flip_consistency_v6/audit'
    out.mkdir(parents=True, exist_ok=True)
    group_path = root/'experiments/lora_zero_weight_v5/weights/train_weights.csv'
    plan = read(root/'logs/zero_weight_v5_plan.json')
    if sha(group_path) != plan['source_pins'][str(group_path)]: raise ValueError('Frozen train OOF groups changed')
    all_rows = csv_rows(group_path)
    manifest = csv_rows(root/'features/manifest.csv')
    if len(all_rows) != 21248 or plan['classes'] != CLASSES: raise ValueError('Train count/classes changed')
    for row in all_rows:
        entry = manifest[int(row['manifest_index'])]
        if entry['split'] != 'train' or entry['path'] != row['path'] or entry['label'] != row['label']:
            raise ValueError('A visual-audit row is not the original train member')
    selected = []
    rng = random.Random(SEED)
    for group, per_class in QUOTAS.items():
        for label in CLASSES:
            pool = sorted((r for r in all_rows if r['difficulty'] == group and r['label'] == label), key=lambda r: int(r['manifest_index']))
            selected += sorted(rng.sample(pool, per_class), key=lambda r: int(r['manifest_index']))
    inventory_path = root/'data/original_splits/image_inventory.csv'
    def image_key(value):
        parts = str(value).replace('\\', '/').split('/')
        return '/'.join(parts[parts.index('fer_data'):]).casefold()
    wanted = {image_key(row['path']) for row in selected}
    inventory = {}
    for entry in csv_rows(inventory_path):
        key = image_key(entry['path'])
        if key in wanted:
            if key in inventory: raise ValueError('Ambiguous original inventory key')
            inventory[key] = entry['sha256']
    records = []
    for row in selected:
        path = Path(row['path']); digest = sha(path)
        if inventory[image_key(path)] != digest: raise ValueError('Original training image content differs')
        with Image.open(path) as source:
            gray = np.asarray(source.convert('L'), dtype=np.float32)
            lap = gray[1:-1,2:]+gray[1:-1,:-2]+gray[2:,1:-1]+gray[:-2,1:-1]-4*gray[1:-1,1:-1]
            records.append({'manifest_index': int(row['manifest_index']), 'path': str(path), 'label': row['label'],
                            'target': int(row['target']), 'difficulty': row['difficulty'], 'sha256': digest,
                            'width': source.width, 'height': source.height, 'mode': source.mode,
                            'laplacian_variance_proxy': float(lap.var()), 'std_intensity': float(gray.std())})
    fields = list(records[0])
    with (out/'sample_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 12) if font_path.exists() else ImageFont.load_default()
    pages = []
    for number, start in enumerate(range(0, len(records), 32), 1):
        batch = records[start:start+32]
        canvas = Image.new('RGB', (8*144, 4*168+32), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((6, 8), f'TRAIN ONLY | page{number} | seed{SEED} | nearest-neighbor display; source unchanged', fill='black', font=font)
        for index, record in enumerate(batch):
            x = (index%8)*144; y = (index//8)*168+32
            with Image.open(record['path']) as source:
                face = source.convert('RGB'); face.thumbnail((132,132), Image.Resampling.NEAREST)
                # Enlarge thumbnails for visual inspection without synthesizing detail.
                scale = min(132/face.width, 132/face.height)
                face = face.resize((round(face.width*scale), round(face.height*scale)), Image.Resampling.NEAREST)
            canvas.paste(face, (x+(144-face.width)//2,y))
            draw.text((x+3, y+134), f"{record['manifest_index']} {record['label']}", font=font, fill='black')
            draw.text((x+3, y+149), record['difficulty'], font=font, fill='#9a2222' if record['difficulty']=='very_uncertain' else '#333333')
        filename = f'contact_{number:02d}.png'; canvas.save(out/filename)
        pages.append({'page': number, 'file': filename, 'manifest_indices': [r['manifest_index'] for r in batch]})
    sampling = {'state': 'prepared', 'classes': CLASSES, 'seed': SEED, 'sampling': 'stratified fixed quotas without replacement',
                'per_class_quotas': QUOTAS, 'sample_rows': len(records), 'pages': pages,
                'sample_manifest_sha256': sha(out/'sample_manifest.csv'),
                'source_pins': {str(group_path): sha(group_path), str(root/'logs/zero_weight_v5_plan.json'): sha(root/'logs/zero_weight_v5_plan.json'),
                                str(root/'features/manifest.csv'): sha(root/'features/manifest.csv'), str(inventory_path): sha(inventory_path), str(Path(__file__).resolve()): sha(__file__)},
                'train_indices': plan['train_indices'], 'val_indices': plan['val_indices'],
                'internal_test_images_read': 0, 'external_answers_used': False}
    write(out/'sampling.json', sampling)
    print(json.dumps({'state': 'prepared', 'sample_rows': len(records), 'pages': pages}, ensure_ascii=False), flush=True)
    return sampling

def finalize(root=ROOT):
    root = Path(root).resolve(); out = root/'experiments/lora_flip_consistency_v6/audit'
    sampling = read(out/'sampling.json'); review = read(out/'visual_review.json')
    records = csv_rows(out/'sample_manifest.csv')
    if sha(out/'sample_manifest.csv') != sampling['sample_manifest_sha256']: raise ValueError('Audit membership changed')
    expected = [int(row['manifest_index']) for row in records]
    if sorted(review['reviewed_manifest_indices']) != sorted(expected) or len(set(expected)) != 192:
        raise ValueError('All sampled images require explicit visual review coverage')
    if review['labels_changed'] or review['images_changed'] or review['weights_changed']:
        raise ValueError('This matched experiment only records visual audit; it does not change data')
    excluded = review['excluded_train_indices']
    if excluded != [7727,10448] or not set(excluded).issubset(expected):
        raise ValueError('Only the two visually confirmed invalid training pictures may be excluded')
    group_rows = csv_rows(root/'experiments/lora_zero_weight_v5/weights/train_weights.csv')
    members = [{'manifest_index':int(r['manifest_index']), 'path':r['path'], 'label':r['label'], 'target':int(r['target'])}
               for r in group_rows if int(r['manifest_index']) not in excluded]
    pins = dict(sampling['source_pins'])
    for path, digest in pins.items():
        if sha(path) != digest: raise ValueError('Frozen visual-audit source changed')
    for row in records:
        if sha(row['path']) != row['sha256']: raise ValueError('Sample image content changed')
        pins[row['path']] = row['sha256']
    files = ['sampling.json','sample_manifest.csv','visual_review.json']+[page['file'] for page in sampling['pages']]
    summary = {'state':'complete', 'format':'train_visual_audit_v6', 'classes':CLASSES,
               'train_rows':21248, 'effective_train_rows':21246, 'sample_rows':192, 'counts':{'very_uncertain':96,'uncertain':48,'normal':48},
               'original_train_indices':sampling['train_indices'], 'excluded_train_indices':excluded,
               'train_indices':[r['manifest_index'] for r in members], 'val_indices':sampling['val_indices'],
               'train_membership_sha256': read(root/'experiments/lora_zero_weight_v5/weights/summary.json')['binding']['train_membership_sha256'],
               'filtered_membership_sha256':hashlib.sha256(json.dumps(members,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest(),
               'labels_changed':False, 'images_changed':False, 'weights_changed':False, 'training_weights':'uniform1',
               'training_membership_changed':True,
               'source_pins':pins, 'file_sha256':{name:sha(out/name) for name in files},
               'visual_review':review, 'internal_test_read':False, 'external_answers_used':False,
               'sampling_caveat':'Uncertainty-stratified sample; not an estimate of total dataset label error rate. Single-image expressions are subjective; doubtful examples are not automatically relabeled.'}
    write(out/'summary.json',summary)
    print(json.dumps({'state':'complete','sample_rows':192,'visual_review_summary':review['summary']},ensure_ascii=False),flush=True)
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['prepare','finalize'],default='prepare'); args=parser.parse_args()
    (prepare if args.stage=='prepare' else finalize)(args.project_root)
