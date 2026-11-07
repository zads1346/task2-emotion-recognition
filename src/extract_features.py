"""Frozen official DINOv2 Base features, resumable by completed batches."""
from __future__ import annotations
import os
os.environ['XFORMERS_DISABLED'] = '1'
import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

class FaceImages(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['path']) as im:
            return self.transform(im.convert('RGB'))

def encode(model, x):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        output = model.forward_features(x)
        z = torch.cat([output['x_norm_clstoken'], output['x_norm_patchtokens'].mean(dim=1)], dim=1)
    if z.shape[1] != 1536 or not torch.isfinite(z).all():
        raise RuntimeError('Invalid DINOv2 Base feature dimensions or values')
    return z.float().cpu().numpy()

def safe_encode(model, x, device):
    try:
        return encode(model, x.to(device, non_blocking=True))
    except torch.cuda.OutOfMemoryError:
        if len(x) <= 1:
            raise
    # Leave the except block before retrying so its traceback releases tensors.
    gc.collect()
    torch.cuda.empty_cache()
    midpoint = len(x) // 2
    print(f'GPU memory pressure: split inference batch {len(x)} into smaller chunks', flush=True)
    return np.concatenate([safe_encode(model, x[:midpoint], device), safe_encode(model, x[midpoint:], device)])

def extract_partition(model, records, name, signature, batch_size, workers, device):
    folder = ROOT / 'features'
    paths = [folder / f'{name}_{view}.npy' for view in ['original', 'flip']]
    progress_path = folder / f'{name}_progress.json'
    completed = 0
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding='utf-8'))
        if previous['signature'] != signature:
            raise RuntimeError('Feature cache belongs to a different model, data, or extractor')
        completed = int(previous['completed'])
        arrays = [np.load(p, mmap_mode='r+', allow_pickle=False) for p in paths]
        if any(a.shape != (len(records), 1536) or a.dtype != np.float32 for a in arrays):
            raise RuntimeError('Invalid partial cache shape')
        if previous.get('complete'):
            if any(sha(p) != previous['sha256'][p.name] for p in paths):
                raise RuntimeError('Completed feature cache checksum mismatch')
            print(f'{name}: reused {completed} verified cached features', flush=True)
            return previous
    else:
        if any(p.exists() for p in paths):
            raise RuntimeError('Orphan feature file without progress: preserve it and choose a clean feature directory')
        arrays = [np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=(len(records), 1536)) for p in paths]
        write_json(progress_path, {'signature': signature, 'completed': 0, 'total': len(records), 'complete': False})
    if not 0 <= completed <= len(records):
        raise ValueError('Invalid progress count')
    dataset = FaceImages(records[completed:])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    started = last_log = time.monotonic()
    initial = completed
    info = {'signature': signature, 'completed': completed, 'total': len(records), 'complete': False}
    for batch in loader:
        original = safe_encode(model, batch, device)
        flipped = safe_encode(model, torch.flip(batch, [-1]), device)
        end = completed + len(batch)
        arrays[0][completed:end] = original
        arrays[1][completed:end] = flipped
        for array in arrays:
            array.flush()
        completed = end
        elapsed = time.monotonic() - started
        speed = (completed - initial) / max(elapsed, 1e-6)
        info = {'signature': signature, 'completed': completed, 'total': len(records), 'complete': False,
                'images_per_second_both_views': speed, 'elapsed_seconds_this_session': elapsed,
                'estimated_seconds_remaining': (len(records)-completed)/max(speed, 1e-6)}
        write_json(progress_path, info)
        if time.monotonic() - last_log >= 15 or completed == len(records):
            print(f'{name}: {completed}/{len(records)} ({100*completed/len(records):.1f}%), '
                  f'{speed:.1f} images/s incl flip, ETA {info["estimated_seconds_remaining"]/60:.1f} min', flush=True)
            last_log = time.monotonic()
    del arrays
    info.update(complete=True, sha256={p.name: sha(p) for p in paths})
    write_json(progress_path, info)
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; stopped instead of silently running on CPU')
    repo = ROOT / 'vendor' / 'dinov2'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    weight_path = ROOT / 'models' / 'dinov2_vitb14_pretrain.pth'
    weight_info = json.loads(weight_path.with_suffix('.json').read_text(encoding='utf-8'))
    if weight_info['sha256'] != sha(weight_path):
        raise ValueError('Checkpoint hash does not match download receipt')
    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitb14
    model = dinov2_vitb14(pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).to('cuda')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Backbone must be fully frozen')
    labeled = read_csv(ROOT / 'features' / 'manifest.csv')
    unlabeled = read_csv(ROOT / 'features' / 'unlabeled.csv')
    audit = json.loads((ROOT / 'data' / 'split_audit.json').read_text(encoding='utf-8'))
    actual_images = hashlib.sha256()
    for record in labeled + unlabeled:
        actual_images.update(sha(record['path']).encode())
    if actual_images.hexdigest() != audit['image_content_sha256']:
        raise RuntimeError('Image bytes changed since data audit; cached features cannot be reused')
    repo_source = hashlib.sha256()
    for source in sorted((repo / 'dinov2').rglob('*.py')):
        repo_source.update(source.relative_to(repo).as_posix().encode())
        repo_source.update(bytes.fromhex(sha(source)))
    settings = {'model': 'dinov2_vitb14', 'repository_commit': commit, 'weight_sha256': weight_info['sha256'],
                'repository_python_sha256': repo_source.hexdigest(),
                'extractor_sha256': sha(__file__), 'image_size': 224, 'rgb': True,
                'preprocessing': 'PIL RGB; bicubic resize 224x224; ImageNet mean/std; no crop',
                'feature': 'concat normalized CLS token and mean normalized patch tokens', 'feature_dim': 1536,
                'classes': CLASSES, 'frozen': True, 'amp': 'float16', 'torch': torch.__version__,
                'manifest_sha256': sha(ROOT / 'features' / 'manifest.csv'),
                'unlabeled_csv_sha256': sha(ROOT / 'features' / 'unlabeled.csv'),
                'image_content_sha256': audit['image_content_sha256']}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    probe = torch.stack([FaceImages(labeled)[i] for i in range(min(args.batch_size, len(labeled)))])
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    probe_out = safe_encode(model, probe, 'cuda')
    probe_flip = safe_encode(model, torch.flip(probe, [-1]), 'cuda')
    torch.cuda.synchronize()
    probe_info = {'gpu': torch.cuda.get_device_name(), 'batch_size': len(probe),
                  'feature_shape': list(probe_out.shape), 'both_views_seconds': time.monotonic()-begin,
                  'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                  'trainable_backbone_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(ROOT / 'logs' / 'gpu_probe.json', probe_info)
    print('GPU probe passed: ' + json.dumps(probe_info), flush=True)
    if args.probe_only:
        return
    del probe, probe_out, probe_flip
    results = {}
    for name, records in [('labeled', labeled), ('unlabeled', unlabeled)]:
        results[name] = extract_partition(model, records, name, signature, args.batch_size, args.num_workers, 'cuda')
    files = ['manifest.csv', 'unlabeled.csv', 'labeled_original.npy', 'labeled_flip.npy',
             'unlabeled_original.npy', 'unlabeled_flip.npy']
    metadata = {**settings, 'signature': signature, 'file_sha256': {n: sha(ROOT/'features'/n) for n in files},
                'partitions': results, 'data_audit': audit, 'gpu_probe': probe_info}
    write_json(ROOT / 'features' / 'metadata.json', metadata)
    print('All frozen features completed and checksummed.', flush=True)

if __name__ == '__main__':
    main()
