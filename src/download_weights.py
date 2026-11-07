"""Download the official DINOv2 Base checkpoint into this project."""
import hashlib
import json
import time
import urllib.request
import concurrent.futures as futures
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
dest = ROOT / 'models' / 'dinov2_vitb14_pretrain.pth'
receipt = dest.with_suffix('.json')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

PARTS = dest.parent / 'dinov2_vitb14_download_parts'
CHUNK_SIZE = 4 * 1024 * 1024
WORKERS, ATTEMPTS, TIMEOUT = 4, 8, 45
LOCK = threading.Lock()

def log(s):
    with LOCK:
        print(s, flush=True)

def request(method='GET', headers=None):
    return urllib.request.Request(URL, method=method, headers={
        'User-Agent': 'DINOv2-local-setup/1.0', 'Accept-Encoding': 'identity', **(headers or {})})

def metadata():
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request('HEAD'), timeout=TIMEOUT) as r:
                n = int(r.headers['Content-Length'])
                etag = r.headers.get('ETag')
                if n < 300_000_000 or not etag:
                    raise RuntimeError('Unexpected remote metadata')
                return dict(url=URL, bytes=n, etag=etag, chunk_size=CHUNK_SIZE,
                            last_modified=r.headers.get('Last-Modified'))
        except Exception as e:
            if attempt + 1 == ATTEMPTS:
                raise
            log(f'HEAD retry {attempt + 1}: {e}')
            time.sleep(min(2 ** attempt, 20))

def open_range(start, end, meta):
    r = urllib.request.urlopen(request(headers={
        'Range': f'bytes={start}-{end}', 'If-Range': meta['etag']}), timeout=TIMEOUT)
    cr = r.headers.get('Content-Range', '')
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', cr)
    if (r.status != 206 or not match or
        tuple(map(int, match.groups())) != (start, end, meta['bytes']) or
        r.headers.get('ETag') != meta['etag'] or
        int(r.headers.get('Content-Length', '-1')) != end-start+1):
        status = r.status
        r.close()
        raise RuntimeError(f'Invalid range response: {status} {cr}')
    return r

def seed_partial(meta):
    old = dest.with_suffix('.download')
    if not old.exists() or not old.stat().st_size:
        return
    size = old.stat().st_size
    if size > meta['bytes']:
        raise RuntimeError('Old partial too large; preserved')
    for offset in sorted({0, max(0, size-16384)}):
        end = min(size-1, offset+16383)
        with open_range(offset, end, meta) as r, old.open('rb') as f:
            f.seek(offset)
            if r.read() != f.read(end-offset+1):
                raise RuntimeError('Old partial differs from official source; preserved')
    with old.open('rb') as f:
        index = 0
        while b := f.read(CHUNK_SIZE):
            part = PARTS / f'{index:04d}.part'
            if not part.exists():
                part.write_bytes(b)
            elif part.stat().st_size < len(b):
                raise RuntimeError('Existing part unexpectedly short')
            index += 1
    log(f'Reused original partial: {size/1e6:.2f} MB')

def download_part(spec, meta):
    index, start, end = spec
    part = PARTS / f'{index:04d}.part'
    expected = end-start+1
    for attempt in range(ATTEMPTS):
        size = part.stat().st_size if part.exists() else 0
        if size == expected:
            return index
        if size > expected:
            raise RuntimeError(f'Oversized part {index}; preserved')
        try:
            with open_range(start+size, end, meta) as r, part.open('ab') as f:
                left = expected-size
                while left:
                    b = r.read(min(64*1024, left))
                    if not b:
                        raise EOFError(f'{left} bytes still missing')
                    f.write(b)
                    left -= len(b)
                f.flush()
                os.fsync(f.fileno())
            return index
        except Exception as e:
            log(f'Chunk {index} attempt {attempt+1}/{ATTEMPTS}: {type(e).__name__}: {e}')
            if attempt+1 == ATTEMPTS:
                raise RuntimeError(f'Chunk {index} retries exhausted; rerun to resume') from e
            time.sleep(min(2 ** attempt, 20))

def validate(path):
    import torch
    os.environ.setdefault('XFORMERS_DISABLED', '1')
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT/'vendor'/'dinov2'))
    from dinov2.hub.backbones import dinov2_vitb14
    log('CPU validation: weights_only=True and strict official ViT-B state dict')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError('Invalid state dict')
    model = dinov2_vitb14(pretrained=False)
    model.load_state_dict(state, strict=True)
    if tuple(state['patch_embed.proj.weight'].shape) != (768, 3, 14, 14):
        raise RuntimeError('Wrong patch shape')
    if tuple(state['cls_token'].shape) != (1, 1, 768):
        raise RuntimeError('Wrong CLS shape')
    return dict(weights_only_load=True, strict_state_dict_load=True,
        tensor_count=len(state), model='dinov2_vitb14',
        parameter_count=sum(p.numel() for p in model.parameters()),
        patch_embedding_shape=list(state['patch_embed.proj.weight'].shape),
        cls_token_shape=list(state['cls_token'].shape))

def save_receipt(meta, validation):
    info = {**meta, 'sha256': digest(dest), 'validation': validation}
    tmp = receipt.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(info, indent=2), encoding='utf-8')
    tmp.replace(receipt)
    log('Official checkpoint verified: ' + json.dumps(info))

def main():
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and receipt.exists():
        info = json.loads(receipt.read_text(encoding='utf-8'))
        if info['sha256'] != digest(dest):
            raise RuntimeError('Existing checkpoint hash mismatch; preserved')
        if info.get('validation', {}).get('strict_state_dict_load'):
            log('Verified existing official checkpoint SHA256')
            return
    meta = metadata()
    log(f'Official object {meta["bytes"]:,} bytes; {WORKERS} parallel ranged workers')
    if dest.exists():
        if dest.stat().st_size != meta['bytes']:
            raise RuntimeError('Existing destination wrong size; preserved')
        save_receipt(meta, validate(dest))
        return
    PARTS.mkdir(exist_ok=True)
    manifest = PARTS/'remote.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if any(saved.get(k) != meta[k] for k in ('url', 'bytes', 'etag', 'chunk_size')):
            raise RuntimeError('Remote object changed; existing parts preserved')
    else:
        manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    seed_partial(meta)
    specs = [(i, s, min(s+CHUNK_SIZE, meta['bytes'])-1)
             for i, s in enumerate(range(0, meta['bytes'], CHUNK_SIZE))]
    begin = time.monotonic()
    initial = sum(p.stat().st_size for p in PARTS.glob('*.part'))
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {pool.submit(download_part, s, meta) for s in specs}
        while pending:
            finished, pending = futures.wait(pending, timeout=10, return_when=futures.FIRST_EXCEPTION)
            for f in finished:
                f.result()
            received = sum(p.stat().st_size for p in PARTS.glob('*.part'))
            rate = (received-initial)/max(time.monotonic()-begin, 1)/1e6
            log(f'Download {received/1e6:.2f}/{meta["bytes"]/1e6:.2f} MB '
                f'({100*received/meta["bytes"]:.1f}%), aggregate {rate:.2f} MB/s')
    assembled = dest.with_suffix('.assembling')
    with assembled.open('wb') as f:
        for i, start, end in specs:
            part = PARTS/f'{i:04d}.part'
            if part.stat().st_size != end-start+1:
                raise RuntimeError(f'Incomplete chunk {i}')
            with part.open('rb') as source:
                for b in iter(lambda: source.read(1024*1024), b''):
                    f.write(b)
        f.flush()
        os.fsync(f.fileno())
    if assembled.stat().st_size != meta['bytes']:
        raise RuntimeError('Wrong assembled size')
    validation = validate(assembled)
    assembled.replace(dest)
    save_receipt(meta, validation)

if __name__ == '__main__':
    main()
