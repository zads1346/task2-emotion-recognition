"""Sequential local frozen-backbone training with durable logs and a single-run lock."""
from __future__ import annotations
import json
import msvcrt
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'logs' / 'status.json'

def status(stage, **extra):
    value = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(), **extra}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATUS)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def run(stage, script, args):
    status(stage, state='running')
    cmd = [sys.executable, '-u', str(ROOT/'src'/script), *args]
    with (ROOT/'logs'/f'{stage}.log').open('a', encoding='utf-8') as log:
        log.write('\nRUN ' + datetime.now().astimezone().isoformat() + '\n')
        log.flush()
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
        status(stage, state='running', child_pid=child.pid, log=str(ROOT/'logs'/f'{stage}.log'))
        code = child.wait()
    if code:
        raise RuntimeError(f'{stage} failed with code {code}; see logs/{stage}.log')

def main():
    lock = (ROOT/'logs'/'training.lock').open('a+b')
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise RuntimeError('Another training process already owns this project')
    started = time.monotonic()
    try:
        import torch, torchvision, numpy, scipy, sklearn, PIL, joblib, cloudpickle
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        environment = {'python': sys.version, 'executable': sys.executable, 'base_prefix': sys.base_prefix,
                       'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(),
                       'cuda_runtime': torch.version.cuda, 'total_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
                       'shared_base_packages': 'The desktop venv reuses the existing dome CUDA/PyTorch installation.',
                       'packages': {m.__name__: {'version': m.__version__, 'path': m.__file__}
                                    for m in [torch, torchvision, numpy, scipy, sklearn, PIL, joblib, cloudpickle]}}
        (ROOT/'logs'/'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
        if (ROOT/'results'/'summary.json').exists():
            summary = json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))
            if summary.get('status') == 'complete':
                from train_classifier import sha256
                if sha256(ROOT/'results'/'submission.csv') != summary['submission_sha256']:
                    raise RuntimeError('Completed submission was changed')
                status('complete', state='complete', reused=True, test_metrics=summary['test_metrics'],
                       submission=summary['submission_path'])
                return
        if '--wait-for-weights' in sys.argv:
            status('waiting_for_download', state='running', log=str(ROOT/'logs'/'download_weights.log'))
            deadline = time.monotonic() + 1800
            while not (ROOT/'models'/'dinov2_vitb14_pretrain.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Official checkpoint not ready after 30 minutes; inspect download log')
                time.sleep(5)
        run('download', 'download_weights.py', [])
        run('extract', 'extract_features.py', ['--batch-size', '32', '--num-workers', '2'])
        run('classifier', 'train_classifier.py', ['--features-dir', str(ROOT/'features'),
            '--output', str(ROOT/'results'), '--selection-metric', 'macro_f1', '--max-iter', '300', '--tol', '0.001'])
        summary = json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))
        status('complete', state='complete', elapsed_seconds=time.monotonic()-started,
               test_metrics=summary['test_metrics'], submission=summary['submission_path'])
    except BaseException as exc:
        status('error', state='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()

if __name__ == '__main__':
    main()
