"""Run cached-feature MLP, four-block LoRA, then validation-selected CNN fusion."""
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def status(stage, **extra):
    obj = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(), **extra}
    path = ROOT/'logs'/'improvement_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
    print(json.dumps(obj), flush=True)

def stage(name, script, args):
    log_path = ROOT/'logs'/f'improvement_{name}.log'
    cmd = [sys.executable, '-u', str(ROOT/'src'/script), *args]
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); log.flush()
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
        status(name, state='running', child_pid=child.pid, log=str(log_path))
        while child.poll() is None:
            time.sleep(15)
            if child.poll() is None:
                status(name, state='running', child_pid=child.pid, log=str(log_path))
        code = child.returncode
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect {log_path.name}')

def main():
    lock = (ROOT/'logs'/'improvements.lock').open('a+b')
    lock.seek(0, 2)
    if not lock.tell(): lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError: raise RuntimeError('An improvement pipeline is already running')
    started = time.monotonic()
    try:
        stages = ['train_mlp.py', 'train_lora.py', 'cnn_candidate.py', 'finalize_improvements.py']
        plan = {'selection_metric': 'accuracy', 'secondary_metric': 'macro_f1',
                'train_images': 21248, 'val_images': 2656,
                'mlp': {'epochs': 40, 'patience': 8, 'batch_size': 256, 'lr': .001},
                'lora': {'epochs': 20, 'patience': 5, 'batch_size': 4, 'accumulation': 8,
                         'last_blocks': 4, 'rank': 8, 'alpha': 16, 'lr_lora': .0001, 'lr_head': .0005},
                'ensemble': 'best DINO candidate + verified CNN; DINO weights .25/.5/.75',
                'external_answers_used_for_selection': False,
                'source_sha256': {name: sha(ROOT/'src'/name) for name in stages}}
        (ROOT/'logs'/'improvement_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        stage('mlp', 'train_mlp.py', ['--project-root', str(ROOT), '--output', str(ROOT/'experiments'/'mlp'),
                                    '--selection-metric', 'accuracy'])
        lora_output = ROOT/'experiments'/'lora'
        if (lora_output/'summary.json').exists():
            info = json.loads((lora_output/'summary.json').read_text(encoding='utf-8'))
            if info['provenance']['source_sha256'] != sha(ROOT/'src'/'train_lora.py'):
                raise RuntimeError('Completed LoRA source changed')
            if info['best_checkpoint_sha256'] != sha(lora_output/'best.pt'):
                raise RuntimeError('Completed LoRA checkpoint changed')
            if info['val_probabilities_sha256'] != sha(lora_output/'val_probabilities.npy'):
                raise RuntimeError('Completed LoRA validation probabilities changed')
            if info['provenance']['manifest_sha256'] != sha(ROOT/'features'/'manifest.csv'):
                raise RuntimeError('Completed LoRA data split changed')
            status('lora', state='complete', reused=True)
        else:
            args = ['--project-root', str(ROOT), '--output', str(lora_output), '--selection-metric', 'accuracy',
                    '--epochs', '20', '--patience', '5', '--batch-size', '4', '--accumulation', '8', '--workers', '2']
            if (lora_output/'last.pt').exists(): args.append('--resume')
            stage('lora', 'train_lora.py', args)
        stage('finalize', 'finalize_improvements.py', ['--project-root', str(ROOT)])
        summary = json.loads((ROOT/'improved_results'/'summary.json').read_text(encoding='utf-8'))
        status('complete', state='complete', elapsed_seconds=time.monotonic()-started,
               selected=summary['selected'], test_metrics=summary['test_metrics'],
               submission=summary['submission_path'])
    except BaseException as exc:
        status('error', state='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()

if __name__ == '__main__': main()
