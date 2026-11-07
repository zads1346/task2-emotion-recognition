"""Durable, locked launcher for the four-epoch downweighted HSE continuation."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'logs'
PRESERVED = 'f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def status(stage, **extra):
    value = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS/'ambiguity_v2_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS/'ambiguity_v2_experiment.lock').open('a+b')
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close(); raise RuntimeError('The weighted HSE experiment is already running')
    started = time.monotonic()
    try:
        sources = ['train_ambiguity_hse_v2.py', 'prepare_ambiguity_weights_v2.py', 'train_hse_specialist.py',
                   'hse_model.py', 'run_ambiguity_v2_experiment.py']
        inputs = ['experiments/hse_ambiguity_v2/weights/summary.json',
                  'experiments/hse_ambiguity_v2/weights/train_weights.csv',
                  'experiments/hse_ambiguity_v2/weights/training_weights.npy',
                  'experiments/hse_specialist_v1/best.pt', 'experiments/hse_specialist_v1/summary.json',
                  'features/manifest.csv', 'hse_results/submission.csv']
        if sha(ROOT/'hse_results'/'submission.csv') != PRESERVED:
            raise RuntimeError('The previous best submission changed')
        plan = {'experiment': 'hse_oof_ambiguity_downweight_v2', 'mode': 'downweighted', 'epochs': 4,
                'source_sha256': {name: sha(ROOT/'src'/name) for name in sources},
                'input_sha256': {name: sha(ROOT/name) for name in inputs},
                'preserved_submission_sha256': PRESERVED, 'internal_test_read': False, 'external_answers_used': False}
        plan_path = LOGS/'ambiguity_v2_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Locked weighted experiment sources or inputs changed')
        if not plan_path.exists():
            write_json(plan_path, plan)
        log_path = LOGS/'ambiguity_v2_train.log'
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); stream.flush()
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src'/'train_ambiguity_hse_v2.py'),
                '--project-root', str(ROOT), '--stage', 'train', '--mode', 'downweighted'],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
            status('train', state='running', child_pid=child.pid, log=str(log_path))
            while child.poll() is None:
                time.sleep(15)
                if child.poll() is None:
                    status('train', state='running', child_pid=child.pid, log=str(log_path))
            if child.returncode:
                raise RuntimeError(f'Weighted HSE training exited {child.returncode}; inspect {log_path}')
        summary_path = ROOT/'experiments'/'hse_ambiguity_v2'/'downweighted'/'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('state') != 'complete' or summary.get('epochs_completed') != 4:
            raise RuntimeError('Weighted HSE training did not complete four epochs')
        if sha(ROOT/'hse_results'/'submission.csv') != PRESERVED:
            raise RuntimeError('The previous best submission was changed')
        status('complete', state='complete', summary=str(summary_path), elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        status('error', state='failed', error=str(error), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc(); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
