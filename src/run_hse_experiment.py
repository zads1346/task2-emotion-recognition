"""Run local HSEmotion smoke, training, fixed fusion selection and CSV export."""
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
LOGS = ROOT / 'logs'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def status(stage, **extra):
    value = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS / 'hse_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)

def run_stage(name, script, *arguments):
    log_path = LOGS / f'hse_{name}.log'
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write('\nSTART ' + datetime.now().astimezone().isoformat() + '\n')
        stream.flush()
        child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'src' / script), '--project-root', str(ROOT), *arguments],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        status(name, state='running', child_pid=child.pid, log=str(log_path))
        while child.poll() is None:
            time.sleep(15)
            if child.poll() is None:
                status(name, state='running', child_pid=child.pid, log=str(log_path))
        if child.returncode:
            raise RuntimeError(f'{name} exited {child.returncode}; inspect {log_path}')

def make_plan():
    baseline_sha = '2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad'
    if sha(ROOT / 'fusion_results' / 'submission.csv') != baseline_sha:
        raise RuntimeError('The previous best submission changed')
    sources = ['hse_model.py', 'train_hse_specialist.py', 'finalize_hse_specialist.py', 'run_hse_experiment.py']
    inputs = ['features/manifest.csv', 'data/split_audit.json', 'fusion_validation_results/selected_recipe.json',
              'fusion_validation_results/selected_val_probabilities.npy', 'fusion_results/summary.json',
              'fusion_results/probabilities.npy', 'fusion_results/submission.csv']
    return {'experiment': 'hsemotion_transfer_and_fixed_fusion_v1', 'classes': ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            'train_rows': 21248, 'validation_rows': 2656, 'submission_rows': 7065,
            'training': {'model': 'tf_efficientnet_b2', 'official_weights': 'enet_b2_7.pt', 'global_pool': 'max',
                         'head_rows_from_official': [0, 2, 3, 5, 6, 4], 'image_size': 224, 'epochs': 12,
                         'batch_size': 8, 'accumulation': 4, 'head_warmup_epochs': 1,
                         'backbone_lr': 5e-5, 'head_lr': 5e-4, 'weight_decay': .01, 'label_smoothing': .05,
                         'cosine_lr_floor_fraction': .1, 'batchnorm_running_stats_frozen': True,
                         'seed': 42, 'workers': 0, 'amp': 'float16', 'cpu_threads': 4},
            'fusion': {'baseline': 'existing .4 LoRA + .4 CNN + .2 cosine MLP', 'hse_weights': [0, .1, .25, .5],
                       'hse_views': ['original', 'original_flip_mean'], 'unique_candidates': 7,
                       'rule': 'strict validation accuracy gain required; ties with baseline keep baseline'},
            'reused_validation': True, 'internal_test_used': False, 'external_answers_used': False,
            'preserved_submission_sha256': baseline_sha,
            'source_sha256': {name: sha(ROOT / 'src' / name) for name in sources},
            'input_sha256': {name: sha(ROOT / name) for name in inputs}}

def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / 'hse_experiment.lock').open('a+b')
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        lock.close()
        raise RuntimeError('An HSE pipeline is already running') from exc
    started = time.monotonic()
    try:
        plan = make_plan()
        path = LOGS / 'hse_plan.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != plan:
                raise RuntimeError('Locked HSE experiment sources/configuration/data changed')
        else:
            write_json(path, plan)
        run_stage('smoke', 'train_hse_specialist.py', '--stage', 'smoke')
        run_stage('train', 'train_hse_specialist.py', '--stage', 'train')
        run_stage('select', 'finalize_hse_specialist.py', '--stage', 'select')
        run_stage('export', 'finalize_hse_specialist.py', '--stage', 'export')
        summary_path = ROOT / 'hse_results' / 'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('state') != 'complete':
            raise RuntimeError('HSE submission export is incomplete')
        if sha(ROOT / 'fusion_results' / 'submission.csv') != plan['preserved_submission_sha256']:
            raise RuntimeError('Previous submission was modified')
        status('complete', state='complete', elapsed_seconds=time.monotonic() - started,
               summary=str(summary_path), submission=str(ROOT / 'hse_results' / 'submission.csv'))
    except BaseException as exc:
        status('error', state='failed', error=str(exc), elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()

if __name__ == '__main__':
    main()
