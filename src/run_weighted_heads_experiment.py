"""Run the prespecified image-weighting and head-regularization comparison."""
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
    value = {'stage': stage, 'pid': os.getpid(),
             'updated_at': datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS / 'weighted_heads_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def make_plan():
    original_submission = ROOT / 'refinement_results/submission.csv'
    expected = 'f662aa9718056f236c8c9be41f27bde9219cb1be5f7fc6ac22df6fb07c5b465d'
    if sha(original_submission) != expected:
        raise RuntimeError('The existing fusion submission changed')
    sources = ['evaluate_weighted_heads.py', 'image_sample_weights.py', 'evaluate_multilayer_cv.py',
               'run_weighted_heads_experiment.py', 'extract_multilayer_features.py', 'group_training_images.py']
    return {
        'experiment': 'frozen_dinov2_image_weights_and_regularization', 'version': 1,
        'classes': ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
        'source_split': 'train', 'expected_images': 21248, 'folds': 3,
        'candidates': ['baseline', 'label_smoothing', 'cosine', 'feature_mixup',
                       'class_weight', 'duplicate_weight', 'class_duplicate_weight'],
        'head': 'last4_mlp_3840_256_6', 'epochs': 12, 'batch_size': 256,
        'learning_rate': .001, 'weight_decay': .001, 'dropout': .3, 'seed': 42,
        'label_smoothing': .05, 'cosine_final_learning_rate': .00001,
        'feature_mixup_alpha': .2, 'feature_mixup_batch_probability': .5,
        'image_weighting': {
            'class_factor': 'clip(sqrt(N_train/(6*n_train_class)), .75, 1.5)',
            'duplicate_factor': '1/sqrt(number_of_training_images_in_this_group)',
            'combined_factor': 'clip(class_factor*duplicate_factor, .5, 1.5)',
            'normalization': 'divide by full training fold image-weight mean; final range can shift',
            'loss': 'mean(image_weight * per_image_cross_entropy)',
            'sampling': 'uniform shuffle once per epoch; no additional weighted sampling',
            'statistics': 'current training fold only', 'labels_changed': False},
        'selection': 'fixed final epoch; unweighted heldout accuracy and macro-F1/log-loss',
        'existing_val_test_or_competition_used': False, 'external_answers_used': False,
        'new_submission_produced': False, 'preserved_submission_sha256': expected,
        'data_receipt_sha256': {
            'features': sha(ROOT/'features_multilayer_train/metadata.json'),
            'groups': sha(ROOT/'experiments/multilayer_groups/groups_summary.json')},
        'source_sha256': {name: sha(ROOT/'src'/name) for name in sources}}


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS/'weighted_heads.lock').open('a+b')
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        lock.close()
        raise RuntimeError('A weighted-heads experiment is already running') from exc
    started = time.monotonic()
    try:
        plan = make_plan()
        plan_path = LOGS/'weighted_heads_plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
                raise RuntimeError('Experiment sources, data or fixed plan changed')
        else:
            write_json(plan_path, plan)
        log_path = LOGS/'weighted_heads_cv.log'
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write('\nSTART ' + datetime.now().astimezone().isoformat() + '\n'); stream.flush()
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src/evaluate_weighted_heads.py'),
                                      '--project-root', str(ROOT)], cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
            status('cv', state='running', child_pid=child.pid, log=str(log_path))
            while child.poll() is None:
                time.sleep(15)
                if child.poll() is None:
                    status('cv', state='running', child_pid=child.pid, log=str(log_path))
            if child.returncode:
                raise RuntimeError(f'Weighted heads exited {child.returncode}; inspect {log_path}')
        path = ROOT/'weighted_results/summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        if (summary.get('state') != 'complete' or summary.get('train_rows') != 21248
                or summary.get('folds') != 3 or summary.get('epochs_per_fold') != 12
                or len(summary.get('candidates', [])) != 7):
            raise RuntimeError('Weighted-head summary is incomplete or differs from the fixed plan')
        if sha(ROOT/'refinement_results/submission.csv') != plan['preserved_submission_sha256']:
            raise RuntimeError('Previous submission changed during experiment')
        status('complete', state='complete', elapsed_seconds=time.monotonic()-started, summary=str(path))
    except BaseException as exc:
        status('error', state='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
