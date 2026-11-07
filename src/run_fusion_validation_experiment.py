"""Refit the selected cosine head, validate fixed fusion recipes, then export."""
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(stage, **extra):
    value = {'stage': stage, 'pid': os.getpid(),
             'updated_at': datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS/'fusion_validation_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_stage(name, script, *arguments):
    log_path = LOGS/f'fusion_validation_{name}.log'
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); stream.flush()
        child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src'/script), '--project-root', str(ROOT),
                                  *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'})
        status(name, state='running', child_pid=child.pid, log=str(log_path))
        while child.poll() is None:
            time.sleep(15)
            if child.poll() is None:
                status(name, state='running', child_pid=child.pid, log=str(log_path))
        if child.returncode:
            raise RuntimeError(f'{name} exited {child.returncode}; inspect {log_path}')


def make_plan():
    expected = 'f662aa9718056f236c8c9be41f27bde9219cb1be5f7fc6ac22df6fb07c5b465d'
    if sha(ROOT/'refinement_results/submission.csv') != expected:
        raise RuntimeError('Existing fusion submission changed')
    source_names = ['train_fusion_head.py', 'extract_fusion_features.py', 'validate_cosine_fusion.py',
                    'run_fusion_validation_experiment.py', 'evaluate_weighted_heads.py', 'image_sample_weights.py',
                    'evaluate_multilayer_cv.py', 'extract_multilayer_features.py', 'finalize_refinement.py',
                    'refine_lora.py', 'finalize_improvements.py', 'train_lora.py', 'train_classifier.py',
                    'train_mlp.py', 'cnn_candidate.py']
    return {'experiment': 'cosine_mlp_fixed_fusion_validation', 'version': 1,
            'classes': ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            'train_rows': 21248, 'validation_rows': 2656, 'submission_rows': 7065,
            'head': {'readout': 'last4_cls_plus_last_patch_mean', 'dimension': 3840, 'hidden': 256,
                     'dropout': .3, 'epochs': 12, 'batch_size': 256, 'lr': .001, 'min_lr': .00001,
                     'weight_decay': .001, 'seed': 42, 'training_split': 'train',
                     'sample_weights': False, 'checkpoint': 'fixed_final_epoch',
                     'scaler': 'all training original-view features only'},
            'fixed_fusions': [
                {'name': 'baseline', 'weights': {'refined_lora': .45, 'cnn': .45, 'old_mlp': .10}},
                {'name': 'replace_mlp_10', 'weights': {'refined_lora': .45, 'cnn': .45, 'new_mlp': .10}},
                {'name': 'replace_mlp_20', 'weights': {'refined_lora': .40, 'cnn': .40, 'new_mlp': .20}},
                {'name': 'split_mlp_05', 'weights': {'refined_lora': .45, 'cnn': .45, 'old_mlp': .05, 'new_mlp': .05}}],
            'selection': 'validation accuracy; a tie with baseline keeps baseline',
            'tta': 'average original/flip probabilities separately per model before fusion',
            'reused_validation': True, 'internal_test_used': False, 'external_answers_used': False,
            'export_policy': 'lock recipe before competition inference; baseline win copies existing submission',
            'preserved_submission_sha256': expected,
            'input_receipt_sha256': {
                'train_features': sha(ROOT/'features_multilayer_train/metadata.json'),
                'head_selection': sha(ROOT/'weighted_results/summary.json'),
                'existing_recipe': sha(ROOT/'refinement_results/selected_recipe.json'),
                'existing_summary': sha(ROOT/'refinement_results/summary.json')},
            'source_sha256': {name: sha(ROOT/'src'/name) for name in source_names}}


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS/'fusion_validation.lock').open('a+b')
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        lock.close()
        raise RuntimeError('Fusion validation is already running') from exc
    started = time.monotonic()
    try:
        plan = make_plan()
        plan_path = LOGS/'fusion_validation_plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
                raise RuntimeError('Fixed fusion experiment sources/data changed')
        else:
            write_json(plan_path, plan)
        run_stage('train_head', 'train_fusion_head.py')
        run_stage('val_features', 'extract_fusion_features.py', '--partition', 'val')
        run_stage('select', 'validate_cosine_fusion.py', '--stage', 'select')
        recipe_path = ROOT/'fusion_validation_results/selected_recipe.json'
        recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
        if recipe.get('state') != 'locked' or not isinstance(recipe.get('requires_new_predictions'), bool):
            raise RuntimeError('Fusion selection returned without a locked recipe')
        if recipe['requires_new_predictions']:
            run_stage('unlabeled_features', 'extract_fusion_features.py', '--partition', 'unlabeled')
        run_stage('export', 'validate_cosine_fusion.py', '--stage', 'export')
        summary_path = ROOT/'fusion_results/summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('state') != 'complete':
            raise RuntimeError('Fusion export summary is incomplete')
        if sha(ROOT/'refinement_results/submission.csv') != plan['preserved_submission_sha256']:
            raise RuntimeError('Original submission was modified')
        status('complete', state='complete', elapsed_seconds=time.monotonic()-started,
               summary=str(summary_path), submission=str(ROOT/'fusion_results/submission.csv'))
    except BaseException as exc:
        status('error', state='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
