"""Run the fixed larger-validation official HSE restart, then export its candidate."""
from datetime import datetime
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import run_hse_sam_v7 as guard
import prepare_hse_expanded_validation_v10 as preparation

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'logs'
PREFIX = 'hse_expanded_validation_v10'
guard.GPU_SCRIPT_NAMES += (
    'prepare_ddamfn_groupcv_v9','train_ddamfn_groupcv_v9','finalize_ddamfn_groupcv_v9',
    'run_ddamfn_groupcv_v9','ddamfn_groupcv_v9_worker','audit_hse_sam_v7_epoch0',
    'finalize_hse_sam_v7_recovered','prepare_hse_expanded_validation_v10',
    'train_expanded_validation_v10','finalize_expanded_validation_v10',
    'run_hse_expanded_validation_v10','hse_expanded_validation_v10_worker',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def status(stage, **values):
    value = {'stage':stage,'pid':os.getpid(),
             'updated_at':datetime.now().astimezone().isoformat(),**values}
    guard.write_json(LOGS/f'{PREFIX}_status.json', value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def verify_review_and_smoke():
    plan_path=LOGS/f'{PREFIX}_plan.json'
    plan=read(plan_path)
    pins=plan['source_pins']
    required={str((ROOT/name).resolve()) for name in preparation.REQUIRED_SOURCES}
    if not required.issubset(pins):
        raise ValueError('V10 plan omits production sources')
    for path,digest in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('V10 source/input changed: '+path)
    plan_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for name in ('review','gpu_smoke'):
        receipt=read(LOGS/f'{PREFIX}_{name}.json')
        if receipt.get('state')!='passed' or receipt.get('source_pins')!=pins or receipt.get('plan_sha256')!=plan_sha:
            raise ValueError('V10 needs a passing review and real GPU smoke bound to these exact sources')
        if name=='gpu_smoke':
            expected={'GPU_used':True,'weights_saved':False,'validation_evaluated':False,
                      'successful_optimizer_steps':2,'amp_skipped_steps':0,'ema_updates':2,
                      'batch_size':16,'effective_batch_size':32,'uniform_CE':True,
                      'batchnorm_buffers_frozen':True,'full_backbone_updated':True,
                      'internal_test_read':False,'external_answers_used':False,
                      'parent_checkpoint_sha256':plan['parent_checkpoint_sha256'],
                      'model_source_receipt_sha256':plan['model_source_receipt_sha256'],
                      'split_manifest_sha256':plan['split_manifest_sha256'],
                      'split_summary_sha256':plan['split_summary_sha256'],
                      'training_manifest_indices':plan['train_indices'][:64]}
            if any(receipt.get(key)!=value for key,value in expected.items()):
                raise ValueError('V10 smoke does not prove two real official-initialized GPU steps')
            if (not receipt.get('all_parameter_steps') or any(value!=2 for value in receipt['all_parameter_steps'].values())
                    or not isinstance(receipt.get('peak_allocated_bytes'),int) or receipt['peak_allocated_bytes']<=0):
                raise ValueError('V10 smoke did not update all parameters on the GPU')
    return plan


def main():
    LOGS.mkdir(exist_ok=True)
    lock=(LOGS/f'{PREFIX}_experiment.lock').open('a+b')
    lock.seek(0,2)
    if not lock.tell():
        lock.write(b'0');lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    child=None
    started=time.monotonic()
    try:
        plan=verify_review_and_smoke()
        stages=(
            ('prepare','prepare_hse_expanded_validation_v10.py',['--stage','verify']),
            ('train','train_expanded_validation_v10.py',['--stage','train','--device','cuda']),
            ('select','finalize_expanded_validation_v10.py',['--stage','select']),
            ('export','finalize_expanded_validation_v10.py',['--stage','export']),
            ('verify','finalize_expanded_validation_v10.py',['--stage','verify']),
        )
        for stage,script,arguments in stages:
            verify_review_and_smoke()
            guard.no_other_gpu_task()
            log=LOGS/f'{PREFIX}_{stage}.log'
            with log.open('a',encoding='utf-8') as stream:
                stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n');stream.flush()
                child=subprocess.Popen(
                    [sys.executable,'-B','-u',str(ROOT/'src'/script),'--project-root',str(ROOT),*arguments],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1',
                         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while True:
                    status(stage,state='running',child_pid=child.pid,log=str(log),total_training_epochs=12,
                           training_rows=len(plan['train_indices']),validation_rows=len(plan['val_indices']))
                    try:
                        code=child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                child=None
                if code:
                    raise RuntimeError(f'{stage} exited {code}; inspect {log}')
        summary_path=ROOT/'hse_expanded_validation_v10_results/summary.json'
        summary=read(summary_path)
        if summary.get('state')!='complete' or summary.get('submission_rows')!=7065:
            raise ValueError('V10 verified candidate export is incomplete')
        status('complete',state='complete',summary=str(summary_path),elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        guard.stop_child_tree(child)
        status('error',state='failed',error=str(error),elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    main()
