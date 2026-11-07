"""One locked, restartable GPU queue; completion includes every export stage."""
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'logs'
JOBS=[
    ('prepare',['--stage','prepare']),
    ('diagnostic',['--stage','diagnostic']),
    ('diagnostic_export',['--stage','export','--family','diagnostic']),
    ('hse_full',['--stage','hse','--mode','full']),
    ('hse_last_stage',['--stage','hse','--mode','last_stage']),
    ('hse_export',['--stage','export','--family','hse']),
    ('head_warmup',['--stage','head']),
    ('lora_last4',['--stage','lora','--blocks','4']),
    ('lora_last8',['--stage','lora','--blocks','8']),
    ('lora_export',['--stage','export','--family','lora']),
]

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def status(stage,**values):
    result={'stage':stage,'pid':os.getpid(),'updated_at':datetime.now().astimezone().isoformat(),**values}
    write_json(LOGS/'improvement_v3_status.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def completed(name):
    experiments=ROOT/'experiments'/'improvement_v3'
    paths={'hse_full':experiments/'hse_full'/'summary.json',
           'hse_last_stage':experiments/'hse_last_stage'/'summary.json',
           'head_warmup':experiments/'head_warmup'/'summary.json',
           'lora_last4':experiments/'lora_last4'/'summary.json',
           'lora_last8':experiments/'lora_last8'/'summary.json'}
    if name.endswith('_export'):
        family=name.removesuffix('_export'); return (ROOT/'improvement_v3_results'/family/'summary.json').is_file()
    if name in paths: return paths[name].is_file()
    if name=='prepare': return (LOGS/'improvement_v3_plan.json').is_file()
    if name=='diagnostic':
        return len(list((experiments/'diagnostic').glob('*/candidate.json')))==8
    return False

def reject_other_training():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(train_hse_specialist|train_ambiguity_hse|run_hse_experiment|run_ambiguity|train_lora\\.py|refine_lora\\.py)' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    if result.stdout.strip(): raise RuntimeError('Another training process exists; do not start duplicate GPU work: '+result.stdout.strip())

def main():
    LOGS.mkdir(exist_ok=True)
    lock=(LOGS/'improvement_v3_experiment.lock').open('a+b'); lock.seek(0,2)
    if lock.tell()==0: lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: lock.close(); raise RuntimeError('This GPU queue is already running')
    started=time.monotonic()
    try:
        reject_other_training()
        for index,(name,args) in enumerate(JOBS,1):
            if completed(name):
                # Execute idempotent entry points to verify completed bindings; these
                # return without retraining, rather than trusting a file's existence.
                print('Revalidate completed '+name,flush=True)
            path=LOGS/f'improvement_v3_{name}.log'
            with path.open('a',encoding='utf-8') as stream:
                stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); stream.flush()
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'src'/'improvement_v3.py'),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                status(name,state='running',child_pid=child.pid,log=str(path),job=index,jobs=len(JOBS))
                while child.poll() is None:
                    time.sleep(15)
                    if child.poll() is None: status(name,state='running',child_pid=child.pid,log=str(path),job=index,jobs=len(JOBS))
                if child.returncode: raise RuntimeError(f'{name} exited {child.returncode}; inspect {path}')
        summary=ROOT/'improvement_v3_results'/'summary.json'
        data=json.loads(summary.read_text(encoding='utf-8'))
        if data.get('state')!='complete' or data.get('submission_rows')!=7065: raise RuntimeError('Final export incomplete')
        status('complete',state='complete',summary=str(summary),elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        status('error',state='failed',error=str(error),elapsed_seconds=time.monotonic()-started); traceback.print_exc(); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

if __name__=='__main__': main()
