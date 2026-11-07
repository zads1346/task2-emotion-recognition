"""Independent, locked six-epoch LoRA queue with validation/export completion."""
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

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def status(stage,**extra):
    data={'stage':stage,'pid':os.getpid(),'updated_at':datetime.now().astimezone().isoformat(),**extra}
    write_json(LOGS/'lora_task_v4_status.json',data); print(json.dumps(data,ensure_ascii=False),flush=True)

def no_other_gpu_task():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(improvement_v3.py|train_hse_specialist.py|train_ambiguity_hse|run_hse_experiment|run_ambiguity|train_lora\\.py|refine_lora\\.py|train_task_lora_v4.py)' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    if result.stdout.strip(): raise RuntimeError('Another GPU task already exists: '+result.stdout.strip())

def main():
    LOGS.mkdir(exist_ok=True); lock=(LOGS/'lora_task_v4_experiment.lock').open('a+b'); lock.seek(0,2)
    if lock.tell()==0: lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: lock.close(); raise RuntimeError('This LoRA queue is already running')
    start=time.monotonic()
    try:
        no_other_gpu_task()
        for name,script in [('prepare','prepare_task_lora_v4.py'),('train','train_task_lora_v4.py'),('finalize','finalize_task_lora_v4.py')]:
            log=LOGS/f'lora_task_v4_{name}.log'
            with log.open('a',encoding='utf-8') as stream:
                stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); stream.flush()
                command=[sys.executable,'-u',str(ROOT/'src'/script)]
                if name=='train': command+=['--project-root',str(ROOT),'--device','cuda']
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
                status(name,state='running',child_pid=child.pid,log=str(log),epochs=6)
                while child.poll() is None:
                    time.sleep(15)
                    if child.poll() is None: status(name,state='running',child_pid=child.pid,log=str(log),epochs=6)
                if child.returncode: raise RuntimeError(f'{name} exited {child.returncode}; inspect {log}')
        path=ROOT/'lora_task_v4_results'/'summary.json'; summary=json.loads(path.read_text(encoding='utf-8'))
        if summary['state']!='complete' or summary['submission_rows']!=7065: raise RuntimeError('Final export incomplete')
        status('complete',state='complete',summary=str(path),elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        status('error',state='failed',error=str(error),elapsed_seconds=time.monotonic()-start); traceback.print_exc(); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

if __name__=='__main__': main()
