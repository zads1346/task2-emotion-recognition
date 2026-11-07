"""Serial, resumable paired CE/flip-consistency LoRA experiment."""
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

def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(path)

def status(stage, **extra):
    data = {'stage': stage, 'pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS/'flip_consistency_v6_status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)

def no_other_gpu_task():
    command = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(improvement_v3.py|train_hse_specialist.py|train_ambiguity_hse|run_hse_experiment|run_ambiguity|train_lora\\.py|refine_lora\\.py|train_task_lora_v4.py|train_zero_weight_lora_v5.py|finalize_zero_weight_v5.py|prepare_zero_weight_v5.py|train_flip_consistency_v6.py|prepare_flip_consistency_v6.py|finalize_flip_consistency_v6.py)' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', check=True)
    if result.stdout.strip(): raise RuntimeError('Another GPU task already exists: '+result.stdout.strip())

def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS/'flip_consistency_v6_experiment.lock').open('a+b'); lock.seek(0, 2)
    if lock.tell() == 0: lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError: lock.close(); raise RuntimeError('This CE/consistency experiment is already running')
    start = time.monotonic()
    try:
        no_other_gpu_task()
        stages = [('prepare', 'prepare_flip_consistency_v6.py', []),
                  ('ce_control', 'train_flip_consistency_v6.py', ['--mode', 'ce_control', '--device', 'cuda']),
                  ('flip_consistency', 'train_flip_consistency_v6.py', ['--mode', 'flip_consistency', '--device', 'cuda']),
                  ('finalize', 'finalize_flip_consistency_v6.py', [])]
        for stage, script, arguments in stages:
            log = LOGS/f'flip_consistency_v6_{stage}.log'
            with log.open('a', encoding='utf-8') as stream:
                stream.write('\nSTART '+datetime.now().astimezone().isoformat()+'\n'); stream.flush()
                child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src'/script),
                                          '--project-root', str(ROOT), *arguments], cwd=ROOT,
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
                                              'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
                status(stage, state='running', child_pid=child.pid, log=str(log), epochs_per_arm=4, total_training_epochs=8)
                while child.poll() is None:
                    time.sleep(15)
                    if child.poll() is None:
                        status(stage, state='running', child_pid=child.pid, log=str(log), epochs_per_arm=4, total_training_epochs=8)
                if child.returncode: raise RuntimeError(f'{stage} exited{child.returncode}; inspect {log}')
        path = ROOT/'lora_flip_consistency_v6_results/summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary['state'] != 'complete' or summary['submission_rows'] != 7065: raise RuntimeError('Final export incomplete')
        status('complete', state='complete', summary=str(path), elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        status('error', state='failed', error=str(error), elapsed_seconds=time.monotonic()-start)
        traceback.print_exc(); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()

if __name__ == '__main__': main()
