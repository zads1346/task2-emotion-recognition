"""Run V8 only after the V7 export; GPU smoke and training are strictly serial."""
from datetime import datetime
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run_hse_sam_v7 import no_other_gpu_task, stop_child_tree

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
POLL_SECONDS = 15


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def status(stage, **values):
    value = {"stage": stage, "pid": os.getpid(), "updated_at": datetime.now().astimezone().isoformat(), **values}
    write_json(LOGS / "ddamfn_v8_status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def require_v7_complete():
    path = ROOT / "hse_sam_v7_results/summary.json"
    if not path.is_file():
        raise RuntimeError("V7 must complete before starting the V8 runner")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("state") != "complete" or summary.get("submission_rows") != 7065:
        raise RuntimeError("V7 must finish its verified submission export before V8 starts")


def main():
    # An idle V8 Python runner is also recognized by the V7 concurrency guard.
    # Do not launch this script as a waiting queue while V7 is still running.
    require_v7_complete()
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / "ddamfn_v8_experiment.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError("The DDAMFN V8 experiment is already running")
    started, child = time.monotonic(), None
    try:
        for script in ("prepare_ddamfn_v8.py", "train_ddamfn_v8.py", "finalize_ddamfn_v8.py"):
            if not (ROOT / "src" / script).is_file():
                raise FileNotFoundError("Missing V8 stage script: " + script)

        def stage(name, script, arguments):
            nonlocal child
            no_other_gpu_task()
            log = LOGS / f"ddamfn_v8_{name}.log"
            with log.open("a", encoding="utf-8") as stream:
                stream.write("\nSTART " + datetime.now().astimezone().isoformat() + "\n"); stream.flush()
                child = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "src" / script), "--project-root", str(ROOT), *arguments],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                         "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                status(name, state="running", child_pid=child.pid, log=str(log), total_training_epochs=6)
                while True:
                    try:
                        code = child.wait(timeout=POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        status(name, state="running", child_pid=child.pid, log=str(log), total_training_epochs=6)
                child = None
                return code

        import prepare_ddamfn_v8 as prepare
        if not prepare.PLAN.exists():
            try:
                probe = prepare.verify_smoke(ROOT)
                status("smoke_reused", state="running", batch_size=probe["batch_size"], total_training_epochs=6)
            except (FileNotFoundError, ValueError, KeyError, TypeError):
                for microbatch in (32, 16, 8):
                    code = stage("smoke", "prepare_ddamfn_v8.py", ("--stage", "smoke", "--batch-size", str(microbatch)))
                    if code == 0:
                        prepare.verify_smoke(ROOT)
                        break
                    if code != 42:
                        raise RuntimeError(f"V8 GPU smoke exited {code}; inspect logs/ddamfn_v8_smoke.log")
                    status("smoke_retry", state="running", previous_microbatch=microbatch, reason="OOM; retry in a fresh process")
                else:
                    raise RuntimeError("V8 GPU smoke exceeded memory at all supported microbatches")
        stages = (("prepare", "prepare_ddamfn_v8.py", ("--stage", "prepare")),
                  ("train", "train_ddamfn_v8.py", ("--device", "cuda")),
                  ("finalize", "finalize_ddamfn_v8.py", ("--stage", "export")))
        for name, script, arguments in stages:
            code = stage(name, script, arguments)
            if code:
                raise RuntimeError(f"{name} exited {code}; inspect logs/ddamfn_v8_{name}.log")
        path = ROOT / "ddamfn_v8_results/summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("state") != "complete" or summary.get("submission_rows") != 7065:
            raise RuntimeError("V8 final verified export is incomplete")
        status("complete", state="complete", summary=str(path), elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        try:
            stop_child_tree(child)
        finally:
            status("error", state="failed", error=str(error), elapsed_seconds=time.monotonic()-started)
            traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == "__main__":
    main()
