"""Serial, resumable HSE AdamW-control/SAM experiment; explicit launch only."""
from datetime import datetime
import json
import msvcrt
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
POLL_SECONDS = 15
STAGES = (
    ("prepare", "prepare_hse_sam_v7.py", ()),
    ("adamw_control", "train_hse_sam_v7.py", ("--mode", "adamw_control", "--device", "cuda")),
    ("sam", "train_hse_sam_v7.py", ("--mode", "sam", "--device", "cuda")),
    ("finalize", "finalize_hse_sam_v7.py", ()),
)
GPU_SCRIPT_NAMES = (
    "improvement_v3", "run_improvement_v3", "run_improvement_v3_experiment", "improvement_v3_worker",
    "train_hse_specialist", "finalize_hse_specialist", "run_hse_experiment", "hse_worker",
    "train_ambiguity_hse", "train_ambiguity_hse_v2", "finalize_ambiguity_training",
    "finalize_ambiguity_training_v2", "prepare_ambiguity_weights", "prepare_ambiguity_weights_v2",
    "run_ambiguity_experiment", "run_ambiguity_v2_experiment", "ambiguity_worker", "ambiguity_v2_worker",
    "train_lora", "refine_lora", "train_task_lora_v4", "prepare_task_lora_v4", "finalize_task_lora_v4",
    "run_lora_task_v4", "lora_task_v4_worker", "train_zero_weight_lora_v5", "prepare_zero_weight_v5",
    "finalize_zero_weight_v5", "run_zero_weight_v5", "zero_weight_v5_worker",
    "train_flip_consistency_v6", "prepare_flip_consistency_v6", "finalize_flip_consistency_v6",
    "run_flip_consistency_v6", "flip_consistency_v6_worker", "train_classifier", "train_mlp",
    "train_fusion_head", "extract_features", "extract_multilayer_features", "extract_fusion_features",
    "run_training", "run_improvements", "run_refinement", "run_fusion_validation_experiment",
    "run_multilayer_experiment", "run_weighted_heads_experiment", "background_worker", "refinement_worker",
    "fusion_validation_worker", "multilayer_worker", "weighted_heads_worker", "train_hse_sam_v7",
    "prepare_hse_sam_v7", "finalize_hse_sam_v7", "run_hse_sam_v7", "hse_sam_v7_worker",
    "smoke_hse_sam_v7", "verify_hse_sam_v7_math", "train_ddamfn_v8", "prepare_ddamfn_v8",
    "finalize_ddamfn_v8", "run_ddamfn_v8", "ddamfn_v8_worker",
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def status(stage, **extra):
    value = {"stage": stage, "pid": os.getpid(),
             "updated_at": datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS / "hse_sam_v7_status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def no_other_gpu_task():
    # The venv launcher is the worker's parent Python process; exclude both.
    pattern = "(?:" + "|".join(re.escape(name) for name in GPU_SCRIPT_NAMES) + r")\.py"
    command = (
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
        f"-and $_.ProcessId -notin @({os.getpid()}, {os.getppid()}) "
        f"-and $_.CommandLine -match '{pattern}' "
        "} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.stdout.strip():
        raise RuntimeError("Another GPU task already exists: " + result.stdout.strip())


def stop_child_tree(child):
    if child is not None and child.poll() is None:
        subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                       capture_output=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Could not stop the owned stage process {child.pid}") from error


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / "hse_sam_v7_experiment.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError("The HSE AdamW/SAM V7 experiment is already running")
    started = time.monotonic()
    child = None
    try:
        missing = [script for _, script, _ in STAGES if not (ROOT / "src" / script).is_file()]
        if missing:
            raise FileNotFoundError("Missing V7 stage scripts: " + ", ".join(missing))
        for stage, script, arguments in STAGES:
            no_other_gpu_task()
            log = LOGS / f"hse_sam_v7_{stage}.log"
            with log.open("a", encoding="utf-8") as stream:
                stream.write("\nSTART " + datetime.now().astimezone().isoformat() + "\n")
                stream.flush()
                child = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "src" / script), "--project-root", str(ROOT), *arguments],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                         "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                status(stage, state="running", child_pid=child.pid, log=str(log),
                       epochs_per_arm=4, total_training_epochs=8)
                while True:
                    try:
                        returncode = child.wait(timeout=POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        status(stage, state="running", child_pid=child.pid, log=str(log),
                               epochs_per_arm=4, total_training_epochs=8)
                if returncode:
                    raise RuntimeError(f"{stage} exited {returncode}; inspect {log}")
                child = None
        summary_path = ROOT / "hse_sam_v7_results" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("state") != "complete" or summary.get("submission_rows") != 7065:
            raise RuntimeError("Final HSE AdamW/SAM V7 export is incomplete")
        status("complete", state="complete", summary=str(summary_path),
               elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        try:
            stop_child_tree(child)
        finally:
            status("error", state="failed", error=str(error), elapsed_seconds=time.monotonic() - started)
            traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()
