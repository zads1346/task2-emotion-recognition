"""Serial training-only grouped CV, locked selection, refit and one dev confirmation."""
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

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PREFIX = "ddamfn_groupcv_v9"
guard.GPU_SCRIPT_NAMES += (
    "prepare_ddamfn_groupcv_v9", "train_ddamfn_groupcv_v9",
    "finalize_ddamfn_groupcv_v9", "run_ddamfn_groupcv_v9", "ddamfn_groupcv_v9_worker",
    "audit_hse_sam_v7_epoch0", "finalize_hse_sam_v7_recovered",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(stage, **values):
    value = {"stage": stage, "pid": os.getpid(),
             "updated_at": datetime.now().astimezone().isoformat(), **values}
    guard.write_json(LOGS / (PREFIX + "_status.json"), value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify_review_and_smoke():
    # Review and smoke are produced explicitly after all sources are frozen.
    plan_path = LOGS / f"{PREFIX}_plan.json"
    plan = read(plan_path)
    plan_pins = plan.get("source_pins")
    required = ["src/prepare_ddamfn_groupcv_v9.py", "src/train_ddamfn_groupcv_v9.py",
                "src/finalize_ddamfn_groupcv_v9.py", "src/run_ddamfn_groupcv_v9.py",
                "src/ddamfn_groupcv_v9_worker.py", "start_ddamfn_groupcv_v9.ps1",
                "run_ddamfn_groupcv_v9_worker.ps1", "src/run_hse_sam_v7.py",
                "src/train_ddamfn_v8.py", "src/ddamfn_v8_model.py",
                "src/train_hse_sam_v7.py", "src/train_hse_specialist.py",
                "src/validate_cosine_fusion.py"]
    if not isinstance(plan_pins, dict):
        raise ValueError("V9 plan has no source pins")
    resolved_pins = {str((ROOT / p).resolve()).casefold(): digest for p, digest in plan_pins.items()}
    if any(str((ROOT / p).resolve()).casefold() not in resolved_pins for p in required):
        raise ValueError("V9 plan omits a required production source")
    for name in ("review", "gpu_smoke"):
        receipt = read(LOGS / f"{PREFIX}_{name}.json")
        if receipt.get("state") != "passed":
            raise RuntimeError(f"V9 {name} is not passed")
        pins = receipt.get("source_pins")
        if pins != plan_pins:
            raise ValueError(f"V9 {name} must bind exactly all plan sources")
        for path, digest in pins.items():
            location = Path(path)
            if not location.is_absolute():
                location = ROOT / location
            if hashlib.sha256(location.read_bytes()).hexdigest() != digest:
                raise ValueError(f"V9 {name} source differs: {path}")
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != receipt.get("plan_sha256"):
            raise ValueError(f"V9 {name} plan binding differs")
        if name == "gpu_smoke":
            if (receipt.get("GPU_used") is not True or receipt.get("weights_saved") is not False
                    or receipt.get("batch_size") != 32 or receipt.get("effective_batch_size") != 32
                    or receipt.get("development_validation_evaluated") is not False
                    or receipt.get("parent_checkpoint_sha256") != plan["parent_checkpoint_sha256"]
                    or receipt.get("model_source_receipt_sha256") != plan["model_source_receipt_sha256"]
                    or receipt.get("training_manifest_indices") != plan["folds"]["0"]["train_indices"][:64]
                    or set(receipt.get("modes", {})) != {"full", "fc_only"}):
                raise ValueError("V9 smoke must use the real GPU, fixed official model and training-only groups")
            for mode, probe in receipt["modes"].items():
                if (probe.get("state") != "passed" or probe.get("successful_optimizer_steps") != 2
                        or probe.get("amp_skipped_steps") != 0 or probe.get("ema_updates") != 2
                        or probe.get("fc_updated") is not True or probe.get("frozen_state_unchanged") is not True
                        or probe.get("batchnorm_buffers_frozen") is not True
                        or not isinstance(probe.get("peak_allocated_bytes"), int) or probe["peak_allocated_bytes"] <= 0):
                    raise ValueError("V9 smoke requires two successful matched GPU steps and frozen-state checks")
                changed = set(probe.get("changed_state_keys", []))
                if not {"fc.weight", "fc.bias"}.issubset(changed):
                    raise ValueError("V9 smoke did not update the full native head")
                if mode == "fc_only" and (changed != {"fc.weight", "fc.bias"}
                        or set(probe.get("trainable_parameter_names", [])) != {"fc.weight", "fc.bias"}):
                    raise ValueError("V9 fc_only smoke must leave every other state unchanged")
                if mode == "full" and (not any(key.startswith("features.") for key in changed)
                        or not any(key.startswith("cat_head") for key in changed)):
                    raise ValueError("V9 full smoke must update backbone and attention")


def main():
    LOGS.mkdir(exist_ok=True)
    baseline = read(ROOT / "ddamfn_v8_results/summary.json")
    if baseline.get("state") != "complete" or baseline.get("submission_rows") != 7065:
        raise RuntimeError("V8 export must be complete before V9")
    lock = (LOGS / (PREFIX + "_experiment.lock")).open("a+b")
    lock.seek(0, 2)
    if not lock.tell():
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    child, started = None, time.monotonic()
    try:
        verify_review_and_smoke()
        stages = [("prepare", "prepare_ddamfn_groupcv_v9.py", ["--stage", "prepare"])]
        for fold in range(3):
            for mode in ("full", "fc_only"):
                stages.append((f"fold_{fold}_{mode}", "train_ddamfn_groupcv_v9.py",
                               ["--phase", "cv", "--mode", mode, "--fold", str(fold), "--device", "cuda"]))
        stages += [
            ("select_cv", "finalize_ddamfn_groupcv_v9.py", ["--stage", "select_cv"]),
            ("refit", "train_ddamfn_groupcv_v9.py", ["--phase", "refit", "--device", "cuda"]),
            ("export", "finalize_ddamfn_groupcv_v9.py", ["--stage", "export"]),
        ]
        for stage, script, arguments in stages:
            verify_review_and_smoke()
            guard.no_other_gpu_task()
            log = LOGS / f"{PREFIX}_{stage}.log"
            with log.open("a", encoding="utf-8") as stream:
                stream.write("\nSTART " + datetime.now().astimezone().isoformat() + "\n"); stream.flush()
                child = subprocess.Popen(
                    [sys.executable, "-B", "-u", str(ROOT / "src" / script), "--project-root", str(ROOT), *arguments],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                         "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while True:
                    status(stage, state="running", child_pid=child.pid, log=str(log), epochs_per_run=6,
                           cv_training_runs=6, refit_training_runs=1, total_training_epochs=42)
                    try:
                        code = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                child = None
                if code:
                    raise RuntimeError(f"{stage} exited {code}; inspect {log}")
        summary_path = ROOT / "ddamfn_groupcv_v9_results/summary.json"
        summary = read(summary_path)
        if summary.get("state") != "complete" or summary.get("submission_rows") != 7065:
            raise ValueError("V9 verified export is incomplete")
        status("complete", state="complete", summary=str(summary_path), elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        guard.stop_child_tree(child)
        status("error", state="failed", error=str(error), elapsed_seconds=time.monotonic()-started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == "__main__":
    main()
