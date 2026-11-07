"""Single-GPU, complete-epoch resumable HSE training-pool grouped comparison."""
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
PREFIX = "hse_groupcv_v11"
REQUIRED = (
    "src/prepare_hse_groupcv_v11.py", "src/train_hse_groupcv_v11.py",
    "src/finalize_hse_groupcv_v11.py", "src/run_hse_groupcv_v11.py",
    "src/hse_groupcv_v11_worker.py", "start_hse_groupcv_v11.ps1",
    "run_hse_groupcv_v11_worker.ps1", "src/run_hse_sam_v7.py",
    "src/hse_model.py", "src/train_expanded_validation_v10.py",
    "src/train_hse_sam_v7.py", "src/train_hse_specialist.py", "src/validate_cosine_fusion.py",
)
guard.GPU_SCRIPT_NAMES += (
    "prepare_ddamfn_groupcv_v9", "train_ddamfn_groupcv_v9", "finalize_ddamfn_groupcv_v9",
    "run_ddamfn_groupcv_v9", "ddamfn_groupcv_v9_worker", "audit_hse_sam_v7_epoch0",
    "finalize_hse_sam_v7_recovered", "prepare_hse_expanded_validation_v10",
    "train_expanded_validation_v10", "finalize_expanded_validation_v10",
    "run_hse_expanded_validation_v10", "hse_expanded_validation_v10_worker",
    "prepare_hse_groupcv_v11", "train_hse_groupcv_v11", "finalize_hse_groupcv_v11",
    "run_hse_groupcv_v11", "hse_groupcv_v11_worker",
    "verify_kaggle_delivery", "kaggle_expression_runtime",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(stage, **values):
    value = {"stage": stage, "pid": os.getpid(),
             "updated_at": datetime.now().astimezone().isoformat(), **values}
    guard.write_json(LOGS / f"{PREFIX}_status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def verify_review_and_smoke():
    plan_path = LOGS / f"{PREFIX}_plan.json"
    plan = read(plan_path)
    pins = plan.get("source_pins")
    if not isinstance(pins, dict):
        raise ValueError("V11 plan lacks source and input bindings")
    resolved = {str((ROOT / path).resolve()).casefold() for path in pins}
    if not {str((ROOT / path).resolve()).casefold() for path in REQUIRED}.issubset(resolved):
        raise ValueError("V11 plan omits a production source")
    for path, digest in pins.items():
        if sha(ROOT / path) != digest:
            raise ValueError("V11 locked source/input changed: " + path)
    plan_sha = sha(plan_path)
    for name in ("review", "gpu_smoke"):
        receipt = read(LOGS / f"{PREFIX}_{name}.json")
        if (receipt.get("state") != "passed" or receipt.get("source_pins") != pins
                or receipt.get("plan_sha256") != plan_sha):
            raise ValueError("V11 requires a passing implementation review and GPU smoke for exact sources")
        if name == "review":
            if receipt.get("implementation_review") is not True or receipt.get("unresolved_blockers") != []:
                raise ValueError("A protocol review alone cannot authorize V11 launch")
            continue
        required = {"GPU_used": True, "weights_saved": False, "validation_evaluated": False,
                    "internal_test_read": False, "external_answers_used": False,
                    "batch_size": 16, "effective_batch_size": 32,
                    "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
                    "model_source_receipt_sha256": plan["model_source_receipt_sha256"],
                    "training_manifest_indices": plan["folds"][0]["train_indices"][:64]}
        if any(receipt.get(key) != value for key, value in required.items()):
            raise ValueError("V11 smoke does not establish real training-only official-initialized GPU updates")
        if set(receipt.get("modes", {})) != {"full", "fc_only"}:
            raise ValueError("Both V11 arms must pass the actual GPU smoke")
        head = {"classifier.weight", "classifier.bias"}
        for mode, probe in receipt["modes"].items():
            expected = {"state": "passed", "successful_optimizer_steps": 2, "amp_skipped_steps": 0,
                        "ema_updates": 2, "head_updated": True, "frozen_state_unchanged": True,
                        "batchnorm_buffers_frozen": True}
            if any(probe.get(key) != value for key, value in expected.items()):
                raise ValueError("V11 smoke lacks successful updates or frozen-state invariance: " + mode)
            names = probe.get("trainable_parameter_names", [])
            steps = probe.get("all_parameter_steps", {})
            changed = set(probe.get("changed_state_keys", []))
            if (not names or set(steps) != set(names) or any(step != 2 for step in steps.values())
                    or not head.issubset(changed) or not isinstance(probe.get("dropout_configuration"), dict)
                    or not isinstance(probe.get("peak_allocated_bytes"), int) or probe["peak_allocated_bytes"] <= 0):
                raise ValueError("V11 smoke lacks exact AdamW update/dropout/GPU evidence")
            if mode == "fc_only" and (set(names) != head or changed != head):
                raise ValueError("V11 fc_only changed something outside the classifier")
            if mode == "full" and (set(names) == head or not (changed - head)):
                raise ValueError("V11 full backbone did not update")
    return plan


def cv_selection():
    # The finalizer verifies its hash and all producer bindings before any refit.
    from finalize_hse_groupcv_v11 import verify_cv_selection
    return verify_cv_selection()


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / f"{PREFIX}_experiment.lock").open("a+b")
    lock.seek(0, 2)
    if not lock.tell():
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    child, started = None, time.monotonic()
    try:
        verify_review_and_smoke()
        stages = [("prepare", "prepare_hse_groupcv_v11.py", ["--stage", "verify"])]
        for fold in range(3):
            for mode in ("full", "fc_only"):
                stages.append((f"fold_{fold}_{mode}", "train_hse_groupcv_v11.py",
                               ["--stage", "train", "--fold", str(fold), "--mode", mode, "--device", "cuda"]))
        stages.extend([
            ("select_cv", "finalize_hse_groupcv_v11.py", ["--stage", "select_cv"]),
            ("refit", "train_hse_groupcv_v11.py", ["--stage", "train", "--refit", "--mode", "fc_only", "--device", "cuda"]),
            ("select", "finalize_hse_groupcv_v11.py", ["--stage", "select"]),
            ("export", "finalize_hse_groupcv_v11.py", ["--stage", "export"]),
            ("verify", "finalize_hse_groupcv_v11.py", ["--stage", "verify"]),
        ])
        for stage, script, arguments in stages:
            verify_review_and_smoke()
            guard.no_other_gpu_task()
            if stage == "refit":
                selection = cv_selection()
                if selection["selected_mode"] == "full":
                    status("refit_skipped", state="running", reason="Full selected; preserve exact-recipe verified V10")
                    continue
                if selection["selected_mode"] != "fc_only":
                    raise ValueError("Unexpected locked CV selection")
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
                    status(stage, state="running", child_pid=child.pid, log=str(log),
                           epochs_per_run=12, cv_training_runs=6, cv_training_epochs=72,
                           maximum_epochs_with_conditional_refit=84)
                    try:
                        code = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                child = None
                if code:
                    raise RuntimeError(f"{stage} exited {code}; inspect {log}")
        summary_path = ROOT / "hse_groupcv_v11_results/summary.json"
        summary = read(summary_path)
        if summary.get("state") != "complete" or summary.get("submission_rows") != 7065:
            raise ValueError("V11 final delivery is incomplete")
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
