"""Run the separate four-epoch LoRA refinement and validation-selected TTA fusion."""
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def status(stage, **extra):
    data = {"stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS / "refinement_status.json", data)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def run_stage(name, script, args):
    log_path = LOGS / f"refinement_{name}.log"
    command = [sys.executable, "-u", str(ROOT / "src" / script), *args]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nSTART " + datetime.now().astimezone().isoformat() + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        status(name, state="running", child_pid=child.pid, log=str(log_path))
        while child.poll() is None:
            time.sleep(15)
            if child.poll() is None:
                status(name, state="running", child_pid=child.pid, log=str(log_path))
        return_code = child.returncode
    if return_code:
        raise RuntimeError(f"{name} exited {return_code}; inspect {log_path}")


def make_plan():
    sources = ["train_lora.py", "refine_lora.py", "train_classifier.py", "train_mlp.py",
               "cnn_candidate.py", "finalize_improvements.py", "finalize_refinement.py", "run_refinement.py"]
    initial = ROOT / "experiments" / "lora" / "best.pt"
    previous_submission = ROOT / "improved_results" / "submission.csv"
    if sha(initial) != "4380488d2dc85ca7a9f2bdae673721cf5f1f28584ab0237cf8decad0c0f0fa91":
        raise RuntimeError("The expected epoch-6 starting checkpoint changed")
    if sha(previous_submission) != "fee0cdb9dbe2e5ed2cbe181162760c8482173d883ef805b8ef90d0bf89fe6b33":
        raise RuntimeError("The preserved first-round submission changed")
    return {
        "stage": "four_epoch_refinement", "classes": ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "initial_checkpoint": str(initial), "initial_checkpoint_sha256": sha(initial),
        "previous_submission_sha256": sha(previous_submission),
        "manifest_sha256": sha(ROOT / "features" / "manifest.csv"),
        "unlabeled_sha256": sha(ROOT / "features" / "unlabeled.csv"),
        "epochs": 4, "lr_lora": 1e-5, "lr_head": 5e-5, "weight_decay": .001,
        "batch_size": 4, "accumulation": 8, "workers": 2, "ema_decay": .995,
        "selection": "validation accuracy, then macro_f1, then lower log loss",
        "tta": "equal arithmetic mean of original and horizontal-flip softmax probabilities",
        "candidates": ["old_lora_0.5_cnn_0.5_original", "old_lora_0.5_cnn_0.5_tta",
                       "refined_lora_0.5_cnn_0.5_tta", "refined_lora_0.45_cnn_0.45_mlp_0.10_tta"],
        "test_used_for_selection": False, "external_answers_used": False,
        "internal_test_has_been_previously_evaluated": True,
        "source_sha256": {name: sha(ROOT / "src" / name) for name in sources},
    }


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / "refinement.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        lock.close()
        raise RuntimeError("A refinement pipeline is already running") from exc
    started = time.monotonic()
    try:
        plan = make_plan()
        plan_path = LOGS / "refinement_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise RuntimeError("Refinement source/config/data changed; use a separate experiment for a new plan")
        else:
            write_json(plan_path, plan)
        output = ROOT / "experiments" / "lora_refinement"
        args = ["--project-root", str(ROOT), "--output", str(output),
                "--origin-checkpoint", str(ROOT / "experiments" / "lora" / "best.pt"),
                "--epochs", "4", "--batch-size", "4", "--accumulation", "8", "--workers", "2",
                "--lr-lora", "1e-5", "--lr-head", "5e-5", "--weight-decay", "0.001", "--ema-decay", "0.995"]
        if (output / "last.pt").exists():
            args.append("--resume")
        run_stage("train", "refine_lora.py", args)
        run_stage("finalize", "finalize_refinement.py", ["--project-root", str(ROOT)])
        summary_path = ROOT / "refinement_results" / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Finalizer returned without a summary")
        if not (ROOT / "refinement_results" / "submission.csv").is_file():
            raise RuntimeError("Finalizer returned without a submission")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if sha(ROOT / "improved_results" / "submission.csv") != plan["previous_submission_sha256"]:
            raise RuntimeError("First-round submission was unexpectedly modified")
        status("complete", state="complete", elapsed_seconds=time.monotonic() - started,
               selected=summary.get("selected"), summary=str(summary_path),
               submission=str(ROOT / "refinement_results" / "submission.csv"))
    except BaseException as exc:
        status("error", state="failed", error=str(exc), elapsed_seconds=time.monotonic() - started)
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()
