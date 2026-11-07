"""Train-only grouped comparison of last-layer and last-four-layer DINOv2 readouts."""
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
LOGS = ROOT / "logs"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def status(stage, **extra):
    value = {"stage": stage, "pid": os.getpid(),
             "updated_at": datetime.now().astimezone().isoformat(), **extra}
    write_json(LOGS / "multilayer_status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_stage(name, script, args):
    log_path = LOGS / f"multilayer_{name}.log"
    command = [sys.executable, "-u", str(ROOT / "src" / script), *args]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\nSTART " + datetime.now().astimezone().isoformat() + "\n")
        stream.flush()
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
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
    baseline = ROOT / "refinement_results" / "submission.csv"
    expected = "f662aa9718056f236c8c9be41f27bde9219cb1be5f7fc6ac22df6fb07c5b465d"
    if sha(baseline) != expected:
        raise RuntimeError("The completed refinement submission changed")
    sources = ["group_training_images.py", "extract_multilayer_features.py", "evaluate_multilayer_cv.py",
               "run_multilayer_experiment.py", "train_lora.py", "train_classifier.py", "extract_features.py"]
    return {
        "experiment": "frozen_dinov2_last_four_layers_grouped_cv", "version": 1,
        "classes": ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        "source_split": "train", "expected_images": 21248,
        "manifest_sha256": sha(ROOT / "features" / "manifest.csv"),
        "preserved_submission_sha256": expected,
        "grouping": {"phash_bits": 64, "max_hamming": 4, "max_pixel_mae_0_255": 8,
                     "min_centered_correlation": .995, "consider_horizontal_flip": True,
                     "labels_used_for_pair_decision": False, "subject_independence_known": False},
        "folds": {"method": "StratifiedGroupKFold", "n_splits": 3, "shuffle": True, "seed": 42},
        "features": {"model": "official_unadapted_dinov2_vitb14", "layers_zero_based": [8, 9, 10, 11],
                     "readout": "four normalized CLS tokens, then mean normalized last-layer patches",
                     "dimension": 3840, "last_layer_baseline_slice": [2304, 3840],
                     "frozen": True, "batch_size": 16, "workers": 2, "storage_dtype": "float32",
                     "views": ["original", "horizontal_flip"], "image_size": 224},
        "classifiers": {"candidates": ["last_layer_linear", "last4_linear", "last_layer_mlp", "last4_mlp"],
                        "epochs": 12, "batch_size": 256, "lr": .001, "weight_decay": .001,
                        "mlp_hidden": 256, "dropout": .3, "seed": 42,
                        "standardization": "fit only each fold's training original-view features",
                        "fold_validation": "two-view average only after fixed final epoch",
                        "early_stopping_on_validation_fold": False},
        "existing_val_images_used": False, "existing_test_images_used": False,
        "competition_images_used": False, "external_answers_used": False,
        "new_submission_produced_by_this_comparison": False,
        "source_sha256": {name: sha(ROOT / "src" / name) for name in sources},
    }


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (LOGS / "multilayer.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        lock.close()
        raise RuntimeError("A multilayer experiment is already running") from exc
    started = time.monotonic()
    try:
        plan = make_plan()
        plan_path = LOGS / "multilayer_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise RuntimeError("Experiment sources, data or plan changed; preserve the previous run")
        else:
            write_json(plan_path, plan)
        run_stage("groups", "group_training_images.py", ["--project-root", str(ROOT)])
        run_stage("features", "extract_multilayer_features.py", ["--project-root", str(ROOT),
                                                               "--batch-size", "16", "--workers", "2"])
        run_stage("cv", "evaluate_multilayer_cv.py", ["--project-root", str(ROOT)])
        summary_path = ROOT / "multilayer_results" / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Cross-validation returned without a summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("state") != "complete" or summary.get("train_rows") != plan["expected_images"]
                or summary.get("folds") != 3 or summary.get("epochs_per_fold") != 12
                or len(summary.get("candidates", [])) != 4):
            raise RuntimeError("Cross-validation summary is incomplete or differs from the fixed plan")
        if sha(ROOT / "refinement_results" / "submission.csv") != plan["preserved_submission_sha256"]:
            raise RuntimeError("Previous submission changed during the experiment")
        status("complete", state="complete", elapsed_seconds=time.monotonic() - started,
               summary=str(summary_path))
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
