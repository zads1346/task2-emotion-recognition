"""Actual offline GPU verification on competition test images only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/kaggle_expression"
BUNDLE = OUT / "expression_bundle"
AUDIT = ROOT / "logs/kaggle_packaging_20261018"


def load_runtime():
    spec = importlib.util.spec_from_file_location("portable", BUNDLE / "inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records():
    with (ROOT/"features/unlabeled.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cache(branch, view):
    if branch == "hse":
        return ROOT/f"hse_results/prediction_cache/hse_unlabeled/{view}.npy"
    if branch == "lora":
        return ROOT/f"lora_task_v4_results/lora_competition_{view}.npy"
    if branch == "ddam":
        return ROOT/f"ddamfn_v8_results/ddamfn_competition_{view}.npy"
    return ROOT/f"fusion_results/prediction_cache/new_mlp/{view}.npy"


def smoke():
    r = load_runtime()
    r.verify_bundle(BUNDLE)
    r.configure(BUNDLE)
    rows = records()
    paths = [Path(row["path"]) for row in rows[:16]]
    report = {"state": "passed", "images": 16, "branches": {}}
    fusion = np.zeros((16, 6))
    cached_fusion = np.zeros((7065, 6))
    for branch, weight in r.WEIGHTS.items():
        a, b = r.predict_branch(BUNDLE, branch, paths)
        item = {}
        for view, value in zip(("original", "flip"), (a, b)):
            expected = r.probability(np.load(cache(branch, view), allow_pickle=False))
            delta = float(np.abs(expected[:16] - value).max())
            item[view] = {"max_absolute_probability_difference": delta,
                          "argmax_differences": int(np.sum(expected[:16].argmax(1) != value.argmax(1)))}
            if delta > .01:
                raise ValueError(f"Portable {branch} differs materially from reference: {item}")
            cached_fusion += weight * expected / 2
        report["branches"][branch] = item
        fusion += weight * (a+b) / 2
        print(branch, item, flush=True)
    expected = r.probability(np.load(ROOT/"ddamfn_v8_results/probabilities.npy", allow_pickle=False))
    report["cached_four_branch_reconstruction_max_error"] = float(np.abs(cached_fusion - expected).max())
    report["cached_four_branch_argmax_differences"] = int(np.sum(cached_fusion.argmax(1) != expected.argmax(1)))
    report["smoke_ensemble_argmax_differences"] = int(np.sum(fusion.argmax(1) != expected[:16].argmax(1)))
    assert report["cached_four_branch_argmax_differences"] == 0
    assert report["smoke_ensemble_argmax_differences"] == 0
    AUDIT.mkdir(exist_ok=True)
    (AUDIT/"gpu_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def execute_notebook():
    from nbclient import NotebookClient

    class LoggingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg["msg_type"] == "stream":
                print(msg["content"]["text"], end="", flush=True)
            return super().process_message(msg, cell, cell_index)

    audit_input = AUDIT/"input/weights"
    audit_input.mkdir(parents=True, exist_ok=True)
    zip_path = OUT/"expression_pretrained_weights.zip"
    target_zip = audit_input/zip_path.name
    if target_zip.exists():
        if os.path.samefile(target_zip, zip_path):
            pass
        else:
            target_zip.unlink()
            os.link(zip_path, target_zip)
    else:
        os.link(zip_path, target_zip)
    working = AUDIT/"notebook_working"
    working.mkdir(exist_ok=True)
    rows = records()
    os.environ.update(EXPRESSION_INPUT_ROOT=str(AUDIT/"input"), EXPRESSION_WORKING=str(working),
                      EXPRESSION_TEST_DIR=str(Path(rows[0]["path"]).parent))
    # An isolated kernel spec points to this verified project interpreter, not a global Python.
    jupyter_data = AUDIT/"jupyter"
    kernel_dir = jupyter_data/"kernels/expression-offline"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir/"kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Expression offline verification", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(jupyter_data)
    os.environ["PYTHONUTF8"] = "1"
    nb = nbformat.read(OUT/"expression_kaggle.ipynb", as_version=4)
    nbformat.validate(nb)
    client = LoggingClient(nb, timeout=3600, kernel_name="expression-offline", resources={"metadata": {"path": str(working)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, AUDIT/"expression_kaggle_executed.ipynb")
    r = load_runtime()
    with (working/"submission.csv").open(encoding="utf-8", newline="") as f:
        actual = list(csv.DictReader(f))
    with (ROOT/"ddamfn_v8_results/submission.csv").open(encoding="utf-8", newline="") as f:
        expected = list(csv.DictReader(f))
    p = np.load(working/"probabilities.npy", allow_pickle=False)
    assert len(actual) == 7065 and [x["ID"] for x in actual] == [x["ID"] for x in expected]
    assert [int(x["Emotion"]) for x in actual] == p.argmax(1).tolist()
    mismatches = sum(a["Emotion"] != b["Emotion"] for a, b in zip(actual, expected))
    branch_differences = {}
    for branch in r.WEIGHTS:
        for view in ("original", "flip"):
            a = np.load(working/f"{branch}_{view}.npy", allow_pickle=False)
            b = r.probability(np.load(cache(branch, view), allow_pickle=False))
            branch_differences[f"{branch}_{view}"] = {"max_absolute_difference": float(np.abs(a-b).max()),
                "argmax_differences": int(np.sum(a.argmax(1) != b.argmax(1)))}
    report = {"state": "passed", "executed_original_notebook_all_cells": True,
              "weights_loaded_from_zip_offline": True, "test_images": len(actual),
              "columns": ["ID", "Emotion"], "id_order_exact": True, "argmax_exact": True,
              "reference_submission_label_differences": mismatches,
              "reference_submission_byte_identical": r.sha256(working/"submission.csv") == r.sha256(ROOT/"ddamfn_v8_results/submission.csv"),
              "branch_reference_comparisons": branch_differences,
              "blur_probe": r.read_json(working/"blur_stability.json"),
              "run_receipt": r.read_json(working/"run_receipt.json"),
              "notebook_sha256": r.sha256(OUT/"expression_kaggle.ipynb"),
              "zip_sha256": r.sha256(zip_path), "local_gpu_execution_only": True,
              "kaggle_cloud_execution_performed": False, "no_kaggle_upload": True}
    if mismatches:
        report["state"] = "numerical_differences_require_review"
    (OUT/"verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "notebook"], required=True)
    args = parser.parse_args()
    smoke() if args.stage == "smoke" else execute_notebook()
