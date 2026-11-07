"""CPU-only preparation and binding of the matched four-epoch HSE V7 experiment.

This module never starts training or allocates a CUDA model. GPU smoke reports
are produced independently by the authorized operator before the plan is locked.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import torch

import train_hse_specialist as h
import train_hse_sam_v7 as t
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(t.SETTINGS)
MODE_SETTINGS = dict(t.MODE_SETTINGS)
EXCLUDED = [7727, 10448]
PARENT_SHA = "ed58a57d691943dc605053ac6182d6b54f05f4ae4133690d65ce5abeb10d20b7"
BASELINE_SHA = "cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd"
SOURCES = ["train_hse_sam_v7.py", "prepare_hse_sam_v7.py", "finalize_hse_sam_v7.py",
           "run_hse_sam_v7.py", "hse_sam_v7_worker.py", "verify_hse_sam_v7_math.py",
           "smoke_hse_sam_v7.py", "train_hse_specialist.py", "hse_model.py", "validate_cosine_fusion.py"]


def _pin(pins, path, expected=None):
    path = Path(path).resolve()
    digest = h.sha256(path) if expected is None else expected
    h.check_hash(path, digest)
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError("Conflicting source checksum")
    pins[str(path)] = digest


def verify_plan(root=ROOT):
    root = Path(root).resolve()
    plan = t.read_json(root/"logs/hse_sam_v7_plan.json")
    if (plan.get("format") != "hse_sam_v7_plan" or plan.get("classes") != h.CLASSES
            or plan.get("settings") != SETTINGS or plan.get("mode_settings") != MODE_SETTINGS
            or plan.get("baseline_correct_count") != 1915 or plan.get("parent_checkpoint_sha256") != PARENT_SHA
            or plan.get("baseline_submission_sha256") != BASELINE_SHA
            or plan.get("internal_test_read") is not False or plan.get("external_answers_used") is not False):
        raise ValueError("Locked V7 settings/classes/parent/incumbent differ")
    if (len(plan["original_train_indices"]) != 21248 or plan["excluded_train_indices"] != EXCLUDED
            or plan["train_indices"] != [i for i in plan["original_train_indices"] if i not in EXCLUDED]
            or len(plan["train_indices"]) != 21246 or len(plan["val_indices"]) != 2656):
        raise ValueError("Locked V7 audited membership differs")
    c.verify_pins(root, plan["source_pins"])
    c.verify_pins(root, plan["baseline_files"])
    h.check_hash(plan["parent_checkpoint"], PARENT_SHA)
    h.check_hash(root/"experiments/lora_flip_consistency_v6/audit/summary.json", plan["audit_summary_sha256"])
    return plan


def prepare(root=ROOT):
    root = Path(root).resolve()
    plan_path = root/"logs/hse_sam_v7_plan.json"
    if plan_path.exists():
        return verify_plan(root)
    audit_path = root/"experiments/lora_flip_consistency_v6/audit/summary.json"
    audit = t.read_json(audit_path)
    if (audit.get("state") != "complete" or audit.get("classes") != h.CLASSES
            or audit.get("sample_rows") != 192 or audit.get("train_rows") != 21248
            or audit.get("effective_train_rows") != 21246 or audit.get("excluded_train_indices") != EXCLUDED
            or audit.get("training_membership_changed") is not True):
        raise ValueError("The fixed train-only192-image review/two exclusions are not complete")
    if any(audit.get(key) is not False for key in
           ("labels_changed", "images_changed", "weights_changed", "internal_test_read", "external_answers_used")):
        raise ValueError("V7 cannot relabel/edit/reweight images or use internal-test/external answers")
    original, receipt = h.load_trainval(root)
    original_indices = [row["manifest_index"] for row in original["train"]]
    train_rows = [row for row in original["train"] if row["manifest_index"] not in EXCLUDED]
    filtered = h.signature(train_rows)
    if (original_indices != audit["original_train_indices"]
            or receipt["train_rows_sha256"] != audit["train_membership_sha256"]
            or filtered != audit["filtered_membership_sha256"]
            or [row["manifest_index"] for row in train_rows] != audit["train_indices"]
            or [row["manifest_index"] for row in original["val"]] != audit["val_indices"]):
        raise ValueError("Input receipt or audited membership differs")
    parent_path = root/"experiments/hse_specialist_v1/best.pt"
    h.check_hash(parent_path, PARENT_SHA)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    provenance = h.model_api().source_receipt(root)
    if (parent.get("best_epoch") != 7 or parent.get("classes") != h.CLASSES
            or parent.get("model_provenance") != provenance or parent.get("input_sha256") != receipt
            or parent.get("train_indices") != original_indices or parent.get("val_indices") != audit["val_indices"]):
        raise ValueError("HSE parent must be the exact original epoch7 model and original audited data")
    baseline_folder = root/"lora_task_v4_results"
    h.check_hash(baseline_folder/"submission.csv", BASELINE_SHA)
    current = np.load(baseline_folder/"selected_val_probabilities.npy", allow_pickle=False)
    targets = np.asarray([row["target"] for row in original["val"]])
    score = c.metrics(targets, current)
    if score["correct_count"] != 1915:
        raise ValueError("Incumbent development validation is no longer1915/2656")
    report_path = root/"logs/hse_sam_v7_math_verification.json"
    math_report = t.read_json(report_path)
    if (math_report.get("state") != "passed" or math_report.get("trainer_sha256") != h.sha256(root/"src/train_hse_sam_v7.py")
            or math_report.get("verifier_sha256") != h.sha256(root/"src/verify_hse_sam_v7_math.py")):
        raise ValueError("Run the independent CPU math verifier against the completed trainer before locking V7")
    smoke_path = root/"logs/hse_sam_v7_gpu_smoke.json"
    gpu_report = t.read_json(smoke_path)
    if (gpu_report.get("state") not in ("passed", "complete")
            or gpu_report.get("trainer_sha256") != h.sha256(root/"src/train_hse_sam_v7.py")
            or gpu_report.get("smoke_source_sha256") != h.sha256(root/"src/smoke_hse_sam_v7.py")
            or gpu_report.get("microbatch") != 16 or gpu_report.get("effective_batch") != 32
            or gpu_report.get("parent_sha256") != PARENT_SHA or gpu_report.get("internal_test_read") is not False):
        raise ValueError("An independently authorized passing GPU smoke is required before plan lock")
    pins = {}
    for name in SOURCES:
        _pin(pins, root/"src"/name)
    for name in ("start_hse_sam_v7.ps1", "run_hse_sam_v7_worker.ps1", "check_hse_sam_v7.ps1"):
        _pin(pins, root/name)
    for path in (audit_path, report_path, smoke_path, parent_path):
        _pin(pins, path)
    for path, digest in audit["source_pins"].items():
        _pin(pins, path, digest)
    for relative, digest in audit["file_sha256"].items():
        child = Path(relative)
        if child.is_absolute() or ".." in child.parts:
            raise ValueError("Audit artifact escapes its directory")
        _pin(pins, audit_path.parent/child, digest)
    baseline_files = {}
    for folder, names in ((baseline_folder, ("summary.json", "selected_recipe.json", "selected_val_probabilities.npy", "submission.csv")),
                          (root/"experiments/hse_specialist_v1", ("best.pt", "val_original.npy", "val_flip.npy")),
                          (root/"hse_results", ("selected_recipe.json", "selected_val_probabilities.npy", "submission.csv"))):
        for name in names:
            _pin(baseline_files, folder/name)
    for relative in ("lora_zero_weight_v5_results/submission.csv", "lora_flip_consistency_v6_results/submission.csv"):
        _pin(baseline_files, root/relative, BASELINE_SHA)
    _pin(baseline_files, root/"fusion_results/submission.csv", "2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad")
    plan = {"format": "hse_sam_v7_plan", "state": "locked", "classes": list(h.CLASSES),
            "settings": SETTINGS, "mode_settings": MODE_SETTINGS,
            "source_pins": pins, "baseline_files": baseline_files, "input_receipt": receipt,
            "original_train_indices": original_indices, "excluded_train_indices": EXCLUDED,
            "train_indices": [row["manifest_index"] for row in train_rows], "val_indices": audit["val_indices"],
            "filtered_membership_sha256": filtered, "audit_summary_sha256": h.sha256(audit_path),
            "parent_checkpoint": str(parent_path), "parent_checkpoint_sha256": PARENT_SHA,
            "model_provenance": provenance, "baseline_submission_sha256": BASELINE_SHA,
            "baseline_correct_count": 1915, "baseline_metrics": score,
            "math_verification_sha256": h.sha256(report_path), "gpu_smoke_sha256": h.sha256(smoke_path),
            "gpu_smoke_batch_size": 16, "training_membership_changed": True,
            "internal_test_read": False, "external_answers_used": False,
            "selection_policy": "fixed candidates only; replace incumbent HSE50% branch; adopt strictly above1915 correct; incumbent wins ties"}
    h.write_json(plan_path, plan)
    print(json.dumps({"stage": "prepare", "state": "locked", "train_rows": 21246,
                      "val_rows": 2656, "batch_size": 16, "effective_batch_size": 32,
                      "epochs_per_arm": 4, "GPU_used_by_preparation": False}), flush=True)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.project_root)
