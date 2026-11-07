"""CPU integration regression with explicitly synthetic OOF predictions.

No training, outer-validation access, GPU, or production selection artifacts.
Exercise the real OOF assembler/writer/lock and independent outer-permission gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import prepare_hse_groupcv_v11 as p
import train_hse_groupcv_v11 as t
import finalize_hse_groupcv_v11 as f


def main():
    plan = p.verify_plan(verify_images=False)
    materialized = p._data_rows(p.read(p.DATA / "summary.json"))
    rows = p.pure(materialized)
    folded = {fold: (p.pure([r for r in materialized if r["fold"] != fold]),
                     p.pure([r for r in materialized if r["fold"] == fold])) for fold in range(3)}
    report = {"state": "passed", "GPU_used": False, "training_started": False,
              "predictions": "synthetic fixtures, not model performance",
              "outer_images_or_labels_read": False, "internal_test_read": False,
              "plan_sha256": p.h.sha256(p.PLAN), "tests": []}
    assert f.CV_RULE == p.CV_RULE == plan["cv_selection_rule"]
    config = t.config_for(ROOT, plan, 0, "full")
    model = nn.Linear(2, 2)
    assert t.model_payload(model, config, 12, "ema")["purpose"] == "fixed_final_epoch"
    assert t.model_payload(model, config, 0, "raw")["purpose"] == "training_checkpoint"
    report["tests"].append("trainer_final_payload_matches_finalizer_purpose")
    originals = (p.verify_plan, p.records_for, p.TRAIN, f.TRAIN, f.SELECTION, f.SELECTION_LOCK,
                 f.initial_reference, f.verify_run)
    try:
        with tempfile.TemporaryDirectory(prefix="v11_integration_", dir=ROOT / "research") as directory:
            temp = Path(directory).resolve()
            assert temp.parent == (ROOT / "research").resolve()
            p.TRAIN = f.TRAIN = temp
            f.SELECTION = temp / "cv_selection.json"
            f.SELECTION_LOCK = temp / "cv_selection_lock.json"
            p.verify_plan = lambda *args, **kwargs: plan

            def records(_plan, fold, root=ROOT, outer=False, verify_images=False):
                if outer:
                    raise AssertionError("Synthetic CV test must never request outer data")
                return (rows, []) if fold is None else folded[fold]

            p.records_for = records
            f.initial_reference = lambda _plan: ({}, set(), set())

            def synthetic_run(_plan, phase, mode, fold, pins, initial, parameter_names, head_names):
                assert phase == "cv"
                train, heldout = folded[fold]
                targets = np.asarray([r["target"] for r in heldout])
                labels = targets.copy()
                if mode == "full" and fold < 2:
                    labels[0] = (labels[0] + 1) % 6
                probabilities = np.full((len(labels), 6), .01, dtype=np.float32)
                probabilities[np.arange(len(labels)), labels] = .95
                folder = f.run_folder("cv", mode, fold)
                folder.mkdir(parents=True, exist_ok=True)
                for view in ("original", "flip"):
                    path = folder / f"{view}.npy"
                    np.save(path, probabilities, allow_pickle=False)
                    f.pin(pins, path)
                return [probabilities.copy(), probabilities.copy()], {
                    "run_validation_indices": [r["manifest_index"] for r in heldout],
                    "run_train_indices": [r["manifest_index"] for r in train],
                    "single_tta_metrics": t.metrics(probabilities, heldout)}

            f.verify_run = synthetic_run
            selection = f.select_cv()
            assert selection["selected_mode"] == "fc_only" and selection["fc_only_fold_wins"] == 2
            assert selection["pooled_oof_metrics"]["fc_only"]["correct_count"] == 19120
            assert selection["pooled_oof_metrics"]["full"]["correct_count"] == 19118
            for mode in ("full", "fc_only"):
                path = temp / f"oof_{mode}.npy"
                assert selection["producer_files"][str(path)] == p.h.sha256(path)
            assert p._outer_permission(plan, materialized) == p.h.sha256(f.SELECTION)
            report["tests"] += ["real_select_cv_assembles_each19120_member_once", "actual_oof_producer_bindings",
                                "actual_selection_lock_matches_independent_outer_permission"]
            original_lock = f.SELECTION_LOCK.read_bytes()
            lock = p.read(f.SELECTION_LOCK)
            lock["cv_selection_sha256"] = "0" * 64
            p.h.write_json(f.SELECTION_LOCK, lock)
            try:
                p._outer_permission(plan, materialized)
            except ValueError:
                report["tests"].append("tampered_selection_lock_rejected")
            else:
                raise AssertionError("Tampered lock accepted")
            f.SELECTION_LOCK.write_bytes(original_lock)
            # Protocol tie default: two fold wins cannot override a pooled tie.
            pooled = {"full": {"correct_count": 10}, "fc_only": {"correct_count": 10}}
            fold_metrics = {"full": {str(i): {"correct_count": 3} for i in range(3)},
                            "fc_only": {str(i): {"correct_count": 4 if i < 2 else 2} for i in range(3)}}
            assert f.cv_decision(pooled, fold_metrics)[0] == "full"
            pooled["fc_only"]["correct_count"] = 11
            assert f.cv_decision(pooled, fold_metrics)[0] == "fc_only"
            fold_metrics["fc_only"]["1"]["correct_count"] = 3
            assert f.cv_decision(pooled, fold_metrics)[0] == "full"
            report["tests"].append("pooled_tie_and_fewer_than_two_wins_keep_full")
    finally:
        (p.verify_plan, p.records_for, p.TRAIN, f.TRAIN, f.SELECTION, f.SELECTION_LOCK,
         f.initial_reference, f.verify_run) = originals
    report["source_pins"] = plan["source_pins"]
    report["test_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = ROOT / "research/hse_groupcv_v11_integration_20261018/cpu_integration.json"
    p.h.write_json(output, report)
    print(json.dumps({k: report[k] for k in ("state", "GPU_used", "tests")}), flush=True)


if __name__ == "__main__":
    main()
