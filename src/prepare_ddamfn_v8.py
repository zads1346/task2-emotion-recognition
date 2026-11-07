"""Separate GPU smoke from CPU plan locking; require a completed V7 incumbent."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import train_hse_specialist as h
import train_hse_sam_v7 as math_v7
import train_ddamfn_v8 as trainer
import ddamfn_v8_model as model_api
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "logs/ddamfn_v8_plan.json"
SMOKE = ROOT / "logs/ddamfn_v8_gpu_smoke.json"
AUDIT = ROOT / "experiments/lora_flip_consistency_v6/audit/summary.json"
SOURCE_NAMES = ("ddamfn_v8_model.py", "train_ddamfn_v8.py", "prepare_ddamfn_v8.py", "finalize_ddamfn_v8.py",
                "run_ddamfn_v8.py", "ddamfn_v8_worker.py", "verify_ddamfn_v8_cpu.py", "train_hse_sam_v7.py", "run_hse_sam_v7.py",
                "train_hse_specialist.py", "validate_cosine_fusion.py")
IMAGE_SHA = "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA = "8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
WEIGHT = ROOT / "models/ddamfn_affectnet7_v8/affecnet7_epoch19_acc0.671.pth"
WEIGHT_SHA = "edf731a458162eef6b2be12c3572bd10cbe92ce8c009da4331184a2943d7ed14"


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Provenance files must stay inside the project")
    digest = h.sha256(path) if expected is None else expected
    h.check_hash(path, digest)
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError("Conflicting producer hashes")
    pins[str(path)] = digest


def smoke_bindings():
    return {str(ROOT / "src" / name): h.sha256(ROOT / "src" / name)
            for name in ("ddamfn_v8_model.py", "train_ddamfn_v8.py", "prepare_ddamfn_v8.py", "train_hse_sam_v7.py", "train_hse_specialist.py", "validate_cosine_fusion.py")}


def first_training_rows():
    """Read only training labels; the smoke opens just32 training image files."""
    selected = []
    manifest = ROOT / "features/manifest.csv"
    audit = c.read_json(ROOT / "data/split_audit.json")
    h.check_hash(manifest, audit["manifest_sha256"])
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("The audited training manifest has unexpected columns")
        for index, row in enumerate(reader):
            if row["split"] != "train" or index in (7727, 10448):
                continue
            if row["label"] not in trainer.CLASSES:
                raise ValueError("Invalid official training class")
            selected.append({"manifest_index": index, "path": row["path"], "label": row["label"], "target": trainer.CLASSES.index(row["label"])})
            if len(selected) == 32:
                break
    if len(selected) != 32:
        raise ValueError("A real32-image training smoke group is required")
    return selected


def verify_smoke(root=ROOT):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Fixed project root required")
    report = c.read_json(SMOKE)
    if report.get("state") != "passed" or report.get("successful_optimizer_steps") != 2 or report.get("amp_skipped_steps") != 0 or report.get("ema_updates") != 2:
        raise ValueError("A passing real two-step DDAM GPU smoke is required")
    if report.get("effective_batch_size") != 32 or report.get("batch_size") not in (8, 16, 32) or report.get("accumulation") != 32//report["batch_size"]:
        raise ValueError("GPU smoke did not preserve effective32")
    if report.get("source_pins") != smoke_bindings() or report.get("model_source_receipt_sha256") != model_api.receipt_sha256(model_api.source_receipt(ROOT)):
        raise ValueError("GPU smoke was performed with another source/model implementation")
    if report.get("manifest_sha256") != h.sha256(ROOT / "features/manifest.csv") or report.get("parent_checkpoint_sha256") != WEIGHT_SHA:
        raise ValueError("GPU smoke used another audited dataset or official checkpoint")
    for key in ("backbone_updated", "attention_updated", "fc_updated", "bn_buffers_frozen", "full_state_finite", "weights_saved"):
        if report.get(key) is not (False if key == "weights_saved" else True):
            raise ValueError("A required GPU smoke invariant failed: " + key)
    return report


def smoke(batch_size=32):
    if batch_size not in (8, 16, 32) or not torch.cuda.is_available():
        raise RuntimeError("CUDA and a supported8/16/32 microbatch are required")
    # Run one batch size per fresh subprocess. An OOM never retains a previous
    # model/optimizer traceback while trying a smaller batch in the same process.
    source_pins = smoke_bindings()
    h.seed_everything(42); torch.set_num_threads(4)
    rows = first_training_rows()
    dataset = trainer.Images(rows, training=False)
    images = torch.stack([dataset[index][0] for index in range(32)])
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.long)
    if images.shape != (32, 3, 112, 112):
        raise ValueError("Smoke preprocessing is not RGB112")
    model, source = model_api.load_initial_model(ROOT, "cuda")
    model.requires_grad_(True); model.train(); math_v7.freeze_batchnorm(model)
    frozen = trainer.buffers(model)
    before = h.cpu_tree(model.state_dict())
    optimizer = trainer.make_optimizer(model, trainer.SETTINGS)
    scaler = math_v7.ManualGradScaler(True, 1024., 2000)
    ema = math_v7.FullModelEMA(model, .995)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for step in range(2):
        trainer.set_learning_rate(optimizer, step, 664*6, trainer.SETTINGS)
        result = math_v7.group_step(model, optimizer, scaler, ema, images, targets, device="cuda", batch_size=batch_size, rho=0.)
        if not result["success"] or result["seen"] != 32:
            raise RuntimeError("DDAM smoke did not produce a successful full-group update")
        trainer.assert_frozen_buffers(model, frozen)
    after = h.cpu_tree(model.state_dict())
    changed = {name for name, value in after.items() if not torch.equal(value, before[name])}
    if not any(name.startswith("features.") for name in changed) or not any(name.startswith("cat_head") for name in changed) or not {"fc.weight", "fc.bias"}.issubset(changed):
        raise ValueError("Active backbone/attention/classifier parameters did not all update")
    model_api.verify_state(ROOT, after); model_api.verify_state(ROOT, ema.state_dict()["shadow"])
    with torch.inference_mode(), h.autocast_for("cuda"):
        logits = model(images[:batch_size].cuda())
    if logits.shape != (batch_size, 6) or not torch.isfinite(logits).all().item():
        raise ValueError("DDAM smoke logits are not finite official six-class outputs")
    peak = int(torch.cuda.max_memory_allocated())
    if peak > 6500*1024*1024:
        raise torch.cuda.OutOfMemoryError("DDAM smoke exceeded the reserved6.5GiB allocation budget")
    report = {"state": "passed", "classes": trainer.CLASSES, "batch_size": batch_size, "accumulation": 32//batch_size,
              "effective_batch_size": 32, "successful_optimizer_steps": 2, "amp_skipped_steps": 0, "ema_updates": ema.updates,
              "source_pins": source_pins, "model_source_receipt_sha256": model_api.receipt_sha256(source),
              "parent_checkpoint_sha256": WEIGHT_SHA, "manifest_sha256": h.sha256(ROOT / "features/manifest.csv"),
              "train_manifest_indices": [row["manifest_index"] for row in rows], "image_size": 112,
              "backbone_updated": True, "attention_updated": True, "fc_updated": True, "bn_buffers_frozen": True,
              "full_state_finite": True, "unused_native_parameters_need_not_receive_gradient": True,
              "logits_dtype": str(logits.dtype), "parameter_dtype": "torch.float32", "peak_allocated_bytes": peak,
              "elapsed_seconds": time.monotonic()-started, "weights_saved": False, "GPU_used": True,
              "internal_test_read": False, "external_answers_used": False}
    if smoke_bindings() != source_pins:
        raise ValueError("Source changed during the smoke; do not bind an ambiguous result")
    h.write_json(SMOKE, report)
    trainer.emit("ddamfn_v8_smoke", **report)
    del model, optimizer, ema, before, after, logits
    gc.collect(); torch.cuda.empty_cache()
    return report


def verify_plan(root=ROOT):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Fixed project root required")
    plan, _ = trainer.checked_plan(ROOT)
    report = verify_smoke(ROOT)
    h.check_hash(SMOKE, plan["smoke_sha256"])
    if report["batch_size"] != plan["settings"]["batch_size"]:
        raise ValueError("Locked microbatch differs from its real GPU probe")
    h.check_hash(plan["audit_summary_path"], plan["audit_summary_sha256"])
    return plan


def prepare(root=ROOT):
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("Fixed project root required")
    if PLAN.exists():
        return verify_plan(ROOT)
    baseline = ROOT / "hse_sam_v7_results"
    summary = c.read_json(baseline / "summary.json")
    recipe = c.read_json(baseline / "selected_recipe.json")
    if summary.get("state") != "complete" or summary.get("classes") != trainer.CLASSES or summary.get("submission_rows") != 7065 or recipe.get("state") != "locked":
        raise ValueError("V7 must finish its validated export before V8 plan generation")
    if recipe.get("fusion_weights") != {"hse": .5, "lora": .2, "cnn": .2, "mlp": .1}:
        raise ValueError("The V7 incumbent must retain the fixed20% original CNN branch")
    if summary.get("competition_image_content_sha256") != IMAGE_SHA or summary.get("competition_row_order_sha256") != ROW_SHA:
        raise ValueError("V7 probabilities use different competition images or IDs")
    probe = verify_smoke(ROOT)
    if shutil.disk_usage(ROOT).free < 3*1024**3:
        raise RuntimeError("V8 needs at least3GiB free for its immutable complete-epoch and candidate artifacts")
    baseline_pins = {}
    for relative, digest in summary["file_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("V7 artifact escapes its result directory")
        pin(baseline_pins, baseline / relative, digest)
    pin(baseline_pins, baseline / "summary.json")
    h.check_hash(baseline / "selected_recipe.json", summary["recipe_sha256"])
    for path, digest in recipe["producer_files"].items():
        pin(baseline_pins, path, digest)
    original, receipt = h.load_trainval(ROOT)
    audit = c.read_json(AUDIT)
    training = [row for row in original["train"] if row["manifest_index"] not in {7727, 10448}]
    members = {"original_train_indices": [row["manifest_index"] for row in original["train"]],
               "excluded_train_indices": [7727, 10448], "train_indices": [row["manifest_index"] for row in training],
               "val_indices": [row["manifest_index"] for row in original["val"]]}
    if audit.get("state") != "complete" or audit.get("classes") != trainer.CLASSES or audit.get("sample_rows") != 192:
        raise ValueError("The fixed training-only review is incomplete")
    if any(audit.get(key) != value for key, value in members.items()) or audit["filtered_membership_sha256"] != h.signature(training):
        raise ValueError("DDAM training/validation members differ from the shared two exclusions")
    if any(audit.get(key) is not False for key in ("labels_changed", "images_changed", "weights_changed", "internal_test_read", "external_answers_used")):
        raise ValueError("The review may not edit images/labels/weights or use test/answers")
    targets = np.asarray([row["target"] for row in original["val"]])
    val = c.check_probability(np.load(baseline / "selected_val_probabilities.npy", allow_pickle=False), 2656)
    score = c.metrics(targets, val)
    if score["correct_count"] != summary["selected_metrics"]["correct_count"] or score["correct_count"] != recipe["selected_metrics"]["correct_count"]:
        raise ValueError("The completed V7 incumbent score cannot be reproduced")
    import finalize_ddamfn_v8 as finalizer
    val_rows, cnn_targets, _, _, baseline_pins = finalizer.old_cnn_inputs(baseline_pins)
    if [int(row["manifest_index"]) for row in val_rows] != members["val_indices"] or not np.array_equal(cnn_targets, targets):
        raise ValueError("The reusable old CNN probabilities have different official member/label order")
    competition = c.unlabeled_rows(ROOT)
    probability = c.check_probability(np.load(baseline / "probabilities.npy", allow_pickle=False), 7065)
    c.submission_check(baseline / "submission.csv", competition, probability)
    image_receipt = c.image_receipt(ROOT, competition, verify_inventory=False)
    if c.signature(competition) != ROW_SHA or image_receipt["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Current competition inputs differ from the reused V7 probabilities")
    pins = {}
    for name in SOURCE_NAMES:
        pin(pins, ROOT / "src" / name)
    for name in ("start_ddamfn_v8.ps1", "run_ddamfn_v8_worker.ps1", "check_ddamfn_v8.ps1"):
        pin(pins, ROOT / name)
    for path in (AUDIT, SMOKE, WEIGHT, ROOT / "vendor/ddamfn_v8/source_receipt.json", ROOT / "models/ddamfn_affectnet7_v8/source_receipt.json"):
        pin(pins, path)
    for entry in c.read_json(ROOT / "vendor/ddamfn_v8/source_receipt.json")["sources"]:
        pin(pins, entry["local_path"], entry["sha256"])
    for path, digest in audit["source_pins"].items():
        pin(pins, path, digest)
    for relative, digest in audit["file_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Audit artifact escapes its directory")
        pin(pins, AUDIT.parent / relative, digest)
    for path in (ROOT / "features/manifest.csv", ROOT / "features/unlabeled.csv", ROOT / "data/raw/submission.csv", ROOT / "data/split_audit.json", ROOT / "data/original_splits/image_inventory.csv"):
        pin(pins, path)
    h.check_hash(WEIGHT, WEIGHT_SHA)
    provenance = model_api.source_receipt(ROOT)
    baseline_files = {str(Path(path).relative_to(ROOT)): digest for path, digest in baseline_pins.items()}
    settings = {**trainer.SETTINGS, "batch_size": probe["batch_size"], "accumulation": probe["accumulation"]}
    plan = {"format": "ddamfn_v8_locked", "classes": trainer.CLASSES, "settings": settings,
            "source_pins": pins, "input_receipt": receipt, **members, "filtered_membership_sha256": h.signature(training),
            "audit_summary_path": str(AUDIT), "audit_summary_sha256": h.sha256(AUDIT),
            "parent_checkpoint": str(WEIGHT), "parent_checkpoint_sha256": WEIGHT_SHA,
            "model_source_receipt": provenance, "model_source_receipt_sha256": model_api.receipt_sha256(provenance),
            "baseline_directory": str(baseline), "baseline_files": baseline_files,
            "baseline_correct_count": score["correct_count"], "baseline_metrics": score,
            "baseline_submission_sha256": h.sha256(baseline / "submission.csv"),
            "baseline_probability_sha256": h.sha256(baseline / "probabilities.npy"),
            "baseline_val_probability_sha256": h.sha256(baseline / "selected_val_probabilities.npy"),
            "smoke_sha256": h.sha256(SMOKE), "smoke": probe, "fusion_weights": {"hse": .5, "lora": .2, "cnn": .2, "mlp": .1},
            "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
            "candidate_count": 13, "selection": "fixed CNN20% replacement; strict correct-count improvement; incumbent wins every tie",
            "license_grant_found": False, "external_pretrain_overlap_audited": False,
            "internal_test_read": False, "external_answers_used": False, "GPU_used_by_plan_generation": False}
    c.verify_pins(ROOT, pins)
    for relative, digest in baseline_files.items():
        h.check_hash(ROOT / relative, digest)
    h.write_json(PLAN, plan)
    trainer.emit("ddamfn_v8_plan_locked", effective_train_rows=21246, val_rows=2656, epochs=6,
                 batch_size=settings["batch_size"], baseline_correct_count=score["correct_count"], GPU_used=False)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("smoke", "prepare", "verify"), default="prepare")
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), default=32)
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("Fixed project root required")
    torch.set_num_threads(4)
    if args.stage == "smoke":
        try:
            smoke(args.batch_size)
        except torch.cuda.OutOfMemoryError as error:
            trainer.emit("ddamfn_v8_smoke_oom", batch_size=args.batch_size, error=str(error))
            raise SystemExit(42)
    elif args.stage == "verify":
        verify_plan(ROOT)
    else:
        prepare(ROOT)


if __name__ == "__main__":
    main()
