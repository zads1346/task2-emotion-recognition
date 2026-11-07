"""Training-only grouped CV and a locked official-initialized DDAMFN refit.

Every run trains six complete epochs. CV never reads development-validation
images or computes per-epoch heldout metrics: only final EMA original/flip
probabilities are exported. Refit is initialized anew after the external OOF
selection is locked; its development validation is evaluated only at final EMA.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

import ddamfn_v8_model as model_api
import train_ddamfn_v8 as previous
import train_hse_sam_v7 as math_v7
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "ddamfn_groupcv_v9"
CLASSES = list(model_api.CLASSES)
MODES = ("full", "fc_only")
SETTINGS = {"epochs": 6, "seed": 42, "augmentation_epoch_offset": 90,
    "batch_size": 32, "effective_batch_size": 32, "accumulation": 1,
    "workers": 0, "cpu_threads": 4, "image_size": 112, "validation_batch_size": 32,
    "backbone_lr": 1e-5, "head_lr": 1e-4, "weight_decay": .01,
    "label_smoothing": 0., "gradient_clip": 1., "warmup_successful_steps": 100,
    "min_lr_ratio": .1, "ema_decay": .995, "amp_initial_scale": 1024.,
    "amp_growth_interval": 2000, "batchnorm_running_stats": "frozen_eval",
    "sample_weights": "uniform1", "mixup": False, "cutmix": False, "rho": 0.,
    "initialization": "official AffectNet7 only; mapped six-class head"}
HEAD_NAMES = {"fc.weight", "fc.bias"}
PLAN_NAME = "logs/ddamfn_groupcv_v9_plan.json"
OUTPUT_NAME = "experiments/ddamfn_groupcv_v9"


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def _location(root, path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root/value).resolve()


def checked_plan(root):
    root = Path(root).resolve()
    path = root/PLAN_NAME
    plan = c.read_json(path)
    if plan.get("format") != FORMAT+"_locked" or plan.get("classes") != CLASSES or plan.get("modes") != list(MODES):
        raise ValueError("V9 plan format/classes/paired modes differ")
    if any(plan.get("settings", {}).get(key) != value for key, value in SETTINGS.items()):
        raise ValueError("V9 fixed six-epoch uniform-CE protocol differs")
    required = {Path(__file__).resolve(), Path(model_api.__file__).resolve(), Path(previous.__file__).resolve(),
        Path(math_v7.__file__).resolve(), Path(h.__file__).resolve(), Path(c.__file__).resolve()}
    pins = plan.get("source_pins")
    if not isinstance(pins, dict) or not required.issubset({_location(root, value) for value in pins}):
        raise ValueError("V9 pins omit a required model/training implementation")
    c.verify_pins(root, pins)
    receipt = model_api.source_receipt(root)
    if receipt != plan.get("model_source_receipt") or model_api.receipt_sha256(receipt) != plan.get("model_source_receipt_sha256"):
        raise ValueError("V9 official DDAMFN initialization/source receipt differs")
    parent = _location(root, plan["parent_checkpoint"])
    if parent != (root/"models/ddamfn_affectnet7_v8/affecnet7_epoch19_acc0.671.pth").resolve():
        raise ValueError("Every V9 run must restart from the official AffectNet checkpoint")
    h.check_hash(parent, plan["parent_checkpoint_sha256"])
    if plan["parent_checkpoint_sha256"] != model_api.OFFICIAL_WEIGHT_SHA256:
        raise ValueError("V9 official initialization SHA differs")
    original, effective, excluded, validation = (plan[key] for key in
        ("original_train_indices", "train_indices", "excluded_train_indices", "val_indices"))
    if any(not isinstance(values, list) or len(values) != len(set(values)) or any(type(index) is not int or index < 0 for index in values)
           for values in (original, effective, excluded, validation)):
        raise ValueError("V9 member arrays require unique nonnegative integer manifest indices")
    if (len(original) != 21248 or excluded != [7727, 10448] or len(effective) != 21246 or len(validation) != 2656
        or effective != [index for index in original if index not in set(excluded)] or set(original) & set(validation)):
        raise ValueError("V9 immutable original/effective/development-validation membership differs")
    h.check_hash(_location(root, plan["audit_summary_path"]), plan["audit_summary_sha256"])
    if set(plan.get("folds", {})) != {"0", "1", "2"}:
        raise ValueError("V9 requires exactly three fixed grouped folds")
    coverage = []
    for fold in range(3):
        contract = plan["folds"][str(fold)]
        train, heldout = contract["train_indices"], contract["validation_indices"]
        if (contract.get("fold") != fold or len(train) != 14164 or len(heldout) != 7082
            or len(set(train)) != len(train) or len(set(heldout)) != len(heldout)
            or set(train) & set(heldout) or set(train) | set(heldout) != set(effective)
            or set(contract["train_groups"]) & set(contract["validation_groups"])):
            raise ValueError("V9 grouped fold contract is incomplete or leaks members/groups")
        train_set, heldout_set = set(train), set(heldout)
        if train != [index for index in effective if index in train_set] or heldout != [index for index in effective if index in heldout_set]:
            raise ValueError("V9 fold members must retain original manifest order")
        coverage.extend(heldout)
    if len(coverage) != len(effective) or len(set(coverage)) != len(effective) or set(coverage) != set(effective):
        raise ValueError("V9 OOF heldout coverage must include each effective training image exactly once")
    inputs = plan.get("input_receipt", {})
    if inputs.get("internal_test_images_read") != 0 or inputs.get("internal_test_labels_parsed") is not False:
        raise ValueError("V9 input receipt must exclude internal-test images/labels")
    for path, digest in plan.get("baseline_files", {}).items():
        h.check_hash(_location(root, path), digest)
    if not Path(plan["baseline_directory"]).is_absolute() or type(plan["baseline_correct_count"]) is not int:
        raise ValueError("V9 incumbent requires a pinned absolute directory and integer correct count")
    selection_path = _location(root, plan["cv_selection_path"])
    if selection_path != (root/OUTPUT_NAME/"cv_selection.json").resolve():
        raise ValueError("V9 selection must stay in its own experiment directory")
    return plan


def training_records(root, plan, verify_images=True):
    """Filter split before labels: no development-validation/test rows are used."""
    manifest = root/"features/manifest.csv"
    h.check_hash(manifest, plan["input_receipt"]["manifest_sha256"])
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row.get("split") != "train":
                continue
            if row.get("label") not in CLASSES or not os.path.isabs(row.get("path", "")):
                raise ValueError("Invalid original training record")
            rows.append({"manifest_index": index, "path": row["path"], "label": row["label"],
                         "target": CLASSES.index(row["label"])})
    if [row["manifest_index"] for row in rows] != plan["original_train_indices"] or h.signature(rows) != plan["input_receipt"]["train_rows_sha256"]:
        raise ValueError("V9 original training membership differs")
    if verify_images:
        digest = hashlib.sha256()
        started = time.monotonic()
        for index, row in enumerate(rows, 1):
            digest.update(h.sha256(row["path"]).encode("ascii"))
            if index % 4000 == 0:
                emit("verify_v9_training_images", images=index, total=len(rows), elapsed_seconds=time.monotonic()-started)
        if digest.hexdigest() != plan["input_receipt"]["train_image_content_sha256"]:
            raise ValueError("V9 original training image bytes differ")
    excluded = set(plan["excluded_train_indices"])
    rows = [row for row in rows if row["manifest_index"] not in excluded]
    if [row["manifest_index"] for row in rows] != plan["train_indices"] or h.signature(rows) != plan["filtered_membership_sha256"]:
        raise ValueError("V9 effective training members differ after the two shared exclusions")
    return rows


def development_records(root, plan):
    """Called only after refit epoch6 is committed, never for grouped CV."""
    manifest = root/"features/manifest.csv"
    h.check_hash(manifest, plan["input_receipt"]["manifest_sha256"])
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row.get("split") != "val":
                continue
            if row.get("label") not in CLASSES or not os.path.isabs(row.get("path", "")):
                raise ValueError("Invalid development-validation record")
            rows.append({"manifest_index": index, "path": row["path"], "label": row["label"],
                         "target": CLASSES.index(row["label"])})
    if [row["manifest_index"] for row in rows] != plan["val_indices"] or h.signature(rows) != plan["input_receipt"]["val_rows_sha256"]:
        raise ValueError("V9 final development-validation membership differs")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(h.sha256(row["path"]).encode("ascii"))
    if digest.hexdigest() != plan["input_receipt"]["val_image_content_sha256"]:
        raise ValueError("V9 final development-validation image bytes differ")
    return rows


def _read_selection(root, plan):
    path = _location(root, plan["cv_selection_path"])
    selection = c.read_json(path)
    if selection.get("classes") != CLASSES or selection.get("plan_sha256") != h.sha256(root/PLAN_NAME):
        raise ValueError("V9 OOF selection class/plan binding differs")
    mode = selection.get("selected_mode")
    if mode not in MODES:
        raise ValueError("V9 refit requires a completed locked OOF selected_mode")
    if selection.get("source_pins") != plan["source_pins"]:
        raise ValueError("V9 OOF selection source binding differs")
    if selection.get("state") not in ("complete", "selected", "locked"):
        raise ValueError("V9 OOF selection has not been finalized")
    return mode, h.sha256(path)


def config_for(root, plan, phase, mode, fold=None):
    if phase not in ("cv", "refit") or mode not in MODES:
        raise ValueError("Unknown V9 phase/mode")
    if phase == "cv":
        if fold not in (0, 1, 2):
            raise ValueError("CV requires an explicit fixed fold0/1/2")
        contract = plan["folds"][str(fold)]
        run_train, run_val = contract["train_indices"], contract["validation_indices"]
        selection_sha = None
    else:
        selected, selection_sha = _read_selection(root, plan)
        if selected != mode or fold is not None:
            raise ValueError("Refit must use exactly the locked selected mode without a fold")
        contract = None
        run_train, run_val = plan["train_indices"], plan["val_indices"]
    config = {"format": FORMAT+"_config", "phase": phase, "mode": mode, "fold": fold,
        "plan_sha256": h.sha256(root/PLAN_NAME), "plan": plan,
        **{key: plan[key] for key in ("classes", "settings", "source_pins", "input_receipt", "original_train_indices",
            "excluded_train_indices", "train_indices", "val_indices", "filtered_membership_sha256",
            "parent_checkpoint", "parent_checkpoint_sha256", "model_source_receipt", "model_source_receipt_sha256",
            "audit_summary_path", "audit_summary_sha256", "baseline_directory", "baseline_correct_count")},
        "run_train_indices": run_train, "run_validation_indices": run_val, "fold_contract": contract,
        "selection_sha256": selection_sha, "effective_train_rows": len(run_train), "evaluation_rows": len(run_val),
        "train_sample_weights": "uniform1", "validation_weighted": False,
        "checkpoint_selection": "none; fixed final epoch6 EMA only", "per_epoch_evaluation": False,
        "development_validation_used_for_cv": False, "internal_test_read": False, "external_answers_used": False,
        "preprocessing": "V8 RGB EXIF transpose PIL bilinear112 ImageNet mean/std",
        "augmentation": "identical V8 flip.5 affine6deg translate.03 scale.95..1.05 brightness/contrast.1",
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    return config


def configure_trainability(model, mode, training=False):
    if mode not in MODES or not isinstance(model.fc, nn.Linear) or model.fc.out_features != 6:
        raise ValueError("Unexpected V9 mode/native six-class fc")
    model.requires_grad_(mode == "full")
    model.fc.requires_grad_(True)
    expected = set(dict(model.named_parameters())) if mode == "full" else HEAD_NAMES
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != expected:
        raise ValueError("V9 trainable parameters differ from the full/fc-only contract")
    model.train(bool(training) and mode == "full")
    if training:
        model.fc.train(True)
    math_v7.freeze_batchnorm(model)
    if mode == "fc_only" and any(module.training for name, module in model.named_modules() if name != "fc"):
        raise ValueError("Every frozen module must remain in eval for fc_only")
    return model


def make_optimizer(model, settings, mode):
    if mode == "full":
        return previous.make_optimizer(model, settings)
    if mode != "fc_only":
        raise ValueError("Unknown V9 optimizer mode")
    return torch.optim.AdamW([{"name": "head", "params": list(model.fc.parameters()),
        "lr": settings["head_lr"], "base_lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])


def frozen_reference(model, mode):
    parameters = dict(model.named_parameters())
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key not in parameters or (mode == "fc_only" and key not in HEAD_NAMES)}


def assert_frozen(model, reference, mode):
    actual = frozen_reference(model, mode)
    if set(actual) != set(reference) or any(not torch.equal(actual[key], reference[key]) for key in reference):
        raise ValueError("V9 frozen BN buffers or fc_only backbone/attention state changed")
    if mode == "fc_only" and any(parameter.grad is not None for name, parameter in model.named_parameters() if name not in HEAD_NAMES):
        raise ValueError("V9 fc_only backbone/attention acquired gradients")


def _frozen_digest(reference):
    digest = hashlib.sha256()
    for name in sorted(reference):
        value = reference[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], ensure_ascii=False).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_payload(model, config, epoch, weight_source):
    fields = ("classes", "phase", "mode", "fold", "source_pins", "parent_checkpoint_sha256", "model_source_receipt",
        "model_source_receipt_sha256", "original_train_indices", "excluded_train_indices", "train_indices", "val_indices",
        "run_train_indices", "run_validation_indices", "audit_summary_sha256", "selection_sha256")
    return {"format": FORMAT, "config": config, **{key: config[key] for key in fields},
        "plan_sha256": config["plan_sha256"], "parent_sha": config["parent_checkpoint_sha256"],
        "epoch": epoch, "weight_source": weight_source, "model_state": h.cpu_tree(model.state_dict())}


def verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("V9 requires its own full1190-state checkpoint format")
    plan = checked_plan(root)
    expected = config_for(root, plan, payload.get("phase"), payload.get("mode"), payload.get("fold"))
    if payload.get("config") != expected or (config is not None and expected != config):
        raise ValueError("V9 checkpoint configuration differs from the immutable phase/fold plan")
    for key in ("classes", "phase", "mode", "fold", "source_pins", "parent_checkpoint_sha256", "model_source_receipt",
                "model_source_receipt_sha256", "original_train_indices", "excluded_train_indices", "train_indices", "val_indices",
                "run_train_indices", "run_validation_indices", "audit_summary_sha256", "selection_sha256", "plan_sha256"):
        if payload.get(key) != expected[key]:
            raise ValueError(f"V9 checkpoint/source/member binding differs: {key}")
    if payload.get("parent_sha") != expected["parent_checkpoint_sha256"]:
        raise ValueError("V9 parent SHA differs")
    model_api.verify_state(root, payload.get("model_state"))
    return expected


def load_checkpoint(root, path, device="cpu"):
    root = Path(root).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True) if isinstance(path, (str, os.PathLike)) else path
    config = verify_payload(root, payload)
    model, receipt = model_api.load_initial_model(root, "cpu")
    if receipt != config["model_source_receipt"]:
        raise ValueError("V9 current official model source differs")
    reference = frozen_reference(configure_trainability(model, config["mode"]), config["mode"])
    model.load_state_dict(payload["model_state"], strict=True)
    assert_frozen(model, reference, config["mode"])
    model.eval().to(device)
    return model, {key: value for key, value in payload.items() if key not in {"model_state", "optimizer", "scaler", "rng", "ema_state"}}


def _atomic_copy(source, destination):
    temporary = Path(str(destination)+".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, frozen_digest):
    relative = f"epochs/epoch_{epoch:02d}.pt"
    payload = {**model_payload(model, config, epoch, "raw"),
        "optimizer": h.cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
        "ema_state": ema.state_dict(), "rng": h.rng_state(), "counters": dict(counters),
        "successful_steps": counters["successful_steps"], "cumulative_skips": counters["amp_skipped_steps"],
        "zero_weight_groups": 0, "history": list(history), "frozen_state_sha256": frozen_digest}
    h.save_torch(output/relative, payload)
    _atomic_copy(output/relative, output/"last.pt")
    h.write_json(output/"last.json", {"epoch": epoch, "checkpoint": relative,
        "sha256": h.sha256(output/relative), "plan_sha256": config["plan_sha256"],
        "phase": config["phase"], "mode": config["mode"], "fold": config["fold"]})
    h.write_json(output/"history.json", history)


def verify_history(history, counters, config, epoch):
    count = len(config["run_train_indices"])
    groups = math.ceil(count/32)
    if len(history) != epoch or [row["epoch"] for row in history] != list(range(1, epoch+1)):
        raise ValueError("V9 complete-epoch history count/order differs")
    for row in history:
        if (row["seen"] != count or row["successful_optimizer_steps"] <= 0
            or row["successful_optimizer_steps"]+row["amp_skipped_steps"] != groups
            or row.get("zero_weight_groups") != 0
            or any(key in row for key in ("val_accuracy", "validation", "heldout_metrics", "raw_original_metrics", "ema_tta_metrics"))):
            raise ValueError("V9 epoch missed members/groups or contains an intermediate validation result")
    if (sum(row["successful_optimizer_steps"] for row in history) != counters["successful_steps"]
        or sum(row["amp_skipped_steps"] for row in history) != counters["amp_skipped_steps"]
        or counters["zero_weight_groups"] != 0):
        raise ValueError("V9 successful-update/AMP counters differ from complete epochs")
    cumulative = 0
    for row in history:
        cumulative += row["successful_optimizer_steps"]
        if row["cumulative_optimizer_steps"] != cumulative or row["ema_updates"] != cumulative:
            raise ValueError("V9 per-epoch cumulative optimizer/EMA counters differ")


def resume_epoch(root, output, config, model, optimizer, scaler, ema, reference):
    pointer = c.read_json(output/"last.json")
    epoch = pointer.get("epoch")
    if type(epoch) is not int or not 0 <= epoch <= 6 or pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt":
        raise ValueError("V9 resume pointer must identify a complete fixed epoch")
    for key in ("plan_sha256", "phase", "mode", "fold"):
        if pointer.get(key) != config[key]:
            raise ValueError("V9 resume pointer phase/source binding differs")
    path = output/pointer["checkpoint"]
    h.check_hash(path, pointer["sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    verify_payload(root, saved, config)
    if saved.get("epoch") != epoch or saved.get("weight_source") != "raw":
        raise ValueError("V9 resume checkpoint is not a raw complete epoch")
    if saved.get("frozen_state_sha256") != _frozen_digest(reference):
        raise ValueError("V9 resume frozen official reference differs")
    counters = saved["counters"]
    verify_history(saved["history"], counters, config, epoch)
    if saved.get("successful_steps") != counters["successful_steps"] or saved.get("cumulative_skips") != counters["amp_skipped_steps"]:
        raise ValueError("V9 resume counter aliases differ")
    model.load_state_dict(saved["model_state"], strict=True)
    assert_frozen(model, reference, config["mode"])
    model_api.verify_state(root, saved["ema_state"]["shadow"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    ema.load_state_dict(saved["ema_state"])
    if ema.updates != counters["successful_steps"]:
        raise ValueError("V9 resumed EMA/optimizer counts differ")
    with ema.apply(model):
        assert_frozen(model, reference, config["mode"])
    h.restore_rng(saved["rng"])
    if not (output/"last.pt").exists() or h.sha256(output/"last.pt") != pointer["sha256"]:
        _atomic_copy(path, output/"last.pt")
    h.write_json(output/"history.json", saved["history"])
    emit("resume_ddamfn_groupcv_v9", phase=config["phase"], mode=config["mode"], fold=config["fold"],
         completed_epoch=epoch, successful_optimizer_steps=counters["successful_steps"])
    return epoch, counters, saved["history"]


def _validate_final_result(root, output, config, targets):
    result = c.read_json(output/"final_result.json")
    for key in ("phase", "mode", "fold", "classes", "plan_sha256", "source_pins", "run_train_indices", "run_validation_indices"):
        if result.get(key) != config[key]:
            raise ValueError("V9 final inference receipt phase/source/member binding differs")
    if result.get("epoch") != 6 or result.get("weight_source") != "ema":
        raise ValueError("V9 final result must use fixed epoch6 EMA")
    expected_files = {"model": "final_ema.pt", "original": "original.npy", "flip": "flip.npy"}
    for key, relative in expected_files.items():
        if result.get(key) != relative:
            raise ValueError("V9 final inference artifact filename differs")
        h.check_hash(output/relative, result["artifact_sha256"][key])
    payload = torch.load(output/"final_ema.pt", map_location="cpu", weights_only=True)
    verify_payload(root, payload, config)
    if payload.get("epoch") != 6 or payload.get("weight_source") != "ema":
        raise ValueError("V9 final model must be epoch6 EMA")
    original = np.load(output/"original.npy", allow_pickle=False)
    flip = np.load(output/"flip.npy", allow_pickle=False)
    for probability in (original, flip):
        if probability.dtype != np.float32:
            raise ValueError("V9 final probability dtype must be float32")
        c.check_probability(probability, len(config["run_validation_indices"]))
    computed = {"single_original_metrics": c.metrics(targets, original),
                "single_tta_metrics": c.metrics(targets, (original+flip)*.5)}
    if any(result[key] != computed[key] or payload.get(key) != computed[key] for key in computed):
        raise ValueError("V9 final probabilities do not reproduce their unweighted metrics")
    pointer = c.read_json(output/"last.json")
    if pointer.get("epoch") != 6:
        raise ValueError("V9 final inference lacks six committed epochs")
    h.check_hash(output/pointer["checkpoint"], pointer["sha256"])
    last = torch.load(output/pointer["checkpoint"], map_location="cpu", weights_only=True)
    if set(payload["model_state"]) != set(last["ema_state"]["shadow"]) or any(not torch.equal(value, last["ema_state"]["shadow"][key]) for key, value in payload["model_state"].items()):
        raise ValueError("V9 final model is not the complete last-epoch EMA shadow")
    return result


def finalize_run(root, output, config, model, ema, reference, evaluation_rows, counters, history):
    if len(history) != 6 or ema.updates != counters["successful_steps"]:
        raise ValueError("V9 final evaluation requires all six committed epochs and matching EMA updates")
    if [row["manifest_index"] for row in evaluation_rows] != config["run_validation_indices"]:
        raise ValueError("V9 final evaluation member order differs")
    targets = np.asarray([row["target"] for row in evaluation_rows], dtype=np.int64)
    if (output/"final_result.json").exists():
        result = _validate_final_result(root, output, config, targets)
    else:
        loader = previous.make_loader(evaluation_rows, False, 0, 32, next(model.parameters()).device)
        with ema.apply(model):
            assert_frozen(model, reference, config["mode"])
            original_metrics, tta_metrics, original, flip = previous.validate(model, loader, next(model.parameters()).device)
            payload = model_payload(model, config, 6, "ema")
            payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics,
                ema_updates=ema.updates, successful_steps=counters["successful_steps"],
                frozen_state_sha256=_frozen_digest(reference))
            h.save_torch(output/"final_ema.pt", payload)
        h.save_array(output/"original.npy", original)
        h.save_array(output/"flip.npy", flip)
        result = {"format": FORMAT+"_final_result", "state": "complete", "epoch": 6, "weight_source": "ema",
            **{key: config[key] for key in ("phase", "mode", "fold", "classes", "plan_sha256", "source_pins",
                "run_train_indices", "run_validation_indices", "selection_sha256")},
            "model": "final_ema.pt", "original": "original.npy", "flip": "flip.npy",
            "artifact_sha256": {"model": h.sha256(output/"final_ema.pt"), "original": h.sha256(output/"original.npy"), "flip": h.sha256(output/"flip.npy")},
            "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
            "validation_weighted": False, "development_validation_evaluated": config["phase"] == "refit"}
        h.write_json(output/"final_result.json", result)
        _validate_final_result(root, output, config, targets)
    paths = ["config.json", "history.json", "last.json", "last.pt", "final_ema.pt", "original.npy", "flip.npy", "final_result.json"]
    paths.extend(f"epochs/epoch_{epoch:02d}.pt" for epoch in range(7))
    verify_history(history, counters, config, 6)
    assert_frozen(model, reference, config["mode"])
    pointer = c.read_json(output/"last.json")
    h.check_hash(output/"last.pt", pointer["sha256"])
    summary = {"format": FORMAT+"_summary", "state": "complete", "epochs_completed": 6, "weight_source": "ema",
        **{key: config[key] for key in ("phase", "mode", "fold", "classes", "plan_sha256", "source_pins",
            "model_source_receipt", "model_source_receipt_sha256", "parent_checkpoint_sha256",
            "original_train_indices", "excluded_train_indices", "train_indices", "val_indices", "run_train_indices", "run_validation_indices", "selection_sha256")},
        "effective_train_rows": len(config["run_train_indices"]), "evaluation_rows": len(evaluation_rows),
        "single_original_metrics": result["single_original_metrics"], "single_tta_metrics": result["single_tta_metrics"],
        "counters": dict(counters), "successful_optimizer_steps": counters["successful_steps"],
        "amp_skipped_steps": counters["amp_skipped_steps"], "zero_weight_groups": 0, "ema_updates": ema.updates,
        "frozen_state_sha256": _frozen_digest(reference), "frozen_state_unchanged": True, "batchnorm_buffers_frozen": True,
        "per_epoch_evaluation": False, "validation_weighted": False,
        "development_validation_evaluated": config["phase"] == "refit", "development_validation_used_for_cv": False,
        "internal_test_read": False, "external_answers_used": False,
        "artifact_sha256": {relative: h.sha256(output/relative) for relative in paths}}
    h.write_json(output/"summary.json", summary)
    emit("ddamfn_groupcv_v9_run_complete", phase=config["phase"], mode=config["mode"], fold=config["fold"],
        correct_count=summary["single_tta_metrics"]["correct_count"], samples=len(evaluation_rows),
        successful_optimizer_steps=counters["successful_steps"], amp_skipped_steps=counters["amp_skipped_steps"])
    return summary


def run_training(root, phase, mode=None, fold=None, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; V9 never silently falls back to CPU")
    torch.set_num_threads(4)
    plan = checked_plan(root)
    if phase == "refit":
        selected, _ = _read_selection(root, plan)
        if mode is not None and mode != selected:
            raise ValueError("Refit CLI mode differs from the locked OOF selection")
        mode = selected
    config = config_for(root, plan, phase, mode, fold)
    all_rows = training_records(root, plan)
    lookup = {row["manifest_index"]: row for row in all_rows}
    run_rows = [lookup[index] for index in config["run_train_indices"]]
    if phase == "cv":
        heldout_rows = [lookup[index] for index in config["run_validation_indices"]]
        if (h.signature(run_rows) != config["fold_contract"]["train_rows_sha256"]
            or h.signature(heldout_rows) != config["fold_contract"]["validation_rows_sha256"]
            or {row["target"] for row in run_rows} != set(range(6))
            or {row["target"] for row in heldout_rows} != set(range(6))):
            raise ValueError("V9 training/heldout rows differ from the group-fold contract")
        output = root/OUTPUT_NAME/"cv"/mode/f"fold_{fold}"
    else:
        heldout_rows = None  # Development labels/images are loaded only after epoch6.
        output = root/OUTPUT_NAME/"refit"/mode
    output.mkdir(parents=True, exist_ok=True)
    (output/"epochs").mkdir(exist_ok=True)
    if (output/"config.json").exists() and c.read_json(output/"config.json") != config:
        raise ValueError("Existing V9 run configuration differs; preserve its artifacts")
    if not (output/"config.json").exists():
        h.write_json(output/"config.json", config)
    if (output/"summary.json").exists():
        existing = c.read_json(output/"summary.json")
        if existing.get("state") != "complete" or existing.get("plan_sha256") != config["plan_sha256"]:
            raise ValueError("Existing V9 summary is incomplete or belongs to another plan")
        for relative, digest in existing["artifact_sha256"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("V9 summary artifact escapes its dedicated run")
            h.check_hash(output/path, digest)
    h.seed_everything(config["settings"]["seed"])
    model, receipt = model_api.load_initial_model(root, device)
    if receipt != config["model_source_receipt"]:
        raise ValueError("V9 model did not initialize from the fixed official source")
    configure_trainability(model, mode)
    settings = config["settings"]
    optimizer = make_optimizer(model, settings, mode)
    scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", settings["amp_initial_scale"], settings["amp_growth_interval"])
    ema = math_v7.FullModelEMA(model, settings["ema_decay"])
    reference = frozen_reference(model, mode)
    reference_digest = _frozen_digest(reference)
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    history, done = [], 0
    if (output/"last.json").exists():
        done, counters, history = resume_epoch(root, output, config, model, optimizer, scaler, ema, reference)
    else:
        commit_epoch(output, model, optimizer, scaler, ema, config, 0, counters, history, reference_digest)
    total_steps = math.ceil(len(run_rows)/32)*6
    for epoch in range(done+1, 7):
        data_epoch = settings["augmentation_epoch_offset"]+epoch-1
        loader = previous.make_loader(run_rows, True, data_epoch, 32, device)
        configure_trainability(model, mode, True)
        seen = success = skips = diagnostic_seen = correct = 0
        loss_sum = 0.
        started = stamp = time.monotonic()
        for index, (images, targets) in enumerate(loader, 1):
            previous.set_learning_rate(optimizer, counters["successful_steps"], total_steps, settings)
            result = math_v7.group_step(model, optimizer, scaler, ema, images, targets,
                device=device, batch_size=32, rho=0.)
            seen += len(targets)
            okay = bool(result["success"])
            success += int(okay); skips += int(not okay)
            counters["successful_steps"] += int(okay); counters["amp_skipped_steps"] += int(not okay)
            if result["loss_sum"] is not None:
                diagnostic_seen += len(targets)
                loss_sum += float(result["loss_sum"])
                correct += int(result["correct"])
            if skips >= 25 and success == 0:
                raise RuntimeError("V9 hit25 AMP skips without any successful update")
            if time.monotonic()-stamp >= 45 or index == len(loader):
                emit("train_ddamfn_groupcv_v9", phase=phase, mode=mode, fold=fold, epoch=epoch, epochs=6,
                    batch=index, batches=len(loader), seen=seen, train_loss=loss_sum/max(1, diagnostic_seen),
                    successful_optimizer_steps=success, amp_skipped_steps=skips,
                    cumulative_optimizer_steps=counters["successful_steps"], elapsed_seconds=time.monotonic()-started)
                stamp = time.monotonic()
        if seen != len(run_rows) or success <= 0 or success+skips != math.ceil(len(run_rows)/32):
            raise ValueError("V9 epoch missed training members/groups or never updated")
        assert_frozen(model, reference, mode)
        history.append({"epoch": epoch, "seen": seen, "augmentation_epoch": data_epoch,
            "train_loss": loss_sum/max(1, diagnostic_seen), "train_accuracy": correct/max(1, diagnostic_seen),
            "diagnostic_samples": diagnostic_seen, "successful_optimizer_steps": success, "amp_skipped_steps": skips,
            "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
            "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-started,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}})
        c.verify_pins(root, config["source_pins"])
        commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, reference_digest)
        emit("ddamfn_groupcv_v9_epoch_committed", phase=phase, mode=mode, fold=fold, **history[-1])
    if phase == "refit":
        heldout_rows = development_records(root, plan)
    return finalize_run(root, output, config, model, ema, reference, heldout_rows, counters, history)


def run_fold(root, mode, fold, device="cuda"):
    return run_training(root, "cv", mode, fold, device)


def run_refit(root, device="cuda"):
    return run_training(root, "refit", None, None, device)


def smoke(root, device="cuda"):
    """Two real optimizer groups for each fresh official mode; never save weights."""
    root = Path(root).resolve()
    torch.set_num_threads(4)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the explicitly requested V9 smoke")
    plan = checked_plan(root)
    rows = training_records(root, plan)
    lookup = {row["manifest_index"]: row for row in rows}
    selected = [lookup[index] for index in plan["folds"]["0"]["train_indices"][:64]]
    reports = {}
    for mode in MODES:
        h.seed_everything(42)
        model, receipt = model_api.load_initial_model(root, device)
        configure_trainability(model, mode, True)
        before = h.cpu_tree(model.state_dict())
        reference = frozen_reference(model, mode)
        optimizer = make_optimizer(model, plan["settings"], mode)
        scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", 1024., 2000)
        ema = math_v7.FullModelEMA(model, .995)
        loader = previous.make_loader(selected, True, 90, 32, device)
        success = skips = 0
        started = time.monotonic()
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for images, targets in loader:
            previous.set_learning_rate(optimizer, success, math.ceil(14164/32)*6, plan["settings"])
            result = math_v7.group_step(model, optimizer, scaler, ema, images, targets,
                device=device, batch_size=32, rho=0.)
            success += int(result["success"]); skips += int(not result["success"])
            assert_frozen(model, reference, mode)
        if success != 2 or skips != 0 or ema.updates != 2:
            raise RuntimeError("V9 smoke requires two successful non-skipped updates in both modes")
        after = h.cpu_tree(model.state_dict())
        changed = {name for name in before if not torch.equal(before[name], after[name])}
        if not HEAD_NAMES.issubset(changed):
            raise ValueError("V9 smoke did not update both native classifier tensors")
        if mode == "fc_only" and changed-HEAD_NAMES:
            raise ValueError("V9 fc_only smoke changed a frozen native state tensor")
        if mode == "full" and (not any(name.startswith("features.") for name in changed) or not any(name.startswith("cat_head") for name in changed)):
            raise ValueError("V9 full smoke did not update backbone and attention")
        model_api.verify_state(root, after)
        with ema.apply(model):
            assert_frozen(model, reference, mode)
        reports[mode] = {"state": "passed", "successful_optimizer_steps": success, "amp_skipped_steps": skips,
            "ema_updates": ema.updates, "changed_state_keys": sorted(changed), "fc_updated": True,
            "frozen_state_unchanged": True, "batchnorm_buffers_frozen": True,
            "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
            "elapsed_seconds": time.monotonic()-started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None}
        del model, optimizer, scaler, ema, before, after, reference
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return {"state": "passed", "classes": CLASSES, "source_pins": plan["source_pins"],
        "plan_sha256": h.sha256(root/PLAN_NAME), "model_source_receipt_sha256": plan["model_source_receipt_sha256"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"], "modes": reports,
        "batch_size": 32, "effective_batch_size": 32, "uniform_CE": True,
        "training_manifest_indices": [row["manifest_index"] for row in selected],
        "GPU_used": torch.device(device).type == "cuda", "weights_saved": False,
        "development_validation_evaluated": False, "internal_test_read": False, "external_answers_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("cv", "refit", "smoke"), required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("V9 only supports the fixed project root")
    if args.phase == "smoke":
        if args.mode is not None or args.fold is not None:
            raise ValueError("V9 smoke always checks both modes")
        report = smoke(args.project_root, args.device)
        kind = "gpu" if torch.device(args.device).type == "cuda" else "cpu"
        h.write_json(args.project_root/f"logs/ddamfn_groupcv_v9_{kind}_smoke.json", report)
        emit("ddamfn_groupcv_v9_smoke_passed", modes=list(MODES), source_pins=report["source_pins"], plan_sha256=report["plan_sha256"])
    elif args.phase == "cv":
        if args.mode is None or args.fold is None:
            raise ValueError("V9 grouped CV needs an explicit mode and fold")
        run_fold(args.project_root, args.mode, args.fold, args.device)
    else:
        if args.fold is not None:
            raise ValueError("V9 refit has no fold")
        run_training(args.project_root, "refit", args.mode, None, args.device)


if __name__ == "__main__":
    main()
