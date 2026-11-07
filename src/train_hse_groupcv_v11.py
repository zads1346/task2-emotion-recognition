"""Matched official-initialized HSE full/head-only training on training-pool folds.

Only final epoch12 EMA is evaluated. CV workers receive a materialized train-only
manifest and never evaluate the expanded outer validation. Only a locked fc_only
OOF winner may be refitted and receive its one final outer-validation evaluation.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import prepare_hse_groupcv_v11 as preparation
import train_expanded_validation_v10 as v10

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "hse_groupcv_v11"
CLASSES = list(v10.CLASSES)
MODES = ("full", "fc_only")
SETTINGS = {**v10.SETTINGS, "checkpoint_selection": "fixed epoch12 EMA only"}
PLAN_NAME = "logs/hse_groupcv_v11_plan.json"
OUTPUT_NAME = "experiments/hse_groupcv_v11"
HEAD_NAMES = {"classifier.weight", "classifier.bias"}
model_api, math_v7, h, c = v10.model_api, v10.math_v7, v10.h, v10.c
BINDINGS = ("classes", "source_pins", "plan_sha256", "parent_checkpoint_sha256",
    "model_source_receipt", "model_source_receipt_sha256", "train_indices", "val_indices",
    "phase", "mode", "fold", "run_train_indices", "run_validation_indices",
    "run_train_rows_sha256", "run_validation_rows_sha256", "run_train_image_content_sha256",
    "run_validation_image_content_sha256", "selection_sha256")


def emit(stage, **values):
    v10.emit(stage, **values)


def full_reuse_contract():
    """CPU/source-only mathematical ABI; never reads outer data or old results."""
    difference = {key: [v10.SETTINGS.get(key), SETTINGS.get(key)]
                  for key in sorted(set(v10.SETTINGS) | set(SETTINGS))
                  if v10.SETTINGS.get(key) != SETTINGS.get(key)}
    if difference != {"checkpoint_selection": ["fixed epoch12 EMA; epoch0 diagnostic only", "fixed epoch12 EMA only"]}:
        raise ValueError("V11 full training settings must equal V10 apart from selection-description text")
    functions = {"seed": h.seed_everything, "official_initialization": model_api.load_initial_model,
        "full_trainability": v10.configure_trainability, "optimizer": v10.make_optimizer,
        "loader_and_augmentation": v10.make_loader, "learning_rate": v10.set_learning_rate,
        "group_step": math_v7.group_step, "scaler": math_v7.ManualGradScaler,
        "ema": math_v7.FullModelEMA, "rng_state": h.rng_state, "restore_rng": h.restore_rng,
        "validation_views": v10.validate_views, "metrics": v10.metrics}
    return {"format": FORMAT+"_full_reuse_math_contract", "state": "passed", "classes": CLASSES,
        "training_settings_equal": True, "settings_text_differences": difference,
        "training_settings": {key: value for key, value in SETTINGS.items() if key != "checkpoint_selection"},
        "shared_helpers": {key: value.__module__+"."+value.__qualname__ for key, value in functions.items()},
        "shared_source_sha256": {str(Path(module.__file__).resolve()): h.sha256(module.__file__)
                                  for module in (v10, model_api, math_v7, h, c)},
        "initialization_order": ["seed_everything42", "official_HSE_load_initial_model", "configure_trainability",
            "frozen_reference_CPU_copy", "make_optimizer", "ManualGradScaler", "FullModelEMA", "epoch00_commit"],
        "uniform_standard_CE": True, "label_smoothing": 0., "head_warmup_epochs": 0,
        "group_step_rho": 0., "group_size": 32, "microbatch_size": 16, "actual_tail_denominator": True,
        "successful_step_scheduler": True, "successful_step_EMA": True,
        "V10_diagnostic_preserves_all_RNG": True, "V11_epoch00_evaluation": False,
        "final_epoch": 12, "final_weight_source": "ema", "final_flip_probability_average": .5,
        "outer_validation_read": False, "old_artifacts_read": False, "GPU_used": False}


def checked_plan(root):
    root = Path(root).resolve()
    plan = preparation.verify_plan(root, verify_images=False)
    if (plan.get("format") != FORMAT+"_locked" or plan.get("classes") != CLASSES
        or plan.get("settings") != SETTINGS):
        raise ValueError("V11 plan differs from the fixed paired twelve-epoch HSE protocol")
    required = {Path(__file__).resolve(), Path(preparation.__file__).resolve(), Path(v10.__file__).resolve(),
        Path(model_api.__file__).resolve(), Path(math_v7.__file__).resolve(), Path(h.__file__).resolve(), Path(c.__file__).resolve()}
    pins = plan.get("source_pins", {})
    if not required.issubset({Path(path).resolve() if Path(path).is_absolute() else (root/path).resolve() for path in pins}):
        raise ValueError("V11 plan omits a required source dependency")
    c.verify_pins(root, pins)
    if (plan.get("model_source_receipt") != model_api.source_receipt(root)
        or plan.get("model_source_receipt_sha256") != h.signature(plan["model_source_receipt"])
        or plan.get("parent_checkpoint_sha256") != model_api.OFFICIAL_SHA):
        raise ValueError("V11 official HSE initialization differs")
    parent = Path(plan["parent_checkpoint"])
    if not parent.is_absolute():
        parent = root/parent
    if parent.resolve() != (root/"models/hsemotion/enet_b2_7.pt").resolve():
        raise ValueError("V11 cannot initialize from a task-trained checkpoint")
    h.check_hash(parent, model_api.OFFICIAL_SHA)
    folds = plan.get("folds")
    if not isinstance(folds, list) or len(folds) != 3 or [fold.get("fold") for fold in folds] != [0, 1, 2]:
        raise ValueError("V11 requires exactly three locked training-only folds")
    return plan


def _selection(root, plan):
    path = Path(root)/OUTPUT_NAME/"cv_selection.json"
    selection = c.read_json(path)
    if (selection.get("state") not in ("locked", "complete") or selection.get("selected_mode") != "fc_only"
        or selection.get("classes") != CLASSES or selection.get("source_pins") != plan["source_pins"]
        or selection.get("plan_sha256") != h.sha256(Path(root)/PLAN_NAME)):
        raise ValueError("V11 refit requires a locked source-bound fc_only OOF winner")
    # The independent data gate also verifies the selection's companion lock receipt.
    return h.sha256(path)


def config_for(root, plan, fold, mode, phase="cv"):
    if phase not in ("cv", "refit") or mode not in MODES:
        raise ValueError("Unknown V11 training phase/mode")
    if phase == "cv":
        if fold not in (0, 1, 2):
            raise ValueError("V11 CV requires an explicit fold0/1/2")
        contract = plan["folds"][fold]
        train, evaluation = contract["train_indices"], contract["val_indices"]
        train_rows_sha, eval_rows_sha = contract["train_rows_sha256"], contract["val_rows_sha256"]
        train_image_sha, eval_image_sha = contract["train_image_content_sha256"], contract["val_image_content_sha256"]
        selection_sha = None
    else:
        if fold is not None or mode != "fc_only":
            raise ValueError("V11 only refits a locked fc_only winner; a full winner reuses verified V10")
        selection_sha = _selection(root, plan)
        contract = None
        train, evaluation = plan["train_indices"], plan["val_indices"]
        train_rows_sha, eval_rows_sha = plan["train_rows_sha256"], plan["val_rows_sha256"]
        train_image_sha, eval_image_sha = plan["train_image_content_sha256"], plan["val_image_content_sha256"]
    return {"format": FORMAT+"_config", "plan": plan, "plan_sha256": h.sha256(Path(root)/PLAN_NAME),
        **{key: plan[key] for key in ("classes", "settings", "source_pins", "parent_checkpoint",
            "parent_checkpoint_sha256", "model_source_receipt", "model_source_receipt_sha256", "train_indices", "val_indices")},
        "model_provenance": plan["model_source_receipt"], "phase": phase, "mode": mode, "fold": fold,
        "run_train_indices": train, "run_validation_indices": evaluation, "run_train_rows_sha256": train_rows_sha,
        "run_validation_rows_sha256": eval_rows_sha, "run_train_image_content_sha256": train_image_sha,
        "run_validation_image_content_sha256": eval_image_sha, "selection_sha256": selection_sha,
        "fold_contract": contract, "effective_train_rows": len(train), "evaluation_rows": len(evaluation),
        "train_sample_weights": "uniform1", "validation_weighted": False, "per_epoch_evaluation": False,
        "checkpoint_selection": "fixed epoch12 EMA only", "epoch0_purpose": "initialization_only",
        "outer_validation_used_for_cv": False, "internal_test_read": False, "external_answers_used": False,
        "preprocessing": "exact frozen V10 EXIF transpose RGB PIL bilinear224 ImageNet mean/std",
        "augmentation": "exact frozen V10 flip.5 affine6deg translate.03 scale.95..1.05 brightness/contrast.1",
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}


def configure_trainability(model, mode, training=False):
    if mode not in MODES:
        raise ValueError("Unknown V11 trainability mode")
    head = model.get_classifier()
    if not isinstance(head, nn.Linear) or head.out_features != 6:
        raise ValueError("V11 requires the native six-class HSE Linear classifier")
    parameters = dict(model.named_parameters())
    head_ids = {id(parameter) for parameter in head.parameters()}
    if {name for name, value in parameters.items() if id(value) in head_ids} != HEAD_NAMES:
        raise ValueError("V11 official classifier names changed")
    if mode == "full":
        v10.configure_trainability(model, training)
    else:
        model.requires_grad_(False)
        head.requires_grad_(True)
        model.eval()
        head.train(bool(training))
        math_v7.freeze_batchnorm(model)
    expected = set(parameters) if mode == "full" else HEAD_NAMES
    if {name for name, parameter in parameters.items() if parameter.requires_grad} != expected:
        raise ValueError("V11 trainable parameter set differs from the fixed regime")
    if mode == "fc_only" and any(module.training for name, module in model.named_modules() if name != "classifier"):
        raise ValueError("V11 frozen backbone must stay entirely in eval mode")
    return model


def make_optimizer(model, settings, mode):
    if mode == "full":
        return v10.make_optimizer(model, settings)
    if mode != "fc_only":
        raise ValueError("Unknown V11 optimizer mode")
    return torch.optim.AdamW([{"params": list(model.get_classifier().parameters()), "name": "head",
        "lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])


def frozen_reference(model, mode):
    if mode not in MODES:
        raise ValueError("Unknown V11 frozen-state mode")
    parameters = set(dict(model.named_parameters()))
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key not in parameters or (mode == "fc_only" and key not in HEAD_NAMES)}


def frozen_digest(reference):
    return v10.frozen_digest(reference)


def assert_frozen(model, reference, mode):
    current = frozen_reference(model, mode)
    if set(current) != set(reference) or any(not torch.equal(current[key], value) for key, value in reference.items()):
        raise ValueError("V11 official buffers or fc_only backbone parameters changed")
    if mode == "fc_only" and any(parameter.grad is not None for name, parameter in model.named_parameters() if name not in HEAD_NAMES):
        raise ValueError("V11 fc_only backbone acquired a gradient")


def dropout_configuration(model):
    dropout, drop_path, rates = [], [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)):
            dropout.append({"name": name, "class": type(module).__name__, "p": float(module.p), "training": module.training})
        if "droppath" in type(module).__name__.lower():
            drop_path.append({"name": name, "class": type(module).__name__, "drop_prob": float(getattr(module, "drop_prob", 0.)), "training": module.training})
        for attribute in ("drop_rate", "drop_path_rate"):
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float)):
                rates.append({"name": name, "attribute": attribute, "value": float(value), "training": module.training})
    return {"dropout_modules": dropout, "drop_path_modules": drop_path, "rate_attributes": rates,
        "positive_rates_present": any(row["p"] > 0 for row in dropout)
            or any(row["drop_prob"] > 0 for row in drop_path) or any(row["value"] > 0 for row in rates)}


def metrics(probabilities, rows):
    return v10.metrics(probabilities, rows)


def validate_views(model, rows, device="cpu", batch_size=16):
    return v10.validate_views(model, rows, device, batch_size)


def model_payload(model, config, epoch, source):
    return {"format": FORMAT, "config": config, **{key: config[key] for key in BINDINGS},
        "parent_sha": config["parent_checkpoint_sha256"], "model_provenance": config["model_source_receipt"],
        "epoch": epoch, "weight_source": source, "model_state": h.cpu_tree(model.state_dict()),
        "purpose": "fixed_final_epoch" if epoch == 12 and source == "ema" else "training_checkpoint"}


def verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("V11 requires its own full HSE state checkpoint format")
    plan = checked_plan(root)
    expected = config_for(root, plan, payload.get("fold"), payload.get("mode"), payload.get("phase"))
    if payload.get("config") != expected or (config is not None and config != expected):
        raise ValueError("V11 checkpoint configuration differs from the immutable phase/fold plan")
    if any(payload.get(key) != expected[key] for key in BINDINGS):
        raise ValueError("V11 checkpoint source/member/selection binding differs")
    if payload.get("parent_sha") != model_api.OFFICIAL_SHA or payload.get("model_provenance") != expected["model_source_receipt"]:
        raise ValueError("V11 checkpoint must derive from the official HSE source")
    if type(payload.get("epoch")) is not int or not 0 <= payload["epoch"] <= 12:
        raise ValueError("V11 checkpoint epoch is outside the fixed protocol")
    return expected


def load_checkpoint(root, path, device="cpu"):
    root = Path(root).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True) if isinstance(path, (str, os.PathLike)) else path
    config = verify_payload(root, payload)
    official = configure_trainability(model_api.load_initial_model(root, "cpu"), config["mode"])
    reference = frozen_reference(official, config["mode"])
    v10.check_state(payload.get("model_state"), official.state_dict())
    model = configure_trainability(model_api.load_trained_model(root, payload, device), config["mode"])
    assert_frozen(model, reference, config["mode"])
    return model.eval(), payload


def commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, reference):
    relative = f"epochs/epoch_{epoch:02d}.pt"
    payload = {**model_payload(model, config, epoch, "raw"), "optimizer": h.cpu_tree(optimizer.state_dict()),
        "scaler": scaler.state_dict(), "ema_state": ema.state_dict(), "rng": h.rng_state(),
        "counters": dict(counters), "successful_steps": counters["successful_steps"],
        "cumulative_skips": counters["amp_skipped_steps"], "zero_weight_groups": 0,
        "history": list(history), "frozen_state_sha256": frozen_digest(reference)}
    h.save_torch(output/relative, payload)
    v10._atomic_copy(output/relative, output/"last.pt")
    h.write_json(output/"last.json", {"epoch": epoch, "checkpoint": relative, "sha256": h.sha256(output/relative),
        **{key: config[key] for key in ("plan_sha256", "phase", "mode", "fold", "selection_sha256")}})
    h.write_json(output/"history.json", history)


def verify_history(history, counters, config, epoch):
    if len(history) != epoch or [row["epoch"] for row in history] != list(range(1, epoch+1)):
        raise ValueError("V11 history must contain exactly the committed complete epochs")
    groups = math.ceil(len(config["run_train_indices"])/32)
    cumulative = skips = 0
    for row in history:
        if (row["seen"] != len(config["run_train_indices"]) or row["successful_optimizer_steps"] <= 0
            or row["successful_optimizer_steps"]+row["amp_skipped_steps"] != groups or row.get("zero_weight_groups") != 0
            or any(key in row for key in ("val_accuracy", "val_metrics", "validation", "single_original_metrics", "single_tta_metrics"))):
            raise ValueError("V11 epoch missed training members/groups or contains an intermediate validation result")
        cumulative += row["successful_optimizer_steps"]; skips += row["amp_skipped_steps"]
        if row["cumulative_optimizer_steps"] != cumulative or row["ema_updates"] != cumulative:
            raise ValueError("V11 history optimizer/EMA cumulative counts differ")
    if counters != {"successful_steps": cumulative, "amp_skipped_steps": skips, "zero_weight_groups": 0}:
        raise ValueError("V11 complete-epoch counters differ")


def resume_epoch(root, output, config, model, optimizer, scaler, ema, reference):
    pointer = c.read_json(output/"last.json")
    epoch = pointer.get("epoch")
    if type(epoch) is not int or not 0 <= epoch <= 12 or pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt":
        raise ValueError("V11 resume pointer must reference a complete fixed epoch")
    if any(pointer.get(key) != config[key] for key in ("plan_sha256", "phase", "mode", "fold", "selection_sha256")):
        raise ValueError("V11 resume pointer phase/source/selection differs")
    path = output/pointer["checkpoint"]
    h.check_hash(path, pointer["sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    verify_payload(root, saved, config)
    if saved.get("epoch") != epoch or saved.get("weight_source") != "raw" or saved.get("frozen_state_sha256") != frozen_digest(reference):
        raise ValueError("V11 resume model epoch/source/frozen reference differs")
    counters = saved["counters"]
    verify_history(saved["history"], counters, config, epoch)
    if saved.get("successful_steps") != counters["successful_steps"] or saved.get("cumulative_skips") != counters["amp_skipped_steps"]:
        raise ValueError("V11 resume counter aliases differ")
    v10.check_state(saved["model_state"], model.state_dict())
    v10.check_state(saved["ema_state"]["shadow"], model.state_dict())
    if epoch == 0:
        initial = model.state_dict()
        if (counters != {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
            or any(not torch.equal(value.cpu(), initial[key].detach().cpu()) for key, value in saved["model_state"].items())
            or any(not torch.equal(value, saved["model_state"][key]) for key, value in saved["ema_state"]["shadow"].items())):
            raise ValueError("V11 epoch00 raw/EMA must equal official initialization with zero updates")
    model.load_state_dict(saved["model_state"], strict=True)
    assert_frozen(model, reference, config["mode"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    ema.load_state_dict(saved["ema_state"])
    if ema.updates != counters["successful_steps"]:
        raise ValueError("V11 resumed EMA/optimizer counts differ")
    with ema.apply(model):
        assert_frozen(model, reference, config["mode"])
    h.restore_rng(saved["rng"])
    if not (output/"last.pt").exists() or h.sha256(output/"last.pt") != pointer["sha256"]:
        v10._atomic_copy(path, output/"last.pt")
    h.write_json(output/"history.json", saved["history"])
    emit("resume_hse_groupcv_v11", phase=config["phase"], mode=config["mode"], fold=config["fold"],
         completed_epoch=epoch, successful_optimizer_steps=counters["successful_steps"])
    return epoch, counters, saved["history"]


def _verify_final(root, output, config, rows):
    result = c.read_json(output/"final_result.json")
    if (result.get("state") != "complete" or result.get("epoch") != 12 or result.get("weight_source") != "ema"
        or result.get("purpose") != "fixed_final_epoch"
        or any(result.get(key) != config[key] for key in BINDINGS)):
        raise ValueError("V11 final result must be the source-bound fixed epoch12 EMA")
    files = {"model": "final_ema.pt", "original": "original.npy", "flip": "flip.npy"}
    for key, relative in files.items():
        if result.get(key) != relative:
            raise ValueError("V11 final artifact filename differs")
        h.check_hash(output/relative, result["artifact_sha256"][key])
    payload = torch.load(output/"final_ema.pt", map_location="cpu", weights_only=True)
    verify_payload(root, payload, config)
    if (payload.get("epoch") != 12 or payload.get("weight_source") != "ema"
        or payload.get("purpose") != "fixed_final_epoch"):
        raise ValueError("V11 final model is not epoch12 EMA")
    original, flip = (np.load(output/name, allow_pickle=False) for name in ("original.npy", "flip.npy"))
    for probability in (original, flip):
        if probability.dtype != np.float32:
            raise ValueError("V11 final probability persistence must be float32")
        c.check_probability(probability, len(rows))
    computed = {"single_original_metrics": metrics(original, rows), "single_tta_metrics": metrics((original+flip)*.5, rows)}
    for key, value in computed.items():
        if result.get(key) != value or payload.get(key) != value:
            raise ValueError("V11 final probabilities do not reproduce their unweighted metrics")
    for alias, key in (("final_original_metrics", "single_original_metrics"), ("final_tta_metrics", "single_tta_metrics")):
        if result.get(alias) != computed[key] or payload.get(alias) != computed[key]:
            raise ValueError("V11 final metric aliases differ")
    last = torch.load(output/"epochs/epoch_12.pt", map_location="cpu", weights_only=True)
    shadow = last["ema_state"]["shadow"]
    if set(payload["model_state"]) != set(shadow) or any(not torch.equal(value, shadow[key]) for key, value in payload["model_state"].items()):
        raise ValueError("V11 final model differs from the complete last-epoch EMA shadow")
    return result


def finalize_run(root, output, config, model, ema, reference, rows, counters, history):
    verify_history(history, counters, config, 12)
    if ema.updates != counters["successful_steps"] or [row["manifest_index"] for row in rows] != config["run_validation_indices"]:
        raise ValueError("V11 final evaluation requires twelve complete epochs and exact evaluation members")
    if h.signature(rows) != config["run_validation_rows_sha256"]:
        raise ValueError("V11 final evaluation rows differ from the fixed fold/outer receipt")
    if (output/"final_result.json").exists():
        result = _verify_final(root, output, config, rows)
    else:
        with ema.apply(model):
            assert_frozen(model, reference, config["mode"])
            original, flip = validate_views(model, rows, next(model.parameters()).device, SETTINGS["validation_batch_size"])
            first, averaged = metrics(original, rows), metrics((original+flip)*.5, rows)
            payload = {**model_payload(model, config, 12, "ema"), "single_original_metrics": first,
                "single_tta_metrics": averaged, "final_original_metrics": first, "final_tta_metrics": averaged,
                "successful_steps": counters["successful_steps"], "ema_updates": ema.updates,
                "frozen_state_sha256": frozen_digest(reference)}
            h.save_torch(output/"final_ema.pt", payload)
        h.save_array(output/"original.npy", original)
        h.save_array(output/"flip.npy", flip)
        files = {"model": "final_ema.pt", "original": "original.npy", "flip": "flip.npy"}
        result = {"format": FORMAT+"_final_result", "state": "complete", "epoch": 12, "weight_source": "ema",
            "purpose": "fixed_final_epoch",
            **{key: config[key] for key in BINDINGS}, **files,
            "single_original_metrics": first, "single_tta_metrics": averaged,
            "final_original_metrics": first, "final_tta_metrics": averaged,
            "validation_weighted": False, "checkpoint_selection": False,
            "outer_validation_evaluated": config["phase"] == "refit", "outer_validation_used_for_cv": False,
            "artifact_sha256": {key: h.sha256(output/value) for key, value in files.items()}}
        h.write_json(output/"final_result.json", result)
        _verify_final(root, output, config, rows)
    assert_frozen(model, reference, config["mode"])
    pointer = c.read_json(output/"last.json")
    if pointer.get("epoch") != 12:
        raise ValueError("V11 final results require twelve committed epochs")
    h.check_hash(output/"last.pt", pointer["sha256"])
    files = ["config.json", "history.json", "last.json", "last.pt", "final_result.json", "final_ema.pt", "original.npy", "flip.npy"]
    files.extend(f"epochs/epoch_{epoch:02d}.pt" for epoch in range(13))
    summary = {"format": FORMAT+"_summary", "state": "complete", "epochs_completed": 12, "weight_source": "ema",
        **{key: config[key] for key in BINDINGS}, "effective_train_rows": len(config["run_train_indices"]),
        "evaluation_rows": len(rows), "single_original_metrics": result["single_original_metrics"],
        "single_tta_metrics": result["single_tta_metrics"], "final_original_metrics": result["single_original_metrics"],
        "final_tta_metrics": result["single_tta_metrics"], "counters": dict(counters), "history": history,
        "successful_optimizer_steps": counters["successful_steps"], "amp_skipped_steps": counters["amp_skipped_steps"],
        "ema_updates": ema.updates, "zero_weight_groups": 0, "frozen_state_sha256": frozen_digest(reference),
        "frozen_state_unchanged": True, "batchnorm_buffers_frozen": True, "validation_weighted": False,
        "per_epoch_evaluation": False, "epoch0_purpose": "initialization_only", "checkpoint_selection": False,
        "outer_validation_evaluated": config["phase"] == "refit", "outer_validation_used_for_cv": False,
        "internal_test_read": False, "external_answers_used": False,
        "artifact_sha256": {relative: h.sha256(output/relative) for relative in files}}
    if (output/"summary.json").exists():
        if c.read_json(output/"summary.json") != summary:
            raise ValueError("V11 complete summary does not reproduce its committed state/artifacts")
    else:
        h.write_json(output/"summary.json", summary)
    emit("hse_groupcv_v11_run_complete", phase=config["phase"], mode=config["mode"], fold=config["fold"],
         correct_count=summary["single_tta_metrics"]["correct_count"], samples=len(rows),
         successful_optimizer_steps=counters["successful_steps"], amp_skipped_steps=counters["amp_skipped_steps"])
    return summary


def run_training(root, fold=None, mode=None, refit=False, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V11 CUDA unavailable; never silently fall back to CPU training")
    torch.set_num_threads(4)
    plan = checked_plan(root)
    phase = "refit" if refit else "cv"
    mode = "fc_only" if refit and mode is None else mode
    config = config_for(root, plan, fold, mode, phase)
    train_rows, eval_rows = preparation.records_for(plan, fold, root=root, outer=False, verify_images=True)
    if ([row["manifest_index"] for row in train_rows] != config["run_train_indices"]
        or h.signature(train_rows) != config["run_train_rows_sha256"]):
        raise ValueError("V11 training records differ from the exact configured fold/pool")
    if refit:
        if eval_rows:
            raise ValueError("V11 refit must not read outer validation before epoch12")
        output = root/OUTPUT_NAME/"refit"/mode
    else:
        if ([row["manifest_index"] for row in eval_rows] != config["run_validation_indices"]
            or h.signature(eval_rows) != config["run_validation_rows_sha256"]):
            raise ValueError("V11 heldout training-fold records differ")
        output = root/OUTPUT_NAME/"cv"/mode/f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    (output/"epochs").mkdir(exist_ok=True)
    if (output/"config.json").exists() and c.read_json(output/"config.json") != config:
        raise ValueError("V11 existing run configuration differs; preserve its artifacts")
    if not (output/"config.json").exists():
        h.write_json(output/"config.json", config)
    if (output/"summary.json").exists():
        existing = c.read_json(output/"summary.json")
        if existing.get("state") != "complete" or any(existing.get(key) != config[key] for key in BINDINGS):
            raise ValueError("V11 existing summary is incomplete or belongs to another run")
        for relative, digest in existing["artifact_sha256"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("V11 completed artifact escapes its dedicated run")
            h.check_hash(output/path, digest)
    # Same initialization/RNG order and mathematical helpers as the frozen V10 full run.
    h.seed_everything(SETTINGS["seed"])
    model = configure_trainability(model_api.load_initial_model(root, device), mode)
    reference = frozen_reference(model, mode)
    optimizer = make_optimizer(model, SETTINGS, mode)
    scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", SETTINGS["amp_initial_scale"], SETTINGS["amp_growth_interval"])
    ema = math_v7.FullModelEMA(model, SETTINGS["ema_decay"])
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    history, done = [], 0
    if (output/"last.json").exists():
        done, counters, history = resume_epoch(root, output, config, model, optimizer, scaler, ema, reference)
    else:
        commit_epoch(output, model, optimizer, scaler, ema, config, 0, counters, history, reference)
    groups = math.ceil(len(train_rows)/32)
    total_steps = groups*12
    for epoch in range(done+1, 13):
        loader = v10.make_loader(train_rows, True, epoch, SETTINGS, device)
        configure_trainability(model, mode, True)
        seen = success = skips = diagnostic_seen = correct = 0
        loss_sum = 0.
        started = stamp = time.monotonic()
        for number, (images, targets) in enumerate(loader, 1):
            v10.set_learning_rate(optimizer, counters["successful_steps"], total_steps, SETTINGS)
            result = math_v7.group_step(model, optimizer, scaler, ema, images, targets,
                                       device=device, batch_size=16, rho=0.)
            seen += len(targets)
            okay = bool(result["success"])
            success += int(okay); skips += int(not okay)
            counters["successful_steps"] += int(okay); counters["amp_skipped_steps"] += int(not okay)
            if result["loss_sum"] is not None:
                diagnostic_seen += len(targets); loss_sum += float(result["loss_sum"]); correct += int(result["correct"])
            if skips >= 25 and success == 0:
                raise RuntimeError("V11 hit25 AMP skips without a successful update")
            if time.monotonic()-stamp >= 45 or number == len(loader):
                emit("train_hse_groupcv_v11", phase=phase, mode=mode, fold=fold, epoch=epoch, epochs=12,
                    batch=number, batches=len(loader), seen=seen, train_loss=loss_sum/max(1, diagnostic_seen),
                    successful_optimizer_steps=success, amp_skipped_steps=skips,
                    cumulative_optimizer_steps=counters["successful_steps"], elapsed_seconds=time.monotonic()-started)
                stamp = time.monotonic()
        if seen != len(train_rows) or success <= 0 or success+skips != groups:
            raise ValueError("V11 epoch missed training members/groups or never updated")
        assert_frozen(model, reference, mode)
        history.append({"epoch": epoch, "seen": seen, "augmentation_epoch": SETTINGS["augmentation_epoch_offset"]+epoch-1,
            "train_loss": loss_sum/max(1, diagnostic_seen), "train_accuracy": correct/max(1, diagnostic_seen),
            "diagnostic_samples": diagnostic_seen, "successful_optimizer_steps": success, "amp_skipped_steps": skips,
            "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
            "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-started,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}})
        c.verify_pins(root, config["source_pins"])
        commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, reference)
        emit("hse_groupcv_v11_epoch_committed", phase=phase, mode=mode, fold=fold, **history[-1])
    if refit:
        # This gate verifies the locked fc_only OOF decision and its companion lock receipt.
        _, eval_rows = preparation.records_for(plan, None, root=root, outer=True, verify_images=True)
        if _selection(root, plan) != config["selection_sha256"]:
            raise ValueError("V11 OOF decision changed before the final outer confirmation")
    return finalize_run(root, output, config, model, ema, reference, eval_rows, counters, history)


def smoke(root, device="cuda", fold=0):
    """Two real updates in each independently official-initialized arm; no model files."""
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V11 CUDA unavailable for the explicitly requested smoke")
    torch.set_num_threads(4)
    plan = checked_plan(root)
    training, _ = preparation.records_for(plan, fold, root=root, outer=False, verify_images=False)
    selected = training[:64]
    if len(selected) != 64:
        raise ValueError("V11 smoke requires two complete effective32 groups")
    reports, augmentation_hashes = {}, []
    for mode in MODES:
        h.seed_everything(42)
        model = configure_trainability(model_api.load_initial_model(root, device), mode, True)
        before = h.cpu_tree(model.state_dict())
        reference = frozen_reference(model, mode)
        stochastic = dropout_configuration(model)
        optimizer = make_optimizer(model, SETTINGS, mode)
        scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", 1024., 2000)
        ema = math_v7.FullModelEMA(model, .995)
        if any(not torch.equal(before[key], ema.shadow[key].cpu()) for key in before):
            raise ValueError("V11 initial EMA differs from official initialization")
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        augmentation_digest = hashlib.sha256()
        success = skips = 0
        started = time.monotonic()
        for images, targets in v10.make_loader(selected, True, 1, SETTINGS, device):
            augmentation_digest.update(images.numpy().tobytes()); augmentation_digest.update(targets.numpy().tobytes())
            v10.set_learning_rate(optimizer, success, math.ceil(len(training)/32)*12, SETTINGS)
            result = math_v7.group_step(model, optimizer, scaler, ema, images, targets, device=device, batch_size=16, rho=0.)
            success += int(result["success"]); skips += int(not result["success"])
            assert_frozen(model, reference, mode)
        if success != 2 or skips or ema.updates != 2:
            raise RuntimeError("V11 smoke requires two successful non-skipped optimizer/EMA updates in each arm")
        after = h.cpu_tree(model.state_dict())
        changed = {key for key in before if not torch.equal(before[key], after[key])}
        if not HEAD_NAMES.issubset(changed) or (mode == "full" and not changed-HEAD_NAMES) or (mode == "fc_only" and changed-HEAD_NAMES):
            raise ValueError("V11 smoke updated the wrong parameter set or did not update the classifier")
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        parameter_steps = {name: int(optimizer.state[value]["step"].item()) if "step" in optimizer.state[value] else 0 for name, value in parameters.items()}
        if any(value != 2 for value in parameter_steps.values()):
            raise ValueError("V11 smoke failed to update every intended trainable parameter twice")
        with ema.apply(model):
            assert_frozen(model, reference, mode)
        augmentation_hashes.append(augmentation_digest.hexdigest())
        reports[mode] = {"state": "passed", "successful_optimizer_steps": success, "amp_skipped_steps": skips,
            "ema_updates": ema.updates, "head_updated": True, "frozen_state_unchanged": True,
            "batchnorm_buffers_frozen": True, "changed_state_keys": sorted(changed),
            "trainable_parameter_names": sorted(parameters), "all_parameter_steps": parameter_steps,
            "dropout_configuration": stochastic, "training_augmented_batches_sha256": augmentation_hashes[-1],
            "elapsed_seconds": time.monotonic()-started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None}
        del model, optimizer, scaler, ema, before, after, reference
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    if len(set(augmentation_hashes)) != 1:
        raise ValueError("V11 two arms did not receive identical first64 augmented training images")
    return {"state": "passed", "classes": CLASSES, "plan_sha256": h.sha256(root/PLAN_NAME),
        "source_pins": plan["source_pins"], "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
        "model_source_receipt_sha256": plan["model_source_receipt_sha256"], "modes": reports,
        "fold": fold, "batch_size": 16, "effective_batch_size": 32, "weights_saved": False,
        "training_manifest_indices": [row["manifest_index"] for row in selected], "paired_augmentation_identical": True,
        "GPU_used": torch.device(device).type == "cuda", "validation_evaluated": False,
        "internal_test_read": False, "external_answers_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("smoke", "train"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("V11 only supports the fixed project root")
    if args.stage == "smoke":
        if args.refit or args.mode is not None:
            raise ValueError("V11 smoke checks both CV arms; do not pass mode/refit")
        report = smoke(args.project_root, args.device, 0 if args.fold is None else args.fold)
        kind = "gpu" if torch.device(args.device).type == "cuda" else "cpu"
        h.write_json(args.project_root/f"logs/hse_groupcv_v11_{kind}_smoke.json", report)
        emit("hse_groupcv_v11_smoke_passed", plan_sha256=report["plan_sha256"], source_pins=report["source_pins"])
    elif args.refit:
        if args.fold is not None or args.mode not in (None, "fc_only"):
            raise ValueError("V11 refit requires fc_only without a fold")
        run_training(args.project_root, None, "fc_only", True, args.device)
    else:
        if args.fold is None or args.mode is None:
            raise ValueError("V11 CV requires an explicit fold and mode")
        run_training(args.project_root, args.fold, args.mode, False, args.device)


if __name__ == "__main__":
    main()
