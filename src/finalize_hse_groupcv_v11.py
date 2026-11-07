"""Training-pool grouped CV selection, gated outer confirmation and audited export.

Only final epoch12 EMA is evaluated. No outer data enter mode selection.
Full-mode selection reuses the exact audited V10 run without a second refit.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import csv
import gc
import math
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageOps
import train_hse_specialist as h
import validate_cosine_fusion as c
import hse_model as model_api
import prepare_hse_groupcv_v11 as p
import train_hse_groupcv_v11 as t

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "logs/hse_groupcv_v11_plan.json"
TRAIN = ROOT / "experiments/hse_groupcv_v11"
OUT = ROOT / "hse_groupcv_v11_results"
BASELINE = ROOT / "hse_expanded_validation_v10_results"
SELECTION = TRAIN / "cv_selection.json"
SELECTION_LOCK = TRAIN / "cv_selection_lock.json"
PROTOCOL = ROOT / "research/hse_groupcv_v11_protocol.json"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODES = ("full", "fc_only")
IMAGE_SHA = "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA = "8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
V10_SUBMISSION_SHA = "214d6bec3b1cd40f204cc8db3ad6c91f2db7516b65cd89d4e582f3195e0bc683"
V8_SUBMISSION_SHA = "55a037351e0677eadc82ebe81ce6a9c208da402f304cd9eeccba70e2121ce4b6"
CV_RULE = p.CV_RULE
OUTER_RULE = "only CV-selected fc_only final epoch12 EMA may replace V10, iff outer correct_count >3240 of4782; otherwise preserve V10"
CAVEAT = ("Training-pool near-duplicate-group OOF is not person-independent; official pretraining overlap is unknown. "
          "The outer development split was previously used, is not Kaggle scoring or a fresh final test, and cannot fairly rank old V8 task weights. "
          "No epoch, temperature, ensemble weight or TTA search is performed. Only the fixed final epoch12 EMA is eligible.")

def emit(stage, **values):
    import json
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def child(folder, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact must remain within its own output folder")
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Resolved artifact escapes its output folder")
    return path


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Producer path is outside the fixed project")
    digest = h.sha256(path)
    if expected is not None and digest != expected:
        raise ValueError("Producer checksum differs: " + str(path))
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError("Conflicting producer checksum: " + str(path))
    pins[str(path)] = digest


def same_state(first, second):
    return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)


def check_state(state, initial, trainable_names):
    if not isinstance(state, dict) or set(state) != set(initial):
        raise ValueError("Require complete native HSE state")
    for name, reference in initial.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != reference.dtype or value.shape != reference.shape
                or not torch.isfinite(value).all().item()):
            raise ValueError("Invalid native HSE tensor: " + name)
        if name not in trainable_names and not torch.equal(value, reference):
            raise ValueError("Frozen HSE parameter or buffer changed: " + name)


def check_fields(actual, expected, context):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(context + ": " + key)


def initial_reference(plan):
    receipt = model_api.source_receipt(ROOT)
    if (receipt != plan["model_source_receipt"] or h.signature(receipt) != plan["model_source_receipt_sha256"]
            or plan["parent_checkpoint_sha256"] != model_api.OFFICIAL_SHA):
        raise ValueError("All runs must start independently from the exact official HSE initialization")
    model = model_api.load_initial_model(ROOT, "cpu")
    initial = h.cpu_tree(model.state_dict())
    names = set(dict(model.named_parameters()))
    head_ids = {id(value) for value in model.get_classifier().parameters()}
    head = {name for name, value in model.named_parameters() if id(value) in head_ids}
    if head != {"classifier.weight", "classifier.bias"}:
        raise ValueError("Unexpected official HSE classifier names")
    del model
    return initial, names, head


def verify_payload(payload, config, initial, trainable_names):
    check_fields(payload, {"format": t.FORMAT, "config": config,
        **{key: config[key] for key in t.BINDINGS}, "parent_sha": model_api.OFFICIAL_SHA,
        "model_provenance": config["model_source_receipt"]}, "Checkpoint binding differs")
    check_state(payload.get("model_state"), initial, trainable_names)


def run_folder(phase, mode, fold):
    if phase == "cv" and mode in MODES and type(fold) is int and fold in range(3):
        return TRAIN / "cv" / mode / f"fold_{fold}"
    if phase == "refit" and mode == "fc_only" and fold is None:
        return TRAIN / "refit" / mode
    raise ValueError("Only the preregistered six CV runs and gated fc_only refit are allowed")


def verify_run(plan, phase, mode, fold, pins, initial, parameter_names, head_names):
    folder = run_folder(phase, mode, fold)
    config = t.config_for(ROOT, plan, fold, mode, phase=phase)
    train_rows, eval_rows = p.records_for(plan, fold, ROOT, outer=phase == "refit", verify_images=False)
    if (config["run_train_indices"] != [r["manifest_index"] for r in train_rows]
            or config["run_validation_indices"] != [r["manifest_index"] for r in eval_rows]
            or c.read_json(folder / "config.json") != config):
        raise ValueError("Run config or ordered fold members differ")
    trainable = parameter_names if mode == "full" else head_names
    frozen = {key: value for key, value in initial.items() if key not in trainable}
    frozen_sha = t.frozen_digest(frozen)
    summary = c.read_json(folder / "summary.json")
    check_fields(summary, {"state": "complete", "epochs_completed": 12, "weight_source": "ema",
        **{key: config[key] for key in t.BINDINGS}}, "Run summary incomplete or differently bound")
    must = {"config.json", "history.json", "last.json", "last.pt", "final_ema.pt", "original.npy", "flip.npy", "final_result.json"}
    must |= {f"epochs/epoch_{epoch:02d}.pt" for epoch in range(13)}
    if not must.issubset(summary.get("artifact_sha256", {})):
        raise ValueError("Run summary omits complete epochs or final outputs")
    for name, digest in summary["artifact_sha256"].items():
        pin(pins, child(folder, name), digest)
    pin(pins, folder / "summary.json")
    pointer = c.read_json(folder / "last.json")
    check_fields(pointer, {"epoch": 12, "checkpoint": "epochs/epoch_12.pt", "plan_sha256": h.sha256(PLAN)}, "Last epoch pointer differs")
    pin(pins, folder / "last.pt", pointer["sha256"])
    pin(pins, folder / "epochs/epoch_12.pt", pointer["sha256"])
    history = c.read_json(folder / "history.json")
    if len(history) != 12 or [entry["epoch"] for entry in history] != list(range(1, 13)):
        raise ValueError("Every run requires twelve committed training epochs")
    successes = skipped = 0
    counters_by_epoch = []
    for entry in history:
        n, k = entry["successful_optimizer_steps"], entry["amp_skipped_steps"]
        if (type(n) is not int or type(k) is not int or n <= 0 or k < 0 or entry["seen"] != len(train_rows)
                or n + k != math.ceil(len(train_rows) / 32)):
            raise ValueError("Training must account for each uniform logical group, including its tail")
        if any(key.startswith(("val_", "validation_", "heldout_", "dev_")) for key in entry):
            raise ValueError("Intermediate heldout metrics are forbidden")
        if entry.get("zero_weight_groups") != 0 or entry.get("augmentation_epoch") != plan["settings"]["augmentation_epoch_offset"] + entry["epoch"] - 1:
            raise ValueError("Uniform groups or paired deterministic augmentation changed")
        successes += n
        skipped += k
        if entry["cumulative_optimizer_steps"] != successes or entry["ema_updates"] != successes:
            raise ValueError("EMA and successful update history differ")
        counters_by_epoch.append({"successful_steps": successes, "amp_skipped_steps": skipped, "zero_weight_groups": 0})
    check_fields(summary, {"history": history, "counters": counters_by_epoch[-1],
        "successful_optimizer_steps": successes, "amp_skipped_steps": skipped, "ema_updates": successes,
        "zero_weight_groups": 0, "effective_train_rows": len(train_rows), "evaluation_rows": len(eval_rows),
        "frozen_state_sha256": frozen_sha, "frozen_state_unchanged": True, "batchnorm_buffers_frozen": True,
        "validation_weighted": False, "per_epoch_evaluation": False, "checkpoint_selection": False,
        "internal_test_read": False, "external_answers_used": False}, "Run counter or policy differs")
    for epoch in range(13):
        payload = torch.load(folder / f"epochs/epoch_{epoch:02d}.pt", map_location="cpu", weights_only=True)
        verify_payload(payload, config, initial, trainable)
        counters = counters_by_epoch[epoch - 1] if epoch else {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
        check_fields(payload, {"epoch": epoch, "weight_source": "raw", "history": history[:epoch], "counters": counters,
            "successful_steps": counters["successful_steps"], "cumulative_skips": counters["amp_skipped_steps"],
            "zero_weight_groups": 0, "frozen_state_sha256": frozen_sha}, "Committed epoch binding differs")
        scaler = payload.get("scaler", {})
        if (scaler.get("enabled") is not True or scaler.get("growth_interval") != plan["settings"]["amp_growth_interval"]
                or not isinstance(scaler.get("scale"), (int, float)) or not math.isfinite(scaler["scale"])
                or not 1 <= scaler["scale"] <= 2. ** 24
                or not 0 <= scaler.get("growth_tracker", -1) < scaler["growth_interval"]):
            raise ValueError("Invalid restorable CUDA AMP state")
        if epoch and scaler["scale"] != history[epoch - 1]["amp_scale"]:
            raise ValueError("AMP history differs from committed scaler")
        if not epoch and (scaler["scale"] != 1024. or scaler["growth_tracker"] != 0):
            raise ValueError("Independent runs require fresh AMP scalers")
        rng = payload.get("rng", {})
        if not {"python", "numpy", "torch", "cuda"}.issubset(rng) or not isinstance(rng["torch"], torch.Tensor) or not rng["cuda"]:
            raise ValueError("Committed epoch lacks restorable CPU/CUDA RNG")
        ema = payload["ema_state"]
        if ema.get("decay") != .995 or ema.get("updates") != counters["successful_steps"]:
            raise ValueError("EMA must update once per successful AdamW step")
        check_state(ema["shadow"], initial, trainable)
        optimizer = payload["optimizer"]["state"]
        if not epoch:
            if optimizer or not same_state(payload["model_state"], initial) or not same_state(ema["shadow"], initial):
                raise ValueError("Every run must start from exact official state with fresh optimizer and EMA")
        elif len(optimizer) != len(trainable) or any(int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"]) != counters["successful_steps"] for value in optimizer.values()):
            raise ValueError("Actual AdamW parameter updates differ from the trainable regime or counters")
        if epoch == 12:
            final_shadow = ema["shadow"]
        del payload
    final = torch.load(folder / "final_ema.pt", map_location="cpu", weights_only=True)
    verify_payload(final, config, initial, trainable)
    if not same_state(final["model_state"], final_shadow):
        raise ValueError("Final model must equal committed epoch12 EMA shadow tensor-for-tensor")
    check_fields(final, {"epoch": 12, "weight_source": "ema", "purpose": "fixed_final_epoch",
        "successful_steps": successes, "ema_updates": successes, "frozen_state_sha256": frozen_sha}, "Final EMA metadata differs")
    views = [np.load(folder / f"{view}.npy", allow_pickle=False) for view in ("original", "flip")]
    for array in views:
        if array.dtype != np.float32 or array.shape != (len(eval_rows), 6):
            raise ValueError("Final probability shape or dtype differs")
        c.check_probability(array, len(eval_rows))
    original = t.metrics(views[0], eval_rows)
    tta = t.metrics((views[0] + views[1]) * .5, eval_rows)
    result = c.read_json(folder / "final_result.json")
    for item in (summary, final, result):
        check_fields(item, {"single_original_metrics": original, "single_tta_metrics": tta}, "Final metrics do not reproduce")
    check_fields(result, {"state": "complete", "epoch": 12, "weight_source": "ema", "purpose": "fixed_final_epoch",
        "validation_weighted": False, "checkpoint_selection": False, **{key: config[key] for key in t.BINDINGS}}, "Final result binding differs")
    for key, name in {"model": "final_ema.pt", "original": "original.npy", "flip": "flip.npy"}.items():
        if result.get(key) != name:
            raise ValueError("Unexpected final artifact filename")
        pin(pins, folder / name, result["artifact_sha256"][key])
    audit = {"phase": phase, "mode": mode, "fold": fold, "single_original_metrics": original, "single_tta_metrics": tta,
        "summary_sha256": h.sha256(folder / "summary.json"), "final_model_sha256": h.sha256(folder / "final_ema.pt"),
        "successful_optimizer_steps": successes, "amp_skipped_steps": skipped, "ema_updates": successes,
        "frozen_state_sha256": frozen_sha, "run_train_indices": config["run_train_indices"],
        "run_validation_indices": config["run_validation_indices"]}
    del final, final_shadow
    gc.collect()
    return views, audit


def cv_decision(pooled, folds):
    wins = sum(folds["fc_only"][str(fold)]["correct_count"] > folds["full"][str(fold)]["correct_count"] for fold in range(3))
    selected = "fc_only" if pooled["fc_only"]["correct_count"] > pooled["full"]["correct_count"] and wins >= 2 else "full"
    return selected, wins


def selection_lock(selection):
    return {"format": "hse_groupcv_v11_cv_selection_lock", "state": "locked",
        "cv_selection_sha256": h.sha256(SELECTION), "plan_sha256": h.sha256(PLAN),
        "protocol_sha256": h.sha256(PROTOCOL), "source_pins": selection["source_pins"],
        "finalizer_sha256": h.sha256(__file__), "selected_mode": selection["selected_mode"]}


def verify_cv_selection(plan=None, require_lock=True):
    plan = plan or p.verify_plan(ROOT, verify_images=False)
    selection = c.read_json(SELECTION)
    check_fields(selection, {"format": "hse_groupcv_v11_cv_selection", "state": "locked", "classes": CLASSES,
        "protocol_sha256": h.sha256(PROTOCOL), "plan_sha256": h.sha256(PLAN), "source_pins": plan["source_pins"],
        "finalizer_sha256": h.sha256(__file__), "selection_rule": CV_RULE,
        "outer_development_used_for_selection": False}, "CV selection binding differs")
    c.verify_pins(ROOT, selection["producer_files"])
    records, empty = p.records_for(plan, None, ROOT, verify_images=False)
    indices = [row["manifest_index"] for row in records]
    if empty or len(indices) != 19120 or len(set(indices)) != 19120 or selection.get("oof_manifest_indices") != indices:
        raise ValueError("OOF must cover only all19120 training-pool rows in their fixed order")
    if set(selection.get("file_sha256", {})) != {"oof_full.npy", "oof_fc_only.npy"}:
        raise ValueError("Exactly two full-coverage OOF arrays must be locked")
    positions = {index: position for position, index in enumerate(indices)}
    folds = {mode: {} for mode in MODES}
    pooled = {}
    coverage = np.zeros(len(records), dtype=np.int64)
    fold_positions = {}
    for fold in range(3):
        train_rows, heldout = p.records_for(plan, fold, ROOT, verify_images=False)
        heldout_ids = [row["manifest_index"] for row in heldout]
        train_ids = [row["manifest_index"] for row in train_rows]
        if set(train_ids) & set(heldout_ids) or set(train_ids) | set(heldout_ids) != set(indices) or len(set(heldout_ids)) != len(heldout_ids):
            raise ValueError("CV fold members overlap, duplicate or omit training-pool rows")
        places = np.asarray([positions[index] for index in heldout_ids], dtype=np.int64)
        coverage[places] += 1
        fold_positions[fold] = (places, heldout)
    if not np.all(coverage == 1):
        raise ValueError("Each training-pool member must be held out exactly once")
    for mode in MODES:
        path = TRAIN / f"oof_{mode}.npy"
        h.check_hash(path, selection["file_sha256"][path.name])
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32:
            raise ValueError("OOF must retain exact float32 view averages")
        c.check_probability(array, len(records))
        pooled[mode] = t.metrics(array, records)
        for fold in range(3):
            places, heldout = fold_positions[fold]
            views = [np.load(run_folder("cv", mode, fold) / f"{view}.npy", allow_pickle=False) for view in ("original", "flip")]
            if not np.array_equal(array[places], (views[0] + views[1]) * .5):
                raise ValueError("OOF rows differ from the corresponding fold model view mean")
            folds[mode][str(fold)] = t.metrics(array[places], heldout)
    selected, wins = cv_decision(pooled, folds)
    check_fields(selection, {"pooled_oof_metrics": pooled, "fold_metrics": folds,
        "selected_mode": selected, "fc_only_fold_wins": wins}, "CV selection decision does not reproduce")
    if require_lock:
        if c.read_json(SELECTION_LOCK) != selection_lock(selection):
            raise ValueError("CV selection lock does not bind the exact selection file")
    return selection


def select_cv():
    plan = p.verify_plan(ROOT, verify_images=True)
    if SELECTION.exists():
        selection = verify_cv_selection(plan, require_lock=SELECTION_LOCK.exists())
        if not SELECTION_LOCK.exists():
            h.write_json(SELECTION_LOCK, selection_lock(selection))
        return verify_cv_selection(plan)
    if SELECTION_LOCK.exists():
        raise ValueError("CV lock exists without its selection; do not replace it")
    records, empty = p.records_for(plan, None, ROOT, verify_images=False)
    indices = [row["manifest_index"] for row in records]
    if empty or len(indices) != 19120 or len(set(indices)) != 19120:
        raise ValueError("CV pool must contain only the19120 frozen training members")
    positions = {index: number for number, index in enumerate(indices)}
    oof = {mode: np.empty((len(records), 6), dtype=np.float32) for mode in MODES}
    coverage = {mode: np.zeros(len(records), dtype=np.int64) for mode in MODES}
    audits = {mode: {} for mode in MODES}
    fold_metrics = {mode: {} for mode in MODES}
    pins = dict(plan["source_pins"])
    pin(pins, PLAN); pin(pins, PROTOCOL); pin(pins, __file__)
    initial, names, head = initial_reference(plan)
    for fold in range(3):
        for mode in MODES:
            views, audit = verify_run(plan, "cv", mode, fold, pins, initial, names, head)
            places = np.asarray([positions[index] for index in audit["run_validation_indices"]], dtype=np.int64)
            if len(set(places.tolist())) != len(places) or np.any(coverage[mode][places]):
                raise ValueError("Duplicate OOF coverage")
            oof[mode][places] = (views[0] + views[1]) * .5
            coverage[mode][places] += 1
            audits[mode][str(fold)] = audit
            fold_metrics[mode][str(fold)] = audit["single_tta_metrics"]
    if any(not np.all(value == 1) for value in coverage.values()):
        raise ValueError("Missing or repeated OOF members")
    pooled = {mode: t.metrics(array, records) for mode, array in oof.items()}
    selected, wins = cv_decision(pooled, fold_metrics)
    TRAIN.mkdir(parents=True, exist_ok=True)
    for mode, array in oof.items():
        h.save_array(TRAIN / f"oof_{mode}.npy", array)
        pin(pins, TRAIN / f"oof_{mode}.npy")
    selection = {"format": "hse_groupcv_v11_cv_selection", "state": "locked", "classes": CLASSES,
        "protocol_sha256": h.sha256(PROTOCOL), "plan_sha256": h.sha256(PLAN), "source_pins": plan["source_pins"],
        "finalizer_sha256": h.sha256(__file__), "producer_files": pins, "selection_rule": CV_RULE,
        "selected_mode": selected, "fc_only_fold_wins": wins, "pooled_oof_metrics": pooled, "fold_metrics": fold_metrics,
        "run_audits": audits, "oof_manifest_indices": indices, "outer_development_used_for_selection": False,
        "file_sha256": {f"oof_{mode}.npy": h.sha256(TRAIN / f"oof_{mode}.npy") for mode in MODES},
        "group_independent": True, "person_independent": False, "internal_test_read": False, "external_answers_used": False}
    c.verify_pins(ROOT, pins)
    h.write_json(SELECTION, selection)
    verify_cv_selection(plan, require_lock=False)
    h.write_json(SELECTION_LOCK, selection_lock(selection))
    emit("v11_cv_selected", selected_mode=selected, pooled_oof_metrics=pooled, fc_only_fold_wins=wins)
    return verify_cv_selection(plan)


def incumbent_receipt(plan):
    contract = p.verify_reuse_contract(plan)
    summary = c.read_json(BASELINE / "summary.json")
    recipe = c.read_json(BASELINE / "selected_recipe.json")
    if (summary.get("state") != "complete" or summary.get("selected_metrics", {}).get("correct_count") != 3240
            or summary["selected_metrics"].get("samples") != 4782 or recipe.get("selected_metrics") != summary["selected_metrics"]):
        raise ValueError("The verified V10 incumbent must remain3240/4782")
    for name, digest in summary["file_sha256"].items():
        h.check_hash(child(BASELINE, name), digest)
    h.check_hash(BASELINE / "submission.csv", V10_SUBMISSION_SHA)
    h.check_hash(ROOT / "ddamfn_v8_results/submission.csv", V8_SUBMISSION_SHA)
    return {"reuse_contract": contract, "reuse_contract_sha256": h.signature(contract),
        "summary_sha256": h.sha256(BASELINE / "summary.json"), "recipe_sha256": h.sha256(BASELINE / "selected_recipe.json"),
        "submission_sha256": V10_SUBMISSION_SHA, "selected_metrics": summary["selected_metrics"],
        "single_original_metrics": summary["single_original_metrics"], "v8_submission_sha256": V8_SUBMISSION_SHA,
        "old_checkpoint_outer_reevaluation": False}


def select():
    if (OUT / "selected_recipe.json").exists():
        return verify_recipe()[1]
    plan = p.verify_plan(ROOT, verify_images=True)
    selection = verify_cv_selection(plan)
    incumbent = incumbent_receipt(plan)
    pins = dict(selection["producer_files"])
    for path in (SELECTION, SELECTION_LOCK, BASELINE / "summary.json", BASELINE / "selected_recipe.json"):
        pin(pins, path)
    for name, digest in c.read_json(BASELINE / "summary.json")["file_sha256"].items():
        pin(pins, child(BASELINE, name), digest)
    pin(pins, ROOT / "ddamfn_v8_results/submission.csv", V8_SUBMISSION_SHA)
    audit = None
    adopted = False
    candidate_metrics = None
    if selection["selected_mode"] == "fc_only":
        initial, names, head = initial_reference(plan)
        views, audit = verify_run(plan, "refit", "fc_only", None, pins, initial, names, head)
        candidate_metrics = audit["single_tta_metrics"]
        if candidate_metrics["samples"] != 4782:
            raise ValueError("Only the fixed4782-image outer split may confirm the selected mode")
        adopted = candidate_metrics["correct_count"] > 3240
        del initial
    elif selection["selected_mode"] != "full":
        raise ValueError("Unknown CV mode")
    OUT.mkdir(parents=True, exist_ok=True)
    if adopted:
        probability = (views[0] + views[1]) * .5
        h.save_array(OUT / "selected_val_probabilities.npy", probability)
        metrics = audit["single_tta_metrics"]
        original_metrics = audit["single_original_metrics"]
        model = TRAIN / "refit/fc_only/final_ema.pt"
    else:
        atomic_copy(BASELINE / "selected_val_probabilities.npy", OUT / "selected_val_probabilities.npy")
        metrics = incumbent["selected_metrics"]
        original_metrics = incumbent["single_original_metrics"]
        model = ROOT / "experiments/hse_expanded_validation_v10/final_ema.pt"
    recipe = {"format": "hse_groupcv_v11_deployment", "state": "locked", "classes": CLASSES,
        "plan_sha256": h.sha256(PLAN), "protocol_sha256": h.sha256(PROTOCOL), "source_pins": plan["source_pins"],
        "finalizer_sha256": h.sha256(__file__), "producer_files": pins,
        "cv_selection_sha256": h.sha256(SELECTION), "cv_selection_lock_sha256": h.sha256(SELECTION_LOCK),
        "cv_selected_mode": selection["selected_mode"], "selection_rule": CV_RULE, "outer_rule": OUTER_RULE,
        "selected": "fc_only_epoch_12_ema" if adopted else "v10_epoch_12_ema", "new_candidate_adopted": adopted,
        "selected_model_path": str(model), "selected_model_sha256": h.sha256(model),
        "selected_metrics": metrics, "single_tta_metrics": metrics, "single_original_metrics": original_metrics,
        "candidate_outer_metrics": candidate_metrics, "refit_audit": audit,
        "outer_confirmation_performed": selection["selected_mode"] == "fc_only", "incumbent": incumbent,
        "candidate_status_vs_v8": "unranked_no_common_independent_split", "competition_new_inference_required": adopted,
        "pooled_oof_metrics": selection["pooled_oof_metrics"], "fold_metrics": selection["fold_metrics"],
        "fc_only_fold_wins": selection["fc_only_fold_wins"], "outer_development_used_for_cv": False,
        "selection_file_sha256": {"selected_val_probabilities.npy": h.sha256(OUT / "selected_val_probabilities.npy")},
        "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
        "fixed_prediction_rule": "equal original/flip probability mean", "old_task_fusion_used": False,
        "internal_test_read": False, "external_answers_used": False, "validation_caveat": CAVEAT}
    pin(pins, model)
    c.verify_pins(ROOT, pins)
    h.write_json(OUT / "selected_recipe.json", recipe)
    verify_recipe()
    emit("v11_deployment_selected", selected=recipe["selected"], cv_selected_mode=selection["selected_mode"],
         candidate_outer_metrics=candidate_metrics, selected_metrics=metrics)
    return recipe


def verify_recipe():
    plan = p.verify_plan(ROOT, verify_images=False)
    selection = verify_cv_selection(plan)
    recipe = c.read_json(OUT / "selected_recipe.json")
    check_fields(recipe, {"format": "hse_groupcv_v11_deployment", "state": "locked", "classes": CLASSES,
        "plan_sha256": h.sha256(PLAN), "protocol_sha256": h.sha256(PROTOCOL), "source_pins": plan["source_pins"],
        "finalizer_sha256": h.sha256(__file__), "cv_selection_sha256": h.sha256(SELECTION),
        "cv_selection_lock_sha256": h.sha256(SELECTION_LOCK), "cv_selected_mode": selection["selected_mode"],
        "selection_rule": CV_RULE, "outer_rule": OUTER_RULE, "incumbent": incumbent_receipt(plan),
        "pooled_oof_metrics": selection["pooled_oof_metrics"], "fold_metrics": selection["fold_metrics"],
        "fc_only_fold_wins": selection["fc_only_fold_wins"], "outer_development_used_for_cv": False,
        "candidate_status_vs_v8": "unranked_no_common_independent_split", "old_task_fusion_used": False,
        "fixed_prediction_rule": "equal original/flip probability mean", "internal_test_read": False,
        "external_answers_used": False, "validation_caveat": CAVEAT,
        "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA}, "Deployment binding differs")
    c.verify_pins(ROOT, recipe["producer_files"])
    if set(recipe.get("selection_file_sha256", {})) != {"selected_val_probabilities.npy"}:
        raise ValueError("Exactly one fixed outer prediction array must be deployed")
    h.check_hash(OUT / "selected_val_probabilities.npy", recipe["selection_file_sha256"]["selected_val_probabilities.npy"])
    candidate = None
    if selection["selected_mode"] == "fc_only":
        folder = run_folder("refit", "fc_only", None)
        _, outer = p.records_for(plan, None, ROOT, outer=True, verify_images=False)
        views = [np.load(folder / f"{view}.npy", allow_pickle=False) for view in ("original", "flip")]
        candidate = t.metrics((views[0] + views[1]) * .5, outer)
        audit = recipe.get("refit_audit")
        if (not isinstance(audit, dict) or audit.get("single_tta_metrics") != candidate
                or audit.get("summary_sha256") != h.sha256(folder / "summary.json")
                or audit.get("final_model_sha256") != h.sha256(folder / "final_ema.pt")
                or candidate["samples"] != 4782):
            raise ValueError("The sole refit confirmation differs from its audited result")
    elif recipe.get("refit_audit") is not None:
        raise ValueError("Full mode must reuse V10 without refit or another outer evaluation")
    adopted = candidate is not None and candidate["correct_count"] > 3240
    check_fields(recipe, {"candidate_outer_metrics": candidate, "new_candidate_adopted": adopted,
        "competition_new_inference_required": adopted, "outer_confirmation_performed": selection["selected_mode"] == "fc_only",
        "selected": "fc_only_epoch_12_ema" if adopted else "v10_epoch_12_ema"}, "Fixed adoption decision differs")
    if adopted:
        expected = (views[0] + views[1]) * .5
        model = TRAIN / "refit/fc_only/final_ema.pt"
        expected_metrics = candidate
        original_metrics = t.metrics(views[0], outer)
        saved = np.load(OUT / "selected_val_probabilities.npy", allow_pickle=False)
        if saved.dtype != np.float32 or not np.array_equal(saved, expected):
            raise ValueError("Deployed outer probabilities differ from final refit EMA views")
    else:
        h.check_hash(OUT / "selected_val_probabilities.npy", h.sha256(BASELINE / "selected_val_probabilities.npy"))
        model = ROOT / "experiments/hse_expanded_validation_v10/final_ema.pt"
        expected_metrics = recipe["incumbent"]["selected_metrics"]
        original_metrics = recipe["incumbent"]["single_original_metrics"]
    check_fields(recipe, {"selected_model_path": str(model), "selected_model_sha256": h.sha256(model),
        "selected_metrics": expected_metrics, "single_tta_metrics": expected_metrics,
        "single_original_metrics": original_metrics}, "Selected model or metric differs")
    return plan, recipe


def atomic_copy(source, destination):
    temporary = Path(str(destination) + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def predict_candidate(recipe, rows):
    if not recipe["new_candidate_adopted"] or recipe["cv_selected_mode"] != "fc_only":
        raise ValueError("Only an adopted fc_only refit may perform new competition inference")
    if not torch.cuda.is_available():
        raise RuntimeError("No silent CPU competition inference")
    checkpoint = Path(recipe["selected_model_path"])
    h.check_hash(checkpoint, recipe["selected_model_sha256"])
    model, _ = t.load_checkpoint(ROOT, checkpoint, "cuda")
    model.eval()
    transform = h.make_eval_transform()
    values = [[], []]
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    started = last = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                h.check_hash(OUT / "selected_recipe.json", recipe_sha)
                h.check_hash(checkpoint, recipe["selected_model_sha256"])
                tensors = []
                for row in rows[start:start + 16]:
                    with Image.open(row["path"]) as image:
                        tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                images = torch.stack(tensors).cuda()
                with h.autocast_for("cuda"):
                    logits = [model(images), model(images.flip(-1))]
                for index, output in enumerate(logits):
                    values[index].append(output.float().softmax(1).cpu().numpy())
                if time.monotonic() - last >= 30 or start + 16 >= len(rows):
                    emit("v11_competition_prediction", seen=min(start + 16, len(rows)), total=len(rows), elapsed_seconds=time.monotonic() - started)
                    last = time.monotonic()
        arrays = [np.concatenate(view) for view in values]
        for array in arrays:
            if array.dtype != np.float32:
                raise ValueError("Inference must preserve float32 probabilities")
            c.check_probability(array, 7065)
        return arrays
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def prediction_binding(recipe):
    return {"format": "hse_groupcv_v11_competition_prediction", "state": "complete", "classes": CLASSES,
        "rows": 7065, "recipe_sha256": h.sha256(OUT / "selected_recipe.json"),
        "checkpoint_sha256": recipe["selected_model_sha256"], "finalizer_sha256": h.sha256(__file__),
        "competition_row_order_sha256": ROW_SHA, "competition_image_content_sha256": IMAGE_SHA,
        "new_inference": recipe["new_candidate_adopted"], "batch_size": 16,
        "prediction_rule": "original and normalized-tensor horizontal flip; float32 softmax; fixed equal mean"}


def verify_prediction_cache(recipe):
    receipt = c.read_json(OUT / "competition_prediction_receipt.json")
    check_fields(receipt, prediction_binding(recipe), "Prediction cache belongs to another model or order")
    names = {view: f"hse_competition_{view}.npy" for view in ("original", "flip")}
    if receipt.get("files") != names or set(receipt.get("artifact_sha256", {})) != set(names):
        raise ValueError("Prediction cache must bind both views")
    arrays = []
    for view, name in names.items():
        h.check_hash(OUT / name, receipt["artifact_sha256"][view])
        if not recipe["new_candidate_adopted"]:
            h.check_hash(OUT / name, h.sha256(BASELINE / name))
        array = np.load(OUT / name, allow_pickle=False)
        if array.dtype != np.float32:
            raise ValueError("Cached probabilities must retain float32")
        c.check_probability(array, 7065)
        arrays.append(array)
    return arrays


def competition_views(recipe, rows):
    path = OUT / "competition_prediction_receipt.json"
    if path.exists():
        return verify_prediction_cache(recipe)
    binding = prediction_binding(recipe)
    names = {view: f"hse_competition_{view}.npy" for view in ("original", "flip")}
    if recipe["new_candidate_adopted"]:
        arrays = predict_candidate(recipe, rows)
        for name, array in zip(names.values(), arrays):
            h.save_array(OUT / name, array)
    else:
        for name in names.values():
            atomic_copy(BASELINE / name, OUT / name)
    h.check_hash(OUT / "selected_recipe.json", binding["recipe_sha256"])
    h.check_hash(recipe["selected_model_path"], binding["checkpoint_sha256"])
    h.write_json(path, {**binding, "files": names, "artifact_sha256": {view: h.sha256(OUT / name) for view, name in names.items()}})
    return verify_prediction_cache(recipe)


def verify_submission(path, rows, probability):
    c.submission_check(path, rows, probability)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "Emotion"] or len(list(reader)) != 7065:
            raise ValueError("Submission requires exactly7065 rows with only ID,Emotion columns")


def output_names():
    return {"submission.csv", "probabilities.npy", "selected_val_probabilities.npy", "selected_recipe.json",
        "competition_prediction_receipt.json", "hse_competition_original.npy", "hse_competition_flip.npy"}


def summary_binding(recipe):
    return {"format": "hse_groupcv_v11_export", "state": "complete", "classes": CLASSES, "submission_rows": 7065,
        "recipe_sha256": h.sha256(OUT / "selected_recipe.json"), "plan_sha256": h.sha256(PLAN),
        "protocol_sha256": h.sha256(PROTOCOL), "finalizer_sha256": h.sha256(__file__),
        "cv_selection_sha256": h.sha256(SELECTION), "cv_selection_lock_sha256": h.sha256(SELECTION_LOCK),
        **{key: recipe[key] for key in ("selected", "cv_selected_mode", "new_candidate_adopted", "selected_metrics",
            "single_tta_metrics", "single_original_metrics", "candidate_outer_metrics", "outer_confirmation_performed",
            "incumbent", "pooled_oof_metrics", "fold_metrics", "fc_only_fold_wins", "candidate_status_vs_v8")},
        "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
        "official_label_order_verified": True, "ID_template_order_verified": True, "probability_argmax_verified": True,
        "new_hse_competition_inference": recipe["new_candidate_adopted"], "old_task_fusion_used": False,
        "outer_development_used_for_cv": False, "internal_test_read": False, "external_answers_used": False,
        "validation_caveat": CAVEAT}


def verify_export():
    """Read-only: every missing output fails; never invokes selection, inference or writes."""
    summary = c.read_json(OUT / "summary.json")
    plan, recipe = verify_recipe()
    check_fields(summary, summary_binding(recipe), "Completed output summary differs")
    if set(summary.get("file_sha256", {})) != output_names():
        raise ValueError("Complete output hashes must cover exactly all exported files")
    for name, digest in summary["file_sha256"].items():
        h.check_hash(child(OUT, name), digest)
    rows = c.unlabeled_rows(ROOT)
    if c.signature(rows) != ROW_SHA or c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition content or template ID order differs")
    arrays = verify_prediction_cache(recipe)
    probability = np.load(OUT / "probabilities.npy", allow_pickle=False)
    if probability.dtype != np.float32 or not np.array_equal(probability, (arrays[0] + arrays[1]) * .5):
        raise ValueError("Competition probabilities must equal the fixed original/flip mean")
    c.check_probability(probability, 7065)
    verify_submission(OUT / "submission.csv", rows, probability)
    if not recipe["new_candidate_adopted"]:
        h.check_hash(OUT / "submission.csv", V10_SUBMISSION_SHA)
        h.check_hash(OUT / "probabilities.npy", h.sha256(BASELINE / "probabilities.npy"))
    incumbent_receipt(plan)
    c.verify_pins(ROOT, recipe["producer_files"])
    return summary


def export():
    if (OUT / "summary.json").exists():
        return verify_export()
    select()
    plan, recipe = verify_recipe()
    rows = c.unlabeled_rows(ROOT)
    if c.signature(rows) != ROW_SHA or c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition content or ID order differs")
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    arrays = competition_views(recipe, rows)
    probability = (arrays[0] + arrays[1]) * .5
    c.check_probability(probability, 7065)
    if recipe["new_candidate_adopted"]:
        h.save_array(OUT / "probabilities.npy", probability)
        with (OUT / "submission.csv.tmp").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ID", "Emotion"])
            writer.writerows((row["ID"], int(label)) for row, label in zip(rows, probability.argmax(1)))
        os.replace(OUT / "submission.csv.tmp", OUT / "submission.csv")
    else:
        atomic_copy(BASELINE / "probabilities.npy", OUT / "probabilities.npy")
        atomic_copy(BASELINE / "submission.csv", OUT / "submission.csv")
    verify_submission(OUT / "submission.csv", rows, probability)
    h.check_hash(OUT / "selected_recipe.json", recipe_sha)
    c.verify_pins(ROOT, recipe["producer_files"])
    incumbent_receipt(plan)
    if c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition content changed during export")
    summary = {**summary_binding(recipe), "file_sha256": {name: h.sha256(OUT / name) for name in sorted(output_names())}}
    h.write_json(OUT / "summary.json", summary)
    verify_export()
    emit("v11_export_complete", selected=recipe["selected"], submission=str(OUT / "submission.csv"),
         new_candidate_adopted=recipe["new_candidate_adopted"], selected_metrics=recipe["selected_metrics"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("select_cv", "select", "export", "verify"), default="export")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT:
        raise ValueError("Fixed project root required")
    torch.set_num_threads(4)
    {"select_cv": select_cv, "select": select, "export": export, "verify": verify_export}[args.stage]()
    if args.stage == "verify":
        emit("v11_export_verified")


if __name__ == "__main__":
    main()


