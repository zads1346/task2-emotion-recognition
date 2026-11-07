"""Audit the fixed official-restarted HSE final EMA and export a new candidate.

The expanded development split cannot fairly score old task-trained models.
No old probability, old validation score, or competition label enters selection.
Epoch zero is diagnostic only; epoch12 EMA with fixed flip averaging is exported.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import csv
import gc
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageOps
import train_hse_specialist as h
import validate_cosine_fusion as c
import hse_model as model_api
import prepare_hse_expanded_validation_v10 as p
import prepare_expanded_validation_v10 as split_api
import train_expanded_validation_v10 as t

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "logs/hse_expanded_validation_v10_plan.json"
TRAIN = ROOT / "experiments/hse_expanded_validation_v10"
OUT = ROOT / "hse_expanded_validation_v10_results"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGE_SHA = "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA = "8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
INCUMBENT_SHA = "55a037351e0677eadc82ebe81ce6a9c208da402f304cd9eeccba70e2121ce4b6"
SELECTION_RULE = "fixed epoch12 EMA only; equal original/flip mean; epoch0 diagnostic only; no old-model comparison"
CAVEAT = (
    "This is a larger near-duplicate-group development split, not an untouched final test. "
    "It contains former training images, so old task models/ensembles cannot provide an independent comparison. "
    "Its score cannot be directly compared with old-split scores or Kaggle scores. "
    "This HSE candidate is unranked against the preserved V8 incumbent. "
    "Person/source independence and overlap with official external pretraining remain unestablished."
)
BINDINGS = ("classes", "source_pins", "plan_sha256", "parent_checkpoint_sha256", "model_source_receipt",
            "model_source_receipt_sha256", "original_pool_indices", "excluded_indices", "train_indices",
            "val_indices", "split_summary_sha256", "split_manifest_sha256")


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


def check_state(state, initial, parameter_names):
    if not isinstance(state, dict) or set(state) != set(initial):
        raise ValueError("Checkpoint must contain the full official HSE state layout")
    for name, reference in initial.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != reference.dtype or value.shape != reference.shape
                or not torch.isfinite(value).all().item()):
            raise ValueError("Invalid HSE state tensor: " + name)
        if name not in parameter_names and not torch.equal(value, reference):
            raise ValueError("Frozen HSE buffer differs from its official value: " + name)


def verify_payload(payload, config, initial, parameter_names):
    if payload.get("format") != t.FORMAT or payload.get("config") != config:
        raise ValueError("Checkpoint format or full run configuration differs")
    for key in BINDINGS:
        if payload.get(key) != config[key]:
            raise ValueError("Checkpoint source/member binding differs: " + key)
    if payload.get("parent_sha") != model_api.OFFICIAL_SHA or payload.get("model_provenance") != config["model_source_receipt"]:
        raise ValueError("Checkpoint must derive from the official HSE source")
    check_state(payload.get("model_state"), initial, parameter_names)


def verify_split(plan, rows):
    """Reconcile current allowed records with the independently created grouped split."""
    independent, summary = split_api.load_records(ROOT, verify_images=True, recompute_groups=False)
    if summary != plan["split_receipt"]:
        raise ValueError("Independent split receipt differs from the training plan")
    h.check_hash(plan["split_summary_path"], plan["split_summary_sha256"])
    h.check_hash(plan["split_manifest_path"], plan["split_manifest_sha256"])
    if (summary.get("original_allowed_rows") != 23904 or summary.get("effective_allowed_rows") != 23902
            or summary.get("excluded_manifest_indices") != [7727, 10448]
            or summary.get("old_task_checkpoints_valid_for_independent_evaluation") is not False
            or summary.get("labels_changed") != 0 or summary.get("original_files_changed") is not False):
        raise ValueError("Expanded split must preserve original labels and exclude only the audited two images")
    if set(rows) != {"train", "val"}:
        raise ValueError("Only the new training and development partitions are allowed")
    for split in ("train", "val"):
        projected = [{key: row[key] for key in ("manifest_index", "path", "label", "target")} for row in independent[split]]
        if projected != rows[split]:
            raise ValueError("Trainer rows differ from the independently verified split rows")
        indices = [row["manifest_index"] for row in rows[split]]
        if indices != plan[f"{split}_indices"] or len(indices) != summary["split_counts"][split]:
            raise ValueError("Expanded evaluation/training member order differs")
        if any(row["label"] not in CLASSES or row["target"] != CLASSES.index(row["label"]) for row in rows[split]):
            raise ValueError("Expanded labels must keep the official six-class order")
    training, validation = set(plan["train_indices"]), set(plan["val_indices"])
    if training & validation or training | validation != set(plan["original_pool_indices"]) - {7727, 10448}:
        raise ValueError("Expanded train/validation members overlap or omit allowed images")
    return summary


def audit_training():
    plan = p.verify_plan(ROOT)
    checked, plan_sha, rows = t.checked_plan(ROOT)
    if checked != plan or plan_sha != h.sha256(PLAN):
        raise ValueError("Trainer and finalizer disagree on the immutable plan")
    verify_split(plan, rows)
    config = t.config_for(ROOT, plan)
    if c.read_json(TRAIN / "config.json") != config:
        raise ValueError("Saved training config differs from the fixed official restart")
    if (plan["parent_checkpoint_sha256"] != model_api.OFFICIAL_SHA
            or plan["model_source_receipt"] != model_api.source_receipt(ROOT)
            or plan["model_source_receipt_sha256"] != h.signature(plan["model_source_receipt"])):
        raise ValueError("Official pretrained model provenance differs")
    initial_model = model_api.load_initial_model(ROOT, "cpu")
    initial = h.cpu_tree(initial_model.state_dict())
    parameter_names = set(dict(initial_model.named_parameters()))
    del initial_model
    pins = dict(plan["source_pins"])
    for path in (PLAN, Path(__file__), Path(plan["split_summary_path"]), Path(plan["split_manifest_path"])):
        pin(pins, path)
    summary = c.read_json(TRAIN / "summary.json")
    for key, value in {"state": "complete", "epochs_completed": 12, "weight_source": "ema", **{key: config[key] for key in BINDINGS}}.items():
        if summary.get(key) != value:
            raise ValueError("Incomplete or differently bound HSE summary: " + key)
    required = {"config.json", "history.json", "last.json", "last.pt", "final_ema.pt", "final_original.npy", "final_flip.npy", "final_result.json",
                "diagnostic_epoch0.pt", "diagnostic_epoch0_original.npy", "diagnostic_epoch0_flip.npy", "epoch0_diagnostic.json"}
    required |= {f"epochs/epoch_{epoch:02d}.pt" for epoch in range(13)}
    if not required.issubset(summary.get("artifact_sha256", {})):
        raise ValueError("Training summary omits immutable epochs or the final EMA artifacts")
    for name, digest in summary["artifact_sha256"].items():
        pin(pins, child(TRAIN, name), digest)
    pin(pins, TRAIN / "summary.json")
    pointer = c.read_json(TRAIN / "last.json")
    if pointer.get("epoch") != 12 or pointer.get("checkpoint") != "epochs/epoch_12.pt" or pointer.get("plan_sha256") != plan_sha:
        raise ValueError("The last pointer must reference committed epoch12")
    pin(pins, TRAIN / "last.pt", pointer["sha256"])
    pin(pins, TRAIN / "epochs/epoch_12.pt", pointer["sha256"])
    history = c.read_json(TRAIN / "history.json")
    if len(history) != 12 or [entry["epoch"] for entry in history] != list(range(1, 13)):
        raise ValueError("Exactly12 completed epochs are required")
    successes = skipped = 0
    counters_by_epoch = []
    for entry in history:
        n, k = entry["successful_optimizer_steps"], entry["amp_skipped_steps"]
        if (entry["seen"] != len(rows["train"]) or n <= 0 or k < 0
                or n + k != math.ceil(len(rows["train"]) / 32)):
            raise ValueError("Training did not account for every uniform logical group")
        if any(key.startswith(("val_", "heldout_", "dev_")) for key in entry):
            raise ValueError("Intermediate development scores are not allowed to enter this fixed run")
        if entry.get("augmentation_epoch") != plan["settings"]["augmentation_epoch_offset"] + entry["epoch"] - 1:
            raise ValueError("Deterministic augmentation epoch differs")
        if entry.get("zero_weight_groups", 0) != 0:
            raise ValueError("The expanded experiment must use uniform nonzero weights")
        successes += n
        skipped += k
        if entry["cumulative_optimizer_steps"] != successes or entry["ema_updates"] != successes:
            raise ValueError("EMA or successful optimizer counters differ")
        counters_by_epoch.append({"successful_steps": successes, "amp_skipped_steps": skipped, "zero_weight_groups": 0})
    for key, value in {"successful_optimizer_steps": successes, "amp_skipped_steps": skipped, "ema_updates": successes}.items():
        if summary.get(key) != value:
            raise ValueError("Summary counters differ: " + key)
    if summary.get("history") != history:
        raise ValueError("Summary and committed history differ")
    frozen_digest = t.frozen_digest({name: value for name, value in initial.items() if name not in parameter_names})
    policies = {"counters": counters_by_epoch[-1], "effective_train_rows": len(rows["train"]), "evaluation_rows": len(rows["val"]),
                "zero_weight_groups": 0, "frozen_state_sha256": frozen_digest, "batchnorm_buffers_frozen": True,
                "frozen_state_unchanged": True, "validation_weighted": False, "per_epoch_evaluation": False,
                "epoch0_purpose": "diagnostic_only", "checkpoint_selection": False, "old_model_validation_or_fusion": False,
                "internal_test_read": False, "external_answers_used": False}
    for key, value in policies.items():
        if summary.get(key) != value:
            raise ValueError("Training/evaluation policy proof differs: " + key)
    for epoch in range(13):
        payload = torch.load(TRAIN / f"epochs/epoch_{epoch:02d}.pt", map_location="cpu", weights_only=True)
        verify_payload(payload, config, initial, parameter_names)
        counters = counters_by_epoch[epoch - 1] if epoch else {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
        if (payload.get("epoch") != epoch or payload.get("weight_source") != "raw" or payload.get("history") != history[:epoch]
                or payload.get("counters") != counters or payload.get("successful_steps") != counters["successful_steps"]
                or payload.get("cumulative_skips") != counters["amp_skipped_steps"] or payload.get("zero_weight_groups") != 0):
            raise ValueError("Immutable epoch metadata/history/counters differ")
        scaler = payload.get("scaler", {})
        if (scaler.get("enabled") is not True or scaler.get("growth_interval") != plan["settings"]["amp_growth_interval"]
                or not isinstance(scaler.get("scale"), (int, float)) or not math.isfinite(scaler["scale"])
                or not 1 <= scaler["scale"] <= 2. ** 24
                or not 0 <= scaler.get("growth_tracker", -1) < scaler["growth_interval"]):
            raise ValueError("Committed CUDA AMP state is invalid")
        if epoch and scaler["scale"] != history[epoch - 1]["amp_scale"]:
            raise ValueError("Committed AMP scale differs from its epoch history")
        if epoch == 0 and (scaler["scale"] != 1024. or scaler["growth_tracker"] != 0):
            raise ValueError("Official restart must initialize a fresh AMP scaler")
        rng = payload.get("rng", {})
        if not {"python", "numpy", "torch", "cuda"}.issubset(rng) or not isinstance(rng["torch"], torch.Tensor) or not rng["cuda"]:
            raise ValueError("Committed epoch lacks restorable CPU/CUDA random state")
        ema = payload["ema_state"]
        if ema.get("decay") != .995 or ema.get("updates") != counters["successful_steps"]:
            raise ValueError("EMA must update exactly once per successful AdamW step")
        check_state(ema["shadow"], initial, parameter_names)
        optimizer = payload["optimizer"]["state"]
        if epoch == 0:
            if optimizer or not same_state(payload["model_state"], initial) or not same_state(ema["shadow"], initial):
                raise ValueError("Epoch0 must exactly equal official HSE state with fresh optimizer and EMA")
        elif len(optimizer) != len(parameter_names) or any(int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"]) != counters["successful_steps"] for value in optimizer.values()):
            raise ValueError("Every actual AdamW parameter step must match the successful update count")
        if payload.get("frozen_state_sha256") != frozen_digest:
            raise ValueError("Frozen source buffer digest changed between epochs")
        if epoch == 12:
            final_shadow = ema["shadow"]
        del payload
    final = torch.load(TRAIN / "final_ema.pt", map_location="cpu", weights_only=True)
    verify_payload(final, config, initial, parameter_names)
    if (final.get("epoch") != 12 or final.get("weight_source") != "ema"
            or not same_state(final["model_state"], final_shadow)):
        raise ValueError("The sole candidate must be the actual final epoch12 EMA")
    if (final.get("ema_updates") != successes or final.get("successful_steps") != successes
            or final.get("frozen_state_sha256") != frozen_digest or final.get("purpose") != "fixed_final_epoch"):
        raise ValueError("Final EMA counters differ")
    views = [np.load(TRAIN / f"final_{view}.npy", allow_pickle=False) for view in ("original", "flip")]
    for array in views:
        if array.shape != (len(rows["val"]), 6) or array.dtype != np.float32:
            raise ValueError("Expanded-validation probability shape/dtype differs")
        c.check_probability(array, len(rows["val"]))
    original = t.metrics(views[0], rows["val"])
    tta = t.metrics((views[0] + views[1]) * .5, rows["val"])
    result = c.read_json(TRAIN / "final_result.json")
    for item in (summary, final, result):
        for key, value in (("single_original_metrics", original), ("single_tta_metrics", tta),
                           ("final_original_metrics", original), ("final_tta_metrics", tta)):
            if item.get(key) != value:
                raise ValueError("Stored final metrics do not reproduce on the new validation rows: " + key)
    if (result.get("state") != "complete" or result.get("epoch") != 12 or result.get("weight_source") != "ema"
            or result.get("purpose") != "fixed_final_epoch" or result.get("validation_weighted") is not False
            or result.get("checkpoint_selection") is not False or result.get("old_model_validation_or_fusion") is not False):
        raise ValueError("Final evaluation receipt must describe fixed epoch12 EMA")
    for key in BINDINGS:
        if result.get(key) != config[key]:
            raise ValueError("Final evaluation source/member receipt differs: " + key)
    for key, name in {"model": "final_ema.pt", "original": "final_original.npy", "flip": "final_flip.npy"}.items():
        if result.get(key) != name:
            raise ValueError("Final evaluation artifact filenames differ")
        pin(pins, TRAIN / name, result["artifact_sha256"][key])
    diagnostic = verify_diagnostic(config, initial, parameter_names, rows["val"], pins)
    del final, final_shadow, initial
    gc.collect()
    return plan, rows, pins, {"single_original_metrics": original, "single_tta_metrics": tta,
        "successful_optimizer_steps": successes, "amp_skipped_steps": skipped, "ema_updates": successes,
        "final_model_sha256": h.sha256(TRAIN / "final_ema.pt"), "training_summary_sha256": h.sha256(TRAIN / "summary.json"),
        "epoch0_diagnostic": diagnostic}, views


def verify_diagnostic(config, initial, parameter_names, rows, pins):
    names = ["diagnostic_epoch0.pt", "diagnostic_epoch0_original.npy", "diagnostic_epoch0_flip.npy", "epoch0_diagnostic.json"]
    present = [(TRAIN / name).exists() for name in names]
    if not all(present):
        raise ValueError("Partially written epoch0 diagnostic must be completed before export")
    for name in names:
        pin(pins, TRAIN / name)
    model = torch.load(TRAIN / names[0], map_location="cpu", weights_only=True)
    verify_payload(model, config, initial, parameter_names)
    if (model.get("epoch") != 0 or model.get("weight_source") != "official" or model.get("purpose") != "diagnostic_only"
            or model.get("successful_steps") != 0 or model.get("ema_updates") != 0 or not same_state(model["model_state"], initial)):
        raise ValueError("Epoch0 diagnostic must represent exact official initialization")
    views = [np.load(TRAIN / name, allow_pickle=False) for name in names[1:3]]
    for view in views:
        if view.dtype != np.float32 or view.shape != (len(rows), 6):
            raise ValueError("Epoch0 diagnostic view shape/dtype differs")
        c.check_probability(view, len(rows))
    score = {"single_original_metrics": t.metrics(views[0], rows), "single_tta_metrics": t.metrics((views[0] + views[1]) * .5, rows)}
    receipt = c.read_json(TRAIN / names[3])
    for key in BINDINGS:
        if receipt.get(key) != config[key]:
            raise ValueError("Epoch0 diagnostic member/source receipt differs: " + key)
    if (receipt.get("state") != "complete" or receipt.get("epoch") != 0 or receipt.get("weight_source") != "official"
            or receipt.get("purpose") != "diagnostic_only" or receipt.get("validation_weighted") is not False):
        raise ValueError("Epoch0 diagnostic receipt is incomplete")
    for key, value in score.items():
        if receipt.get(key) != value or model.get(key) != value:
            raise ValueError("Epoch0 diagnostic metric does not reproduce")
    for key, name in zip(("model", "original", "flip"), names[:3]):
        if receipt.get(key) != name:
            raise ValueError("Epoch0 diagnostic artifact filenames differ")
        pin(pins, TRAIN / name, receipt["artifact_sha256"][key])
    return {**score, "checkpoint_sha256": h.sha256(TRAIN / names[0]), "receipt_sha256": h.sha256(TRAIN / names[3]),
            "used_for_selection": False, "interpretation": "official initialization on the same new split; diagnostic only"}


def preserved_incumbent(plan):
    directory = Path(plan["old_incumbent_directory"]).resolve()
    if directory != ROOT / "ddamfn_v8_results":
        raise ValueError("The preserved incumbent directory differs")
    path = directory / "submission.csv"
    if plan["incumbent_submission_sha256"] != INCUMBENT_SHA:
        raise ValueError("The preserved V8 incumbent checksum differs")
    h.check_hash(path, INCUMBENT_SHA)
    return {"path": str(path), "sha256": INCUMBENT_SHA, "evaluation_on_expanded_validation": "not performed: prior task-training overlap",
            "comparison_with_v10": "unranked; no common independent evaluation", "overwritten": False}


def verify_recipe():
    plan = p.verify_plan(ROOT)
    recipe = c.read_json(OUT / "selected_recipe.json")
    for key, value in {"state": "locked", "classes": CLASSES, "plan_sha256": h.sha256(PLAN),
                       "finalizer_sha256": h.sha256(__file__), "source_pins": plan["source_pins"],
                       "selected": "epoch_12_ema", "candidate_count": 1, "candidate_status": "unranked_against_v8",
                       "selection_rule": SELECTION_RULE, "old_task_probabilities_used": False,
                       "old_split_score_used_for_selection": False, "epoch0_used_for_selection": False}.items():
        if recipe.get(key) != value:
            raise ValueError("Fixed V10 candidate recipe differs: " + key)
    if recipe.get("preserved_incumbent") != preserved_incumbent(plan):
        raise ValueError("Preserved incumbent receipt differs")
    c.verify_pins(ROOT, recipe["producer_files"])
    for name, digest in recipe["selection_file_sha256"].items():
        h.check_hash(child(OUT, name), digest)
    checked, _, rows = t.checked_plan(ROOT)
    if checked != plan or recipe.get("train_indices") != plan["train_indices"] or recipe.get("val_indices") != plan["val_indices"]:
        raise ValueError("Locked candidate members differ from the new split")
    views = [np.load(TRAIN / f"final_{view}.npy", allow_pickle=False) for view in ("original", "flip")]
    expected = (views[0] + views[1]) * .5
    saved = np.load(OUT / "selected_val_probabilities.npy", allow_pickle=False)
    if saved.dtype != np.float32 or not np.array_equal(saved, expected):
        raise ValueError("Saved validation output is not the fixed final-EMA view average")
    c.check_probability(saved, len(rows["val"]))
    if (recipe.get("selected_metrics") != t.metrics(saved, rows["val"])
            or recipe.get("single_tta_metrics") != t.metrics(expected, rows["val"])
            or recipe.get("single_original_metrics") != t.metrics(views[0], rows["val"])):
        raise ValueError("Locked candidate metrics do not reproduce on the expanded validation split")
    return plan, recipe


def select():
    if (OUT / "selected_recipe.json").exists():
        return verify_recipe()[1]
    plan, rows, pins, audit, views = audit_training()
    incumbent = preserved_incumbent(plan)
    pin(pins, incumbent["path"], incumbent["sha256"])
    OUT.mkdir(parents=True, exist_ok=True)
    probability = (views[0] + views[1]) * .5
    c.check_probability(probability, len(rows["val"]))
    h.save_array(OUT / "selected_val_probabilities.npy", probability)
    recipe = {"format": "hse_expanded_validation_v10_candidate", "state": "locked", "classes": CLASSES,
        "selected": "epoch_12_ema", "candidate_count": 1, "candidate_status": "unranked_against_v8",
        "selection_rule": SELECTION_RULE, "prediction_rule": "equal probability average of deterministic original and horizontal flip",
        "selected_metrics": audit["single_tta_metrics"], "single_original_metrics": audit["single_original_metrics"],
        "single_tta_metrics": audit["single_tta_metrics"], "training_audit": audit,
        "plan_sha256": h.sha256(PLAN), "source_pins": plan["source_pins"], "producer_files": pins,
        "split_manifest_sha256": plan["split_manifest_sha256"], "split_summary_sha256": plan["split_summary_sha256"],
        "train_indices": plan["train_indices"], "val_indices": plan["val_indices"],
        "parent_checkpoint_sha256": model_api.OFFICIAL_SHA, "model_source_receipt_sha256": plan["model_source_receipt_sha256"],
        "finalizer_sha256": h.sha256(__file__), "preserved_incumbent": incumbent,
        "old_task_probabilities_used": False, "old_split_score_used_for_selection": False, "epoch0_used_for_selection": False,
        "selection_file_sha256": {"selected_val_probabilities.npy": h.sha256(OUT / "selected_val_probabilities.npy")},
        "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
        "internal_test_read": False, "external_answers_used": False, "validation_caveat": CAVEAT}
    c.verify_pins(ROOT, pins)
    h.write_json(OUT / "selected_recipe.json", recipe)
    verify_recipe()
    emit("v10_candidate_locked", selected="epoch_12_ema", new_validation_metrics=recipe["selected_metrics"], status="unranked_against_v8")
    return recipe


def predict_candidate(recipe, rows):
    if not torch.cuda.is_available():
        raise RuntimeError("No silent CPU competition prediction")
    checkpoint = TRAIN / "final_ema.pt"
    model, _ = t.load_checkpoint(ROOT, checkpoint, "cuda")
    model.eval()
    values = [[], []]
    transform = h.make_eval_transform()
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    started = last = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                h.check_hash(OUT / "selected_recipe.json", recipe_sha)
                h.check_hash(checkpoint, recipe["training_audit"]["final_model_sha256"])
                tensors = []
                for row in rows[start:start + 16]:
                    with Image.open(row["path"]) as image:
                        tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                images = torch.stack(tensors).cuda()
                with h.autocast_for("cuda"):
                    logits = [model(images), model(images.flip(-1))]
                for index, value in enumerate(logits):
                    values[index].append(value.float().softmax(1).cpu().numpy())
                if time.monotonic() - last >= 30 or start + 16 >= len(rows):
                    emit("v10_competition_prediction", seen=min(start + 16, len(rows)), total=len(rows), elapsed_seconds=time.monotonic() - started)
                    last = time.monotonic()
        arrays = [np.concatenate(view) for view in values]
        for array in arrays:
            if array.dtype != np.float32:
                raise ValueError("Competition probabilities must retain float32 inference output")
            c.check_probability(array, 7065)
        return arrays
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def verify_submission(rows, probability):
    c.submission_check(OUT / "submission.csv", rows, probability)
    with (OUT / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "Emotion"] or len(list(reader)) != 7065:
            raise ValueError("Final CSV must contain7065 rows and only ID,Emotion")


def competition_binding(recipe):
    return {"format": "hse_expanded_validation_v10_prediction", "state": "complete", "classes": CLASSES,
        "rows": 7065, "recipe_sha256": h.sha256(OUT / "selected_recipe.json"),
        "checkpoint_sha256": recipe["training_audit"]["final_model_sha256"],
        "finalizer_sha256": h.sha256(__file__), "competition_row_order_sha256": ROW_SHA,
        "competition_image_content_sha256": IMAGE_SHA, "batch_size": 16,
        "prediction_rule": "original plus normalized-tensor horizontal flip; float32 softmax; fixed mean"}


def verify_prediction_cache(recipe):
    """Read a complete cached result; missing or changed files fail without inference or writes."""
    receipt = c.read_json(OUT / "competition_prediction_receipt.json")
    binding = competition_binding(recipe)
    names = {view: f"hse_competition_{view}.npy" for view in ("original", "flip")}
    if (any(receipt.get(key) != value for key, value in binding.items()) or receipt.get("files") != names
            or set(receipt.get("artifact_sha256", {})) != set(names)):
        raise ValueError("Cached competition inference belongs to another candidate or image order")
    views = []
    for view, name in names.items():
        h.check_hash(OUT / name, receipt["artifact_sha256"][view])
        array = np.load(OUT / name, allow_pickle=False)
        if array.dtype != np.float32:
            raise ValueError("Cached competition views must preserve float32")
        c.check_probability(array, 7065)
        views.append(array)
    return views


def competition_views(recipe, rows):
    """Reuse only a complete inference result bound to this exact candidate and input order."""
    receipt_path = OUT / "competition_prediction_receipt.json"
    if receipt_path.exists():
        return verify_prediction_cache(recipe)
    binding = competition_binding(recipe)
    names = {view: f"hse_competition_{view}.npy" for view in ("original", "flip")}
    views = predict_candidate(recipe, rows)
    for (view, name), array in zip(names.items(), views):
        h.save_array(OUT / name, array)
    h.check_hash(OUT / "selected_recipe.json", binding["recipe_sha256"])
    h.check_hash(TRAIN / "final_ema.pt", binding["checkpoint_sha256"])
    receipt = {**binding, "files": names, "artifact_sha256": {view: h.sha256(OUT / name) for view, name in names.items()}}
    h.write_json(receipt_path, receipt)
    return views


def verify_export():
    """Strictly read-only verification of a previously completed candidate export."""
    summary = c.read_json(OUT / "summary.json")
    plan, recipe = verify_recipe()
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    rows = c.unlabeled_rows(ROOT)
    if c.signature(rows) != ROW_SHA or c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition ID order or image content changed")
    if (summary.get("state") != "complete" or summary.get("recipe_sha256") != recipe_sha
            or summary.get("candidate_status") != "unranked_against_v8" or summary.get("classes") != CLASSES):
        raise ValueError("Completed export belongs to another candidate")
    expected_fields = {"format": "hse_expanded_validation_v10_export", "selected": "epoch_12_ema", "submission_rows": 7065,
            "selected_metrics": recipe["selected_metrics"], "single_original_metrics": recipe["single_original_metrics"],
            "single_tta_metrics": recipe["single_tta_metrics"], "plan_sha256": h.sha256(PLAN),
            "finalizer_sha256": h.sha256(__file__), "split_manifest_sha256": plan["split_manifest_sha256"],
            "split_summary_sha256": plan["split_summary_sha256"], "preserved_incumbent": preserved_incumbent(plan),
            "new_training_rows": len(plan["train_indices"]), "new_validation_rows": len(plan["val_indices"]),
            "old_task_probabilities_used": False, "old_split_score_used_for_selection": False,
            "epoch0_diagnostic": recipe["training_audit"]["epoch0_diagnostic"],
            "official_label_order_verified": True, "ID_template_order_verified": True, "probability_argmax_verified": True,
            "new_hse_competition_inference": True, "internal_test_read": False, "external_answers_used": False,
            "validation_caveat": CAVEAT,
            "competition_row_order_sha256": ROW_SHA, "competition_image_content_sha256": IMAGE_SHA}
    if any(summary.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Completed export summary metadata differs from the immutable candidate")
    must = {"submission.csv", "probabilities.npy", "selected_val_probabilities.npy", "selected_recipe.json",
            "competition_prediction_receipt.json", "hse_competition_original.npy", "hse_competition_flip.npy"}
    if set(summary.get("file_sha256", {})) != must:
        raise ValueError("Completed export must hash every candidate output")
    for name, digest in summary["file_sha256"].items():
        h.check_hash(child(OUT, name), digest)
    views = verify_prediction_cache(recipe)
    probability = np.load(OUT / "probabilities.npy", allow_pickle=False)
    if probability.dtype != np.float32 or not np.array_equal(probability, (views[0] + views[1]) * .5):
        raise ValueError("Completed competition output differs from the fixed final-EMA view average")
    c.check_probability(probability, 7065)
    verify_submission(rows, probability)
    preserved_incumbent(plan)
    h.check_hash(OUT / "selected_recipe.json", recipe_sha)
    c.verify_pins(ROOT, recipe["producer_files"])
    return summary


def export():
    if (OUT / "summary.json").exists():
        return verify_export()
    select()
    plan, recipe = verify_recipe()
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    rows = c.unlabeled_rows(ROOT)
    if c.signature(rows) != ROW_SHA or c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition ID order or image content changed")
    views = competition_views(recipe, rows)
    probability = (views[0] + views[1]) * .5
    c.check_probability(probability, 7065)
    h.save_array(OUT / "probabilities.npy", probability)
    with (OUT / "submission.csv.tmp").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Emotion"])
        writer.writerows((row["ID"], int(label)) for row, label in zip(rows, probability.argmax(1)))
    os.replace(OUT / "submission.csv.tmp", OUT / "submission.csv")
    verify_submission(rows, probability)
    h.check_hash(OUT / "selected_recipe.json", recipe_sha)
    c.verify_pins(ROOT, recipe["producer_files"])
    if c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition bytes changed during export")
    incumbent = preserved_incumbent(plan)
    names = ["submission.csv", "probabilities.npy", "selected_val_probabilities.npy", "selected_recipe.json", "competition_prediction_receipt.json",
             "hse_competition_original.npy", "hse_competition_flip.npy"]
    summary = {"format": "hse_expanded_validation_v10_export", "state": "complete", "classes": CLASSES,
        "submission_rows": 7065, "selected": "epoch_12_ema", "candidate_status": "unranked_against_v8",
        "selected_metrics": recipe["selected_metrics"], "single_original_metrics": recipe["single_original_metrics"],
        "single_tta_metrics": recipe["single_tta_metrics"], "epoch0_diagnostic": recipe["training_audit"]["epoch0_diagnostic"],
        "new_training_rows": len(plan["train_indices"]), "new_validation_rows": len(plan["val_indices"]),
        "recipe_sha256": recipe_sha, "plan_sha256": h.sha256(PLAN), "finalizer_sha256": h.sha256(__file__),
        "split_manifest_sha256": plan["split_manifest_sha256"], "split_summary_sha256": plan["split_summary_sha256"],
        "official_label_order_verified": True, "ID_template_order_verified": True, "probability_argmax_verified": True,
        "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
        "preserved_incumbent": incumbent, "new_hse_competition_inference": True, "old_task_probabilities_used": False,
        "old_split_score_used_for_selection": False, "internal_test_read": False, "external_answers_used": False,
        "validation_caveat": CAVEAT, "file_sha256": {name: h.sha256(OUT / name) for name in names}}
    h.write_json(OUT / "summary.json", summary)
    emit("v10_candidate_complete", submission=str(OUT / "submission.csv"), candidate_status="unranked_against_v8", new_validation_metrics=summary["selected_metrics"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("select", "export", "verify"), default="export")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT:
        raise ValueError("Fixed project root required")
    torch.set_num_threads(4)
    if args.stage == "select":
        select()
    elif args.stage == "verify":
        verify_export()
        emit("v10_export_verified")
    else:
        export()


if __name__ == "__main__":
    main()
