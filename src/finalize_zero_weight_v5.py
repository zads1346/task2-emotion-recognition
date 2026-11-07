"""Audit the matched V5 arms, lock a fixed fusion, then export one submission.

Selection uses the reused development validation set only. It never fits fusion
weights, reads internal-test labels, or reads an external answer file.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_hse_specialist as h
import train_lora as l
import validate_cosine_fusion as c
import prepare_zero_weights_v5 as weights_module

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "logs/zero_weight_v5_plan.json"
TRAIN = ROOT / "experiments/lora_zero_weight_v5"
OUT = ROOT / "lora_zero_weight_v5_results"
BASELINE = ROOT / "lora_task_v4_results"
PARENT_FOLDER = ROOT / "experiments/lora_task_v4"
PARENT = PARENT_FOLDER / "candidates/epoch_05_ema.pt"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODES = ("uniform_control", "weighted")
IMAGE_SHA = "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA = "8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
PRESERVED = {
    "lora_task_v4_results/submission.csv": "cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd",
    "hse_results/submission.csv": "f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7",
    "fusion_results/submission.csv": "2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad",
}
ALL_NAMES = {f"backbone.blocks.{index}.attn.qkv.lora_{letter}"
             for index in range(4, 12) for letter in ("A", "B")} | {"head.weight", "head.bias"}
CAVEAT = ("Repeated development validation, not Kaggle or an independent accuracy estimate. "
          "The matched uniform arm distinguishes the observed weighting change from continued training; "
          "one seed and selection on the same validation set do not establish a general causal benefit. "
          "OOF uncertainty is a model proxy, not verified visual blur or label error.")


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def safe_child(folder, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("An artifact path escapes its dedicated directory")
    result = (folder / path).resolve()
    if not result.is_relative_to(folder.resolve()):
        raise ValueError("An artifact resolves outside its dedicated directory")
    return result


def pin(pins, path, expected=None):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else safe_child(ROOT, path)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Producer files must remain inside the project")
    digest = h.sha256(path) if expected is None else expected
    h.check_hash(path, digest)
    key = str(path)
    if key in pins and pins[key] != digest:
        raise ValueError("Conflicting hashes for the same producer file")
    pins[key] = digest


def merged_pins(*maps):
    result = {}
    for mapping in maps:
        if not isinstance(mapping, dict):
            raise ValueError("Producer pins must be a path-to-SHA256 mapping")
        for path, digest in mapping.items():
            pin(result, path, digest)
    return result


def verify_preserved():
    for relative, digest in PRESERVED.items():
        h.check_hash(ROOT / relative, digest)


def load_plan():
    plan = c.read_json(PLAN)
    if plan.get("classes") != CLASSES or plan.get("baseline_correct_count") != 1915:
        raise ValueError("Plan official classes or baseline correct count differ")
    if plan.get("settings", {}).get("epochs") != 4:
        raise ValueError("Only the two fixed four-epoch arms are allowed")
    if Path(plan["parent_checkpoint"]).resolve() != PARENT.resolve():
        raise ValueError("V5 must continue the selected V4 epoch05 EMA parent")
    if plan.get("baseline_submission_sha256") != PRESERVED["lora_task_v4_results/submission.csv"]:
        raise ValueError("The locked V4 submission binding differs")
    if plan.get("internal_test_read", False) or plan.get("external_answers_used", False):
        raise ValueError("Internal test or external answers are forbidden")
    if plan.get("competition_image_content_sha256", IMAGE_SHA) != IMAGE_SHA or plan.get("competition_row_order_sha256", ROW_SHA) != ROW_SHA:
        raise ValueError("Locked competition image/ID binding differs")
    source = plan.get("parent_candidate_source", "epoch_05_ema")
    if (source.get("name") if isinstance(source, dict) else source) != "epoch_05_ema":
        raise ValueError("Wrong parent candidate source")
    pins = merged_pins(plan["source_pins"], plan.get("baseline_files", {}))
    for path in (Path(__file__).resolve(), ROOT / "src/train_zero_weight_lora_v5.py",
                 ROOT / "src/prepare_zero_weights_v5.py"):
        if str(path.resolve()) not in pins:
            raise ValueError("Plan omits a required V5 producer implementation")
    h.check_hash(PARENT, plan["parent_checkpoint_sha256"])
    weight_path = TRAIN / "weights/summary.json"
    h.check_hash(weight_path, plan["weight_summary_sha256"])
    weight_summary = weights_module.verify_existing(ROOT, c.read_json(weight_path))
    pin(pins, weight_path, plan["weight_summary_sha256"])
    for relative, digest in weight_summary["file_sha256"].items():
        pin(pins, safe_child(weight_path.parent, relative), digest)
    for path, digest in weight_summary["binding"]["source_sha256"].items():
        pin(pins, path, digest)
    pin(pins, PLAN)
    verify_preserved()
    return plan, pins, weight_summary


def load_baseline(plan, pins, targets):
    summary = c.read_json(BASELINE / "summary.json")
    recipe = c.read_json(BASELINE / "selected_recipe.json")
    if summary.get("state") != "complete" or summary.get("classes") != CLASSES or summary.get("submission_rows") != 7065:
        raise ValueError("The completed V4 export is not the expected six-class result")
    if summary.get("selected") != "epoch_05_ema" or recipe.get("selected", {}).get("name") != "epoch_05_ema":
        raise ValueError("The V4 incumbent uses another LoRA branch")
    if summary.get("competition_image_content_sha256") != IMAGE_SHA or summary.get("competition_row_order_sha256") != ROW_SHA:
        raise ValueError("V4 competition probabilities use different image content/IDs")
    for relative, digest in summary["file_sha256"].items():
        pin(pins, safe_child(BASELINE, relative), digest)
    h.check_hash(BASELINE / "selected_recipe.json", summary["recipe_sha256"])
    for path, digest in recipe["producer_files"].items():
        pin(pins, path, digest)
    pin(pins, BASELINE / "summary.json")
    pin(pins, PARENT, plan["parent_checkpoint_sha256"])
    parent_item = c.read_json(PARENT_FOLDER / "candidates/epoch_05_ema.json")
    if parent_item != recipe["selected"]:
        raise ValueError("Parent candidate JSON differs from the incumbent recipe")
    for key in ("model", "original", "flip"):
        pin(pins, safe_child(PARENT_FOLDER, parent_item[key]), parent_item["artifact_sha256"][key])
    pin(pins, PARENT_FOLDER / parent_item["candidate_json"])
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    if parent.get("classes") != CLASSES or parent.get("epoch") != 5 or parent.get("weight_source") != "ema":
        raise ValueError("Parent LoRA checkpoint metadata differs")
    for split in ("train", "val"):
        if parent[split + "_indices"] != plan[split + "_indices"]:
            raise ValueError("The parent used different train/validation members")
    check_state(parent["adapter_state"])
    current = c.check_probability(np.load(BASELINE / "selected_val_probabilities.npy", allow_pickle=False), 2656)
    current_metrics = c.metrics(targets, current)
    if current_metrics["correct_count"] != 1915 or current_metrics["correct_count"] != summary["selected_metrics"]["correct_count"]:
        raise ValueError("V4 incumbent validation predictions no longer reproduce1915/2656")
    old = [c.check_probability(np.load(PARENT_FOLDER / parent_item[view], allow_pickle=False), 2656)
           for view in ("original", "flip")]
    return current, current_metrics, old, parent, summary


def check_state(state):
    if not isinstance(state, dict) or set(state) != ALL_NAMES:
        raise ValueError("Require the complete last8 LoRA adapter and six-class Linear head")
    for name, value in state.items():
        shape = (6, 1536) if name == "head.weight" else (6,) if name == "head.bias" else (8, 768) if name.endswith("lora_A") else (2304, 8)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all().item():
            raise ValueError("A LoRA/head tensor has an invalid dtype, shape or nonfinite value: " + name)


def same_state(actual, expected):
    check_state(actual)
    check_state(expected)
    return all(torch.equal(actual[name], expected[name]) for name in ALL_NAMES)


def candidate_files(mode, item):
    if mode not in MODES:
        raise ValueError("Unexpected V5 arm")
    return {key: safe_child(TRAIN / mode, item[key]) for key in ("model", "original", "flip")}


def bound_payload(payload, config, mode, plan):
    if payload.get("config") != config or payload.get("classes") != CLASSES:
        raise ValueError("Checkpoint configuration/official class order differs")
    if payload.get("mode", mode) != mode:
        raise ValueError("Checkpoint belongs to the other arm")
    if payload.get("source_pins") != plan["source_pins"] or payload.get("parent_sha") != plan["parent_checkpoint_sha256"]:
        raise ValueError("Checkpoint source/parent binding differs")
    if payload.get("weight_summary_sha256", config.get("weight_summary_sha256")) != plan["weight_summary_sha256"]:
        raise ValueError("Checkpoint sample weights are bound to another policy")
    for key in ("dino_repository", "official_dino_sha256"):
        if payload.get(key) != config.get(key):
            raise ValueError("Checkpoint official DINO/source binding differs")
    for split in ("train", "val"):
        if payload.get(split + "_indices") != plan[split + "_indices"]:
            raise ValueError("Checkpoint train/validation members differ")
    check_state(payload.get("adapter_state"))


def verify_training(mode, plan, pins, parent):
    folder = TRAIN / mode
    summary, config = c.read_json(folder / "summary.json"), c.read_json(folder / "config.json")
    if summary.get("state") != "complete" or summary.get("epochs_completed") != 4 or summary.get("classes") != CLASSES or summary.get("mode") != mode:
        raise ValueError("Four complete V5 epochs are required for each arm")
    for key in ("classes", "source_pins", "input_receipt", "train_indices", "val_indices", "weight_summary_sha256", "parent_checkpoint_sha256"):
        if config.get(key) != plan.get(key):
            raise ValueError("Arm configuration differs from the locked plan: " + key)
    if config.get("settings") != plan["settings"] or config.get("parent_checkpoint") != plan["parent_checkpoint"]:
        raise ValueError("Arm hyperparameters or parent path differ from the locked plan")
    if config.get("architecture") != parent["config"]["architecture"]:
        raise ValueError("V5 cannot change the selected V4 LoRA architecture")
    for key in ("dino_repository", "official_dino_sha256"):
        if config.get(key) != parent.get(key):
            raise ValueError("V5 must retain the parent's exact official DINO/source binding")
    if config.get("mode") != mode or config.get("plan_sha256") != h.sha256(PLAN) or summary.get("plan_sha256") != h.sha256(PLAN):
        raise ValueError("Arm mode or plan SHA differs")
    artifacts = summary["artifact_sha256"]
    required = {"config.json", "last.json", "last.pt", "history.json", "candidates.json", "epochs/epoch_04.pt"}
    if not required.issubset(artifacts):
        raise ValueError("Training summary omits a required committed artifact")
    for relative, digest in artifacts.items():
        pin(pins, safe_child(folder, relative), digest)
    pin(pins, folder / "summary.json")
    pointer = c.read_json(folder / "last.json")
    if pointer.get("epoch") != 4 or pointer.get("checkpoint") != "epochs/epoch_04.pt" or pointer.get("plan_sha256") != h.sha256(PLAN):
        raise ValueError("Last pointer does not identify the fourth complete epoch")
    h.check_hash(folder / pointer["checkpoint"], pointer["sha256"])
    h.check_hash(folder / "last.pt", pointer["sha256"])
    final = torch.load(folder / pointer["checkpoint"], map_location="cpu", weights_only=True)
    bound_payload(final, config, mode, plan)
    history = final["history"]
    if len(history) != 4 or history != c.read_json(folder / "history.json"):
        raise ValueError("Final checkpoint history is incomplete or differs from its JSON")
    steps = skips = zero = 0
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or row.get("seen") != 21248:
            raise ValueError("An epoch did not visit the full, matched training membership")
        successful, skipped, empty = (row[key] for key in ("successful_optimizer_steps", "amp_skipped_steps", "zero_weight_groups"))
        if successful <= 0 or min(skipped, empty) < 0 or successful + skipped + empty != 664:
            raise ValueError("Effective-group success/AMP/zero-weight counts do not cover664 groups")
        if mode == "uniform_control" and empty != 0:
            raise ValueError("Uniform weights cannot produce an empty-weight group")
        steps += successful
        skips += skipped
        zero += empty
        if row["cumulative_optimizer_steps"] != steps or row["ema_updates"] != steps:
            raise ValueError("EMA or cumulative optimizer updates include a skipped/zero group")
    if steps != final["successful_steps"] or steps != final["ema_state"]["updates"] or steps != summary["successful_optimizer_steps"] or steps != summary["ema_updates"]:
        raise ValueError("Final optimizer/EMA update counts differ")
    if skips != final["cumulative_skips"] or skips != summary["amp_skipped_steps"]:
        raise ValueError("Final AMP skip count differs")
    if summary.get("zero_weight_groups", zero) != zero:
        raise ValueError("Final zero-weight-group count differs")
    expected_counters = {"successful_steps": steps, "amp_skipped_steps": skips, "zero_weight_groups": zero}
    if final.get("counters") != expected_counters or summary.get("counters") != expected_counters:
        raise ValueError("Committed detailed success/skip/zero-group counters differ")
    candidates = c.read_json(folder / "candidates.json")
    expected = ["epoch_00_baseline"] + [f"epoch_{epoch:02d}_{source}" for epoch in range(1, 5) for source in ("raw", "ema")]
    if [item["name"] for item in candidates] != expected or candidates != final["candidates"] or candidates != summary.get("candidates"):
        raise ValueError("Fixed baseline/raw/EMA candidate set is incomplete or reordered")
    committed = {}
    for epoch in range(1, 5):
        path = folder / f"epochs/epoch_{epoch:02d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bound_payload(payload, config, mode, plan)
        if payload.get("epoch") != epoch or payload["history"] != history[:epoch] or payload["candidates"] != candidates[:1+2*epoch]:
            raise ValueError("An immutable complete-epoch checkpoint disagrees with the final history")
        check_state(payload["ema_state"]["shadow"])
        committed[epoch] = {"raw": payload["adapter_state"], "ema": payload["ema_state"]["shadow"]}
        pin(pins, path)
    for item in candidates:
        epoch, source = item["epoch"], item["weight_source"]
        if item.get("classes") != CLASSES or item.get("val_indices") != plan["val_indices"] or item.get("plan_sha256") != h.sha256(PLAN) or item.get("mode", mode) != mode:
            raise ValueError("Candidate classes/member order/plan/arm differ")
        if item.get("validation_weighted") is not False:
            raise ValueError("Validation metrics must be unweighted")
        if item != c.read_json(safe_child(folder, item["candidate_json"])):
            raise ValueError("Candidate JSON differs from the committed list")
        files = candidate_files(mode, item)
        for key, path in files.items():
            pin(pins, path, item["artifact_sha256"][key])
        pin(pins, safe_child(folder, item["candidate_json"]))
        payload = torch.load(files["model"], map_location="cpu", weights_only=True)
        bound_payload(payload, config, mode, plan)
        if payload.get("epoch") != epoch or payload.get("weight_source") != source:
            raise ValueError("Candidate checkpoint belongs to another epoch/source")
        expected_state = parent["adapter_state"] if epoch == 0 else committed[epoch][source]
        if not same_state(payload["adapter_state"], expected_state):
            raise ValueError("Candidate parameters differ from the committed raw/EMA/parent state")
        for key in ("single_original_metrics", "single_tta_metrics"):
            if payload.get(key) != item[key]:
                raise ValueError("Candidate metrics differ from their checkpoint")
    return candidates, summary, config, history


def candidate_views(mode, item, targets):
    files = candidate_files(mode, item)
    raw = [np.load(files[view], allow_pickle=False) for view in ("original", "flip")]
    for value in raw:
        if value.dtype != np.float32 or value.shape != (2656, 6):
            raise ValueError("Candidate validation probability dtype/shape differs")
    views = [c.check_probability(value, 2656) for value in raw]
    original, single = c.metrics(targets, raw[0]), c.metrics(targets, (raw[0] + raw[1]) * .5)
    if original != item["single_original_metrics"] or single != item["single_tta_metrics"]:
        raise ValueError("Saved candidate probabilities do not reproduce its metrics")
    return views, single


def fixed_replacement(incumbent, old_views, new_views):
    raw = incumbent - .1 * old_views[0] - .1 * old_views[1] + .1 * new_views[0] + .1 * new_views[1]
    if raw.min() < -1e-10:
        raise ValueError("Fixed20% LoRA replacement produced negative probabilities")
    return c.check_probability(np.maximum(raw, 0), len(incumbent))


def metric_record(name, mode, item, single, fused):
    return {"name": name, "mode": mode, "epoch": item["epoch"], "weight_source": item["weight_source"],
            "single_accuracy": single["accuracy"], "single_macro_f1": single["macro_f1"],
            "single_log_loss": single["log_loss"], "single_correct_count": single["correct_count"],
            **fused}


def write_metrics(entries):
    path = OUT / "candidate_metrics.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["name", "mode", "epoch", "weight_source", "single_accuracy", "single_macro_f1",
              "single_log_loss", "single_correct_count", "accuracy", "macro_f1", "log_loss", "correct_count"]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    os.replace(temporary, path)


def verify_locked():
    plan, _, _ = load_plan()
    recipe = c.read_json(OUT / "selected_recipe.json")
    if recipe.get("state") != "locked" or recipe.get("classes") != CLASSES or recipe.get("plan_sha256") != h.sha256(PLAN) or recipe.get("finalizer_sha256") != h.sha256(__file__):
        raise ValueError("The locked V5 recipe has another plan/source/class order")
    if recipe.get("selection_order") != ["incumbent", *MODES] or recipe.get("fusion_weights") != {"hse": .5, "lora": .2, "cnn": .2, "mlp": .1}:
        raise ValueError("The fixed candidate order/fusion changed")
    c.verify_pins(ROOT, recipe["producer_files"])
    h.check_hash(OUT / "selected_val_probabilities.npy", recipe["selected_val_sha256"])
    h.check_hash(OUT / "candidate_metrics.csv", recipe["candidate_metrics_sha256"])
    verify_preserved()
    return plan, recipe


def select():
    plan, pins, weight_summary = load_plan()
    OUT.mkdir(exist_ok=True)
    if (OUT / "selected_recipe.json").exists():
        return verify_locked()[1]
    rows, receipt = h.load_trainval(ROOT)
    if receipt != plan["input_receipt"]:
        raise ValueError("Training/validation image bytes or membership changed")
    for split in ("train", "val"):
        if [row["manifest_index"] for row in rows[split]] != plan[split + "_indices"]:
            raise ValueError("Training/validation member order differs from the plan")
    if weights_module.original.signature(rows["train"]) != weight_summary["binding"]["train_membership_sha256"]:
        raise ValueError("Difficulty weights bind to different training members")
    targets = np.asarray([row["target"] for row in rows["val"]])
    incumbent, incumbent_metrics, old_views, parent, baseline_summary = load_baseline(plan, pins, targets)
    best_p, best, selected = incumbent, incumbent_metrics, None
    entries = [{"name": "previous_best", "mode": "incumbent", **incumbent_metrics}]
    arm_comparison, histories, configs, paired = {}, {}, {}, {}
    for mode in MODES:
        candidates, training_summary, config, history = verify_training(mode, plan, pins, parent)
        histories[mode], configs[mode] = history, config
        arm_best_single = arm_best_fused = None
        arm_entries = []
        for item in candidates:
            views, single = candidate_views(mode, item, targets)
            if item["epoch"] == 0:
                if any(not np.allclose(view, old, atol=1e-5, rtol=1e-5) for view, old in zip(views, old_views)):
                    raise ValueError("An arm's epoch-zero predictions differ from the shared parent")
                probability = incumbent.copy()
            else:
                probability = fixed_replacement(incumbent, old_views, views)
            fused = c.metrics(targets, probability)
            name = f"{mode}/{item['name']}"
            entry = metric_record(name, mode, item, single, fused)
            entries.append(entry)
            arm_entries.append(entry)
            paired[(mode, item["epoch"], item["weight_source"])] = entry
            if arm_best_single is None or single["correct_count"] > arm_best_single["metrics"]["correct_count"]:
                arm_best_single = {"name": name, "epoch": item["epoch"], "weight_source": item["weight_source"], "metrics": single}
            if arm_best_fused is None or fused["correct_count"] > arm_best_fused["metrics"]["correct_count"]:
                arm_best_fused = {"name": name, "epoch": item["epoch"], "weight_source": item["weight_source"], "metrics": fused}
            if fused["correct_count"] > best["correct_count"]:
                selected, best, best_p = {**item, "mode": mode}, fused, probability
        arm_comparison[mode] = {"best_single": arm_best_single, "best_fixed_fusion": arm_best_fused,
                                "training_counts": {key: training_summary[key] for key in ("successful_optimizer_steps", "amp_skipped_steps", "ema_updates")},
                                "zero_weight_groups": sum(row["zero_weight_groups"] for row in history), "candidates": arm_entries}
    if configs[MODES[0]]["settings"] != configs[MODES[1]]["settings"]:
        raise ValueError("The matched arms used different training hyperparameters")
    if [row["augmentation_epoch"] for row in histories[MODES[0]]] != [row["augmentation_epoch"] for row in histories[MODES[1]]]:
        raise ValueError("The matched arms used different augmentation epochs")
    pairs = []
    for epoch in range(1, 5):
        for source in ("raw", "ema"):
            uniform, weighted = (paired[(mode, epoch, source)] for mode in MODES)
            pairs.append({"epoch": epoch, "weight_source": source,
                          "weighted_minus_uniform_single_correct": weighted["single_correct_count"] - uniform["single_correct_count"],
                          "weighted_minus_uniform_fused_correct": weighted["correct_count"] - uniform["correct_count"]})
    arm_comparison["paired_same_epoch_source"] = pairs
    arm_comparison["weighted_minus_uniform_best_fused_correct"] = (arm_comparison["weighted"]["best_fixed_fusion"]["metrics"]["correct_count"]
                                                                  - arm_comparison["uniform_control"]["best_fixed_fusion"]["metrics"]["correct_count"])
    write_metrics(entries)
    if selected is None:
        atomic_copy(BASELINE / "selected_val_probabilities.npy", OUT / "selected_val_probabilities.npy")
    else:
        h.save_array(OUT / "selected_val_probabilities.npy", best_p)
    recipe = {"format": "zero_weight_v5_fixed_fusion", "state": "locked", "classes": CLASSES,
              "selected": selected, "improved": selected is not None, "baseline_metrics": incumbent_metrics,
              "selected_metrics": best, "selection_order": ["incumbent", *MODES], "candidate_count": 18,
              "selection_rule": "strict correct-count increase; every tie retains the earlier incumbent/candidate",
              "fusion_weights": {"hse": .5, "lora": .2, "cnn": .2, "mlp": .1},
              "replacement": "current-.1*old_original-.1*old_flip+.1*new_original+.1*new_flip",
              "validation_candidates": entries, "arm_comparison": arm_comparison, "producer_files": pins,
              "plan_sha256": h.sha256(PLAN), "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
              "weight_summary_sha256": plan["weight_summary_sha256"], "actual_training_weights": {"normal": 1., "uncertain": .15, "very_uncertain": 0.},
              "val_indices": plan["val_indices"], "val_rows_sha256": receipt["val_rows_sha256"],
              "selected_val_sha256": h.sha256(OUT / "selected_val_probabilities.npy"),
              "candidate_metrics_sha256": h.sha256(OUT / "candidate_metrics.csv"),
              "competition_image_content_sha256": baseline_summary["competition_image_content_sha256"],
              "competition_row_order_sha256": baseline_summary["competition_row_order_sha256"],
              "finalizer_sha256": h.sha256(__file__), "internal_test_read": False, "external_answers_used": False,
              "preserved_submission_sha256": PRESERVED, "validation_caveat": CAVEAT}
    c.verify_pins(ROOT, pins)
    h.write_json(OUT / "selected_recipe.json", recipe)
    emit("select_zero_weight_v5", improved=recipe["improved"], selected=(f"{selected['mode']}/{selected['name']}" if selected else "previous_best"),
         correct_count=best["correct_count"], accuracy=best["accuracy"], weighted_minus_uniform_best_fused_correct=arm_comparison["weighted_minus_uniform_best_fused_correct"])
    return recipe


def atomic_copy(source, destination):
    import shutil
    temporary = Path(str(destination) + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def verify_submission(rows, probability):
    c.submission_check(OUT / "submission.csv", rows, probability)
    with (OUT / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "Emotion"] or len(list(reader)) != 7065:
            raise ValueError("Final submission requires exactly7065 rows and ID,Emotion columns")


def predict_winner(recipe, rows, batch_size=16):
    import train_zero_weight_lora_v5 as trainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU prediction fallback")
    item = recipe["selected"]
    files = candidate_files(item["mode"], item)
    model, _ = trainer.load_checkpoint(ROOT, files["model"], "cuda")
    model.eval()
    loader = DataLoader(l.FaceImages([{**row, "target": 0} for row in rows], training=False),
                        batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
    outputs = [[], []]
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    started = last = time.monotonic()
    try:
        with torch.inference_mode():
            for index, (images, _) in enumerate(loader):
                images = images.cuda()
                with h.autocast_for("cuda"):
                    logits = (model(images), model(images.flip(-1)))
                for view, value in enumerate(logits):
                    outputs[view].append(value.float().softmax(1).cpu().numpy())
                if time.monotonic() - last >= 30 or index + 1 == len(loader):
                    h.check_hash(OUT / "selected_recipe.json", recipe_sha)
                    h.check_hash(files["model"], item["artifact_sha256"]["model"])
                    emit("export_zero_weight_v5", seen=min((index+1)*batch_size, len(rows)), total=len(rows), elapsed_seconds=time.monotonic()-started)
                    last = time.monotonic()
        return [c.check_probability(np.concatenate(values), 7065) for values in outputs]
    finally:
        del model
        torch.cuda.empty_cache()


def export(batch_size=16):
    select()
    _, recipe = verify_locked()
    recipe_sha = h.sha256(OUT / "selected_recipe.json")
    rows = c.unlabeled_rows(ROOT)
    if c.signature(rows) != ROW_SHA:
        raise ValueError("Competition IDs/order differ from the reused V4 probabilities")
    images = c.image_receipt(ROOT, rows, verify_inventory=False)
    if images["image_content_sha256"] != IMAGE_SHA:
        raise ValueError("Competition image content differs from the reused V4 probabilities")
    if (OUT / "summary.json").exists():
        summary = c.read_json(OUT / "summary.json")
        if summary.get("state") != "complete" or summary.get("classes") != CLASSES or summary.get("recipe_sha256") != recipe_sha:
            raise ValueError("Completed export uses another locked recipe")
        for relative, digest in summary["file_sha256"].items():
            h.check_hash(safe_child(OUT, relative), digest)
        probability = c.check_probability(np.load(OUT / "probabilities.npy", allow_pickle=False), 7065)
        verify_submission(rows, probability)
        verify_preserved()
        return summary
    if recipe["improved"]:
        new_views = predict_winner(recipe, rows, batch_size)
        for view, array in zip(("original", "flip"), new_views):
            h.save_array(OUT / f"lora_competition_{view}.npy", array)
        baseline = c.check_probability(np.load(BASELINE / "probabilities.npy", allow_pickle=False), 7065)
        old_views = [c.check_probability(np.load(BASELINE / f"lora_competition_{view}.npy", allow_pickle=False), 7065)
                     for view in ("original", "flip")]
        probability = fixed_replacement(baseline, old_views, new_views)
        h.save_array(OUT / "probabilities.npy", probability)
        temporary = OUT / "submission.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ID", "Emotion"])
            writer.writerows((row["ID"], int(label)) for row, label in zip(rows, probability.argmax(1)))
        os.replace(temporary, OUT / "submission.csv")
    else:
        atomic_copy(BASELINE / "probabilities.npy", OUT / "probabilities.npy")
        atomic_copy(BASELINE / "submission.csv", OUT / "submission.csv")
        h.check_hash(OUT / "submission.csv", PRESERVED["lora_task_v4_results/submission.csv"])
        probability = c.check_probability(np.load(OUT / "probabilities.npy", allow_pickle=False), 7065)
    verify_submission(rows, probability)
    h.check_hash(OUT / "selected_recipe.json", recipe_sha)
    c.verify_pins(ROOT, recipe["producer_files"])
    load_plan()
    if c.image_receipt(ROOT, rows, verify_inventory=False)["image_content_sha256"] != images["image_content_sha256"]:
        raise ValueError("Competition image bytes changed during inference/export")
    names = ["submission.csv", "probabilities.npy", "selected_val_probabilities.npy", "selected_recipe.json", "candidate_metrics.csv"]
    if recipe["improved"]:
        names.extend(["lora_competition_original.npy", "lora_competition_flip.npy"])
    summary = {"format": "zero_weight_v5_export", "state": "complete", "classes": CLASSES,
               "submission_rows": 7065, "improved": recipe["improved"],
               "selected": f"{recipe['selected']['mode']}/{recipe['selected']['name']}" if recipe["selected"] else "previous_best",
               "selected_mode": recipe["selected"]["mode"] if recipe["selected"] else "incumbent",
               "baseline_metrics": recipe["baseline_metrics"], "selected_metrics": recipe["selected_metrics"],
               "arm_comparison": recipe["arm_comparison"], "recipe_sha256": recipe_sha,
               "plan_sha256": recipe["plan_sha256"], "finalizer_sha256": h.sha256(__file__),
               "official_label_order_verified": True, "ID_template_order_verified": True, "probability_argmax_verified": True,
               "competition_image_content_sha256": IMAGE_SHA, "competition_row_order_sha256": ROW_SHA,
               "other_model_inference_repeated": False, "new_lora_competition_inference": recipe["improved"],
               "preserved_submission_sha256": PRESERVED, "internal_test_read": False, "external_answers_used": False,
               "validation_caveat": CAVEAT, "file_sha256": {name: h.sha256(OUT / name) for name in names}}
    verify_preserved()
    h.write_json(OUT / "summary.json", summary)
    emit("complete_zero_weight_v5", improved=summary["improved"], selected=summary["selected"],
         correct_count=summary["selected_metrics"]["correct_count"], accuracy=summary["selected_metrics"]["accuracy"], submission=str(OUT / "submission.csv"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("select", "export"), default="export")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8, 16), default=16)
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("This locked finalizer only supports its fixed project root")
    torch.set_num_threads(4)
    if args.stage == "select":
        select()
    else:
        export(args.batch_size)


if __name__ == "__main__":
    main()
