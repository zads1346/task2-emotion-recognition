"""Seven fixed HSE validation fusions and locked-recipe submission export.

Selection uses only the original validation split. Existing DINO/CNN baseline
predictions are verified and reused. HSE competition inference occurs only after
a recipe is locked and only if it strictly improves validation accuracy. Neither
internal-test labels nor external answers are read, and no model is trained here.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import validate_cosine_fusion as base

CLASSES = base.CLASSES
EXPECTED_BASELINE = {"accuracy": .6957831325301205, "macro_f1": .6739671930971368,
                     "log_loss": .8396570359300835, "correct_count": 1848, "samples": 2656}
BASELINE_WEIGHTS = {"refined_lora": .4, "cnn": .4, "old_mlp": 0., "new_mlp": .2}
BASELINE_SUBMISSION_SHA256 = "2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad"
BASELINE_PROBABILITY_SHA256 = "5f160d95354766fcccbb965f4ef7c250e5e2f59128e4576c1c503ba52bba71e8"
CANDIDATES = [{"name": "baseline", "alpha": 0., "hse_mode": "plain"}] + [
    {"name": f"hse_{int(alpha * 100):02d}_{mode}", "alpha": alpha, "hse_mode": mode}
    for alpha in [.1, .25, .5] for mode in ["plain", "flip"]
]


def hse_blend(baseline, original, flipped, alpha, mode):
    if alpha not in [0., .1, .25, .5] or mode not in ["plain", "flip"]:
        raise ValueError("HSE fusion is outside the seven prespecified candidates")
    baseline = base.check_probability(baseline)
    if alpha == 0:
        return baseline
    original = base.check_probability(original, len(baseline))
    value = original if mode == "plain" else (original + base.check_probability(flipped, len(baseline))) / 2
    return base.check_probability((1 - alpha) * baseline + alpha * value)


def evaluate_candidates(targets, baseline, original, flipped):
    evaluated, arrays = [], {}
    baseline_correct = base.check_probability(baseline, len(targets)).argmax(1) == targets
    for specification in CANDIDATES:
        p = hse_blend(baseline, original, flipped, specification["alpha"], specification["hse_mode"])
        arrays[specification["name"]] = p
        correct = p.argmax(1) == targets
        evaluated.append({**specification, "weights": {"baseline": 1 - specification["alpha"], "hse": specification["alpha"]},
                          **base.metrics(targets, p),
                          "paired_vs_baseline": {"both_correct": int(np.sum(baseline_correct & correct)),
                                                 "baseline_only_correct": int(np.sum(baseline_correct & ~correct)),
                                                 "candidate_only_correct": int(np.sum(~baseline_correct & correct)),
                                                 "both_wrong": int(np.sum(~baseline_correct & ~correct)),
                                                 "net_correct_change": int(np.sum(correct) - np.sum(baseline_correct))}})
    best = max(evaluated[1:], key=lambda item: (item["correct_count"], item["macro_f1"], -item["log_loss"],
                                               -item["alpha"], item["hse_mode"] == "plain"))
    selected = best if best["correct_count"] > evaluated[0]["correct_count"] else evaluated[0]
    return evaluated, selected, arrays


def baseline_inputs(root):
    """Reconstruct the exact current validation baseline from pinned view caches."""
    recipe_path, recipe, recipe_sha = base.validate_locked(root)
    if recipe["selected"].get("name") != "replace_mlp_20" or recipe["selected"].get("weights") != BASELINE_WEIGHTS:
        raise ValueError("Expected the frozen .4 refined-LoRA + .4 CNN + .2 cosine-MLP baseline")
    rows, targets, images, old_probability, _, old_pins, _, _ = base.load_old_baseline(root)
    cache_dir = root / "fusion_validation_results" / "prediction_cache" / "new_mlp_val"
    cache_meta = base.read_json(cache_dir / "metadata.json")
    binding = cache_meta.get("binding", {})
    if (cache_meta.get("complete") is not True or cache_meta.get("signature") != base.signature(binding) or
            binding.get("partition") != "val" or binding.get("rows") != len(rows) or binding.get("classes") != CLASSES or
            binding.get("model_sha256") != recipe["artifacts"]["new_mlp"]["sha256"] or
            binding.get("row_order_sha256") != base.signature(rows) or
            binding.get("image_content_sha256") != images["image_content_sha256"] or
            binding.get("producer_files") != recipe.get("producer_files")):
        raise ValueError("Baseline cosine-MLP validation cache has different source/model/VAL membership")
    pins = {**old_pins, **recipe["producer_files"]}
    base.add_pin(pins, recipe_path, recipe_sha)
    base.add_pin(pins, cache_dir / "metadata.json")
    new_views = []
    for view in ["original", "flip"]:
        path = cache_dir / f"{view}.npy"
        base.add_pin(pins, path, cache_meta.get("file_sha256", {}).get(f"{view}.npy"))
        new_views.append(base.check_probability(np.load(path, allow_pickle=False), len(rows)))
    reconstructed = base.blend({**old_probability, "new_mlp": tuple(new_views)}, BASELINE_WEIGHTS)
    probability_path = root / "fusion_validation_results" / "selected_val_probabilities.npy"
    baseline = base.check_probability(np.load(probability_path, allow_pickle=False), len(rows))
    maximum_difference = float(np.max(np.abs(reconstructed - baseline)))
    if maximum_difference > 1e-12:
        raise ValueError("Stored baseline VAL probabilities differ from their pinned producer reconstruction")
    score = base.metrics(targets, baseline)
    for key, expected in EXPECTED_BASELINE.items():
        if not np.isclose(score[key], expected, atol=1e-12, rtol=0) or not np.isclose(recipe["selected"][key], expected, atol=1e-12, rtol=0):
            raise ValueError(f"Frozen HSE baseline metric changed: {key}")
    base.add_pin(pins, probability_path)
    baseline_export = root / "fusion_results"
    export_summary = base.read_json(baseline_export / "summary.json")
    base.official_classes(export_summary.get("classes"))
    if export_summary.get("state") != "complete" or export_summary.get("recipe_sha256") != recipe_sha or export_summary.get("selected") != recipe["selected"]:
        raise ValueError("Existing baseline competition export differs from locked validation recipe")
    for name, expected in [("submission.csv", BASELINE_SUBMISSION_SHA256), ("probabilities.npy", BASELINE_PROBABILITY_SHA256)]:
        if export_summary.get("file_sha256", {}).get(name) != expected:
            raise ValueError("Existing baseline competition artifact identity changed")
        base.add_pin(pins, baseline_export / name, expected)
    base.add_pin(pins, baseline_export / "summary.json")
    base.add_pin(pins, root / "src" / "validate_cosine_fusion.py", recipe["validator_sha256"])
    base.add_pin(pins, root / "src" / "finalize_hse_specialist.py", base.sha(__file__))
    receipt = {"format": "hse_baseline_validation_v1", "finalizer_sha256": base.sha(__file__),
               "baseline_recipe_sha256": recipe_sha, "weights": BASELINE_WEIGHTS,
               "baseline_val_probability_sha256": base.sha(probability_path), "baseline_metrics": score,
               "reconstruction_max_absolute_difference": maximum_difference,
               "val_rows": len(rows), "val_row_order_sha256": base.signature(rows),
               "val_image_content_sha256": images["image_content_sha256"],
               "source_manifest_sha256": base.sha(root / "features" / "manifest.csv"),
               "baseline_submission_sha256": BASELINE_SUBMISSION_SHA256,
               "baseline_competition_probability_sha256": BASELINE_PROBABILITY_SHA256,
               "producer_files": pins, "internal_test_read": False, "external_answers_used": False}
    return rows, targets, baseline, receipt


def load_hse_validation(root, rows, targets):
    trainer = base.module(root, "train_hse_specialist")
    receipt = trainer.load_receipt(root)  # Hash/metadata validation only; does not build a GPU model.
    base.official_classes(receipt.get("classes"))
    if receipt.get("state") != "complete" or receipt.get("epochs_completed") != 12 or receipt.get("checkpoint_selection_view") != "original":
        raise ValueError("HSE must complete twelve epochs with original-view checkpoint selection")
    if not 1 <= int(receipt.get("best_epoch", 0)) <= 12:
        raise ValueError("Invalid HSE selected epoch")
    base.membership(receipt.get("val_indices"), receipt.get("val_paths"), rows)
    manifest_sha = base.sha(root / "features" / "manifest.csv")
    if receipt.get("manifest_sha256") != manifest_sha:
        raise ValueError("HSE source manifest differs from fixed baseline")
    folder = root / "experiments" / "hse_specialist_v1"
    pins = {}
    base.add_pin(pins, folder / "summary.json")
    for name, key in [("best.pt", "best_checkpoint_sha256"), ("val_original.npy", "val_original_sha256"), ("val_flip.npy", "val_flip_sha256")]:
        base.add_pin(pins, folder / name, receipt.get(key))
    for name, digest in receipt.get("source_sha256", {}).items():
        base.add_pin(pins, root / "src" / name, digest)
    if {"train_hse_specialist.py", "hse_model.py"} - set(receipt.get("source_sha256", {})):
        raise ValueError("HSE receipt does not bind its trainer and model loader sources")
    original = base.check_probability(np.load(folder / "val_original.npy", allow_pickle=False), len(rows))
    flipped = base.check_probability(np.load(folder / "val_flip.npy", allow_pickle=False), len(rows))
    original_score = base.metrics(targets, original)
    for key in ["accuracy", "macro_f1", "log_loss"]:
        if not np.isclose(original_score[key], receipt.get("best_validation", {}).get(key, np.nan), atol=1e-9, rtol=0):
            raise ValueError("HSE selected-checkpoint original validation cache differs from training receipt")
    return original, flipped, receipt, pins


def verify_locked(root):
    folder = root / "hse_results"
    path = folder / "selected_recipe.json"
    recipe = base.read_json(path)
    base.official_classes(recipe.get("classes"))
    if recipe.get("state") != "locked" or recipe.get("candidate_configurations") != CANDIDATES or recipe.get("finalizer_sha256") != base.sha(__file__):
        raise ValueError("HSE recipe is not locked or protocol/source changed")
    selected = recipe.get("selected", {})
    definition = next((item for item in CANDIDATES if item["name"] == selected.get("name")), None)
    if (definition is None or any(selected.get(key) != definition[key] for key in ["alpha", "hse_mode"]) or
            selected.get("weights") != {"baseline": 1 - selected["alpha"], "hse": selected["alpha"]} or
            recipe.get("requires_new_predictions") != (selected.get("name") != "baseline")):
        raise ValueError("Locked HSE selection/weights/gating disagree")
    base.verify_pins(root, recipe["producer_files"])
    rows = base.val_rows(root)
    if base.signature(rows) != recipe.get("val_inputs", {}).get("row_order_sha256"):
        raise ValueError("HSE locked validation membership/order changed")
    baseline_receipt = base.read_json(folder / "baseline_validation_receipt.json")
    if base.sha(folder / "baseline_validation_receipt.json") != recipe.get("baseline_receipt_sha256"):
        raise ValueError("Saved baseline validation provenance receipt changed")
    if baseline_receipt.get("baseline_recipe_sha256") != base.sha(root / "fusion_validation_results" / "selected_recipe.json"):
        raise ValueError("Original locked baseline recipe changed")
    return path, recipe, base.sha(path)


def select(root):
    output = root / "hse_results"
    if (output / "selected_recipe.json").exists():
        _, recipe, _ = verify_locked(root)
        print("Locked HSE selection verified; no selection repeated.", flush=True)
        return recipe
    rows, targets, baseline, baseline_receipt = baseline_inputs(root)
    output.mkdir(parents=True, exist_ok=True)
    baseline_receipt_path = output / "baseline_validation_receipt.json"
    if baseline_receipt_path.exists():
        if base.read_json(baseline_receipt_path) != baseline_receipt:
            raise ValueError("Saved baseline validation provenance differs; refuse overwrite")
    else:
        base.write_json(baseline_receipt_path, baseline_receipt)
    original, flipped, hse_receipt, hse_pins = load_hse_validation(root, rows, targets)
    evaluated, selected, probabilities = evaluate_candidates(targets, baseline, original, flipped)
    pins = {**baseline_receipt["producer_files"], **hse_pins}
    base.add_pin(pins, baseline_receipt_path)
    recipe = {"format": "hse_specialist_fusion_v1", "state": "locked", "classes": CLASSES,
              "finalizer_sha256": base.sha(__file__), "candidate_configurations": CANDIDATES, "candidate_count": 7,
              "selected": selected, "requires_new_predictions": selected["name"] != "baseline",
              "selection_rule": "Strict accuracy gain required; ties retain baseline. New-candidate ties: macro-F1, lower log-loss, lower alpha, then plain.",
              "validation_candidates": evaluated, "producer_files": pins,
              "baseline_receipt_sha256": base.sha(baseline_receipt_path),
              "baseline_metrics": baseline_receipt["baseline_metrics"],
              "baseline_submission_sha256": BASELINE_SUBMISSION_SHA256,
              "baseline_competition_probability_sha256": BASELINE_PROBABILITY_SHA256,
              "hse_checkpoint": {"path": str(root / "experiments" / "hse_specialist_v1" / "best.pt"),
                                 "sha256": hse_receipt["best_checkpoint_sha256"], "best_epoch": hse_receipt["best_epoch"]},
              "val_inputs": {"rows": len(rows), "row_order_sha256": base.signature(rows),
                             "image_content_sha256": baseline_receipt["val_image_content_sha256"],
                             "manifest_indices": [int(row["manifest_index"]) for row in rows]},
              "checkpoint_selection_view": "original", "fusion_TTA_selection": True,
              "baseline_model_inference_repeated": False, "internal_test_read": False, "external_answers_used": False,
              "validation_caveat": "Reused validation selected HSE checkpoint and these seven fixed fusions; this is not independent generalization performance."}
    base.verify_pins(root, pins)
    base.write_json(output / "selected_recipe.json", recipe)
    with (output / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["name", "alpha", "hse_mode", "accuracy", "macro_f1", "log_loss", "correct_count"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({key: item[key] for key in fields} for item in evaluated)
    np.save(output / "selected_val_probabilities.npy", probabilities[selected["name"]], allow_pickle=False)
    print(json.dumps({"stage": "select", "state": "locked", "selected": selected["name"],
                      "correct": selected["correct_count"], "rows": len(rows), "accuracy": selected["accuracy"]}), flush=True)
    return recipe


def predict_hse_views(root, checkpoint, rows, include_flip, device, batch_size, lock_check):
    import torch
    from PIL import Image, ImageOps
    trainer = base.module(root, "train_hse_specialist")
    model_module = base.module(root, "hse_model")
    model = model_module.load_trained_model(root, checkpoint, device)
    model.eval()
    transform = trainer.make_eval_transform()
    outputs = {"original": []}
    if include_flip:
        outputs["flip"] = []
    last_log = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                lock_check()
                tensors = []
                for row in rows[start:start + batch_size]:
                    with Image.open(row["path"]) as image:
                        tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                x = torch.stack(tensors).to(device)
                for view in outputs:
                    value = x if view == "original" else x.flip(-1)
                    with trainer.autocast_for(device):
                        logits = model(value)
                    p = logits.float().softmax(1).cpu().numpy()
                    outputs[view].append(base.check_probability(p, len(value)))
                if time.monotonic() - last_log >= 30 or start + batch_size >= len(rows):
                    print(f"Locked HSE competition prediction {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
                    last_log = time.monotonic()
        original = np.concatenate(outputs["original"])
        flipped = np.concatenate(outputs["flip"]) if include_flip else original.copy()
        lock_check()
        return original, flipped
    finally:
        del model; gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def export(root, device="cuda", batch_size=16):
    recipe_path, recipe, recipe_sha = verify_locked(root)
    output = root / "hse_results"
    rows = base.unlabeled_rows(root)
    baseline_folder = root / "fusion_results"
    base.checked_hash(baseline_folder / "submission.csv", recipe["baseline_submission_sha256"])
    base.checked_hash(baseline_folder / "probabilities.npy", recipe["baseline_competition_probability_sha256"])
    baseline = base.check_probability(np.load(baseline_folder / "probabilities.npy", allow_pickle=False), len(rows))
    base.submission_check(baseline_folder / "submission.csv", rows, baseline)

    def lock_check():
        base.checked_hash(recipe_path, recipe_sha)

    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = base.read_json(summary_path)
        if (summary.get("state") != "complete" or summary.get("recipe_sha256") != recipe_sha or
                summary.get("finalizer_sha256") != base.sha(__file__) or summary.get("row_order_sha256") != base.signature(rows)):
            raise ValueError("Completed HSE export is incomplete or belongs to different inputs/source/recipe")
        for name in ["submission.csv", "probabilities.npy"]:
            base.checked_hash(output / name, summary.get("file_sha256", {}).get(name))
        base.submission_check(output / "submission.csv", rows, np.load(output / "probabilities.npy", allow_pickle=False))
        print("Completed HSE export verified; no inference repeated.", flush=True)
        return summary
    start_receipt = {"recipe_sha256": recipe_sha, "finalizer_sha256": base.sha(__file__), "row_order_sha256": base.signature(rows)}
    start_path = output / "export_started.json"
    if start_path.exists() and base.read_json(start_path) != start_receipt:
        raise ValueError("Partial HSE export belongs to a different locked recipe/source/ID order")
    if not start_path.exists() and any((output / name).exists() for name in ["submission.csv", "probabilities.npy"]):
        raise ValueError("Orphan HSE export without identity receipt")
    base.write_json(start_path, start_receipt)
    prediction_receipt = None
    image_content_sha = None
    if not recipe["requires_new_predictions"]:
        lock_check()
        import shutil
        shutil.copyfile(baseline_folder / "submission.csv", output / "submission.csv")
        shutil.copyfile(baseline_folder / "probabilities.npy", output / "probabilities.npy")
        probability = baseline
        export_mode, inference = "unchanged_verified_baseline_copy", False
    else:
        images = base.image_receipt(root, rows, verify_inventory=True)
        image_content_sha = images["image_content_sha256"]
        selected = recipe["selected"]
        checkpoint = recipe["hse_checkpoint"]
        base.checked_hash(checkpoint["path"], checkpoint["sha256"])
        binding = {"partition": "unlabeled", "rows": len(rows), "classes": CLASSES, "recipe_sha256": recipe_sha,
                   "checkpoint_sha256": checkpoint["sha256"], "finalizer_sha256": base.sha(__file__),
                   "row_order_sha256": base.signature(rows), "image_content_sha256": image_content_sha,
                   "producer_files_sha256": base.signature(recipe["producer_files"]), "hse_mode": selected["hse_mode"],
                   "preprocessing": "trainer.make_eval_transform; EXIF transpose RGB; flip normalized tensor when selected"}
        cache_folder = output / "prediction_cache" / "hse_unlabeled"
        original, flipped = base.cached_prediction(cache_folder, binding,
            lambda: predict_hse_views(root, checkpoint["path"], rows, selected["hse_mode"] == "flip", device, batch_size, lock_check), lock_check)
        prediction_receipt = {"metadata_sha256": base.sha(cache_folder / "metadata.json"), "binding_sha256": base.signature(binding)}
        probability = hse_blend(baseline, original, flipped, selected["alpha"], selected["hse_mode"])
        lock_check()
        np.save(output / "probabilities.npy", probability, allow_pickle=False)
        with (output / "submission.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["ID", "Emotion"])
            writer.writerows((row["ID"], int(target)) for row, target in zip(rows, probability.argmax(1)))
        export_mode, inference = "locked_HSE_fusion", True
    lock_check()
    base.submission_check(output / "submission.csv", rows, probability)
    summary = {"format": "hse_specialist_export_v1", "state": "complete", "classes": CLASSES,
               "recipe_sha256": recipe_sha, "finalizer_sha256": base.sha(__file__), "selected": recipe["selected"],
               "row_order_sha256": base.signature(rows), "submission_rows": len(rows), "export_mode": export_mode,
               "hse_competition_inference": inference, "baseline_model_inference_repeated": False,
               "improvement_on_reused_validation": recipe["requires_new_predictions"],
               "baseline_submission_sha256": recipe["baseline_submission_sha256"],
               "baseline_competition_probability_sha256": recipe["baseline_competition_probability_sha256"],
               "unlabeled_image_content_sha256": image_content_sha, "prediction_receipt": prediction_receipt,
               "internal_test_read": False, "external_answers_used": False,
               "file_sha256": {name: base.sha(output / name) for name in ["submission.csv", "probabilities.npy"]}}
    base.write_json(summary_path, summary)
    print(json.dumps({"stage": "export", "state": "complete", "mode": export_mode, "rows": len(rows)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["select", "export"])
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    root = args.project_root.resolve()
    select(root) if args.stage == "select" else export(root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
