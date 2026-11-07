"""Select four fixed validation fusions, then export only the locked recipe.

No internal-test labels/images are inspected. The original manifest is streamed
and only split=val labels are parsed. The new head is already fixed and trained
on original train only; validation does not fit weights, calibration, or epochs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EXPECTED_VAL, EXPECTED_UNLABELED = 2656, 7065
BASELINE_EXPECTED = {"accuracy": .6935240963855421, "macro_f1": .6696894939182741}
CANDIDATES = [
    {"name": "baseline", "weights": {"refined_lora": .45, "cnn": .45, "old_mlp": .10, "new_mlp": 0.}},
    {"name": "replace_mlp_10", "weights": {"refined_lora": .45, "cnn": .45, "old_mlp": 0., "new_mlp": .10}},
    {"name": "replace_mlp_20", "weights": {"refined_lora": .40, "cnn": .40, "old_mlp": 0., "new_mlp": .20}},
    {"name": "split_mlp_05", "weights": {"refined_lora": .45, "cnn": .45, "old_mlp": .05, "new_mlp": .05}},
]
_NO_EXPECTATION = object()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def csv_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV fields {required}: {path}")
        result = list(reader)
    if not result:
        raise ValueError(f"Empty CSV: {path}")
    return result


def checked_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return expected


def add_pin(pins, path, expected=_NO_EXPECTATION):
    path = Path(path)
    actual = sha(path)
    if expected is not _NO_EXPECTATION and (not isinstance(expected, str) or len(expected) != 64 or actual != expected):
        raise ValueError(f"Checksum mismatch: {path}")
    key = str(path)
    if key in pins and pins[key] != actual:
        raise ValueError(f"Conflicting file provenance: {path}")
    pins[key] = actual
    return actual


def verify_pins(root, pins):
    last = time.monotonic()
    for i, (path, expected) in enumerate(pins.items(), 1):
        # These are source/artifact paths, not thousands of image paths.
        absolute = Path(path).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError(f"Provenance path lies outside project: {path}")
        checked_hash(absolute, expected)
        if time.monotonic() - last >= 30:
            print(f"Provenance verified {i}/{len(pins)} files", flush=True)
            last = time.monotonic()


def official_classes(value):
    if value != CLASSES:
        raise ValueError("Probability/model classes do not match the official order")


def check_probability(value, count=None):
    p = np.asarray(value, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[1] != 6 or (count is not None and len(p) != count):
        raise ValueError("Probability dimensions/order mismatch")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Invalid probability values or row sums")
    return p / p.sum(1, keepdims=True)


def metrics(targets, probability):
    p = check_probability(probability, len(targets))
    pred = p.argmax(1)
    report = classification_report(targets, pred, labels=np.arange(6), target_names=CLASSES, zero_division=0, output_dict=True)
    return {"accuracy": float(accuracy_score(targets, pred)),
            "macro_f1": float(f1_score(targets, pred, labels=np.arange(6), average="macro", zero_division=0)),
            "log_loss": float(log_loss(targets, p, labels=np.arange(6))),
            "correct_count": int(np.sum(pred == targets)), "samples": len(targets),
            "per_class": {name: report[name] for name in CLASSES},
            "confusion_matrix": confusion_matrix(targets, pred, labels=np.arange(6)).tolist()}


def blend(probabilities, weights):
    if set(weights) != {"refined_lora", "cnn", "old_mlp", "new_mlp"} or not np.isclose(sum(weights.values()), 1, atol=1e-12, rtol=0):
        raise ValueError("Fusion weights must have four official keys and sum to one")
    combined = None
    for name, weight in weights.items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid fusion weight")
        if weight == 0:
            continue
        original, flipped = probabilities[name]
        value = (check_probability(original) + check_probability(flipped)) / 2
        combined = weight * value if combined is None else combined + weight * value
    return check_probability(combined)


def compare_candidates(targets, probabilities):
    evaluated, fused = [], {}
    baseline_correct = None
    for spec in CANDIDATES:
        p = blend(probabilities, spec["weights"])
        fused[spec["name"]] = p
        correct = p.argmax(1) == targets
        if baseline_correct is None:
            baseline_correct = correct
        paired = {"both_correct": int(np.sum(baseline_correct & correct)),
                  "baseline_only_correct": int(np.sum(baseline_correct & ~correct)),
                  "candidate_only_correct": int(np.sum(~baseline_correct & correct)),
                  "both_wrong": int(np.sum(~baseline_correct & ~correct)),
                  "net_correct_change": int(np.sum(correct) - np.sum(baseline_correct)),
                  "predictions_changed": int(np.sum(p.argmax(1) != fused["baseline"].argmax(1)))}
        evaluated.append({**spec, "tta_flip": True, **metrics(targets, p), "paired_vs_baseline": paired})
    best_new = max(evaluated[1:], key=lambda item: (item["correct_count"], item["macro_f1"], -item["log_loss"], -[s["name"] for s in CANDIDATES].index(item["name"])))
    chosen = best_new if best_new["correct_count"] > evaluated[0]["correct_count"] else evaluated[0]
    return evaluated, chosen, fused


def val_rows(root):
    result = []
    with (root / "features" / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest lacks path,label,split")
        for index, row in enumerate(reader):
            if row.get("split") != "val":
                continue
            label = row.get("label")
            if label not in CLASSES or not row.get("path"):
                raise ValueError(f"Invalid VAL row {index}")
            result.append({"manifest_index": str(index), "path": row["path"], "label": label, "target": str(CLASSES.index(label))})
    if len(result) != EXPECTED_VAL or {r["label"] for r in result} != set(CLASSES):
        raise ValueError(f"Require exactly {EXPECTED_VAL} VAL rows and all six classes")
    return result


def image_key(path):
    parts = str(path).replace("\\", "/").split("/")
    if "fer_data" not in parts:
        raise ValueError("Image is outside audited FER archive layout")
    return "/".join(parts[parts.index("fer_data"):]).casefold()


def image_receipt(root, rows, verify_inventory=False):
    inventory_path = root / "data" / "original_splits" / "image_inventory.csv"
    inventory = None
    if verify_inventory:
        inventory = {image_key(r["path"]): r["sha256"] for r in csv_rows(inventory_path, ["path", "sha256"])}
    ordered, keyed = hashlib.sha256(), hashlib.sha256()
    hashes = []
    for row in rows:
        digest = sha(row["path"])
        if inventory is not None and digest != inventory.get(image_key(row["path"])):
            raise ValueError("Selected image bytes differ from original audited archive")
        hashes.append(digest)
        ordered.update(digest.encode("ascii"))
        if verify_inventory:
            keyed.update(image_key(row["path"]).encode())
            keyed.update(bytes.fromhex(digest))
    result = {"image_content_sha256": ordered.hexdigest(), "image_sha256": hashes}
    if verify_inventory:
        result["legacy_image_receipt"] = {"image_rows_sha256": keyed.hexdigest(), "inventory_sha256": sha(inventory_path)}
    return result


def membership(indices, paths, rows):
    if indices != [int(r["manifest_index"]) for r in rows] or paths != [r["path"] for r in rows]:
        raise ValueError("VAL membership/order differs from the original manifest")


def load_old_baseline(root):
    """Verify the locked old producers and reconstruct only their VAL fusion."""
    folder = root / "refinement_results"
    recipe_path, summary_path = folder / "selected_recipe.json", folder / "summary.json"
    recipe, summary = read_json(recipe_path), read_json(summary_path)
    official_classes(recipe.get("classes")); official_classes(summary.get("classes"))
    if summary.get("status") != "complete" or summary.get("selected") != recipe.get("selected"):
        raise ValueError("Old refinement summary is incomplete or differs from recipe")
    checked_hash(recipe_path, summary.get("recipe_sha256"))
    checked_hash(recipe_path, summary.get("file_sha256", {}).get("selected_recipe.json"))
    if recipe["selected"].get("weights") != {"refined_lora": .45, "cnn": .45, "mlp": .1} or recipe["selected"].get("tta_flip") is not True:
        raise ValueError("Expected frozen .45 refined-LoRA + .45 CNN + .10 old-MLP TTA baseline")
    rows = val_rows(root)
    membership(recipe.get("val_indices"), recipe.get("val_paths"), rows)
    manifest_sha = checked_hash(root / "features" / "manifest.csv", recipe["input_sha256"].get("manifest.csv"))
    if summary.get("source_sha256") != recipe.get("source_sha256"):
        raise ValueError("Old producer source receipts differ")
    pins = dict(recipe["producer_files"])
    for name, digest in recipe["source_sha256"].items():
        add_pin(pins, root / "src" / name, digest)
    add_pin(pins, recipe_path); add_pin(pins, summary_path)
    add_pin(pins, root / "features" / "manifest.csv", manifest_sha)
    verify_pins(root, pins)
    images = image_receipt(root, rows, verify_inventory=True)
    if images["legacy_image_receipt"] != recipe.get("val_image_receipt"):
        raise ValueError("Old validation image fingerprint differs")
    add_pin(pins, root / "data" / "original_splits" / "image_inventory.csv")
    artifacts = {"refined_lora": recipe["artifacts"]["refined_lora"], "cnn": recipe["artifacts"]["cnn"], "old_mlp": recipe["artifacts"]["mlp"]}
    for artifact in artifacts.values():
        add_pin(pins, artifact["checkpoint"], artifact["sha256"])
    refined = Path(artifacts["refined_lora"]["checkpoint"]).parent
    mlp = Path(artifacts["old_mlp"]["checkpoint"]).parent
    refined_receipt = read_json(refined / "summary.json")
    official_classes(refined_receipt.get("classes"))
    if (refined_receipt.get("state") != "complete" or refined_receipt.get("epochs_completed") != 4 or
            refined_receipt.get("val_indices") != [int(r["manifest_index"]) for r in rows] or
            refined_receipt.get("provenance", {}).get("manifest_sha256") != manifest_sha or
            refined_receipt.get("test_labels_used") is not False or refined_receipt.get("external_answers_used") is not False):
        raise ValueError("Refined LoRA receipt has different training/validation provenance")
    old_mlp_receipt = read_json(mlp / "summary.json")
    official_classes(old_mlp_receipt.get("classes"))
    if old_mlp_receipt.get("status") != "complete" or old_mlp_receipt.get("val_indices") != [int(r["manifest_index"]) for r in rows]:
        raise ValueError("Old MLP validation membership differs")
    for name in ["config.json", "val_metadata.json", "val_original.npy", "val_flip.npy", "val_predictions.npy"]:
        add_pin(pins, mlp / name, old_mlp_receipt["artifact_sha256"].get(name))
    mlp_config, mlp_meta = read_json(mlp / "config.json"), read_json(mlp / "val_metadata.json")
    official_classes(mlp_meta.get("classes")); official_classes(mlp_config.get("classes"))
    membership(mlp_meta.get("manifest_indices"), mlp_meta.get("paths"), rows)
    if mlp_config.get("input_sha256", {}).get("manifest.csv") != manifest_sha or mlp_meta.get("manifest_sha256") != manifest_sha:
        raise ValueError("Old MLP source manifest differs")
    cnn_receipt_path = root / "experiments" / "cnn_reference" / "summary.json"
    cnn = read_json(cnn_receipt_path)
    official_classes(cnn.get("classes"))
    if cnn.get("val_indices") != [int(r["manifest_index"]) for r in rows] or cnn.get("manifest_sha256") != manifest_sha or cnn.get("checkpoint_sha256") != artifacts["cnn"]["sha256"]:
        raise ValueError("CNN checkpoint/manifest/VAL membership differs")
    if cnn.get("probability_column_permutation") != [cnn["checkpoint_classes"].index(name) for name in CLASSES]:
        raise ValueError("CNN probability-column mapping differs")
    flip_path = root / "experiments" / "refinement_tta_cache" / "cnn_flip.npy"
    flip_receipt = read_json(flip_path.with_suffix(".json"))
    flip_signature = flip_receipt["signature"]
    legacy_signature = hashlib.sha256(json.dumps(flip_signature, sort_keys=True).encode()).hexdigest()
    if flip_receipt.get("signature_sha256") != legacy_signature:
        raise ValueError("CNN flip cache signature mismatch")
    official_classes(flip_signature.get("classes"))
    membership(flip_signature.get("manifest_indices"), flip_signature.get("paths"), rows)
    if flip_signature.get("image_receipt") != images["legacy_image_receipt"] or flip_signature.get("source_sha256") != recipe["source_sha256"] or flip_signature.get("input_sha256", {}).get("manifest.csv") != manifest_sha:
        raise ValueError("CNN flip source/data/image provenance mismatch")
    add_pin(pins, flip_path, flip_receipt["probability_sha256"])
    cached = {"refined_lora": (refined / "val_original.npy", refined / "val_flip.npy"),
              "cnn": (root / "experiments" / "cnn_reference" / "val_probabilities.npy", flip_path),
              "old_mlp": (mlp / "val_predictions.npy", mlp / "val_flip.npy")}
    probabilities = {}
    for name, files in cached.items():
        for path in files:
            if str(path) not in pins:
                raise ValueError(f"Validation probability is not pinned by old recipe: {path}")
            checked_hash(path, pins[str(path)])
        probabilities[name] = tuple(check_probability(np.load(p, allow_pickle=False), len(rows)) for p in files)
    if not np.array_equal(np.load(mlp / "val_original.npy", allow_pickle=False), np.load(mlp / "val_predictions.npy", allow_pickle=False)):
        raise ValueError("Old MLP original probability aliases differ")
    targets = np.array([int(r["target"]) for r in rows], dtype=np.int64)
    baseline_p = blend(probabilities, CANDIDATES[0]["weights"])
    baseline_metrics = metrics(targets, baseline_p)
    for name, expected in {**BASELINE_EXPECTED, "log_loss": recipe["selected"]["log_loss"]}.items():
        if not np.isclose(baseline_metrics[name], expected, atol=1e-9, rtol=0):
            raise ValueError(f"Frozen baseline reconstruction differs: {name}")
    return rows, targets, images, probabilities, artifacts, pins, recipe, baseline_metrics


def module(root, name):
    source_dir = str(root / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    imported = importlib.import_module(name)
    if Path(imported.__file__).resolve() != (root / "src" / f"{name}.py").resolve():
        raise ValueError("Imported model helper belongs to another project")
    return imported


def check_rows(actual, expected, fields):
    if len(actual) != len(expected) or any(any(str(a.get(key)) != str(b.get(key)) for key in fields) for a, b in zip(actual, expected)):
        raise ValueError("Feature row membership/order/class mapping mismatch")


def fusion_features(root, partition, rows, images, recipe_sha=None):
    folder = root / "features_multilayer_fusion" / partition
    metadata_path = folder / "metadata.json"
    metadata = read_json(metadata_path)
    official_classes(metadata.get("classes"))
    extractor = module(root, "extract_fusion_features")
    _, contract = extractor.verify_training_contract(root, 16)  # CPU provenance only; never loads the backbone.
    if metadata.get("format") != extractor.FORMAT:
        raise ValueError("Unsupported fusion feature receipt format")
    for key, expected in contract.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Fusion feature readout/model/runtime contract differs: {key}")
    settings = {key: value for key, value in metadata.items() if key not in ("signature", "complete", "file_sha256", "progress")}
    if extractor.signature(settings) != metadata.get("signature"):
        raise ValueError("Fusion feature metadata signature differs")
    if set(metadata.get("file_sha256", {})) != {"rows.csv", "original.npy", "flip.npy"}:
        raise ValueError("Fusion feature checksum receipt is incomplete")
    if metadata.get("partition") != partition or metadata.get("complete") is not True or metadata.get("feature_dim") != 3840 or metadata.get("num_rows") != len(rows):
        raise ValueError("Fusion features are incomplete or have different partition/dimensions")
    if metadata.get("image_content_sha256") != images["image_content_sha256"]:
        raise ValueError("Fusion feature image-content fingerprint differs")
    pins = {}
    add_pin(pins, metadata_path)
    for name in ["rows.csv", "original.npy", "flip.npy"]:
        add_pin(pins, folder / name, metadata.get("file_sha256", {}).get(name))
    if metadata.get("rows_sha256") != sha(folder / "rows.csv"):
        raise ValueError("Fusion feature row-file signature differs")
    for name, field in [("extract_fusion_features.py", "extractor_sha256"), ("extract_multilayer_features.py", "base_extractor_sha256")]:
        add_pin(pins, root / "src" / name, metadata.get(field))
    add_pin(pins, root / "features_multilayer_train" / "metadata.json", metadata.get("train_metadata_sha256"))
    source_field = "source_manifest_sha256" if partition == "val" else "unlabeled_csv_sha256"
    source_csv = root / "features" / ("manifest.csv" if partition == "val" else "unlabeled.csv")
    add_pin(pins, source_csv, metadata.get(source_field))
    if partition == "unlabeled":
        if recipe_sha is None or metadata.get("recipe_sha256") != recipe_sha:
            raise ValueError("Unlabeled features were not extracted for this exact locked recipe")
        add_pin(pins, root / "data" / "raw" / "submission.csv", metadata.get("template_sha256"))
    fields = ["manifest_index", "path", "label", "target"] if partition == "val" else ["inference_index", "ID", "path"]
    cached_rows = csv_rows(folder / "rows.csv", fields + ["image_sha256"])
    check_rows(cached_rows, rows, fields)
    if [r["image_sha256"] for r in cached_rows] != images["image_sha256"]:
        raise ValueError("Fusion cache per-image fingerprints/order differ")
    arrays = {}
    for view in ["original", "flip"]:
        array = np.load(folder / f"{view}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(rows), 3840) or array.dtype != np.float32:
            raise ValueError("Fusion feature shape/dtype differs")
        for start in range(0, len(rows), 1024):
            if not np.isfinite(array[start:start + 1024]).all():
                raise ValueError("Fusion features contain NaN/Inf")
        arrays[view] = array
    return arrays, metadata, pins


def predict_features(model, arrays, batch_size):
    import torch
    device = next(model.parameters()).device
    model.eval()
    result = []
    with torch.inference_mode():
        for view in ["original", "flip"]:
            parts = []
            for start in range(0, len(arrays[view]), batch_size):
                x = torch.from_numpy(np.array(arrays[view][start:start + batch_size], dtype=np.float32, copy=True)).to(device)
                parts.append(model(x).float().softmax(1).cpu().numpy())
            result.append(check_probability(np.concatenate(parts), len(arrays[view])))
    return tuple(result)


def cached_prediction(folder, binding, producer, lock_check=None):
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "metadata.json"
    digest = signature(binding)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("binding") != binding or receipt.get("signature") != digest:
            raise ValueError("Prediction cache belongs to a different recipe/model/input/source")
        if receipt.get("complete") is True:
            for name in ["original.npy", "flip.npy"]:
                checked_hash(folder / name, receipt.get("file_sha256", {}).get(name))
            if lock_check:
                lock_check()
            return tuple(check_probability(np.load(folder / f"{v}.npy", allow_pickle=False), binding["rows"]) for v in ["original", "flip"])
    elif any(folder.iterdir()):
        raise ValueError("Orphan prediction cache without binding receipt")
    write_json(receipt_path, {"binding": binding, "signature": digest, "complete": False})
    if lock_check:
        lock_check()
    values = producer()
    if lock_check:
        lock_check()
    for view, value in zip(["original", "flip"], values):
        p = check_probability(value, binding["rows"])
        temporary = folder / f"{view}.npy.tmp"
        with temporary.open("wb") as f:
            np.save(f, p, allow_pickle=False)
        temporary.replace(folder / f"{view}.npy")
    write_json(receipt_path, {"binding": binding, "signature": digest, "complete": True,
                             "file_sha256": {f"{v}.npy": sha(folder / f"{v}.npy") for v in ["original", "flip"]}})
    return tuple(check_probability(np.load(folder / f"{v}.npy", allow_pickle=False), binding["rows"]) for v in ["original", "flip"])


def validate_locked(root):
    path = root / "fusion_validation_results" / "selected_recipe.json"
    recipe = read_json(path)
    official_classes(recipe.get("classes"))
    if recipe.get("state") != "locked" or recipe.get("candidate_configurations") != CANDIDATES:
        raise ValueError("Fusion recipe is not locked or candidate protocol changed")
    name = recipe.get("selected", {}).get("name")
    definition = next((item for item in CANDIDATES if item["name"] == name), None)
    if definition is None or recipe["selected"].get("weights") != definition["weights"] or recipe.get("requires_new_predictions") != (name != "baseline"):
        raise ValueError("Locked selection/weights/gating disagree")
    if recipe.get("validator_sha256") != sha(__file__):
        raise ValueError("Locked validator source changed")
    verify_pins(root, recipe["producer_files"])
    rows = val_rows(root)
    if signature(rows) != recipe.get("val_inputs", {}).get("row_order_sha256"):
        raise ValueError("Locked validation membership changed")
    return path, recipe, sha(path)


def select(root, device="cuda", batch_size=16):
    output = root / "fusion_validation_results"
    recipe_path = output / "selected_recipe.json"
    if recipe_path.exists():
        _, recipe, _ = validate_locked(root)
        images = image_receipt(root, val_rows(root), verify_inventory=True)
        if images["image_content_sha256"] != recipe["val_inputs"]["image_content_sha256"]:
            raise ValueError("Locked validation image bytes changed")
        print("Locked selection verified; no prediction or re-selection repeated.", flush=True)
        return recipe
    rows, targets, images, probabilities, artifacts, pins, old_recipe, baseline = load_old_baseline(root)
    arrays, metadata, feature_pins = fusion_features(root, "val", rows, images)
    pins.update(feature_pins)
    loader = module(root, "train_fusion_head")
    model, head_receipt = loader.load_trained_head(root, device)
    official_classes(head_receipt.get("classes"))
    if head_receipt.get("state") != "complete" or head_receipt.get("epochs_completed") != 12 or head_receipt.get("epoch_selection") is not False or head_receipt.get("validation_used") is not False:
        raise ValueError("New head did not finish its fixed train-only twelve epochs")
    head_folder = root / "experiments" / "cosine_full_train"
    add_pin(pins, head_folder / "summary.json")
    add_pin(pins, head_folder / "model.pt", head_receipt["model_sha256"])
    for name, digest in head_receipt["source_sha256"].items():
        add_pin(pins, root / "src" / name, digest)
    for path, digest in head_receipt["data_receipt"]["data_file_sha256"].items():
        add_pin(pins, path, digest)
    if head_receipt["data_receipt"]["feature_metadata"].get("manifest_sha256") != old_recipe["input_sha256"]["manifest.csv"]:
        raise ValueError("New head training source manifest differs from baseline")
    add_pin(pins, root / "src" / "validate_cosine_fusion.py", sha(__file__))
    output.mkdir(parents=True, exist_ok=True)
    binding = {"partition": "val", "rows": len(rows), "classes": CLASSES, "model_sha256": head_receipt["model_sha256"],
               "row_order_sha256": signature(rows), "image_content_sha256": images["image_content_sha256"],
               "producer_files": pins, "candidate_configurations": CANDIDATES, "tta": "mean original/flip probabilities"}
    probabilities["new_mlp"] = cached_prediction(output / "prediction_cache" / "new_mlp_val", binding,
                                                  lambda: predict_features(model, arrays, max(batch_size, 512)))
    del model, arrays; gc.collect()
    evaluated, selected, fused = compare_candidates(targets, probabilities)
    if any(not np.isclose(evaluated[0][key], baseline[key], atol=1e-12, rtol=0) for key in ["accuracy", "macro_f1", "log_loss"]):
        raise ValueError("Baseline changed during new candidate comparison")
    recipe = {"format": "cosine_fusion_validation_v1", "state": "locked", "classes": CLASSES,
              "validator_sha256": sha(__file__), "candidate_configurations": CANDIDATES,
              "candidate_count": 4, "selection_rule": "Strict accuracy improvement required; accuracy ties retain baseline. New-candidate ties rank macro-F1 then lower log-loss, then fixed candidate order.",
              "selected": selected, "validation_candidates": evaluated,
              "requires_new_predictions": selected["name"] != "baseline", "producer_files": pins,
              "artifacts": {**artifacts, "new_mlp": {"checkpoint": str(head_folder / "model.pt"), "sha256": head_receipt["model_sha256"]}},
              "legacy_input_sha256": old_recipe["input_sha256"],
              "val_inputs": {"rows": len(rows), "row_order_sha256": signature(rows), "image_content_sha256": images["image_content_sha256"],
                             "manifest_indices": [int(r["manifest_index"]) for r in rows], "new_feature_metadata_sha256": sha(root / "features_multilayer_fusion" / "val" / "metadata.json")},
              "baseline_reconstructed": baseline, "new_head_training_summary_sha256": sha(head_folder / "summary.json"),
              "validation_used_to_train_new_head": False, "epoch_selection": False, "temperatures_fitted": False,
              "internal_test_read": False, "external_answers_used": False,
              "validation_caveat": "This is reused selection validation with four prespecified fusions, not an independent generalization estimate."}
    verify_pins(root, pins)
    write_json(recipe_path, recipe)
    with (output / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "accuracy", "macro_f1", "log_loss", "correct_count"])
        writer.writeheader(); writer.writerows({key: item[key] for key in writer.fieldnames} for item in evaluated)
    np.save(output / "selected_val_probabilities.npy", fused[selected["name"]], allow_pickle=False)
    print(json.dumps({"stage": "select", "state": "locked", "selected": selected["name"], "accuracy": selected["accuracy"], "requires_new_predictions": recipe["requires_new_predictions"]}), flush=True)
    return recipe


def unlabeled_rows(root):
    raw = csv_rows(root / "features" / "unlabeled.csv", ["ID", "path"])
    template = csv_rows(root / "data" / "raw" / "submission.csv", ["ID"])
    if len(raw) != EXPECTED_UNLABELED or [r["ID"] for r in raw] != [r["ID"] for r in template]:
        raise ValueError("Competition ID count/order differs from original template")
    if len({r["ID"] for r in raw}) != len(raw) or any(not r["ID"] or not r["path"] for r in raw):
        raise ValueError("Invalid competition ID/path")
    return [{"inference_index": str(i), "ID": row["ID"], "path": row["path"]} for i, row in enumerate(raw)]


def submission_check(path, rows, probability):
    written = csv_rows(path, ["ID", "Emotion"])
    if [r["ID"] for r in written] != [r["ID"] for r in rows] or any(r["Emotion"] not in {str(i) for i in range(6)} for r in written):
        raise ValueError("Export ID order or emotion code differs")
    if [int(r["Emotion"]) for r in written] != check_probability(probability, len(rows)).argmax(1).tolist():
        raise ValueError("Submission codes differ from saved probability argmax")


def old_unlabeled_features(root, recipe, count):
    checked_hash(root / "features" / "metadata.json", recipe["legacy_input_sha256"]["metadata.json"])
    official_classes(read_json(root / "features" / "metadata.json").get("classes"))
    arrays = {}
    for view in ["original", "flip"]:
        name = f"unlabeled_{view}.npy"
        checked_hash(root / "features" / name, recipe["legacy_input_sha256"][name])
        value = np.load(root / "features" / name, mmap_mode="r", allow_pickle=False)
        if value.shape != (count, 1536) or value.dtype != np.float32:
            raise ValueError("Old unlabeled MLP feature shape/dtype differs")
        arrays[view] = value
    return arrays


def export(root, device="cuda", batch_size=16):
    recipe_path, recipe, recipe_sha = validate_locked(root)
    rows = unlabeled_rows(root)
    checked_hash(root / "features" / "unlabeled.csv", recipe["legacy_input_sha256"]["unlabeled.csv"])
    output = root / "fusion_results"
    summary_path = output / "summary.json"

    def lock_check():
        checked_hash(recipe_path, recipe_sha)

    if summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("state") != "complete" or previous.get("recipe_sha256") != recipe_sha or previous.get("validator_sha256") != sha(__file__):
            raise ValueError("Completed export has different recipe/source or is incomplete")
        for name in ["submission.csv", "probabilities.npy"]:
            checked_hash(output / name, previous.get("file_sha256", {}).get(name))
        p = check_probability(np.load(output / "probabilities.npy", allow_pickle=False), len(rows))
        submission_check(output / "submission.csv", rows, p)
        print("Completed locked export verified; no inference repeated.", flush=True)
        return previous
    output.mkdir(parents=True, exist_ok=True)
    started_path = output / "export_started.json"
    start_binding = {"recipe_sha256": recipe_sha, "validator_sha256": sha(__file__), "row_order_sha256": signature(rows)}
    if started_path.exists() and read_json(started_path) != start_binding:
        raise ValueError("Partial export belongs to a different locked recipe/source/ID order")
    if not started_path.exists() and any(output.iterdir()):
        raise ValueError("Nonempty export folder lacks a binding receipt")
    write_json(started_path, start_binding)
    cache_receipts = {}
    if not recipe["requires_new_predictions"]:
        old = root / "refinement_results"
        old_summary = read_json(old / "summary.json")
        checked_hash(old / "submission.csv", old_summary["file_sha256"]["submission.csv"])
        checked_hash(old / "submission_probabilities.npy", old_summary["file_sha256"]["submission_probabilities.npy"])
        probability = check_probability(np.load(old / "submission_probabilities.npy", allow_pickle=False), len(rows))
        submission_check(old / "submission.csv", rows, probability)
        lock_check()
        shutil.copyfile(old / "submission.csv", output / "submission.csv")
        shutil.copyfile(old / "submission_probabilities.npy", output / "probabilities.npy")
        mode, new_inference = "verified_copy_of_unchanged_baseline", False
    else:
        images = image_receipt(root, rows)
        probabilities = {}
        for name, weight in recipe["selected"]["weights"].items():
            if weight == 0:
                continue
            artifact = recipe["artifacts"][name]
            checked_hash(artifact["checkpoint"], artifact["sha256"])
            feature_pins = {}
            if name == "new_mlp":
                arrays, _, feature_pins = fusion_features(root, "unlabeled", rows, images, recipe_sha)
            elif name == "old_mlp":
                arrays = old_unlabeled_features(root, recipe, len(rows))
                feature_pins = {str(root / "features" / f"unlabeled_{view}.npy"): recipe["legacy_input_sha256"][f"unlabeled_{view}.npy"] for view in ["original", "flip"]}
            binding = {"partition": "unlabeled", "rows": len(rows), "classes": CLASSES, "recipe_sha256": recipe_sha,
                       "model": name, "checkpoint_sha256": artifact["sha256"], "validator_sha256": sha(__file__),
                       "row_order_sha256": signature(rows), "image_content_sha256": images["image_content_sha256"],
                       "feature_files": feature_pins, "source_producer_files_sha256": signature(recipe["producer_files"]),
                       "tta": "mean original/flip probabilities; locked original preprocessing"}

            def producer(name=name, artifact=artifact, arrays=arrays if name in ["new_mlp", "old_mlp"] else None):
                if name in ["cnn", "refined_lora"]:
                    return module(root, "finalize_refinement").predict_image_views(root, name, Path(artifact["checkpoint"]), rows,
                                                                                  original=True, flip=True, device=device, batch_size=batch_size)
                if name == "new_mlp":
                    model, receipt = module(root, "train_fusion_head").load_trained_head(root, device)
                    if receipt["model_sha256"] != artifact["sha256"]:
                        raise ValueError("New head loader checkpoint differs from locked recipe")
                else:
                    model, _ = module(root, "train_mlp").load_mlp_checkpoint(artifact["checkpoint"], device)
                try:
                    return predict_features(model, arrays, max(batch_size, 512))
                finally:
                    del model; gc.collect()

            cache_dir = output / "prediction_cache" / name
            probabilities[name] = cached_prediction(cache_dir, binding, producer, lock_check)
            cache_receipts[name] = {"metadata_sha256": sha(cache_dir / "metadata.json"), "binding_sha256": signature(binding)}
        probability = blend(probabilities, recipe["selected"]["weights"])
        lock_check()
        np.save(output / "probabilities.npy", probability, allow_pickle=False)
        with (output / "submission.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["ID", "Emotion"])
            writer.writerows((row["ID"], int(target)) for row, target in zip(rows, probability.argmax(1)))
        mode, new_inference = "locked_new_fusion", True
    lock_check()
    submission_check(output / "submission.csv", rows, probability)
    summary = {"format": "cosine_fusion_export_v1", "state": "complete", "classes": CLASSES,
               "recipe_sha256": recipe_sha, "validator_sha256": sha(__file__), "selected": recipe["selected"],
               "submission_rows": len(rows), "row_order_sha256": signature(rows), "export_mode": mode,
               "new_competition_inference": new_inference, "improvement_on_reused_validation": recipe["requires_new_predictions"],
               "prediction_cache_receipts": cache_receipts, "internal_test_read": False, "external_answers_used": False,
               "file_sha256": {name: sha(output / name) for name in ["submission.csv", "probabilities.npy"]}}
    write_json(summary_path, summary)
    print(json.dumps({"stage": "export", "state": "complete", "mode": mode, "rows": len(rows)}), flush=True)
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
    (select if args.stage == "select" else export)(root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
