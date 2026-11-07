"""Reuse frozen OOF difficulty groups with exact weights1/.15/0, without class normalization."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import prepare_ambiguity_weights as original

ROOT = Path(__file__).resolve().parents[1]
CLASSES = original.CLASSES
WEIGHTS = {"normal": 1., "uncertain": .15, "very_uncertain": 0.}
COUNTS = {"normal": 17139, "uncertain": 3811, "very_uncertain": 298}
POLICY = {"version": 5, "raw_weights": WEIGHTS,
          "grouping": "unchanged verified train-only grouped OOF V1 categories",
          "normalization": "none; exact weights1/.15/0; no per-class normalization",
          "sampling": "uniform identical full training shuffle; no sample removal or resampling",
          "loss": "sum(w_i*CE_i)/sum(w_i) over one effective32-image group; zero groups skip all updates",
          "reported_share": "weight sum fraction, not measured loss fraction",
          "labels_changed": False, "groups_changed": False, "deleted_images": 0,
          "validation_weight": 1.}


def check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or original.sha(path) != expected:
        raise ValueError(f"Frozen difficulty source/artifact changed: {path}")


def _write_csv(path, entries, fields):
    temporary = Path(str(path)+".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    os.replace(temporary, path)


def verify_existing(root, summary):
    root = Path(root).resolve()
    folder = root/"experiments/lora_zero_weight_v5/weights"
    if summary.get("state") != "complete" or summary.get("classes") != CLASSES or summary.get("counts") != COUNTS:
        raise ValueError("Zero-weight summary state/classes/group counts differ")
    if summary.get("binding", {}).get("policy") != POLICY or summary.get("train_rows") != 21248:
        raise ValueError("Zero-weight policy/training count differs")
    for path, digest in summary["binding"]["source_sha256"].items():
        check_hash(path, digest)
    for relative, digest in summary["file_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Weight artifact escapes its dedicated directory")
        check_hash(folder/relative, digest)
    entries = original.csv_rows(folder/"train_weights.csv")
    values = np.load(folder/"training_weights.npy", allow_pickle=False)
    expected = np.asarray([WEIGHTS[row["difficulty"]] for row in entries], dtype=np.float64)
    if values.dtype != np.float64 or values.shape != (21248,) or not np.array_equal(values, expected):
        raise ValueError("Exact1/.15/0 weights differ from the saved float64 array")
    if not np.array_equal(values, np.asarray([float(row["training_weight"]) for row in entries])):
        raise ValueError("Weight CSV and array differ")
    if {name: sum(row["difficulty"] == name for row in entries) for name in COUNTS} != COUNTS:
        raise ValueError("Saved difficulty membership counts differ")
    return summary


def build(root):
    root = Path(root).resolve()
    v1_folder, v2_folder = (root/f"experiments/hse_ambiguity_v{version}/weights" for version in (1, 2))
    previous = [original.read(folder/"summary.json") for folder in (v1_folder, v2_folder)]
    pins = {}
    for folder, summary in zip((v1_folder, v2_folder), previous):
        if summary.get("state") != "complete" or summary.get("classes") != CLASSES or summary.get("train_rows") != 21248:
            raise ValueError("Original difficulty experiment is incomplete or has different classes/members")
        if summary.get("counts") != COUNTS or summary.get("internal_test_read") is not False or summary.get("external_answers_used") is not False:
            raise ValueError("Original groups/counts/train-only provenance differ")
        for path, digest in summary["binding"]["source_sha256"].items():
            if path in pins and pins[path] != digest:
                raise ValueError("The two previous source chains disagree")
            pins[path] = digest
        pins[str(folder/"summary.json")] = original.sha(folder/"summary.json")
        for name, digest in summary["file_sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Previous weight artifact escapes its directory")
            pins[str(folder/name)] = digest
    pins[str(Path(__file__).resolve())] = original.sha(__file__)
    for path, digest in pins.items():
        check_hash(path, digest)
    manifest = root/"features/manifest.csv"
    audit = original.read(root/"data/split_audit.json")
    check_hash(manifest, audit["manifest_sha256"])
    if audit.get("classes") != CLASSES:
        raise ValueError("Audited official class order differs")
    training = []
    for index, record in enumerate(original.csv_rows(manifest)):
        if record.get("split") != "train":
            continue  # Do not parse labels of validation/internal-test rows.
        if record.get("label") not in CLASSES:
            raise ValueError("Unexpected training label")
        training.append({"manifest_index": index, "path": record["path"], "label": record["label"],
                         "target": CLASSES.index(record["label"])})
    membership = original.signature(training)
    if len(training) != 21248 or any(summary["binding"]["train_membership_sha256"] != membership for summary in previous):
        raise ValueError("Frozen difficulty training membership differs from the original manifest")
    entries = original.csv_rows(v1_folder/"train_weights.csv")
    old_v2 = original.csv_rows(v2_folder/"train_weights.csv")
    v1_array = np.load(v1_folder/"training_weights.npy", allow_pickle=False)
    v2_array = np.load(v2_folder/"training_weights.npy", allow_pickle=False)
    if any(len(value) != 21248 for value in (entries, old_v2, v1_array, v2_array)):
        raise ValueError("Old difficulty CSV/array count differs")
    if not np.array_equal(v1_array, np.asarray([float(row["training_weight"]) for row in entries])) or not np.array_equal(v2_array, np.asarray([float(row["training_weight"]) for row in old_v2])):
        raise ValueError("Previous weight CSV and probability-independent weight array disagree")
    targets = np.asarray([row["target"] for row in training], dtype=np.int64)
    buckets = np.asarray([row["difficulty"] for row in entries])
    if {name: int(np.sum(buckets == name)) for name in COUNTS} != COUNTS:
        raise ValueError("Difficulty membership changed")
    for row, other, member in zip(entries, old_v2, training):
        for key in ("manifest_index", "target"):
            if int(row[key]) != member[key] or int(other[key]) != member[key]:
                raise ValueError("Weight row train membership/target order differs")
        if row["path"] != member["path"] or row["label"] != member["label"]:
            raise ValueError("Weight row training path/label differs")
        stable_fields = set(row)-{"raw_weight", "training_weight"}
        if any(other.get(key) != row[key] for key in stable_fields):
            raise ValueError("V1 and V2 grouping or unchanged OOF evidence differs")
    # Validate both historical class normalization formulas without refitting groups.
    for summary, array, raw_values in [(previous[0], v1_array, {"normal": 1., "uncertain": .75, "very_uncertain": .5}),
                                       (previous[1], v2_array, {"normal": 1., "uncertain": .5, "very_uncertain": .2})]:
        raw = np.asarray([raw_values[bucket] for bucket in buckets])
        for target in range(6):
            selected = targets == target
            if not np.array_equal(array[selected], raw[selected]/raw[selected].mean()):
                raise ValueError("Historical class-normalized weights no longer reproduce their arrays")
    values = np.asarray([WEIGHTS[bucket] for bucket in buckets], dtype=np.float64)
    hard = buckets != "normal"
    old_share, new_share = float(v2_array[hard].sum()/v2_array.sum()), float(values[hard].sum()/values.sum())
    if not new_share < old_share or not np.all(values[hard] < v2_array[hard]):
        raise ValueError("Hard-image weight share/contribution was not strictly reduced")
    binding = {"policy": POLICY, "classes": CLASSES, "train_membership_sha256": membership,
               "source_sha256": pins}
    output = root/"experiments/lora_zero_weight_v5/weights"
    output.mkdir(parents=True, exist_ok=True)
    if (output/"summary.json").exists():
        existing = original.read(output/"summary.json")
        if existing.get("binding") != binding:
            raise ValueError("Existing zero-weight experiment has different frozen sources/policy")
        return verify_existing(root, existing)
    updated = []
    for index, row in enumerate(entries):
        updated.append({**row, "previous_v1_training_weight": float(v1_array[index]),
                        "previous_v2_training_weight": float(v2_array[index]),
                        "raw_weight": float(values[index]), "training_weight": float(values[index])})
    fields = list(updated[0])
    artifacts = []
    for name, subset in [("train_weights.csv", updated)] + [(bucket+".csv", [row for row in updated if row["difficulty"] == bucket]) for bucket in WEIGHTS]:
        _write_csv(output/name, subset, fields)
        artifacts.append(name)
    temporary = output/"training_weights.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, output/"training_weights.npy")
    artifacts.append("training_weights.npy")
    per_class = {}
    for target, label in enumerate(CLASSES):
        selected = targets == target
        per_class[label] = {"images": int(selected.sum()),
            "counts": {bucket: int(np.sum(selected & (buckets == bucket))) for bucket in WEIGHTS},
            "weight_sum": float(values[selected].sum()), "mean_training_weight": float(values[selected].mean()),
            "previous_v2_weight_sum": float(v2_array[selected].sum()),
            "hard_weight_share": float(values[selected & hard].sum()/values[selected].sum())}
    shares = {bucket: {"images": COUNTS[bucket], "weight": WEIGHTS[bucket],
              "physical_image_share": COUNTS[bucket]/21248,
              "previous_v2_weight_share": float(v2_array[buckets == bucket].sum()/v2_array.sum()),
              "new_weight_share": float(values[buckets == bucket].sum()/values.sum()),
              "weight_sum": float(values[buckets == bucket].sum())} for bucket in WEIGHTS}
    summary = {"format": "zero_weights_v5", "state": "complete", "classes": CLASSES,
        "binding": binding, "train_rows": 21248, "counts": COUNTS, "per_class": per_class,
        "group_weight_shares": shares, "previous_v2_uncertain_weight_share": old_share,
        "uncertain_weight_share": new_share, "training_weight_sum": float(values.sum()),
        "training_weight_mean": float(values.mean()), "min_training_weight": 0., "max_training_weight": 1.,
        "class_normalization": False, "groups_changed": False, "labels_changed": False,
        "validation_weights_used": False, "internal_test_read": False, "external_answers_used": False,
        "review_images_copied": 0, "review_images_directory": str(root/"experiments/hse_ambiguity_v1/review_images"),
        "uncertainty_caveat": "Frozen OOF uncertainty is a model proxy, not verified label error or human judgment.",
        "file_sha256": {name: original.sha(output/name) for name in artifacts}}
    original.write_json(output/"summary.json", summary)
    verify_existing(root, summary)
    print(__import__("json").dumps({"stage": "zero_weights_v5_complete", "counts": COUNTS,
          "previous_v2_hard_weight_share": old_share, "new_hard_weight_share": new_share}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.project_root)


if __name__ == "__main__":
    main()
