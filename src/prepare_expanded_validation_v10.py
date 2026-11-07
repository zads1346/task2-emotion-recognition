"""Create a larger grouped development split from original TRAIN + VAL only.

CPU-only dataset preparation. Original internal-test rows are filtered before
their paths/labels are accessed. No original image, label, manifest or experiment
is changed. Near-duplicate grouping reuses the SHA-pinned original pixel-only
implementation; original split and class labels never enter pair decisions.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
import csv
import hashlib
import io
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
import scipy
import sklearn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import group_training_images as grouping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/expanded_validation_v10"
FORMAT = "expanded_validation_v10"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SEED = 20261016
EXCLUDED = [7727, 10448]
EXPECTED_MANIFEST_SHA = "1c6929af352703962c3e382aa8a1186fb75baf4ea2d807f48f9708ced40763a9"
GROUP_SOURCE_SHA = "c5679f1eb9a6f01c91eeb1dd605cd35aef242806dc7b16ef9ac2c29edfaa8b60"
AUDIT = ROOT / "experiments/lora_flip_consistency_v6/audit/summary.json"
FIELDS = ("manifest_index", "path", "label", "target", "split", "group", "group_id", "fold", "original_split", "image_sha256")
PAIR_FIELDS = ("left_manifest_index", "right_manifest_index", "kind", "left_view", "right_view", "phash_hamming",
               "pixel_mae", "pixel_mae_255", "centered_correlation", "exact_pixels_bypass", "union_new_component")
PARAMETERS = {**grouping.PARAMETERS, "fold_method": "StratifiedGroupKFold", "folds": 5,
              "random_state": SEED, "validation_fold": 0, "original_splits_used": ["train", "val"],
              "excluded_manifest_indices": EXCLUDED, "groups_use_labels": False, "groups_use_original_split": False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signature(value):
    """Exact train_hse_specialist.signature JSON ABI, without importing torch."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def training_bindings(original, rows):
    result = {"original_pool_indices": [row["manifest_index"] for row in original],
              "excluded_indices": EXCLUDED}
    for split in ("train", "val"):
        members = [row for row in rows if row["split"] == split]
        result[split + "_indices"] = [row["manifest_index"] for row in members]
        result[split + "_rows_sha256"] = signature([{key: row[key] for key in ("manifest_index", "path", "label", "target")} for row in members])
        result[split + "_image_rows_sha256"] = signature([(row["manifest_index"], row["image_sha256"]) for row in members])
    return result


def pin(path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Source pin outside the fixed project")
    actual = grouping.file_sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Source/input hash changed: {path}")
    return str(path), actual


def source_contract():
    if Path(grouping.__file__).resolve() != ROOT / "src/group_training_images.py":
        raise ValueError("Grouping helper belongs to another project")
    if grouping.CLASSES != CLASSES:
        raise ValueError("Grouping helper class order differs")
    sources = dict([pin(__file__), pin(grouping.__file__, GROUP_SOURCE_SHA),
                    pin(ROOT / "features/manifest.csv", EXPECTED_MANIFEST_SHA), pin(AUDIT)])
    for relative in ("data/split_audit.json", "data/original_splits/image_inventory.csv",
                     "data/original_splits/split_audit_original.json", "data/original_splits/archive_audit.json"):
        sources.update([pin(ROOT / relative)])
    return {"format": FORMAT, "classes": CLASSES, "parameters": PARAMETERS, "source_pins": sources,
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pillow": PIL.__version__}}


def records_and_inventory():
    original = []
    with (ROOT / "features/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split", "subject_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Original manifest schema changed")
        for index, row in enumerate(reader):
            # Do not access paths, labels or image files of any other split.
            if row.get("split") not in ("train", "val"):
                continue
            path, label = row["path"], row["label"]
            if label not in CLASSES or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(ROOT / "data/raw"):
                raise ValueError("Invalid allowed original row")
            original.append({"manifest_index": index, "path": path, "label": label, "target": CLASSES.index(label),
                             "original_split": row["split"], "subject_id": row.get("subject_id", "").strip()})
    if Counter(row["original_split"] for row in original) != {"train": 21248, "val": 2656}:
        raise ValueError("Original TRAIN/VAL membership counts changed")
    if len({row["path"].casefold() for row in original}) != 23904:
        raise ValueError("Repeated original TRAIN/VAL path")
    audit = read_json(AUDIT)
    if (audit.get("state") != "complete" or audit.get("classes") != CLASSES
            or audit.get("excluded_train_indices") != EXCLUDED
            or audit.get("original_train_indices") != [row["manifest_index"] for row in original if row["original_split"] == "train"]
            or audit.get("val_indices") != [row["manifest_index"] for row in original if row["original_split"] == "val"]):
        raise ValueError("Previously reviewed two-image exclusions or source members differ")
    effective = [row for row in original if row["manifest_index"] not in EXCLUDED]
    if len(effective) != 23902 or any(row["original_split"] != "train" for row in original if row["manifest_index"] in EXCLUDED):
        raise ValueError("Only the two previously reviewed TRAIN images may be excluded")
    expected = {image_key(row["path"]) for row in effective}
    inventory = {}
    with (ROOT / "data/original_splits/image_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            # Inventory paths are metadata only; never interpret its labels.
            key = image_key(row["path"])
            if key not in expected:
                continue
            if key in inventory:
                raise ValueError("Repeated allowed inventory path")
            inventory[key] = row["sha256"]
    if set(inventory) != expected:
        raise ValueError("Original inventory does not cover the allowed image members")
    return original, effective, inventory


def image_key(path):
    parts = str(path).replace("\\", "/").split("/")
    if "fer_data" not in parts:
        raise ValueError("Image metadata lacks the established fer_data prefix")
    return "/".join(parts[parts.index("fer_data"):]).casefold()


def content(records, inventory, decode, progress):
    digest = hashlib.sha256()
    pixels = np.empty((len(records), 48, 48), dtype=np.uint8) if decode else None
    hashes, modes = [], Counter()
    for index, row in enumerate(records):
        raw = Path(row["path"]).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != inventory[image_key(row["path"])]:
            raise ValueError("Allowed image differs from original inventory")
        digest.update(actual.encode("ascii")); hashes.append(actual)
        if decode:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.size != (48, 48):
                    raise ValueError("Expected original 48x48 image")
                modes[image.mode] += 1
                pixels[index] = np.asarray(image.convert("L"), dtype=np.uint8)
        progress.log("Allowed TRAIN+VAL content" + (" + decode" if decode else " verification"), index + 1, len(records))
    progress.log("Allowed TRAIN+VAL content complete", len(records), len(records), force=True)
    return digest.hexdigest(), hashes, pixels, dict(sorted(modes.items()))


def make_folds(records, group_ids):
    labels = np.asarray([row["target"] for row in records], dtype=np.int64)
    folds = np.full(len(records), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, val) in enumerate(splitter.split(np.zeros(len(records)), labels, group_ids)):
        if set(group_ids[train]) & set(group_ids[val]) or set(labels[train]) != set(range(6)) or set(labels[val]) != set(range(6)):
            raise ValueError("Five-fold class coverage or group isolation failed")
        folds[val] = fold
    if np.any(folds < 0) or not .19 <= np.mean(folds == 0) <= .21:
        raise ValueError("Fixed fold0 is not approximately20% of effective allowed members")
    return folds


def isolation(rows):
    partitions = {split: [row for row in rows if row["split"] == split] for split in ("train", "val")}
    checks = {}
    for field in ("manifest_index", "path", "image_sha256", "group_id"):
        overlap = set(row[field] for row in partitions["train"]) & set(row[field] for row in partitions["val"])
        checks[field + "_cross_split_overlap"] = len(overlap)
        if overlap:
            raise ValueError("New TRAIN/VAL overlap: " + field)
    group_folds = defaultdict(set)
    for row in rows:
        group_folds[row["group_id"]].add(row["fold"])
    if any(len(value) != 1 for value in group_folds.values()):
        raise ValueError("A detected near-duplicate component crosses folds")
    checks["detected_near_duplicate_groups_cross_any_fold"] = 0
    return checks


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def read_manifest():
    with (OUT / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Expanded manifest schema differs")
        rows = list(reader)
    for row in rows:
        for field in ("manifest_index", "target", "group", "group_id", "fold"):
            row[field] = int(row[field])
    return rows


def verify_evidence_components(rows):
    positions = {row["manifest_index"]: index for index, row in enumerate(rows)}
    union = grouping.UnionFind(len(rows))
    with (OUT / "grouping_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(PAIR_FIELDS):
            raise ValueError("Grouping evidence schema differs")
        for edge in reader:
            left, right = int(edge["left_manifest_index"]), int(edge["right_manifest_index"])
            if left not in positions or right not in positions:
                raise ValueError("Grouping evidence references a disallowed member")
            union.union(positions[left], positions[right])
    components = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    ordered = sorted(components.values(), key=lambda members: min(rows[index]["manifest_index"] for index in members))
    for group, members in enumerate(ordered):
        if any(rows[index]["group_id"] != group for index in members):
            raise ValueError("Saved group does not match accepted-edge connected components")


def verify(recompute_groups=False, verify_images=True):
    contract = source_contract(); summary = read_json(OUT / "summary.json")
    if summary.get("state") != "complete" or any(summary.get(key) != value for key, value in contract.items()):
        raise ValueError("Expanded split source/parameter contract differs")
    if set(summary["file_sha256"]) != {"manifest.csv", "grouping_pairs.csv"}:
        raise ValueError("Expanded split artifact hash set differs")
    for name, digest in summary["file_sha256"].items():
        if name not in ("manifest.csv", "grouping_pairs.csv") or grouping.file_sha(OUT / name) != digest:
            raise ValueError("Expanded split artifact hash differs")
    original, records, inventory = records_and_inventory()
    if verify_images or recompute_groups:
        actual_content, hashes, pixels, _ = content(records, inventory, recompute_groups, grouping.Progress())
    else:
        hashes = [inventory[image_key(row["path"])] for row in records]
        actual_content = hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()
        pixels = None
    if actual_content != summary["allowed_image_content_sha256"]:
        raise ValueError("Allowed image content binding differs")
    rows = read_manifest()
    if len(rows) != len(records):
        raise ValueError("Expanded manifest member count differs")
    for saved, record, digest in zip(rows, records, hashes):
        for field in ("manifest_index", "path", "label", "target", "original_split"):
            if saved[field] != record[field]:
                raise ValueError("Expanded row differs from original allowed member: " + field)
        if saved["image_sha256"] != digest or saved["fold"] not in range(5) or saved["group_id"] < 0 or saved["group"] != saved["group_id"]:
            raise ValueError("Expanded row hash/fold/group differs")
        if saved["split"] != ("val" if saved["fold"] == 0 else "train"):
            raise ValueError("Only fold0 may form validation")
    group_ids = np.asarray([row["group_id"] for row in rows], dtype=np.int64)
    if not np.array_equal(make_folds(records, group_ids), [row["fold"] for row in rows]):
        raise ValueError("Fixed seeded stratified group allocation does not reproduce")
    if isolation(rows) != summary["isolation_checks"] or grouping.canonical_sha(rows) != summary["expanded_membership_sha256"]:
        raise ValueError("Expanded isolation/member receipt differs")
    if grouping.canonical_sha(original) != summary["original_trainval_membership_sha256"]:
        raise ValueError("Original allowed source membership differs")
    if any(summary.get(key) != value for key, value in training_bindings(original, rows).items()):
        raise ValueError("Training API membership/row/image binding differs")
    verify_evidence_components(rows)
    if recompute_groups:
        groups, _, evidence, stats = grouping.label_blind_groups(pixels, [row["manifest_index"] for row in records], grouping.Progress())
        if not np.array_equal(groups, group_ids) or grouping.canonical_sha(evidence) != summary["grouping_evidence_sha256"] or stats != summary["candidate_statistics"]:
            raise ValueError("Label-blind grouping does not reproduce")
    if source_contract() != contract:
        raise ValueError("Source changed during verification")
    print(json.dumps({"stage": "expanded_validation_v10_verified", "split_counts": summary["split_counts"],
                      "grouping_recomputed": recompute_groups, "images_rehashed": bool(verify_images or recompute_groups), "GPU_used": False}), flush=True)
    return summary


def load_records(root=ROOT, verify_images=True, recompute_groups=False):
    """Return full rows in original manifest order, plus the verified split receipt.

    verify_images=False still checks source pins, all output hashes, membership,
    inventory bindings, grouping components and the fixed SGKF assignment. It is
    for repeated checkpoint metadata checks after a full image-byte preflight.
    """
    if Path(root).resolve() != ROOT:
        raise ValueError("Fixed project root required")
    summary = verify(recompute_groups=recompute_groups, verify_images=verify_images)
    rows = read_manifest()
    return {split: [row for row in rows if row["split"] == split] for split in ("train", "val")}, summary


read_records = load_records


def prepare():
    contract = source_contract()
    if OUT.exists() and any(OUT.iterdir()):
        if (OUT / "summary.json").exists():
            return verify()
        raise ValueError("Incomplete nonempty V10 output preserved; do not overwrite it")
    original, records, inventory = records_and_inventory()
    started = time.monotonic(); progress = grouping.Progress()
    content_sha, image_hashes, pixels, modes = content(records, inventory, True, progress)
    group_ids, components, evidence, stats = grouping.label_blind_groups(pixels, [row["manifest_index"] for row in records], progress)
    del pixels
    folds = make_folds(records, group_ids)
    rows = [{"manifest_index": row["manifest_index"], "path": row["path"], "label": row["label"], "target": row["target"],
             "group": int(group), "group_id": int(group), "split": "val" if fold == 0 else "train", "fold": int(fold),
             "original_split": row["original_split"], "image_sha256": digest}
            for row, group, fold, digest in zip(records, group_ids, folds, image_hashes)]
    checks = isolation(rows)
    by_index = {row["manifest_index"]: row for row in records}
    cross_old_edges = [edge for edge in evidence if by_index[edge["left_manifest_index"]]["original_split"] != by_index[edge["right_manifest_index"]]["original_split"]]
    cross_label = [group for group, members in enumerate(components) if len({records[i]["label"] for i in members}) > 1]
    split_counts = dict(Counter(row["split"] for row in rows))
    split_class_counts = {split: {label: sum(row["split"] == split and row["label"] == label for row in rows) for label in CLASSES} for split in ("train", "val")}
    split_content = {split: hashlib.sha256("".join(row["image_sha256"] for row in rows if row["split"] == split).encode("ascii")).hexdigest() for split in ("train", "val")}
    summary = {**contract, **training_bindings(original, rows), "state": "complete", "original_allowed_rows": 23904, "effective_allowed_rows": 23902,
               "excluded_manifest_indices": EXCLUDED, "source_manifest_sha256": EXPECTED_MANIFEST_SHA,
               "original_trainval_membership_sha256": grouping.canonical_sha(original), "expanded_membership_sha256": grouping.canonical_sha(rows),
               "allowed_image_content_sha256": content_sha, "split_image_content_sha256": split_content,
               "image_content_hash_definition": "SHA256 of concatenated lowercase per-image file SHA256 hex in original manifest order",
               "training_api_hash_definition": "train/val_rows_sha256 = h.signature of ordered manifest_index,path,label,target dictionaries; train/val_image_rows_sha256 = h.signature of ordered (manifest_index,image_sha256) pairs; h.signature uses json.dumps(sort_keys=True,ensure_ascii=False) with default separators",
               "source_image_mode_counts": modes, "split_counts": split_counts, "split_class_counts": split_class_counts,
               "new_split_original_source_counts": {split: dict(Counter(row["original_split"] for row in rows if row["split"] == split)) for split in ("train", "val")},
               "split_membership_sha256": {split: grouping.canonical_sha([row for row in rows if row["split"] == split]) for split in ("train", "val")},
               "validation_fraction": split_counts["val"] / len(rows), "validation_fold": 0,
               "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in range(5)},
               "group_count": len(components), "group_size_histogram": dict(sorted(Counter(str(len(m)) for m in components).items(), key=lambda pair: int(pair[0]))),
               "cross_label_group_count": len(cross_label), "cross_label_groups": cross_label,
               "accepted_edges_crossing_original_train_val": len(cross_old_edges), "original_cross_split_edges": cross_old_edges,
               "grouping_evidence_sha256": grouping.canonical_sha(evidence), "candidate_statistics": stats,
               "isolation_checks": checks, "subject_ids_available": all(row["subject_id"] for row in records),
               "subject_independence_established": False, "labels_changed": 0, "original_files_changed": False,
               "old_task_checkpoints_valid_for_independent_evaluation": False,
               "old_task_checkpoint_warning": "New validation includes formerly trained images. Restart every task-trained branch from external initialization; old task checkpoints/predictions cannot be scored as independent on this split.",
               "validation_caveat": "A larger grouped development split is not a new untouched final test. Earlier research used these allowed data; original internal test remains completely excluded. Person/source independence and external pretraining overlap remain unestablished.",
               "limitations": ["Conservative pHash/pixel groups do not identify all related images, subjects or sources.",
                               "Connected components are transitive; every pair in a group need not satisfy the direct edge threshold.",
                               "Five-fold stratification uses labels only after label-blind image grouping; neither labels nor old split affect merging."],
               "GPU_used": False, "internal_test_images_read": False, "internal_test_labels_used": False,
               "competition_images_or_labels_read": False, "external_answers_used": False,
               "elapsed_seconds": time.monotonic() - started}
    if source_contract() != contract:
        raise ValueError("Source changed while preparing the expanded split")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "manifest.csv", rows, FIELDS)
    write_csv(OUT / "grouping_pairs.csv", evidence, PAIR_FIELDS)
    summary["file_sha256"] = {name: grouping.file_sha(OUT / name) for name in ("manifest.csv", "grouping_pairs.csv")}
    grouping.write_json(OUT / "summary.json", summary)
    print(json.dumps({"stage": "expanded_validation_v10_complete", "split_counts": split_counts,
                      "groups": len(components), "original_cross_split_edges": len(cross_old_edges),
                      "summary": str(OUT / "summary.json"), "GPU_used": False}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("prepare", "verify"), default="prepare")
    parser.add_argument("--recompute-groups", action="store_true", help="Verify by redoing the complete CPU-only grouping")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT:
        raise ValueError("Fixed project root required")
    if args.stage == "prepare":
        prepare()
    else:
        verify(args.recompute_groups)


if __name__ == "__main__":
    main()
