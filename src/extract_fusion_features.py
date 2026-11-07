"""Extract audited validation or locked-recipe inference features for fusion.

Uses the unchanged train extractor's official frozen backbone and readout.
No internal-test images, labels, external answers, or old caches are used/edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

import extract_multilayer_features as base

ROOT = Path(__file__).resolve().parents[1]
CLASSES = base.CLASSES
FEATURE_DIM = 3840
FORMAT = "dinov2_multilayer_fusion_v1"
FILES = ["rows.csv", "original.npy", "flip.npy"]
ROW_FIELDS = {
    "val": ["manifest_index", "path", "label", "target", "image_sha256"],
    "unlabeled": ["inference_index", "ID", "path", "image_sha256"],
}
CONTRACT = {
    "model": "dinov2_vitb14", "frozen": True, "trainable_parameters": 0,
    "feature_dim": FEATURE_DIM, "dtype": "float32", "layer_indices": [8, 9, 10, 11],
    "feature_definition": "concat normalized CLS of last four blocks then mean normalized last-block patch tokens",
    "readout_layout": [{"name": f"cls_block_{block}", "start": i * 768, "end": (i + 1) * 768}
                       for i, block in enumerate([8, 9, 10, 11])]
                      + [{"name": "mean_patch_block_11", "start": 3072, "end": 3840}],
    "last_layer_feature_slice": [2304, 3840], "image_size": 224, "rgb": True,
    "preprocessing": "PIL RGB; bicubic resize 224x224; ImageNet mean/std; no crop",
    "flip_definition": "horizontal flip of normalized input tensor; flip(-1)", "amp": "float16",
}
sha256, array_sha256, write_json = base.sha256, base.array_sha256, base.write_json
load_backbone, make_transform, safe_encode = base.load_backbone, base.make_transform, base.safe_encode


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {path}")
        return list(reader)


def image_key(path):
    # Original inventory paths precede the project move; use their archive suffix.
    parts = str(path).replace("\\", "/").split("/")
    if "fer_data" not in parts or any(part in (".", "..") for part in parts):
        raise ValueError(f"Invalid original FER archive path: {path}")
    return "/".join(parts[parts.index("fer_data"):]).casefold()


def check_recipe(root, expected_sha=None):
    path = Path(root) / "fusion_validation_results" / "selected_recipe.json"
    if not path.is_file():
        raise ValueError("Unlabeled extraction requires an existing locked fusion recipe")
    content = path.read_bytes()
    recipe = json.loads(content)
    if recipe.get("state") != "locked" or recipe.get("requires_new_predictions") is not True:
        raise ValueError("Fusion recipe must be locked and require new predictions")
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Locked fusion recipe changed during extraction")
    return digest


def verify_training_contract(root, batch_size=16):
    root = Path(root)
    folder = root / "features_multilayer_train"
    metadata = read_json(folder / "metadata.json")
    if (metadata.get("format") != base.FORMAT or metadata.get("partition") != "train"
            or metadata.get("complete") is not True or metadata.get("classes") != CLASSES):
        raise ValueError("Invalid completed training feature receipt")
    for key, expected in CONTRACT.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Training feature readout/preprocessing mismatch: {key}")
    runtime = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "Pillow": PIL_VERSION}
    for key, expected in {**runtime, "batch_size": batch_size}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Training feature runtime/configuration mismatch: {key}")
    train_settings = {key: value for key, value in metadata.items()
                      if key not in ("signature", "complete", "file_sha256", "progress")}
    if signature(train_settings) != metadata.get("signature"):
        raise ValueError("Training metadata signature mismatch")
    current = base.official_source_info(root)
    for key, value in current.items():
        if metadata.get(key) != value:
            raise ValueError(f"Official model or base extractor differs from training cache: {key}")
    if set(metadata.get("file_sha256", {})) != set(base.FILES):
        raise ValueError("Training cache file receipt is incomplete")
    for name in base.FILES:
        if sha256(folder / name) != metadata["file_sha256"][name]:
            raise ValueError(f"Training cache checksum mismatch: {name}")
    return metadata, {**CONTRACT, **runtime, "batch_size": batch_size,
                      "official_weight_sha256": current["official_weight_sha256"],
                      "repository_commit": current["repository_commit"],
                      "repository_python_sha256": current["repository_python_sha256"],
                      "base_extractor_sha256": current["extractor_sha256"],
                      "extractor_sha256": sha256(__file__),
                      "train_metadata_sha256": sha256(folder / "metadata.json"),
                      "train_cache_signature": metadata["signature"]}


def load_partition_rows(root, partition, train_metadata):
    """Filter first; other manifest partitions' labels and image files are untouched."""
    root = Path(root)
    if partition not in ROW_FIELDS:
        raise ValueError("Only val and unlabeled partitions are supported")
    recipe_sha = check_recipe(root) if partition == "unlabeled" else None
    audit_path = root / "data" / "split_audit.json"
    audit = read_json(audit_path)
    if audit.get("classes") != CLASSES:
        raise ValueError("Split audit has a different official class order")
    info = {"split_audit_sha256": sha256(audit_path)}
    if partition == "val":
        manifest = root / "features" / "manifest.csv"
        manifest_sha = sha256(manifest)
        if manifest_sha != audit.get("manifest_sha256") or manifest_sha != train_metadata.get("manifest_sha256"):
            raise ValueError("Validation manifest differs from audited training manifest")
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("Validation manifest requires path,label,split")
            rows = [{"manifest_index": index, "path": row["path"], "label": row["label"]}
                    for index, row in enumerate(reader) if row["split"].strip().lower() == "val"]
        if len(rows) != audit.get("splits", {}).get("val"):
            raise ValueError("Validation row count differs from audit")
        for row in rows:
            if row["label"] not in CLASSES:
                raise ValueError("Unknown selected validation label")
            row["target"] = CLASSES.index(row["label"])
        info["source_manifest_sha256"] = manifest_sha
    else:
        manifest, template = root / "features" / "unlabeled.csv", root / "data" / "raw" / "submission.csv"
        manifest_sha = sha256(manifest)
        if manifest_sha != audit.get("inference_sha256"):
            raise ValueError("Unlabeled manifest differs from original split audit")
        rows = [{"inference_index": index, "ID": row["ID"], "path": row["path"]}
                for index, row in enumerate(read_rows(manifest, ["ID", "path"]))]
        ids = [row["ID"] for row in rows]
        if len(rows) != audit.get("inference_samples") or len(set(ids)) != len(ids) or any(not value for value in ids):
            raise ValueError("Unlabeled IDs/count are invalid")
        if ids != [row["ID"] for row in read_rows(template, ["ID", "Emotion"])]:
            raise ValueError("Unlabeled ID order differs from submission template")
        info.update(unlabeled_csv_sha256=manifest_sha, template_sha256=sha256(template), recipe_sha256=recipe_sha)
    if not rows:
        raise ValueError("Selected partition is empty")
    inventory_path = root / "data" / "original_splits" / "image_inventory.csv"
    inventory = {}
    # Only path and SHA are accessed; labels in this inventory are never interpreted.
    for source in read_rows(inventory_path, ["path", "sha256"]):
        key, digest = image_key(source["path"]), source["sha256"]
        if key in inventory or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid or duplicate inventory image key/hash")
        inventory[key] = digest
    content_digest, keyed_digest, seen, seen_keys = hashlib.sha256(), hashlib.sha256(), set(), set()
    for row in rows:
        if not Path(row["path"]).is_absolute():
            raise ValueError("Image paths must be absolute")
        canonical, key = os.path.normcase(os.path.normpath(row["path"])), image_key(row["path"])
        if canonical in seen or key in seen_keys:
            raise ValueError("Duplicate selected image path/key")
        seen.add(canonical)
        seen_keys.add(key)
        digest = sha256(row["path"])
        if digest != inventory.get(key):
            raise ValueError(f"Selected image bytes differ from original inventory: {row['path']}")
        row["image_sha256"] = digest
        content_digest.update(digest.encode("ascii"))
        keyed_digest.update(key.encode("utf-8"))
        keyed_digest.update(bytes.fromhex(digest))
    if recipe_sha is not None:
        check_recipe(root, recipe_sha)
    info.update(inventory_sha256=sha256(inventory_path), image_content_sha256=content_digest.hexdigest(),
                image_rows_sha256=keyed_digest.hexdigest(),
                image_content_hash_definition="SHA256 of ordered concatenated lowercase image SHA256 hex strings")
    return rows, info


def rows_bytes(rows, partition):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS[partition], extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class PartitionImages(Dataset):
    def __init__(self, rows):
        self.rows, self.transform = rows, make_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        content = Path(row["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Selected image changed during extraction: {row['path']}")
        with Image.open(io.BytesIO(content)) as source:
            return self.transform(source.convert("RGB"))


def extract_partition_cache(model, rows, output, settings, batch_size, workers, device, root):
    partition = settings["partition"]
    if partition not in ROW_FIELDS or len(rows) != settings["num_rows"] or settings["feature_dim"] != FEATURE_DIM:
        raise ValueError("Invalid extraction partition/configuration")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature model must be fully frozen in evaluation mode")
    def gate():
        if partition == "unlabeled":
            check_recipe(root, settings["recipe_sha256"])
    gate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_rows = rows_bytes(rows, partition)
    settings = {**settings, "rows_sha256": hashlib.sha256(expected_rows).hexdigest()}
    expected_signature = signature(settings)
    row_path, progress_path, metadata_path = output / "rows.csv", output / "progress.json", output / "metadata.json"
    if row_path.exists():
        if row_path.read_bytes() != expected_rows:
            raise ValueError("Existing cache rows differ from selected source/order")
    else:
        temporary = row_path.with_suffix(".csv.tmp")
        temporary.write_bytes(expected_rows)
        os.replace(temporary, row_path)
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        actual_settings = {key: value for key, value in metadata.items()
                           if key not in ("signature", "complete", "file_sha256", "progress")}
        if (metadata.get("complete") is not True or actual_settings != settings
                or metadata.get("signature") != expected_signature or signature(actual_settings) != expected_signature):
            raise ValueError("Completed cache signature/source/configuration changed")
        if set(metadata.get("file_sha256", {})) != set(FILES):
            raise ValueError("Completed cache file receipt is incomplete")
        for name in FILES:
            if sha256(output / name) != metadata["file_sha256"][name]:
                raise ValueError(f"Completed cache checksum mismatch: {name}")
        for name in FILES[1:]:
            array = np.load(output / name, mmap_mode="r", allow_pickle=False)
            if array.shape != (len(rows), FEATURE_DIM) or array.dtype != np.float32:
                raise ValueError("Completed cache array shape/dtype mismatch")
            del array
        gate()
        return metadata
    paths = [output / name for name in FILES[1:]]
    if progress_path.exists():
        progress = read_json(progress_path)
        if (progress.get("signature") != expected_signature or progress.get("total") != len(rows)
                or progress.get("partition") != partition):
            raise ValueError("Partial cache signature/source/configuration changed")
    else:
        if any(path.exists() for path in paths):
            raise ValueError("Orphan arrays without signed progress; choose a fresh output directory")
        progress = {"signature": expected_signature, "partition": partition, "completed": 0,
                    "total": len(rows), "complete": False, "batches": []}
        write_json(progress_path, progress)
    completed = progress.get("completed")
    if type(completed) is not int or not 0 <= completed <= len(rows):
        raise ValueError("Invalid completed row count")
    arrays = []
    for path in paths:
        if path.exists():
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
        elif completed == 0:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), FEATURE_DIM))
        else:
            raise ValueError("Partial array is missing")
        if array.shape != (len(rows), FEATURE_DIM) or array.dtype != np.float32:
            raise ValueError("Partial array shape/dtype mismatch")
        arrays.append(array)
    cursor = 0
    for batch in progress["batches"]:
        if batch["start"] != cursor or not cursor < batch["end"] <= completed:
            raise ValueError("Invalid partial checksum ledger")
        for index, view in enumerate(("original", "flip")):
            if array_sha256(arrays[index][cursor:batch["end"]]) != batch[view + "_sha256"]:
                raise ValueError("Partial cache batch checksum mismatch")
        cursor = batch["end"]
    if cursor != completed or (progress.get("complete") and completed != len(rows)):
        raise ValueError("Partial ledger does not cover completed rows")
    loader = DataLoader(PartitionImages(rows[completed:]), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=torch.device(device).type == "cuda", persistent_workers=workers > 0)
    started = last_log = time.monotonic()
    initial = completed
    for images in loader:
        gate()
        original, flipped = safe_encode(model, images, device), safe_encode(model, images.flip(-1), device)
        end = completed + len(images)
        arrays[0][completed:end], arrays[1][completed:end] = original, flipped
        for array in arrays:
            array.flush()
        gate()
        progress["batches"].append({"start": completed, "end": end,
                                    "original_sha256": array_sha256(original), "flip_sha256": array_sha256(flipped)})
        completed = end
        progress["completed"] = completed
        progress["elapsed_seconds_this_session"] = time.monotonic() - started
        progress["images_per_second_both_views"] = (completed - initial) / max(progress["elapsed_seconds_this_session"], 1e-6)
        write_json(progress_path, progress)
        if time.monotonic() - last_log >= 15 or completed == len(rows):
            print(json.dumps({"type": "fusion_feature_extraction", "partition": partition, "completed": completed,
                              "total": len(rows), "images_per_second_both_views": progress["images_per_second_both_views"]}), flush=True)
            last_log = time.monotonic()
    del arrays
    hashes = {name: sha256(output / name) for name in FILES}
    gate()
    progress.update(complete=True, file_sha256=hashes)
    write_json(progress_path, progress)
    metadata = {**settings, "signature": expected_signature, "complete": True, "file_sha256": hashes,
                "progress": {key: value for key, value in progress.items() if key != "batches"}}
    write_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--partition", choices=tuple(ROW_FIELDS), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size != 16 or args.workers != 2:
        raise ValueError("Production extraction uses the frozen training settings: batch size 16, workers 2")
    root = args.project_root.resolve()
    if args.partition == "unlabeled":
        check_recipe(root)
    print(f"Verifying {args.partition} sources and frozen training feature contract...", flush=True)
    train_metadata, contract = verify_training_contract(root, args.batch_size)
    rows, sources = load_partition_rows(root, args.partition, train_metadata)
    settings = {"format": FORMAT, "partition": args.partition, "num_rows": len(rows), "classes": CLASSES,
                **contract, **sources, "workers": args.workers, "other_partition_images_read": 0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent full-extraction CPU fallback")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = load_backbone(root, "cuda")
    output = root / "features_multilayer_fusion" / args.partition
    metadata = extract_partition_cache(model, rows, output, settings, args.batch_size, args.workers, "cuda", root)
    print(json.dumps({"complete": True, "partition": args.partition, "rows": len(rows),
                      "signature": metadata["signature"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
