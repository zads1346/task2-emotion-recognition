"""Frozen official DINOv2 last-four-layer features, TRAIN partition only.

3840 dimensions = CLS blocks 8/9/10/11 + mean patch tokens of block 11.
The contiguous slice [:, 2304:3840] is the 1536-D last-layer baseline.
No validation, internal-test or competition image is opened by this module.
"""
from __future__ import annotations

import os
os.environ.setdefault("XFORMERS_DISABLED", "1")
import argparse
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FEATURE_DIM = 3840
ROW_FIELDS = ["manifest_index", "path", "label", "target"]
FILES = ["train_rows.csv", "train_original.npy", "train_flip.npy"]
FORMAT = "dinov2_multilayer_train_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_train_rows(root):
    manifest = Path(root) / "features" / "manifest.csv"
    manifest_sha = sha256(manifest)
    audit = json.loads((Path(root) / "data" / "split_audit.json").read_text(encoding="utf-8"))
    if audit.get("manifest_sha256") != manifest_sha or audit.get("classes") != CLASSES:
        raise ValueError("Manifest or official class order differs from the audited split")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,split")
        # Ignore other partitions before parsing their labels or touching paths.
        rows = [{"manifest_index": index, "path": row["path"], "label": row["label"]}
                for index, row in enumerate(reader) if row["split"].strip().lower() == "train"]
    if not rows or len(rows) != audit["splits"]["train"]:
        raise ValueError("Training row count differs from the audited split")
    seen_paths = set()
    digest = hashlib.sha256()
    last_log = time.monotonic()
    for index, row in enumerate(rows):
        if row["label"] not in CLASSES:
            raise ValueError("Unknown training label")
        if not Path(row["path"]).is_absolute():
            raise ValueError("Training image paths must be absolute")
        # Keep exact manifest path text; avoid expensive per-image Path.resolve.
        canonical = os.path.normcase(os.path.normpath(row["path"]))
        if canonical in seen_paths:
            raise ValueError("Duplicate training image path")
        seen_paths.add(canonical)
        row["target"] = CLASSES.index(row["label"])
        row["image_sha256"] = sha256(row["path"])
        digest.update(row["image_sha256"].encode("ascii"))
        if time.monotonic() - last_log >= 60:
            print(json.dumps({"type": "train_source_verification", "completed": index + 1, "total": len(rows)}), flush=True)
            last_log = time.monotonic()
    if {row["target"] for row in rows} != set(range(6)):
        raise ValueError("Training partition must contain all six classes")
    return rows, {"manifest_sha256": manifest_sha, "train_image_content_sha256": digest.hexdigest(),
                  "train_image_content_hash_definition": "SHA256 of ordered concatenated lowercase image SHA256 hex strings"}


def rows_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def make_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


class TrainImages(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = make_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Hash and decode the same bytes, preventing mutation between audit/read.
        content = Path(row["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Training image changed during extraction: {row['path']}")
        with Image.open(io.BytesIO(content)) as source:
            image = self.transform(source.convert("RGB"))
        return image


def official_source_info(root):
    weight = Path(root) / "models" / "dinov2_vitb14_pretrain.pth"
    receipt = json.loads(weight.with_suffix(".json").read_text(encoding="utf-8"))
    weight_sha = sha256(weight)
    if receipt.get("sha256") != weight_sha:
        raise ValueError("Official pretrained weights differ from their download receipt")
    repo = Path(root) / "vendor" / "dinov2"
    digest = hashlib.sha256()
    for source in sorted((repo / "dinov2").rglob("*.py")):
        digest.update(source.relative_to(repo).as_posix().encode())
        digest.update(bytes.fromhex(sha256(source)))
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return {"official_weight_sha256": weight_sha, "repository_commit": commit,
            "repository_python_sha256": digest.hexdigest(), "extractor_sha256": sha256(__file__)}


def load_backbone(root, device):
    repo = str(Path(root) / "vendor" / "dinov2")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dinov2.hub.backbones import dinov2_vitb14
    model = dinov2_vitb14(pretrained=False)
    state = torch.load(Path(root) / "models" / "dinov2_vitb14_pretrain.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).to(device)
    if int(model.embed_dim) != 768 or len(model.blocks) != 12:
        raise ValueError("Expected ViT-B/14 with 768 embedding and 12 blocks")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Backbone must remain fully frozen")
    return model


@torch.inference_mode()
def encode(model, images, device):
    amp = torch.device(device).type == "cuda"
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
        layers = model.get_intermediate_layers(images.to(device, non_blocking=True), n=4,
                                               return_class_token=True, norm=True)
        if len(layers) != 4:
            raise RuntimeError("Expected four intermediate outputs")
        features = torch.cat([layer[1] for layer in layers] + [layers[-1][0].mean(1)], dim=1)
    if features.shape != (len(images), FEATURE_DIM) or not torch.isfinite(features).all():
        raise RuntimeError("Invalid multilayer feature shape or nonfinite values")
    return features.float().cpu().numpy()


def safe_encode(model, images, device):
    try:
        return encode(model, images, device)
    except torch.cuda.OutOfMemoryError:
        if len(images) <= 1:
            raise
    # Retry only after leaving except, releasing traceback-held CUDA tensors.
    gc.collect()
    torch.cuda.empty_cache()
    midpoint = len(images) // 2
    print(f"Inference memory pressure: splitting batch {len(images)}", flush=True)
    return np.concatenate([safe_encode(model, images[:midpoint], device), safe_encode(model, images[midpoint:], device)])


def extract_cache(model, rows, output, settings, batch_size, workers, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_rows = rows_bytes(rows)
    settings = {**settings, "train_rows_sha256": hashlib.sha256(expected_rows).hexdigest()}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    row_path = output / FILES[0]
    if row_path.exists():
        if row_path.read_bytes() != expected_rows:
            raise ValueError("Existing cache rows differ from the current training manifest")
    else:
        temporary = row_path.with_suffix(".csv.tmp")
        temporary.write_bytes(expected_rows)
        os.replace(temporary, row_path)
    progress_path, metadata_path = output / "progress.json", output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature or not metadata.get("complete"):
            raise ValueError("Completed cache settings/source differ from this extraction")
        for filename in FILES:
            if sha256(output / filename) != metadata["file_sha256"][filename]:
                raise ValueError(f"Completed cache checksum mismatch: {filename}")
        print(f"Reused {len(rows)} verified completed training features", flush=True)
        return metadata
    paths = [output / name for name in FILES[1:]]
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["signature"] != signature or progress["total"] != len(rows):
            raise ValueError("Partial cache source/configuration changed")
    else:
        if any(path.exists() for path in paths):
            raise ValueError("Orphan cache arrays without progress; choose a new output directory")
        progress = {"signature": signature, "completed": 0, "total": len(rows), "complete": False, "batches": []}
        write_json(progress_path, progress)
    completed = int(progress["completed"])
    if not 0 <= completed <= len(rows):
        raise ValueError("Invalid completed row count")
    arrays = []
    for path in paths:
        if path.exists():
            array = np.load(path, mmap_mode="r+", allow_pickle=False)
        elif completed == 0:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(rows), FEATURE_DIM))
        else:
            raise ValueError("A partial cache array is missing")
        if array.shape != (len(rows), FEATURE_DIM) or array.dtype != np.float32:
            raise ValueError("Invalid partial cache shape or dtype")
        arrays.append(array)
    cursor = 0
    for batch in progress["batches"]:
        if batch["start"] != cursor or not cursor < batch["end"] <= completed:
            raise ValueError("Invalid partial cache checksum ledger")
        for index, view in enumerate(("original", "flip")):
            if array_sha256(arrays[index][cursor:batch["end"]]) != batch[view + "_sha256"]:
                raise ValueError("Partial cache batch checksum mismatch")
        cursor = batch["end"]
    if cursor != completed:
        raise ValueError("Partial cache ledger does not cover completed rows")
    dataset = TrainImages(rows[completed:])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=torch.device(device).type == "cuda", persistent_workers=workers > 0)
    started = last_log = time.monotonic()
    initial = completed
    for images in loader:
        original = safe_encode(model, images, device)
        flipped = safe_encode(model, images.flip(-1), device)
        end = completed + len(images)
        arrays[0][completed:end], arrays[1][completed:end] = original, flipped
        for array in arrays:
            array.flush()
        progress["batches"].append({"start": completed, "end": end,
                                    "original_sha256": array_sha256(original), "flip_sha256": array_sha256(flipped)})
        completed = end
        progress["completed"] = completed
        progress["elapsed_seconds_this_session"] = time.monotonic() - started
        progress["images_per_second_both_views"] = (completed - initial) / max(progress["elapsed_seconds_this_session"], 1e-6)
        write_json(progress_path, progress)
        if time.monotonic() - last_log >= 15 or completed == len(rows):
            print(json.dumps({"type": "multilayer_extraction", "completed": completed, "total": len(rows),
                              "images_per_second_both_views": progress["images_per_second_both_views"],
                              "estimated_seconds_remaining": (len(rows) - completed) / max(progress["images_per_second_both_views"], 1e-6)}), flush=True)
            last_log = time.monotonic()
    del arrays
    file_hashes = {name: sha256(output / name) for name in FILES}
    progress.update(complete=True, file_sha256=file_hashes)
    write_json(progress_path, progress)
    metadata = {**settings, "signature": signature, "complete": True, "file_sha256": file_hashes,
                "progress": {k: value for k, value in progress.items() if k != "batches"}}
    write_json(metadata_path, metadata)
    return metadata


@torch.inference_mode()
def probe(model, rows, batch_size, device):
    images = torch.stack([TrainImages(rows)[i] for i in range(min(batch_size, len(rows)))])
    amp = torch.device(device).type == "cuda"
    if amp:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    errors = {}
    for name, pixels in [("original", images), ("flip", images.flip(-1))]:
        features = safe_encode(model, pixels, device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            result = model.forward_features(pixels.to(device))
            baseline = torch.cat([result["x_norm_clstoken"], result["x_norm_patchtokens"].mean(1)], dim=1)
        baseline = baseline.float().cpu().numpy()
        errors[name] = float(np.max(np.abs(features[:, 2304:3840] - baseline)))
        if not np.allclose(features[:, 2304:3840], baseline, rtol=2e-3, atol=2e-3):
            raise RuntimeError("Multilayer last-layer slice does not match the original readout")
    if amp:
        torch.cuda.synchronize()
    return {"passed": True, "images": len(images), "feature_shape": [len(images), FEATURE_DIM],
            "last_layer_shape": [len(images), 1536], "last_layer_max_absolute_error": errors,
            "comparison_tolerance": {"rtol": 0.002, "atol": 0.002}, "seconds": time.monotonic() - started,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3 if amp else None,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "extractor_sha256": sha256(__file__), "images_partition": "train"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or workers")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU full-extraction fallback")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = args.project_root.resolve()
    output = (args.output or root / "features_multilayer_train").resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Verifying train-only image sources and official pretrained model...", flush=True)
    rows, data_info = load_train_rows(root)
    settings = {"format": FORMAT, "partition": "train", "num_rows": len(rows), "classes": CLASSES,
                "model": "dinov2_vitb14", "frozen": True, "trainable_parameters": 0,
                "feature_dim": FEATURE_DIM, "dtype": "float32", "layer_indices": [8, 9, 10, 11],
                "feature_definition": "concat normalized CLS of last four blocks then mean normalized last-block patch tokens",
                "readout_layout": [{"name": f"cls_block_{block}", "start": i * 768, "end": (i + 1) * 768}
                                   for i, block in enumerate([8, 9, 10, 11])] + [{"name": "mean_patch_block_11", "start": 3072, "end": 3840}],
                "last_layer_feature_slice": [2304, 3840], "image_size": 224, "rgb": True,
                "preprocessing": "PIL RGB; bicubic resize 224x224; ImageNet mean/std; no crop",
                "flip_definition": "horizontal flip of normalized input tensor; flip(-1)",
                "amp": "float16", "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "Pillow": PIL_VERSION, "batch_size": args.batch_size,
                "other_partition_images_read": 0, **data_info, **official_source_info(root)}
    model = load_backbone(root, "cuda")
    if args.probe_only:
        result = probe(model, rows, args.batch_size, "cuda")
        result.update(gpu=torch.cuda.get_device_name(), **data_info)
        write_json(output / "gpu_probe.json", result)
        print(json.dumps({"gpu_probe": result}), flush=True)
        return
    metadata = extract_cache(model, rows, output, settings, args.batch_size, args.workers, "cuda")
    print(json.dumps({"complete": True, "rows": len(rows), "feature_dim": FEATURE_DIM,
                      "signature": metadata["signature"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
