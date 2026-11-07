"""Portable, offline six-class expression inference. No training data is read."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
WEIGHTS = {"hse": .5, "lora": .2, "ddam": .2, "mlp": .1}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
FORMAT = "expression_offline_bundle_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_extract(archive, destination, members=None):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        selected = z.infolist() if members is None else [z.getinfo(x) for x in members]
        for info in selected:
            rel = PurePosixPath(info.filename.replace("\\", "/"))
            if rel.is_absolute() or ".." in rel.parts or not rel.parts or ":" in rel.parts[0]:
                raise ValueError("Unsafe ZIP path: " + info.filename)
            target = (destination / str(rel)).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP entry")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("classes") != CLASSES:
        raise ValueError("Wrong weight bundle or class order")
    for relative, expected in manifest["files"].items():
        p = (bundle / relative).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or sha256(p) != expected["sha256"]:
            raise ValueError("Missing/modified bundle file: " + relative)
    print(f"Verified {len(manifest['files'])} bundled files and model weights.", flush=True)
    return manifest


def locate_bundle(input_root, working, explicit=None):
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            verify_bundle(candidate)
            return candidate
        archives = [candidate]
    else:
        found = [p.parent for p in Path(input_root).rglob("manifest.json")
                 if p.parent.name == "expression_bundle"]
        if len(found) == 1:
            verify_bundle(found[0])
            return found[0]
        archives = list(Path(input_root).rglob("expression_pretrained_weights.zip"))
    if len(archives) != 1:
        raise FileNotFoundError("Attach expression_pretrained_weights.zip as a Kaggle Dataset, "
                                "or set BUNDLE_PATH to its ZIP/extracted expression_bundle directory.")
    safe_extract(archives[0], Path(working) / "offline_weights")
    bundle = Path(working) / "offline_weights" / "expression_bundle"
    verify_bundle(bundle)
    return bundle


def configure(bundle, device="cuda"):
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    for path in [Path(bundle) / "vendor", Path(bundle) / "vendor" / "dinov2"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if "timm" in sys.modules:
        origin = Path(sys.modules["timm"].__file__).resolve()
        if not origin.is_relative_to(Path(bundle).resolve()):
            raise RuntimeError("Restart kernel: another timm version was imported before this notebook.")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator (T4/P100). CPU inference is slow; "
                           "set DEVICE='cpu' explicitly if desired.")
    print("Runtime:", torch.__version__, "device:", device,
          torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "CPU", flush=True)


def template_ids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["ID", "Emotion"]:
            raise ValueError("Expected template columns ID,Emotion")
        ids = [r["ID"] for r in reader]
    if len(ids) != 7065 or len(set(ids)) != len(ids) or any(Path(x).name != x for x in ids):
        raise ValueError("Expected 7065 unique template image IDs")
    return ids


def locate_images(input_root, working, ids, test_dir=None):
    """Only inspect test paths / test ZIP entries; never extract or open train images."""
    if test_dir:
        folders = [Path(test_dir)]
    else:
        folders = [p for p in Path(input_root).rglob("test") if p.is_dir()]
    valid = [p for p in folders if all((p / x).is_file() for x in ids)]
    if len(valid) == 1:
        return [valid[0] / x for x in ids]
    if len(valid) > 1:
        raise ValueError("Multiple complete test folders. Set TEST_DIR explicitly.")
    if test_dir:
        raise FileNotFoundError("TEST_DIR does not contain every template image")
    for archive in sorted(Path(input_root).rglob("*.zip")):
        if archive.name == "expression_pretrained_weights.zip":
            continue
        with zipfile.ZipFile(archive) as z:
            entries = {PurePosixPath(x).name: x for x in z.namelist()
                       if not x.endswith("/") and "test" in PurePosixPath(x).parts[:-1]
                       and PurePosixPath(x).name in set(ids)}
        if set(entries) == set(ids):
            out = Path(working) / "competition_test"
            safe_extract(archive, out, [entries[x] for x in ids])
            return [out / entries[x] for x in ids]
    raise FileNotFoundError("Attach the competition Dataset with test/*.jpg or the original "
                            "2026task2.zip. Weights alone do not include competition images.")


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.dropout = nn.Dropout(.05)
        self.lora_A = nn.Parameter(torch.zeros(8, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * 2.


class DinoLoRA(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        for index in range(4, 12):
            attention = self.backbone.blocks[index].attn
            attention.qkv = LoRALinear(attention.qkv)
        self.head = nn.Linear(1536, 6)

    def forward(self, x):
        z = self.backbone.forward_features(x)
        return self.head(torch.cat((z["x_norm_clstoken"], z["x_norm_patchtokens"].mean(1)), 1))


class MultilayerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(3840))
        self.register_buffer("feature_scale", torch.ones(3840))
        self.head = nn.Sequential(nn.Linear(3840, 256), nn.LayerNorm(256), nn.GELU(),
                                  nn.Dropout(.3), nn.Linear(256, 6))

    def forward(self, raw):
        return self.head((raw.float() - self.feature_mean) / self.feature_scale)


def state(path):
    result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or not result or not all(isinstance(v, torch.Tensor) for v in result.values()):
        raise ValueError("Expected a tensor-only state dict: " + str(path))
    return result


def load_branch(bundle, branch, device):
    bundle = Path(bundle)
    head = None
    if branch in ("hse", "hse_v10"):
        timm = importlib.import_module("timm")
        if timm.__version__ != "0.9.16":
            raise RuntimeError("Expected bundled timm 0.9.16")
        model = timm.create_model("tf_efficientnet_b2", pretrained=False, num_classes=6, global_pool="max")
        model.load_state_dict(state(bundle / "weights" / (branch + ".pt")), strict=True)
    elif branch == "ddam":
        model = importlib.import_module("ddam_networks.DDAM").DDAMNet(num_class=6, num_head=2, pretrained=False)
        model.load_state_dict(state(bundle / "weights/ddam.pt"), strict=True)
    elif branch in ("lora", "mlp"):
        from dinov2.hub.backbones import dinov2_vitb14
        model = dinov2_vitb14(pretrained=False)
        model.load_state_dict(state(bundle / "weights/dinov2_vitb14.pth"), strict=True)
        if branch == "lora":
            model = DinoLoRA(model)
            adapter = state(bundle / "weights/lora.pt")
            expected = {n for n, _ in model.named_parameters()
                        if n.startswith("head.") or n.endswith((".lora_A", ".lora_B"))}
            if set(adapter) != expected:
                raise ValueError("LoRA adapter keys differ")
            named = dict(model.named_parameters())
            with torch.no_grad():
                for key, tensor in adapter.items():
                    if named[key].shape != tensor.shape:
                        raise ValueError("LoRA tensor shape differs")
                    named[key].copy_(tensor)
        else:
            head = MultilayerHead()
            head.load_state_dict(state(bundle / "weights/mlp.pt"), strict=True)
            head.eval().requires_grad_(False).to(device)
    else:
        raise ValueError("Unknown branch: " + branch)
    return model.eval().requires_grad_(False).to(device), head


def probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid six-class probabilities")
    if not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probability sums differ")
    return p / p.sum(1, keepdims=True)


def transform_for(branch):
    return transforms.Compose([
        transforms.Resize((112, 112) if branch == "ddam" else (224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC if branch in ("mlp", "lora")
                          else transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


@torch.inference_mode()
def predict_branch(bundle, branch, paths, device="cuda", batch_size=16, blur_radius=0.):
    model, head = load_branch(bundle, branch, device)
    transform = transform_for(branch)
    outputs = [[], []]
    last_log = time.monotonic()
    try:
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as source:
                    im = source.convert("RGB") if branch in ("lora", "mlp") else ImageOps.exif_transpose(source).convert("RGB")
                    if blur_radius:
                        im = im.filter(ImageFilter.GaussianBlur(blur_radius))
                    tensors.append(transform(im))
            x = torch.stack(tensors).to(device)
            for view in (0, 1):
                value = x if view == 0 else x.flip(-1)
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                    enabled=str(device).startswith("cuda")):
                    if branch == "mlp":
                        layers = model.get_intermediate_layers(value, n=4, return_class_token=True, norm=True)
                        features = torch.cat([layer[1] for layer in layers] + [layers[-1][0].mean(1)], 1)
                    else:
                        logits = model(value)
                        if branch == "ddam":
                            logits = logits[0]
                if branch == "mlp":
                    # Original pipeline stores float32 features and evaluates its head without AMP.
                    logits = head(features.float())
                outputs[view].append(logits.float().softmax(1).cpu().numpy())
            if time.monotonic() - last_log >= 25 or start + batch_size >= len(paths):
                print(f"{branch}: {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
                last_log = time.monotonic()
        return tuple(probability(np.concatenate(parts)) for parts in outputs)
    finally:
        del model, head
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def predict(bundle, paths, output, mode="ensemble_v8", device="cuda", batch_size=16, blur_probe=0):
    if mode not in ("ensemble_v8", "hse_v10") or batch_size < 1:
        raise ValueError("Invalid model mode or batch size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = WEIGHTS if mode == "ensemble_v8" else {"hse_v10": 1.}
    result = np.zeros((len(paths), 6), dtype=np.float64)
    votes, blurred = {}, np.zeros((min(blur_probe, len(paths)), 6), dtype=np.float64)
    started = time.monotonic()
    for branch, weight in weights.items():
        views = predict_branch(bundle, branch, paths, device, batch_size)
        mean = probability((views[0] + views[1]) / 2)
        for view, value in zip(("original", "flip"), views):
            np.save(output / f"{branch}_{view}.npy", value, allow_pickle=False)
        votes[branch] = mean.argmax(1)
        result += weight * mean
        if len(blurred):
            print(f"Synthetic blur probe: {branch}, Gaussian radius 1 at native resolution", flush=True)
            a, b = predict_branch(bundle, branch, paths[:len(blurred)], device, batch_size, blur_radius=1.)
            blurred += weight * probability((a + b) / 2)
    result = probability(result)
    np.save(output / "probabilities.npy", result, allow_pickle=False)
    if len(blurred):
        blurred = probability(blurred)
        np.save(output / "synthetic_blur_probabilities.npy", blurred, allow_pickle=False)
        probe = {"images": len(blurred), "gaussian_radius_at_native_resolution": 1.,
                 "prediction_agreement": float(np.mean(blurred.argmax(1) == result[:len(blurred)].argmax(1))),
                 "mean_original_max_probability": float(result[:len(blurred)].max(1).mean()),
                 "mean_blurred_max_probability": float(blurred.max(1).mean()),
                 "interpretation": "Unlabeled corruption stability only, NOT accuracy or robustness certification."}
        (output / "blur_stability.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
        print("Blur stability (not accuracy):", probe, flush=True)
    return result, votes, {"elapsed_seconds": time.monotonic() - started, "mode": mode, "weights": weights}


def write_outputs(bundle, ids, paths, p, votes, output, runtime):
    output = Path(output)
    p = probability(p)
    if p.shape != (7065, 6) or len(ids) != 7065 or len(paths) != len(ids):
        raise ValueError("Submission must have 7065 image predictions")
    labels = p.argmax(1)
    with (output / "submission.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Emotion"])
        writer.writerows(zip(ids, labels.tolist()))
    ranked = p.argsort(1)[:, ::-1]
    fields = ["ID", "Emotion", "class_name", "top1_probability", "top2_class", "top2_probability", "margin",
              "entropy", "native_width", "native_height", "laplacian_variance_48", "low_sharpness_heuristic",
              "uncertain_heuristic", "branch_disagreements"]
    with (output / "prediction_diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                width, height = image.size
                a = np.asarray(image.convert("L").resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32)
            lap = -4*a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
            sharpness = float(lap.var())
            top, second = ranked[i, :2]
            margin = float(p[i, top] - p[i, second])
            writer.writerow(dict(ID=ids[i], Emotion=int(top), class_name=CLASSES[top], top1_probability=p[i, top],
                                 top2_class=CLASSES[second], top2_probability=p[i, second], margin=margin,
                                 entropy=float(-(p[i]*np.log(np.clip(p[i], 1e-12, 1))).sum()),
                                 native_width=width, native_height=height, laplacian_variance_48=sharpness,
                                 low_sharpness_heuristic=sharpness < 35.,
                                 uncertain_heuristic=bool(p[i, top] < .6 or margin < .15),
                                 branch_disagreements=sum(int(v[i] != top) for v in votes.values())))
    with (output / "submission.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == ["ID", "Emotion"]
    assert [r["ID"] for r in rows] == ids
    assert [r["Emotion"] for r in rows] == [str(x) for x in labels]
    manifest = read_json(Path(bundle) / "manifest.json")
    receipt = {"state": "complete", "rows": len(rows), "classes": CLASSES, **runtime,
               "torch": torch.__version__, "numpy": np.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "bundle_manifest_sha256": sha256(Path(bundle) / "manifest.json"),
               "submission_sha256": sha256(output / "submission.csv"),
               "probabilities_sha256": sha256(output / "probabilities.npy"),
               "reference_submission_sha256": manifest["reference_submissions"][runtime["mode"]],
               "reference_byte_identical": sha256(output / "submission.csv") == manifest["reference_submissions"][runtime["mode"]],
               "caveat": "GPU/software numerical differences can affect borderline cases. "
                         "Probabilities and sharpness flags are not calibrated correctness guarantees."}
    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Verified submission:", output / "submission.csv", "rows:", len(rows), flush=True)
    return receipt
