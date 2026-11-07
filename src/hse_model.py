"""Pinned HSEmotion EfficientNet-B2 runtime, verified conversion and six-class head.

Only prepare_assets() may deserialize the exact SHA-verified official pickle.
All normal training/inference loading uses a converted tensor-only state dict.
No dataset is opened and no pretrained network download is triggered here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickletools
import sys
import zipfile

import numpy as np
import torch
from torch import nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SOURCE_CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
HEAD_ROW_INDICES = [0, 2, 3, 5, 6, 4]
MODEL_NAME = "tf_efficientnet_b2"
TIMM_VERSION = "0.9.16"
OFFICIAL_URL = "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b2_7.pt"
OFFICIAL_SHA = "e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f"
OFFICIAL_SIZE = 31367173
FORMAT = "hsemotion_efficientnet_b2_six_class_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def private_runtime(root):
    runtime = (Path(root) / "vendor" / "hse_runtime").resolve()
    package = runtime / "timm"
    if not (package / "__init__.py").is_file():
        raise RuntimeError("Install timm==0.9.16 only into vendor/hse_runtime with pip --target --no-deps")
    if "timm" in sys.modules:
        module_path = Path(sys.modules["timm"].__file__).resolve()
        if not module_path.is_relative_to(package):
            raise RuntimeError("A shared/system timm is already imported; start a fresh HSE process")
    if str(runtime) in sys.path:
        sys.path.remove(str(runtime))
    sys.path.insert(0, str(runtime))
    timm = importlib.import_module("timm")
    if timm.__version__ != TIMM_VERSION or not Path(timm.__file__).resolve().is_relative_to(package):
        raise ValueError("Expected the private timm==0.9.16 runtime")
    return timm


def runtime_fingerprint(root):
    timm = private_runtime(root)
    runtime = Path(root) / "vendor" / "hse_runtime"
    digest = hashlib.sha256()
    sources = sorted((runtime / "timm").rglob("*.py"))
    if not sources:
        raise ValueError("Private timm Python sources are missing")
    for source in sources:
        digest.update(source.relative_to(runtime).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(source)))
    metadata = runtime / f"timm-{TIMM_VERSION}.dist-info" / "METADATA"
    wheel = runtime / f"timm-{TIMM_VERSION}.dist-info" / "WHEEL"
    return {"timm_version": timm.__version__, "timm_source_sha256": digest.hexdigest(),
            "timm_python_files": len(sources), "timm_metadata_sha256": sha256(metadata),
            "timm_wheel_metadata_sha256": sha256(wheel), "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__), "numpy_version": str(np.__version__)}


def verify_official_weight(root):
    folder = Path(root) / "models" / "hsemotion"
    path = folder / "enet_b2_7.pt"
    if path.stat().st_size != OFFICIAL_SIZE or sha256(path) != OFFICIAL_SHA:
        raise ValueError("Official HSE checkpoint differs from pinned SHA256/size")
    receipt = load_json(folder / "official_weight.json")
    if (receipt.get("sha256") != OFFICIAL_SHA or receipt.get("size") != OFFICIAL_SIZE
            or receipt.get("official_url") != OFFICIAL_URL or receipt.get("download_verified") is not True):
        raise ValueError("Official HSE download receipt differs from pinned source")
    return path


def _base_model(root):
    timm = private_runtime(root)
    return timm.create_model(MODEL_NAME, pretrained=False, num_classes=7, global_pool="max")


def _verified_official_pickle(path):
    # Hard-coded byte identity is checked again immediately before the only unsafe load.
    if Path(path).stat().st_size != OFFICIAL_SIZE or sha256(path) != OFFICIAL_SHA:
        raise ValueError("Refusing to deserialize an unverified official checkpoint")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(names) != 1:
            raise ValueError("Unexpected official torch archive structure")
        operations = list(pickletools.genops(archive.read(names[0])))
    globals_ = sorted({argument for operation, argument, _ in operations if operation.name == "GLOBAL"})
    allowed = ("torch ", "torch._utils ", "torch.nn.modules.", "timm.layers.", "timm.models.",
               "collections OrderedDict", "__builtin__ set", "_codecs encode")
    if any(not name.startswith(allowed) for name in globals_):
        raise ValueError(f"Unexpected official pickle globals: {globals_}")
    original = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(original, nn.Module):
        raise ValueError("Expected the original official seven-class nn.Module")
    return original.eval(), globals_


def _replace_head(model):
    previous = model.get_classifier()
    if not isinstance(previous, nn.Linear) or previous.out_features != 7:
        raise ValueError("Unexpected seven-class HSE classifier layout")
    replacement = nn.Linear(previous.in_features, len(CLASSES), bias=previous.bias is not None,
                            device=previous.weight.device, dtype=previous.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(previous.weight[HEAD_ROW_INDICES])
        if previous.bias is not None:
            replacement.bias.copy_(previous.bias[HEAD_ROW_INDICES])
    model.classifier = replacement
    model.num_classes = len(CLASSES)
    return model


def _validate_conversion(root, runtime):
    folder = Path(root) / "models" / "hsemotion"
    receipt = load_json(folder / "conversion_receipt.json")
    if (receipt.get("format") != FORMAT or receipt.get("official_weight_sha256") != OFFICIAL_SHA
            or receipt.get("model_source_sha256") != sha256(__file__)
            or receipt.get("runtime") != runtime or receipt.get("strict_full_restore") is not True
            or receipt.get("cpu_forward_equivalence", {}).get("passed") is not True
            or receipt.get("six_class_head_equivalence", {}).get("passed") is not True
            or receipt.get("tensor_only_exact_roundtrip") is not True
            or receipt.get("source_classes7") != SOURCE_CLASSES or receipt.get("head_row_indices") != HEAD_ROW_INDICES):
        raise ValueError("HSE conversion receipt/source/runtime is incompatible")
    state_path = folder / "enet_b2_7_state.pt"
    if sha256(state_path) != receipt.get("state_sha256") or state_path.stat().st_size != receipt.get("state_size"):
        raise ValueError("Converted HSE tensor-only state differs from its receipt")
    return receipt


def source_receipt(root):
    """Return strict, path-independent provenance to store as model_provenance."""
    verify_official_weight(root)
    runtime = runtime_fingerprint(root)
    converted = _validate_conversion(root, runtime)
    folder = Path(root) / "models" / "hsemotion"
    return {"format": FORMAT, "architecture": MODEL_NAME, "global_pool": "max", "classes": CLASSES,
            "source_classes7": SOURCE_CLASSES, "head_row_indices": HEAD_ROW_INDICES,
            "head_initialization": "copy official classifier rows; drop Disgust; retain official six-class order",
            "official_url": OFFICIAL_URL, "official_weight_sha256": OFFICIAL_SHA, "official_weight_size": OFFICIAL_SIZE,
            "official_weight_receipt_sha256": sha256(folder / "official_weight.json"),
            "state_sha256": converted["state_sha256"], "conversion_receipt_sha256": sha256(folder / "conversion_receipt.json"),
            "model_source_sha256": sha256(__file__), "runtime": runtime,
            "pretraining_overlap": "unknown; not independently audited"}


def prepare_assets(root):
    """CPU-only strict restoration and synthetic forward verification; no datasets."""
    root = Path(root)
    original_path = verify_official_weight(root)
    runtime = runtime_fingerprint(root)
    folder = root / "models" / "hsemotion"
    state_path, receipt_path = folder / "enet_b2_7_state.pt", folder / "conversion_receipt.json"
    if state_path.exists() or receipt_path.exists():
        if not (state_path.exists() and receipt_path.exists()):
            raise ValueError("Incomplete previous HSE conversion; inspect before preparing again")
        _validate_conversion(root, runtime)
        return source_receipt(root)
    original, globals_ = _verified_official_pickle(original_path)
    if not isinstance(original.get_classifier(), nn.Linear) or original.get_classifier().out_features != 7:
        raise ValueError("Official checkpoint must have seven output classes")
    state = {name: tensor.detach().cpu().clone() for name, tensor in original.state_dict().items()}
    rebuilt = _base_model(root).eval()
    rebuilt.load_state_dict(state, strict=True)
    generator = torch.Generator().manual_seed(20261013)
    inputs = torch.randn(2, 3, 260, 260, generator=generator)
    with torch.inference_mode():
        expected, actual = original(inputs), rebuilt(inputs)
    if expected.shape != (2, 7) or not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("Invalid official/restored CPU predictions")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    max_error = float((actual - expected).abs().max())
    _replace_head(rebuilt)
    with torch.inference_mode():
        initialized = rebuilt(inputs)
    torch.testing.assert_close(initialized, actual[:, HEAD_ROW_INDICES], rtol=1e-5, atol=1e-6)
    head_error = float((initialized - actual[:, HEAD_ROW_INDICES]).abs().max())
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, state_path)
    safe_state = torch.load(state_path, map_location="cpu", weights_only=True)
    if set(state) != set(safe_state) or any(not torch.equal(state[name], safe_state[name]) for name in state):
        raise ValueError("Converted tensor-only checkpoint did not round-trip exactly")
    receipt = {"format": FORMAT, "official_weight_sha256": OFFICIAL_SHA, "model_source_sha256": sha256(__file__),
               "state_sha256": sha256(state_path), "state_size": state_path.stat().st_size,
               "strict_full_restore": True, "source_classes7": SOURCE_CLASSES, "head_row_indices": HEAD_ROW_INDICES,
               "runtime": runtime, "pickle_globals": globals_,
               "cpu_forward_equivalence": {"passed": True, "synthetic_shape": list(inputs.shape),
                                           "max_absolute_error": max_error, "rtol": 1e-5, "atol": 1e-6},
               "six_class_head_equivalence": {"passed": True, "max_absolute_error": head_error},
               "tensor_only_exact_roundtrip": True}
    write_json(receipt_path, receipt)
    return source_receipt(root)


def load_initial_model(root, device="cpu"):
    """Return the official backbone with six supervised class rows copied into its head."""
    source_receipt(root)
    model = _base_model(root)
    state = torch.load(Path(root) / "models" / "hsemotion" / "enet_b2_7_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return _replace_head(model).eval().to(device)


def load_trained_model(root, checkpoint, device="cpu"):
    """Load an audited full six-class model_state; never execute training checkpoint pickle."""
    provenance = source_receipt(root)
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("classes") != CLASSES:
        raise ValueError("HSE training checkpoint class order is incompatible")
    if checkpoint.get("model_provenance") != provenance:
        raise ValueError("HSE training checkpoint provenance differs from current sources")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("HSE checkpoint requires a tensor-only complete model_state")
    model = _replace_head(_base_model(root))
    model.load_state_dict(state, strict=True)
    if model.classifier.out_features != 6 or any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("HSE trained model contains invalid weights")
    return model.eval().to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    receipt = prepare_assets(args.project_root) if args.prepare else source_receipt(args.project_root)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
