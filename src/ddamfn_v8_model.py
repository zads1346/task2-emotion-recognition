"""Pinned official DDAMFN++ AffectNet7 model with a native six-class state.

Only tensor-only checkpoints are deserialized. The official feature/network
modules execute in a private namespace; the generic ``networks`` import is
resolved locally without modifying that name in sys.modules or editing vendor
source. No model downloads, extra MFN checkpoint, dataset reads or GPU probes
are performed by this module.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "ddamfn_v8"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SOURCE_CLASSES = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]
HEAD_ROW_INDICES = [6, 4, 1, 2, 3, 0]
REPOSITORY = "SainingZhang/DDAMFN"
COMMIT = "cebfe5c5969b7d1fd64ecd775e318d3834ca76bb"
OFFICIAL_WEIGHT_SHA256 = "edf731a458162eef6b2be12c3572bd10cbe92ce8c009da4331184a2943d7ed14"
OFFICIAL_WEIGHT_SIZE = 17499403
OFFICIAL_REPO_PATH = "DDAMFN++/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth"
OFFICIAL_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/DDAMFN%2B%2B/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth"
SOURCE_SHA256 = {
    "README.md": "30b0b9d18360b61852762f3634231806de01e2c8daa34bedf5b9ba722394d5d3",
    "DDAMFN++/networks/DDAM.py": "fa73e9b8a0a04e634b04d33f8d100beecfac05c28135b187fbab823983ce921d",
    "DDAMFN++/networks/MixedFeatureNet.py": "7e1186767d3cdee975ad099ae218046a38949f0ceb2f99e4d611e900863451d5",
    "DDAMFN++/affectnet7_test.py": "479d59558a51c29d7c48bd38b4d0a6111655bcfc10c9f89878734e64308fa760",
    "DDAMFN++/affectnet8_test.py": "92ed3d96e51293c84aefef814a3972d400978d226ed764e975a91cf5353f4d92",
    "DDAMFN++/affectnet_train_sam_opt_v2.0.py": "fb18817504ea028e60f01e9111807b438fd845337b08d2266939202987416ef7",
    "DDAMFN++/sam.py": "48a0dfd1093f251d2f895c7a5cb9ba7a3844f6ece6c8be32653dc7c13f3bd75e",
}
SLICED_STATE_KEYS = {"fc.weight", "fc.bias", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"}
PREPROCESSING = {"image_size": [112, 112], "color": "RGB", "exif_transpose": True,
    "resize_interpolation": "bilinear", "antialias": True, "tensor_range_before_normalization": [0., 1.],
    "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "official_reference": "DDAMFN++/affectnet7_test.py: Resize112, ToTensor, ImageNet mean/std",
    "validation_flip": "deterministic normalized tensor.flip(-1)"}
_IMPORT_LOCK = threading.RLock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_sha256(receipt):
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"DDAMFN source/checkpoint SHA256 differs: {path}")


def _paths(root):
    root = Path(root).resolve()
    vendor = root/"vendor/ddamfn_v8"
    weights = root/"models/ddamfn_affectnet7_v8"
    return vendor, weights, weights/"affecnet7_epoch19_acc0.671.pth"


def _verified_asset_receipts(root):
    vendor, weights, weight_path = _paths(root)
    source = _read_json(vendor/"source_receipt.json")
    weight = _read_json(weights/"source_receipt.json")
    if source.get("repository") != REPOSITORY or source.get("commit") != COMMIT or source.get("license_paths") != []:
        raise ValueError("Pinned DDAMFN source repository/commit/license inventory differs")
    entries = source.get("sources")
    if not isinstance(entries, list) or {entry.get("repo_path") for entry in entries} != set(SOURCE_SHA256) or len(entries) != len(SOURCE_SHA256):
        raise ValueError("Complete pinned DDAMFN source inventory differs")
    for entry in entries:
        relative = entry["repo_path"]
        path = vendor/relative
        _check_hash(path, SOURCE_SHA256[relative])
        if entry.get("sha256") != SOURCE_SHA256[relative] or entry.get("commit") != COMMIT or entry.get("size") != path.stat().st_size:
            raise ValueError("DDAMFN source receipt hash/commit/size differs")
    python_files = {path.relative_to(vendor).as_posix() for path in vendor.rglob("*.py")}
    if python_files != {name for name in SOURCE_SHA256 if name.endswith(".py")}:
        raise ValueError("Unexpected or missing DDAMFN vendor Python module")
    if (weight.get("repository") != REPOSITORY or weight.get("commit") != COMMIT
        or weight.get("repo_path") != OFFICIAL_REPO_PATH or weight.get("url") != OFFICIAL_URL
        or weight.get("sha256") != OFFICIAL_WEIGHT_SHA256 or weight.get("bytes") != OFFICIAL_WEIGHT_SIZE
        or weight.get("native_classes") != SOURCE_CLASSES or weight.get("official_classes") != CLASSES
        or weight.get("official_class_row_indices") != HEAD_ROW_INDICES
        or weight.get("pickle_unsafe_load_used") is not False
        or weight.get("external_pretrain_overlap_audited") is not False):
        raise ValueError("Official DDAMFN weight receipt/source/classes/safe-load provenance differs")
    safe = weight.get("safe_load", {})
    if (safe.get("weights_only") is not True or safe.get("map_location") != "cpu"
        or safe.get("state_keys") != 1190 or safe.get("fc_shape") != [7, 512]
        or safe.get("all_state_tensors_finite") is not True
        or not isinstance(safe.get("state_key_shape_dtype"), dict)
        or len(safe["state_key_shape_dtype"]) != 1190):
        raise ValueError("Official DDAMFN tensor-only state inventory differs")
    if weight_path.stat().st_size != OFFICIAL_WEIGHT_SIZE:
        raise ValueError("Official DDAMFN checkpoint size differs")
    _check_hash(weight_path, OFFICIAL_WEIGHT_SHA256)
    return source, weight


def source_receipt(root):
    """Deterministic, path-independent model/source/weight/preprocessing binding."""
    vendor, weights, _ = _paths(root)
    source, weight = _verified_asset_receipts(root)
    entries = {entry["repo_path"]: entry for entry in source["sources"]}
    source_files = {name: {"sha256": SOURCE_SHA256[name], "bytes": (vendor/name).stat().st_size,
                          "git_blob_sha1": entries[name]["git_blob_sha1"], "url": entries[name]["url"]}
                    for name in sorted(SOURCE_SHA256)}
    python_digest = hashlib.sha256()
    for name in sorted(SOURCE_SHA256):
        if name.endswith(".py"):
            python_digest.update(name.encode("utf-8"))
            python_digest.update(bytes.fromhex(SOURCE_SHA256[name]))
    return {"format": "ddamfn_v8_model_source", "architecture": "DDAMFN++ DDAMNet",
        "repository": REPOSITORY, "commit": COMMIT, "source_files": source_files,
        "vendor_python_sha256": python_digest.hexdigest(),
        "vendor_source_receipt_sha256": sha256(vendor/"source_receipt.json"),
        "official_weight_receipt_sha256": sha256(weights/"source_receipt.json"),
        "official_weight_url": OFFICIAL_URL, "official_weight_sha256": OFFICIAL_WEIGHT_SHA256,
        "official_weight_git_blob_sha1": weight["git_blob_sha1"], "class_order_source": "DDAMFN++/affectnet7_test.py",
        "official_weight_bytes": OFFICIAL_WEIGHT_SIZE, "official_checkpoint_epoch": weight["safe_load"]["iter"],
        "official_tensor_only_inventory_sha256": receipt_sha256(weight["safe_load"]["state_key_shape_dtype"]),
        "official_weight_safe_load": {"weights_only": True, "map_location": "cpu", "unsafe_pickle_used": False},
        "native_classes": SOURCE_CLASSES, "classes": CLASSES, "head_row_indices": HEAD_ROW_INDICES,
        "classifier_mapping": "official7 to six: fc and unused classifier bn rows[6,4,1,2,3,0]; bn.num_batches_tracked unchanged",
        "native_state_keys": 1190, "num_heads": 2, "num_classes": 6,
        "extra_mfn_pretrained_checkpoint_loaded": False,
        "official_constructor": "DDAMNet(num_class=6,num_head=2,pretrained=False)",
        "forward": "unmodified official network forward tuple[0]; native parameter/state keys retained",
        "vendor_import": "private module namespace; official networks import redirected only inside DDAM globals; no generic sys.modules pollution",
        "preprocessing": PREPROCESSING, "model_source_sha256": sha256(__file__),
        "runtime": {"torch_version": str(torch.__version__)},
        "license_paths": source["license_paths"], "license_status": "no LICENSE in pinned repository inventory; terms not independently established",
        "external_pretrain_overlap_audited": False,
        "external_pretrain_overlap": "unknown; pretrained AffectNet/MS-Celeb overlap with task not independently audited"}


def _private_vendor_modules(root):
    """Load verified native classes without inserting/replacing generic networks."""
    vendor, _, _ = _paths(root)
    prefix = "_task_ddamfn_v8_vendor_"+COMMIT
    package_name, mixed_name, ddam_name = prefix+".networks", prefix+".networks.MixedFeatureNet", prefix+".networks.DDAM"
    with _IMPORT_LOCK:
        if ddam_name in sys.modules:
            for name, relative in ((mixed_name, "DDAMFN++/networks/MixedFeatureNet.py"), (ddam_name, "DDAMFN++/networks/DDAM.py")):
                if Path(sys.modules[name].__file__).resolve() != (vendor/relative).resolve():
                    raise ValueError("An incompatible private DDAMFN namespace is already imported")
            return sys.modules[ddam_name], sys.modules[mixed_name]
        parent = types.ModuleType(prefix)
        parent.__path__ = [str(vendor/"DDAMFN++")]
        package = types.ModuleType(package_name)
        package.__path__ = [str(vendor/"DDAMFN++/networks")]
        package.__package__ = package_name
        sys.modules[prefix], sys.modules[package_name] = parent, package
        try:
            mixed_path = vendor/"DDAMFN++/networks/MixedFeatureNet.py"
            mixed_spec = importlib.util.spec_from_file_location(mixed_name, mixed_path)
            mixed = importlib.util.module_from_spec(mixed_spec)
            sys.modules[mixed_name] = mixed
            # Compile/exec verified file bytes directly; do not write vendor pyc.
            exec(compile(mixed_path.read_bytes(), str(mixed_path), "exec"), mixed.__dict__)
            package.MixedFeatureNet = mixed
            ddam_path = vendor/"DDAMFN++/networks/DDAM.py"
            ddam_spec = importlib.util.spec_from_file_location(ddam_name, ddam_path)
            ddam = importlib.util.module_from_spec(ddam_spec)
            def private_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "networks" and level == 0:
                    if tuple(fromlist) != ("MixedFeatureNet",):
                        raise ImportError("Unexpected generic networks import in pinned DDAM source")
                    return package
                return builtins.__import__(name, globals, locals, fromlist, level)
            ddam.__dict__["__builtins__"] = {**builtins.__dict__, "__import__": private_import}
            sys.modules[ddam_name] = ddam
            exec(compile(ddam_path.read_bytes(), str(ddam_path), "exec"), ddam.__dict__)
            return ddam, mixed
        except Exception:
            for name in (ddam_name, mixed_name, package_name, prefix):
                sys.modules.pop(name, None)
            raise


def _new_native_model(root, classes=6):
    module, _ = _private_vendor_modules(root)
    model = module.DDAMNet(num_class=classes, num_head=2, pretrained=False)
    if model.fc.in_features != 512 or model.fc.out_features != classes or len(model.state_dict()) != 1190:
        raise ValueError("Native DDAMFN architecture/classifier/state key count differs")
    return model.requires_grad_(True)


def _native_logits_forward(self, images):
    output = self._ddamfn_native_forward(images)
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError("Pinned DDAMFN native forward tuple differs")
    logits = output[0]
    if logits.ndim != 2 or logits.shape[1] != 6:
        raise ValueError("Expected six official DDAMFN logits")
    return logits


def _logits_only(model):
    # A method substitution avoids adding an nn.Module wrapper/state prefix.
    object.__setattr__(model, "_ddamfn_native_forward", model.forward)
    model.forward = types.MethodType(_native_logits_forward, model)
    return model


def verify_state(root, state):
    """Strictly validate all1190 native six-class tensor keys/shapes/dtypes/finite values."""
    _, official = _verified_asset_receipts(root)
    inventory = official["safe_load"]["state_key_shape_dtype"]
    if not isinstance(state, dict) or set(state) != set(inventory) or len(state) != 1190:
        raise ValueError("DDAMFN checkpoint requires the complete native1190-key state")
    for name, expected in inventory.items():
        value = state[name]
        shape = list(expected["shape"])
        if name in SLICED_STATE_KEYS:
            if not shape or shape[0] != 7:
                raise ValueError("Unexpected official seven-class tensor shape")
            shape[0] = 6
        if not isinstance(value, torch.Tensor) or list(value.shape) != shape or str(value.dtype) != expected["dtype"]:
            raise ValueError(f"DDAMFN tensor shape/dtype differs: {name}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"DDAMFN tensor is nonfinite: {name}")
    return True


def load_initial_model(root, device="cpu"):
    """Return (native model with six logits, full deterministic source receipt)."""
    root = Path(root).resolve()
    receipt = source_receipt(root)
    _, _, weight_path = _paths(root)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Official DDAMFN checkpoint lacks tensor-only model_state_dict")
    if checkpoint.get("iter") != receipt["official_checkpoint_epoch"]:
        raise ValueError("Official DDAMFN checkpoint epoch differs")
    original = checkpoint["model_state_dict"]
    converted = {name: tensor.detach().cpu().clone() for name, tensor in original.items()}
    for name in SLICED_STATE_KEYS:
        converted[name] = converted[name][HEAD_ROW_INDICES].clone()
    verify_state(root, converted)
    model = _new_native_model(root, classes=6)
    model.load_state_dict(converted, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), converted[name]):
            raise ValueError(f"Native six-class DDAMFN state did not restore exactly: {name}")
    return _logits_only(model).eval().to(device), receipt


def _verify_checkpoint(root, checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT or checkpoint.get("classes") != CLASSES:
        raise ValueError("Unsupported full DDAMFN V8 checkpoint/official class order")
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("classes") != CLASSES:
        raise ValueError("DDAMFN config official class order differs")
    receipt = source_receipt(root)
    signature = receipt_sha256(receipt)
    for owner in (config, checkpoint):
        if owner.get("model_source_receipt") != receipt or owner.get("model_source_receipt_sha256") != signature:
            raise ValueError("DDAMFN checkpoint current source/weight/preprocessing receipt differs")
    plan_path = root/"logs/ddamfn_v8_plan.json"
    _check_hash(plan_path, config.get("plan_sha256"))
    plan = _read_json(plan_path)
    fields = ("classes", "parent_checkpoint", "parent_checkpoint_sha256", "source_pins",
              "train_indices", "val_indices", "original_train_indices", "excluded_train_indices",
              "input_receipt", "audit_summary_path", "audit_summary_sha256", "baseline_directory", "baseline_correct_count")
    if any(config.get(key) != plan.get(key) for key in fields):
        raise ValueError("DDAMFN config differs from the immutable plan")
    for key in ("settings", "filtered_membership_sha256", "model_source_receipt", "model_source_receipt_sha256"):
        if key in plan and config.get(key) != plan[key]:
            raise ValueError(f"DDAMFN optional config/plan binding differs: {key}")
    pins = config.get("source_pins")
    if not isinstance(pins, dict) or Path(__file__).resolve() not in {Path(path).resolve() for path in pins}:
        raise ValueError("DDAMFN source pins omit the active model implementation")
    for path, digest in pins.items():
        if not Path(path).is_absolute():
            raise ValueError("DDAMFN source pins require absolute paths")
        _check_hash(path, digest)
    if checkpoint.get("source_pins") != pins:
        raise ValueError("DDAMFN checkpoint/config source pins differ")
    if not isinstance(config["audit_summary_path"], str) or not Path(config["audit_summary_path"]).is_absolute():
        raise ValueError("DDAMFN audit summary requires an absolute string path")
    _check_hash(config["audit_summary_path"], config["audit_summary_sha256"])
    _, _, official_path = _paths(root)
    if Path(config["parent_checkpoint"]).resolve() != official_path.resolve() or config["parent_checkpoint_sha256"] != OFFICIAL_WEIGHT_SHA256:
        raise ValueError("DDAMFN parent must be the pinned official seven-class AffectNet checkpoint")
    _check_hash(official_path, config["parent_checkpoint_sha256"])
    if checkpoint.get("parent_sha") != config["parent_checkpoint_sha256"]:
        raise ValueError("DDAMFN checkpoint parent SHA binding differs")
    if not isinstance(config["baseline_directory"], str) or not Path(config["baseline_directory"]).is_absolute():
        raise ValueError("DDAMFN baseline_directory must be an absolute string path")
    if type(config["baseline_correct_count"]) is not int or not 0 <= config["baseline_correct_count"] <= len(config["val_indices"]):
        raise ValueError("DDAMFN dynamic baseline correct count is invalid")
    inputs = config["input_receipt"]
    if (not isinstance(inputs, dict) or inputs.get("internal_test_images_read") != 0
        or inputs.get("internal_test_labels_parsed") is not False):
        raise ValueError("DDAMFN input receipt must verify train/validation only")
    original, effective, excluded, validation = (config[key] for key in ("original_train_indices", "train_indices", "excluded_train_indices", "val_indices"))
    if any(not isinstance(values, list) or any(type(index) is not int or index < 0 for index in values) or len(values) != len(set(values)) for values in (original, effective, excluded, validation)):
        raise ValueError("DDAMFN member arrays must contain unique integer manifest indices")
    if not set(excluded).issubset(original) or effective != [index for index in original if index not in set(excluded)] or set(original) & set(validation):
        raise ValueError("DDAMFN effective/original/excluded/validation members are inconsistent")
    for key in ("train_indices", "val_indices", "original_train_indices", "excluded_train_indices", "audit_summary_sha256"):
        if checkpoint.get(key) != config[key]:
            raise ValueError(f"DDAMFN checkpoint/config member/audit binding differs: {key}")
    verify_state(root, checkpoint.get("model_state"))
    return config


def load_trained_model(root, path, device="cpu"):
    """Return (model, metadata) after full native-state/source/plan/member verification."""
    root = Path(root).resolve()
    checkpoint = (torch.load(path, map_location="cpu", weights_only=True)
                  if isinstance(path, (str, os.PathLike)) else path)
    _verify_checkpoint(root, checkpoint)
    model = _new_native_model(root, classes=6)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), checkpoint["model_state"][name].detach().cpu()):
            raise ValueError(f"DDAMFN trained state did not restore exactly: {name}")
    excluded = {"model_state", "optimizer", "scaler", "rng", "ema_state"}
    metadata = {key: value for key, value in checkpoint.items() if key not in excluded}
    return _logits_only(model).eval().to(device), metadata
