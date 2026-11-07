"""Task-specific, eight-block DINOv2 LoRA continuation with audited candidates.

Only this new implementation is used; existing experiments stay immutable.
Training reads train/validation members only. Candidate selection and competition
export are deliberately owned by a separate runner/finalizer.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "4"
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import types

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

import train_lora as base
import train_hse_specialist as hse
from refine_lora import AdapterEMA
from validate_cosine_fusion import metrics, val_rows

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FORMAT = "task_lora_v4"
ARCHITECTURE = {"model": "dinov2_vitb14", "last_blocks": 8, "rank": 8,
                "alpha": 16.0, "lora_dropout": .05, "pooling": "CLS+mean_patch",
                "feature_dim": 1536, "classes": CLASSES}
PARENT_ARCHITECTURE = {**ARCHITECTURE, "last_blocks": 4}
FIXED_SETTINGS = {"epochs": 6, "seed": 42, "augmentation_epoch_offset": 20,
                  "label_smoothing": .02, "weight_decay": .001,
                  "lr_new": 2e-5, "lr_inherited": 5e-6, "lr_head": 2e-5,
                  "warmup_successful_steps": 100, "min_lr_ratio": .1,
                  "ema_decay": .995}


def sha256(path):
    return base.sha256(path)


def check_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_array(path, array):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_copy(source, destination):
    temporary = Path(str(destination) + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def adapter_names(blocks):
    return {f"backbone.blocks.{index}.attn.qkv.lora_{letter}"
            for index in blocks for letter in ("A", "B")}


INHERITED_NAMES = adapter_names(range(8, 12)) | {"head.weight", "head.bias"}
NEW_NAMES = adapter_names(range(4, 8))
ALL_NAMES = INHERITED_NAMES | NEW_NAMES


def new_adapter_names(model):
    """Return the verified blocks4-7 adapter names for the independent GPU smoke."""
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if names != ALL_NAMES:
        raise ValueError("Smoke model trainable parameter keys differ")
    return set(NEW_NAMES)


def _validate_adapter_tensors(state):
    if not isinstance(state, dict) or set(state) != ALL_NAMES:
        raise ValueError("Complete eight-block adapter/head state required")
    for name, value in state.items():
        if name == "head.weight":
            expected_shape = (6, 1536)
        elif name == "head.bias":
            expected_shape = (6,)
        elif name.endswith(".lora_A"):
            expected_shape = (8, 768)
        else:
            expected_shape = (2304, 8)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
            raise ValueError(f"Invalid checkpoint adapter dtype/shape: {name}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite checkpoint adapter: {name}")


def _strict_state(model, state, expected):
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("Adapter/head key set differs from the fixed architecture")
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"Adapter must be a float32 tensor: {name}")
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter/head shape or nonfinite tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)
    for name, value in state.items():
        if not torch.equal(parameters[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"Adapter/head copy was not exact: {name}")


def _checkpointed_forward(self, images):
    # Evaluation delegates to the unchanged official feature path.
    if not self.training or not torch.is_grad_enabled() or not self.activation_checkpointed:
        return base.DinoLoRAClassifier.forward(self, images)
    tokens = self.backbone.prepare_tokens_with_masks(images, masks=None)
    for index, block in enumerate(self.backbone.blocks):
        if index in self.adapted_blocks:
            tokens = activation_checkpoint(block, tokens, use_reentrant=False,
                                           preserve_rng_state=True)
        else:
            tokens = block(tokens)
    normalized = self.backbone.norm(tokens)
    patch_start = 1 + self.backbone.num_register_tokens
    features = torch.cat((normalized[:, 0], normalized[:, patch_start:].mean(1)), dim=1)
    return self.head(features)


def _attach_checkpointing(model, checkpointed):
    if model.backbone.chunked_blocks or model.backbone.num_register_tokens != 0:
        raise ValueError("Require the original non-chunked, no-register ViT-B/14")
    model.activation_checkpointed = bool(checkpointed)
    model.forward = types.MethodType(_checkpointed_forward, model)
    model.assert_trainable()
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if actual != ALL_NAMES or model.adapted_blocks != list(range(4, 12)):
        raise ValueError("Unexpected trainable parameters/adapted blocks")
    return model


def _parent_payload(root, parent_path):
    parent_path = Path(parent_path).resolve()
    expected_path = (root / "experiments/lora_refinement/best.pt").resolve()
    if parent_path != expected_path:
        raise ValueError("Only the audited original refined LoRA parent is allowed")
    payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "dinov2_lora_refinement_v1" or payload.get("classes") != CLASSES:
        raise ValueError("Parent format/official class order differs")
    if payload.get("config", {}).get("architecture") != PARENT_ARCHITECTURE:
        raise ValueError("Parent must have last4 qkv rank8 alpha16 dropout.05 and a 1536 Linear head")
    provenance = payload["provenance"]
    source_paths = {"original_model_source_sha256": Path(base.__file__),
                    "refinement_source_sha256": root / "src/refine_lora.py"}
    for field, source in source_paths.items():
        check_hash(source, provenance.get(field))
    official = base.verify_official_weights(root)
    if official != provenance.get("official_weight_sha256"):
        raise ValueError("Parent official DINOv2 weights differ")
    repository = base.repository_info(root)
    if any(repository[key] != provenance.get(key) for key in repository):
        raise ValueError("Parent DINOv2 vendor source differs")
    if provenance.get("test_labels_used") is not False or provenance.get("test_images_read") != 0:
        raise ValueError("Parent lacks a train/validation-only provenance")
    return payload, {"parent_checkpoint": str(parent_path), "parent_sha256": sha256(parent_path),
                     "parent_source_pins": {str(path.resolve()): sha256(path) for path in source_paths.values()},
                     "dino_repository": repository, "official_dino_sha256": official}


def build_model(root, parent_path, device="cpu", checkpointed=True):
    """Return (model, inheritance receipt); new blocks contribute exactly zero."""
    root = Path(root).resolve()
    parent, receipt = _parent_payload(root, parent_path)
    model = base.make_model(root, ARCHITECTURE, "cpu")
    _strict_state(model, parent["adapter_state"], INHERITED_NAMES)
    named = dict(model.named_parameters())
    actual = {name for name, parameter in named.items() if parameter.requires_grad}
    if actual - set(parent["adapter_state"]) != NEW_NAMES or actual != ALL_NAMES:
        raise ValueError("Only blocks4-7 A/B may be newly added")
    for name in sorted(NEW_NAMES):
        value = named[name].detach()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite new adapter")
        if name.endswith(".lora_B") and torch.count_nonzero(value).item() != 0:
            raise ValueError("New LoRA B must be initialized to exactly zero")
        if name.endswith(".lora_A") and torch.count_nonzero(value).item() == 0:
            raise ValueError("New LoRA A must be randomly initialized")
    receipt.update(inherited_keys=sorted(INHERITED_NAMES), new_keys=sorted(NEW_NAMES),
                   inherited_parameters_exact=True, new_B_exactly_zero=True,
                   architecture=ARCHITECTURE)
    model = _attach_checkpointing(model, checkpointed).to(device).eval()
    return model, receipt


def _verify_config_environment(root, config):
    if config.get("format") != "task_lora_v4_config" or config.get("classes") != CLASSES:
        raise ValueError("V4 configuration format/class order differs")
    if config.get("architecture") != ARCHITECTURE:
        raise ValueError("V4 architecture differs")
    plan_path = root / "logs/lora_task_v4_plan.json"
    check_hash(plan_path, config["plan_sha256"])
    plan = read_json(plan_path)
    bound_fields = ("classes", "settings", "parent_checkpoint", "parent_checkpoint_sha256",
                    "source_pins", "input_receipt", "train_indices", "val_indices",
                    "dino_repository", "official_dino_sha256")
    if any(config.get(key) != plan.get(key) for key in bound_fields):
        raise ValueError("V4 configuration fields differ from the locked plan")
    pins = config.get("source_pins")
    if not isinstance(pins, dict):
        raise ValueError("Missing V4 source pins")
    required = {Path(__file__).resolve(), Path(base.__file__).resolve(),
                (root / "src/refine_lora.py").resolve(), Path(hse.__file__).resolve(),
                (root / "src/validate_cosine_fusion.py").resolve()}
    if not required.issubset({Path(path).resolve() for path in pins}):
        raise ValueError("V4 pins omit a required model/training implementation")
    for path, digest in pins.items():
        if not Path(path).is_absolute():
            raise ValueError("Source pins require absolute paths")
        check_hash(path, digest)
    if config.get("dino_repository") != base.repository_info(root):
        raise ValueError("V4 vendor repository changed")
    if config.get("official_dino_sha256") != base.verify_official_weights(root):
        raise ValueError("V4 official weights changed")
    check_hash(config["parent_checkpoint"], config["parent_checkpoint_sha256"])
    if Path(config["parent_checkpoint"]).resolve() != (root / "experiments/lora_refinement/best.pt").resolve():
        raise ValueError("V4 parent path differs")


def _verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("Unsupported task LoRA checkpoint/class order")
    saved = payload["config"]
    _verify_config_environment(root, saved)
    if config is not None and saved != config:
        raise ValueError("Resume configuration changed")
    for key in ("source_pins", "dino_repository", "official_dino_sha256"):
        if payload.get(key) != saved[key]:
            raise ValueError(f"V4 checkpoint/config binding differs: {key}")
    if payload.get("parent_sha") != saved["parent_checkpoint_sha256"]:
        raise ValueError("V4 parent binding differs")
    _validate_adapter_tensors(payload.get("adapter_state"))
    return saved


def load_checkpoint(root, path, device="cpu"):
    """Return (model, metadata), checking the complete new adapter/source format."""
    root = Path(root).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = _verify_payload(root, payload)
    model = base.make_model(root, config["architecture"], "cpu")
    _strict_state(model, payload["adapter_state"], ALL_NAMES)
    model = _attach_checkpointing(model, True).to(device).eval()
    excluded = {"adapter_state", "optimizer", "scaler", "rng", "ema_state"}
    return model, {key: value for key, value in payload.items() if key not in excluded}


def checked_plan(root):
    path = root / "logs/lora_task_v4_plan.json"
    plan = read_json(path)
    if plan.get("classes") != CLASSES:
        raise ValueError("Plan official class order differs")
    settings = plan["settings"]
    if any(settings.get(key) != value for key, value in FIXED_SETTINGS.items()):
        raise ValueError("Plan fixed training settings differ")
    batch_size = settings.get("batch_size")
    if batch_size not in (1, 2, 4, 8) or settings.get("accumulation") != 32 // batch_size:
        raise ValueError("GPU-probed microbatch must preserve an effective batch of32")
    config = {"format": "task_lora_v4_config", "classes": CLASSES,
              "architecture": ARCHITECTURE, "settings": settings,
              "plan_sha256": sha256(path), "parent_checkpoint": plan["parent_checkpoint"],
              "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
              "source_pins": plan["source_pins"], "input_receipt": plan["input_receipt"],
              "train_indices": plan["train_indices"], "val_indices": plan["val_indices"],
              "dino_repository": plan["dino_repository"],
              "official_dino_sha256": plan["official_dino_sha256"],
              "validation_batch_size": 4, "workers": 0, "sample_weights": "uniform",
              "activation_checkpointing": "adapted blocks only; non-reentrant; preserve RNG",
              "amp": "float16", "initial_grad_scale": 1024., "gradient_clip": 1.,
              "preprocessing": "RGB; bicubic resize224; ImageNet mean/std",
              "augmentation": "original FaceImages flip.5; rotation +/-5deg; brightness[.95,1.05]",
              "checkpoint_selection": "none; export every raw/EMA candidate for locked fusion selection",
              "internal_test_read": False, "external_answers_used": False,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    _verify_config_environment(root, config)
    return plan, config


@torch.inference_mode()
def validate_views(model, loader, device):
    model.eval()
    original, flipped, targets = [], [], []
    amp = torch.device(device).type == "cuda"
    started, last_progress = time.monotonic(), time.monotonic()
    for batch_index, (images, target) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            flip_logits = model(images.flip(-1))
        original.append(logits.float().softmax(1).cpu().numpy())
        flipped.append(flip_logits.float().softmax(1).cpu().numpy())
        targets.extend(target.tolist())
        if time.monotonic() - last_progress >= 45:
            print(json.dumps({"stage": "validate_lora_v4", "batch": batch_index,
                              "batches": len(loader), "seen": len(targets),
                              "elapsed_seconds": time.monotonic()-started}), flush=True)
            last_progress = time.monotonic()
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    for value in (original, flipped):
        if value.shape != (len(targets), 6) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid unweighted validation probability array")
        if not np.allclose(value.sum(1), 1., atol=2e-6, rtol=0):
            raise ValueError("Validation probabilities do not sum to one")
    return metrics(np.asarray(targets), original), metrics(np.asarray(targets), (original+flipped)*.5), original, flipped


def model_payload(model, config, epoch, source):
    return {"format": FORMAT, "config": config, "classes": CLASSES, "epoch": epoch,
            "weight_source": source, "adapter_state": base.adapter_state(model),
            "source_pins": config["source_pins"], "dino_repository": config["dino_repository"],
            "official_dino_sha256": config["official_dino_sha256"],
            "parent_sha": config["parent_checkpoint_sha256"],
            "train_indices": config["train_indices"], "val_indices": config["val_indices"]}


def evaluate_candidate(model, loader, device, output, config, epoch, source):
    name = f"epoch_{epoch:02d}_{source}"
    print(json.dumps({"stage": "validate_candidate", "name": name}), flush=True)
    original_metrics, tta_metrics, original, flipped = validate_views(model, loader, device)
    paths = {"model": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy",
             "flip": f"candidates/{name}_flip.npy"}
    payload = model_payload(model, config, epoch, source)
    payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics)
    base.save_torch(output / paths["model"], payload)
    save_array(output / paths["original"], original)
    save_array(output / paths["flip"], flipped)
    candidate = {"name": name, "epoch": epoch, "weight_source": source,
                 **paths, "path": paths["model"], "candidate_json": f"candidates/{name}.json",
                 "val_indices": config["val_indices"], "classes": CLASSES,
                 "artifact_sha256": {key: sha256(output / path) for key, path in paths.items()},
                 "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
                 "plan_sha256": config["plan_sha256"], "validation_weighted": False}
    base.write_json(output / candidate["candidate_json"], candidate)
    print(json.dumps({"stage": "candidate_complete", "name": name,
                      "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics}), flush=True)
    return candidate


def verify_candidate(root, output, candidate, config):
    if candidate.get("classes") != CLASSES or candidate.get("val_indices") != config["val_indices"]:
        raise ValueError("Candidate official classes/validation member order differs")
    if candidate.get("plan_sha256") != config["plan_sha256"] or candidate.get("validation_weighted") is not False:
        raise ValueError("Candidate plan/validation weighting differs")
    expected_name = f"epoch_{candidate['epoch']:02d}_{candidate['weight_source']}"
    if candidate.get("name") != expected_name or candidate.get("model") != f"candidates/{expected_name}.pt":
        raise ValueError("Candidate name/model filename binding differs")
    if candidate.get("path") != candidate["model"] or candidate.get("candidate_json") != f"candidates/{expected_name}.json":
        raise ValueError("Candidate checkpoint alias/JSON filename differs")
    if read_json(output / candidate["candidate_json"]) != candidate:
        raise ValueError("Candidate JSON differs from the committed pointer")
    for key in ("model", "original", "flip"):
        relative = Path(candidate[key])
        if relative.is_absolute() or ".." in relative.parts or not str(relative).startswith("candidates"):
            raise ValueError("Candidate file escapes the dedicated candidates directory")
        check_hash(output / relative, candidate["artifact_sha256"][key])
    payload = torch.load(output / candidate["model"], map_location="cpu", weights_only=True)
    _verify_payload(root, payload, config)
    if payload["epoch"] != candidate["epoch"] or payload["weight_source"] != candidate["weight_source"]:
        raise ValueError("Candidate checkpoint epoch/source binding differs")
    if payload["val_indices"] != config["val_indices"] or payload["train_indices"] != config["train_indices"]:
        raise ValueError("Candidate checkpoint train/validation members differ")
    for key in ("single_original_metrics", "single_tta_metrics"):
        if payload[key] != candidate[key]:
            raise ValueError("Candidate checkpoint metrics differ")
    original = np.load(output / candidate["original"], allow_pickle=False)
    flipped = np.load(output / candidate["flip"], allow_pickle=False)
    for value in (original, flipped):
        if value.dtype != np.float32 or value.shape != (len(config["val_indices"]), 6):
            raise ValueError("Candidate validation probability dtype/shape differs")
        if not np.isfinite(value).all() or np.any(value < 0) or not np.allclose(value.sum(1), 1., atol=2e-6, rtol=0):
            raise ValueError("Candidate contains invalid validation probabilities")
    records = val_rows(root)
    if [int(row["manifest_index"]) for row in records] != config["val_indices"]:
        raise ValueError("Candidate metric validation membership differs")
    targets = np.asarray([int(row["target"]) for row in records], dtype=np.int64)
    computed = {"single_original_metrics": metrics(targets, original),
                "single_tta_metrics": metrics(targets, (original+flipped)*.5)}
    if any(computed[key] != candidate[key] for key in computed):
        raise ValueError("Candidate saved probabilities do not reproduce its metrics")
    return candidate


def make_optimizer(model, settings):
    named = {name: value for name, value in model.named_parameters() if value.requires_grad}
    groups = [("new", NEW_NAMES, "lr_new"),
              ("inherited", INHERITED_NAMES - {"head.weight", "head.bias"}, "lr_inherited"),
              ("head", {"head.weight", "head.bias"}, "lr_head")]
    return torch.optim.AdamW([{"name": name, "params": [named[key] for key in sorted(keys)],
                              "lr": settings[field], "base_lr": settings[field]}
                             for name, keys, field in groups], weight_decay=settings["weight_decay"])


def set_learning_rate(optimizer, successful_steps, total_steps, settings):
    warmup = settings["warmup_successful_steps"]
    if successful_steps < warmup:
        factor = (successful_steps + 1) / max(1, warmup)
    else:
        position = min(1., (successful_steps-warmup) / max(1, total_steps-warmup-1))
        floor = settings["min_lr_ratio"]
        factor = floor + (1-floor) * .5 * (1 + math.cos(math.pi*position))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def optimizer_step(model, optimizer, scaler, ema):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    scaler.unscale_(optimizer)
    finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                 for parameter in parameters)
    if finite:
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        finite = bool(torch.isfinite(norm).item())
    if not finite and not scaler.is_enabled():
        raise RuntimeError("Nonfinite gradients in non-AMP training")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    successful = finite and scaler.get_scale() >= old_scale
    optimizer.zero_grad(set_to_none=True)
    ema.update(model, successful)
    return successful


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def commit_epoch(root, output, model, optimizer, scaler, ema, config, epoch,
                 successful_steps, cumulative_skips, history, candidates):
    _verify_config_environment(root, config)
    payload = {**model_payload(model, config, epoch, "raw"),
               "optimizer": cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
               "ema_state": ema.state_dict(), "rng": base.rng_state(),
               "successful_steps": successful_steps, "cumulative_skips": cumulative_skips,
               "history": history, "candidates": candidates}
    relative = f"epochs/epoch_{epoch:02d}.pt"
    checkpoint_path = output / relative
    base.save_torch(checkpoint_path, payload)
    # Immutable complete-epoch file makes a crash between last.pt and last.json
    # recoverable: last.json always identifies one fully committed checkpoint.
    atomic_copy(checkpoint_path, output / "last.pt")
    base.write_json(output / "last.json", {"epoch": epoch, "checkpoint": relative,
                                         "sha256": sha256(checkpoint_path),
                                         "plan_sha256": config["plan_sha256"]})
    base.write_json(output / "history.json", history)
    base.write_json(output / "candidates.json", candidates)


def resume_checkpoint(root, output, config):
    pointer = read_json(output / "last.json")
    if pointer.get("plan_sha256") != config["plan_sha256"]:
        raise ValueError("Resume pointer plan changed")
    epoch = pointer.get("epoch")
    if not isinstance(epoch, int) or not 0 <= epoch <= config["settings"]["epochs"]:
        raise ValueError("Resume pointer epoch invalid")
    if pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt":
        raise ValueError("Resume pointer must identify its complete epoch file")
    path = output / pointer["checkpoint"]
    check_hash(path, pointer["sha256"])
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _verify_payload(root, payload, config)
    if payload["epoch"] != epoch or len(payload["history"]) != epoch:
        raise ValueError("Resume epoch/history differs")
    expected_names = ["epoch_00_baseline"] + [f"epoch_{index:02d}_{source}"
                    for index in range(1, epoch+1) for source in ("raw", "ema")]
    if [candidate["name"] for candidate in payload["candidates"]] != expected_names:
        raise ValueError("Resume committed candidate list is incomplete")
    for candidate in payload["candidates"]:
        verify_candidate(root, output, candidate, config)
    if not (output / "last.pt").exists() or sha256(output / "last.pt") != pointer["sha256"]:
        atomic_copy(path, output / "last.pt")
    return payload


def write_summary(root, output, config, history, candidates, successful_steps, skips, ema_updates):
    if len(history) != config["settings"]["epochs"]:
        raise ValueError("Cannot finalize an incomplete training")
    for candidate in candidates:
        verify_candidate(root, output, candidate, config)
    if len(candidates) != 1 + 2*config["settings"]["epochs"]:
        raise ValueError("Final raw/EMA candidate count differs")
    if sum(row["successful_optimizer_steps"] for row in history) != successful_steps or ema_updates != successful_steps:
        raise ValueError("Optimizer/EMA successful-update counts differ")
    if any(row["successful_optimizer_steps"] <= 0 for row in history):
        raise ValueError("An epoch has no successful optimizer update")
    base.write_json(output / "history.json", history)
    base.write_json(output / "candidates.json", candidates)
    pointer = read_json(output / "last.json")
    paths = ["config.json", "history.json", "candidates.json", "last.json", "last.pt", pointer["checkpoint"]]
    for candidate in candidates:
        paths.extend([candidate["candidate_json"], candidate["model"], candidate["original"], candidate["flip"]])
    summary = {"format": "task_lora_v4_summary", "state": "complete", "classes": CLASSES,
               "epochs_completed": len(history), "candidate_count": len(candidates),
               "plan_sha256": config["plan_sha256"], "successful_optimizer_steps": successful_steps,
               "amp_skipped_steps": skips, "ema_updates": ema_updates,
               "validation_weighted": False, "train_sample_weights": "uniform",
               "internal_test_read": False, "external_answers_used": False,
               "candidates": candidates, "artifact_sha256": {path: sha256(output/path) for path in paths}}
    base.write_json(output / "summary.json", summary)
    print(json.dumps({"stage": "lora_task_v4_complete", "epochs_completed": len(history),
                      "candidate_count": len(candidates), "successful_optimizer_steps": successful_steps,
                      "amp_skipped_steps": skips}), flush=True)
    return summary


def run_training(root, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    plan, config = checked_plan(root)
    rows, receipt = hse.load_trainval(root)
    if receipt != plan["input_receipt"]:
        raise ValueError("Training/validation image receipt differs from locked inputs")
    for split in ("train", "val"):
        if [row["manifest_index"] for row in rows[split]] != config[split+"_indices"]:
            raise ValueError("Locked train/validation member order differs")
    parent, _ = _parent_payload(root, config["parent_checkpoint"])
    if parent["train_indices"] != config["train_indices"] or parent["val_indices"] != config["val_indices"]:
        raise ValueError("Parent training/validation membership differs")
    if parent["train_paths"] != [row["path"] for row in rows["train"]] or parent["val_paths"] != [row["path"] for row in rows["val"]]:
        raise ValueError("Parent training/validation paths differ")
    del parent
    output = root / "experiments/lora_task_v4"
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidates").mkdir(exist_ok=True)
    (output / "epochs").mkdir(exist_ok=True)
    if (output / "config.json").exists():
        if read_json(output / "config.json") != config:
            raise ValueError("Existing V4 configuration changed; preserve its artifacts")
    else:
        base.write_json(output / "config.json", config)
    base.seed_everything(config["settings"]["seed"])
    model, inheritance = build_model(root, config["parent_checkpoint"], device, True)
    if inheritance["parent_sha256"] != config["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed during model build")
    settings = config["settings"]
    optimizer = make_optimizer(model, settings)
    amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=amp)
    ema = AdapterEMA(model, settings["ema_decay"])
    steps_per_epoch = math.ceil(len(rows["train"])/32)
    total_steps = steps_per_epoch * settings["epochs"]
    val_loader = base.make_loader(rows["val"], 4, 0, settings["seed"], device=device)
    start_epoch = successful_steps = cumulative_skips = 0
    history, candidates = [], []
    if (output / "last.json").exists():
        saved = resume_checkpoint(root, output, config)
        _strict_state(model, saved["adapter_state"], ALL_NAMES)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        ema.load_state_dict(saved["ema_state"])
        base.restore_rng(saved["rng"])
        start_epoch, successful_steps = saved["epoch"], saved["successful_steps"]
        cumulative_skips = saved["cumulative_skips"]
        history, candidates = saved["history"], saved["candidates"]
        if ema.updates != successful_steps:
            raise ValueError("Resumed EMA update count differs from successful optimizer count")
        del saved
        print(json.dumps({"stage": "resume_lora_v4", "completed_epoch": start_epoch,
                          "successful_optimizer_steps": successful_steps}), flush=True)
    else:
        baseline_json = output / "candidates/epoch_00_baseline.json"
        if baseline_json.exists():
            candidate = verify_candidate(root, output, read_json(baseline_json), config)
            payload = torch.load(output / candidate["model"], map_location="cpu", weights_only=True)
            current = base.adapter_state(model)
            if any(not torch.equal(current[key], payload["adapter_state"][key]) for key in INHERITED_NAMES):
                raise ValueError("Uncommitted baseline differs from the original inherited parent")
            if any(torch.count_nonzero(payload["adapter_state"][key]).item() for key in NEW_NAMES if key.endswith(".lora_B")):
                raise ValueError("Uncommitted baseline new B is not zero")
            _strict_state(model, payload["adapter_state"], ALL_NAMES)
            # Reusing a fully evaluated baseline also restores its original new A.
            ema = AdapterEMA(model, settings["ema_decay"])
            del payload
        else:
            candidate = evaluate_candidate(model, val_loader, device, output, config, 0, "baseline")
        candidates.append(candidate)
        commit_epoch(root, output, model, optimizer, scaler, ema, config, 0,
                     successful_steps, cumulative_skips, history, candidates)
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()
                       if not parameter.requires_grad}
    if start_epoch == settings["epochs"]:
        return write_summary(root, output, config, history, candidates, successful_steps,
                             cumulative_skips, ema.updates)
    for epoch in range(start_epoch+1, settings["epochs"]+1):
        started, last_progress = time.monotonic(), time.monotonic()
        data_epoch = settings["augmentation_epoch_offset"] + epoch-1
        loader = base.make_loader(rows["train"], settings["batch_size"], 0, settings["seed"],
                                  epoch=data_epoch, training=True, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = correct = seen = epoch_steps = epoch_skips = 0
        for batch_index, (images, targets) in enumerate(loader):
            group_start = (batch_index // settings["accumulation"]) * 32
            group_images = min(32, len(loader.dataset)-group_start)
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                raw_loss = F.cross_entropy(logits, targets, reduction="sum", label_smoothing=settings["label_smoothing"])
                loss = raw_loss/group_images
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite loss at epoch{epoch}, batch{batch_index+1}")
            scaler.scale(loss).backward()
            loss_sum += raw_loss.detach().float().item()
            correct += (logits.detach().argmax(1)==targets).sum().item()
            seen += len(targets)
            if (batch_index+1) % settings["accumulation"] == 0 or batch_index+1 == len(loader):
                set_learning_rate(optimizer, successful_steps, total_steps, settings)
                successful = optimizer_step(model, optimizer, scaler, ema)
                successful_steps += int(successful)
                epoch_steps += int(successful)
                epoch_skips += int(not successful)
                cumulative_skips += int(not successful)
                if epoch_skips >= 25 and epoch_steps == 0:
                    raise RuntimeError("25 AMP skips without a successful optimizer update")
            if time.monotonic()-last_progress >= 45:
                print(json.dumps({"stage": "train_lora_v4", "epoch": epoch, "batch": batch_index+1,
                                  "batches": len(loader), "seen": seen, "train_loss": loss_sum/seen,
                                  "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": epoch_skips,
                                  "cumulative_optimizer_steps": successful_steps,
                                  "elapsed_seconds": time.monotonic()-started}), flush=True)
                last_progress = time.monotonic()
        if seen != len(rows["train"]) or epoch_steps <= 0:
            raise RuntimeError("Epoch did not visit every training image or did not update")
        if any(parameter._version != frozen_versions[name] or parameter.grad is not None
               for name, parameter in model.named_parameters() if not parameter.requires_grad):
            raise RuntimeError("Frozen DINOv2 weights changed or received gradients")
        raw_candidate = evaluate_candidate(model, val_loader, device, output, config, epoch, "raw")
        raw_state = base.adapter_state(model)
        try:
            _strict_state(model, ema.state_dict()["shadow"], ALL_NAMES)
            ema_candidate = evaluate_candidate(model, val_loader, device, output, config, epoch, "ema")
        finally:
            _strict_state(model, raw_state, ALL_NAMES)
        candidates.extend((raw_candidate, ema_candidate))
        entry = {"epoch": epoch, "train_loss": loss_sum/seen, "train_accuracy": correct/seen,
                 "seen": seen, "augmentation_epoch": data_epoch,
                 "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": epoch_skips,
                 "cumulative_optimizer_steps": successful_steps, "ema_updates": ema.updates,
                 "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-started,
                 "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                 "raw_original_metrics": raw_candidate["single_original_metrics"],
                 "raw_tta_metrics": raw_candidate["single_tta_metrics"],
                 "ema_original_metrics": ema_candidate["single_original_metrics"],
                 "ema_tta_metrics": ema_candidate["single_tta_metrics"]}
        history.append(entry)
        commit_epoch(root, output, model, optimizer, scaler, ema, config, epoch,
                     successful_steps, cumulative_skips, history, candidates)
        print(json.dumps({"stage": "epoch_committed", **entry}), flush=True)
    return write_summary(root, output, config, history, candidates, successful_steps,
                         cumulative_skips, ema.updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_training(args.project_root, args.device)


if __name__ == "__main__":
    main()
