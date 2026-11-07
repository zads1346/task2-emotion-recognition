"""Matched HSE AdamW/SAM continuation; train/validation only, no test access.

SAM reuses materialized augmented groups and replays Torch RNG. Manual dynamic
loss scaling avoids two-unscale GradScaler state-machine ambiguity. Only an
atomic complete-epoch checkpoint may be resumed.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import train_hse_specialist as h

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "hse_sam_v7"
CLASSES = list(h.CLASSES)
MODES = ("adamw_control", "sam")
MODE_SETTINGS = {"adamw_control": {"rho": 0.0}, "sam": {"rho": 0.05}}
SETTINGS = {"epochs": 4, "seed": 42, "augmentation_epoch_offset": 70,
    "batch_size": 16, "accumulation": 2, "effective_batch_size": 32,
    "workers": 0, "cpu_threads": 4, "image_size": 224, "validation_batch_size": 16,
    "backbone_lr": 5e-6, "head_lr": 5e-5, "weight_decay": .01,
    "label_smoothing": 0.0, "gradient_clip": 1.0, "warmup_successful_steps": 50,
    "min_lr_ratio": .1, "ema_decay": .995, "amp_initial_scale": 1024.,
    "amp_growth_interval": 2000, "batchnorm_running_stats": "frozen_eval",
    "sample_weights": "uniform1", "mixup": False, "cutmix": False,
    "sam_adaptive": False, "sampling": "same epoch shuffle and augmentation for both arms"}


class ManualGradScaler:
    """Serializable dynamic scaling with one decision per logical optimizer group."""
    def __init__(self, enabled=True, init_scale=1024., growth_interval=2000):
        if not math.isfinite(init_scale) or init_scale < 1 or growth_interval < 1:
            raise ValueError("Invalid manual gradient-scale settings")
        self.enabled = bool(enabled)
        self.scale = float(init_scale if self.enabled else 1.)
        self.growth_interval = int(growth_interval)
        self.growth_tracker = 0

    def get_scale(self):
        return self.scale

    def update(self, success):
        if not self.enabled:
            return
        if success:
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * 2., 2. ** 24)
                self.growth_tracker = 0
        else:
            self.scale = max(self.scale * .5, 1.)
            self.growth_tracker = 0

    def state_dict(self):
        return {"enabled": self.enabled, "scale": self.scale,
                "growth_interval": self.growth_interval, "growth_tracker": self.growth_tracker}

    def load_state_dict(self, state):
        if bool(state["enabled"]) != self.enabled or int(state["growth_interval"]) != self.growth_interval:
            raise ValueError("Manual scaler mode/growth interval differs on resume")
        scale, tracker = float(state["scale"]), int(state["growth_tracker"])
        if not math.isfinite(scale) or not 1 <= scale <= 2. ** 24 or not 0 <= tracker < self.growth_interval:
            raise ValueError("Invalid saved manual gradient scale")
        self.scale, self.growth_tracker = scale, tracker


class FullModelEMA:
    """Average learned parameters; copy all buffers after successful updates.

    BN running statistics remain bitwise frozen throughout training. Shadow is
    a complete state_dict so the evaluated EMA can be saved without ambiguity.
    """
    def __init__(self, model, decay=.995):
        if not 0 <= decay < 1:
            raise ValueError("Invalid EMA decay")
        self.decay, self.updates = float(decay), 0
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.parameter_names = set(dict(model.named_parameters()))

    @torch.no_grad()
    def update(self, model):
        for key, source in model.state_dict().items():
            target = self.shadow[key]
            if key in self.parameter_names:
                target.lerp_(source.detach(), 1. - self.decay)
            else:
                target.copy_(source.detach())
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": h.cpu_tree(self.shadow)}

    def load_state_dict(self, state):
        if float(state["decay"]) != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA configuration/layout differs on resume")
        for key, target in self.shadow.items():
            value = state["shadow"][key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError("EMA tensor layout differs")
            target.copy_(value.to(target.device))
        self.updates = int(state["updates"])

    @contextmanager
    def apply(self, model):
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        training_modes = [(module, module.training) for module in model.modules()]
        try:
            model.load_state_dict(self.shadow, strict=True)
            model.eval()
            yield model
        finally:
            model.load_state_dict(original, strict=True)
            for module, flag in training_modes:
                module.training = flag


def parameter_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}


@torch.no_grad()
def restore_parameters(model, snapshot):
    parameters = dict(model.named_parameters())
    for name, original in snapshot.items():
        parameters[name].copy_(original)


def torch_rng(device):
    result = {"cpu": torch.get_rng_state().clone()}
    if torch.device(device).type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def restore_torch_rng(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _unscale_and_norm(model, scale):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No gradients were produced")
    torch._foreach_div_(gradients, scale)
    result = torch.stack([torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients]).norm()
    value = float(result)
    return value if math.isfinite(value) else None


def group_step(model, optimizer, scaler, ema, images, targets,
               device="cuda", batch_size=16, rho=.05):
    """Update one actual1..32-image group; rho0 is the matched AdamW control.

    Two-pass gradients are normalized by actual group size, including a tail30.
    Inputs are already augmented; no dataset loader is called inside this API.
    Nonfinite passes/grads skip the whole group, halve scale once, and leave the
    optimizer/EMA untouched. Ordinary exceptions roll perturbations back and
    propagate so the last complete-epoch checkpoint remains authoritative.
    """
    count = len(targets)
    if len(images) != count or not 1 <= count <= 32 or batch_size < 1 or not math.isfinite(rho) or rho < 0:
        raise ValueError("Invalid logical group, microbatch or SAM rho")
    freeze_batchnorm(model)
    optimizer.zero_grad(set_to_none=True)
    amp = torch.device(device).type == "cuda"
    scale = scaler.get_scale()
    first_before = torch_rng(device)
    first_after, snapshot = None, None
    microbatches = math.ceil(count / batch_size)
    first_loss = second_loss = correct = 0.
    first_norm = second_norm = None
    perturbation_norm = 0.
    skip_reason = None

    def backward_pass():
        loss_sum, right = 0., 0
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            x = images[start:stop].to(device, non_blocking=True)
            y = targets[start:stop].to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, reduction="sum", label_smoothing=0.)
            if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
                return None, None
            (loss * (scale / count)).backward()
            loss_sum += float(loss.detach())
            right += int((logits.detach().argmax(1) == y).sum())
        return loss_sum, right

    try:
        first_loss, correct = backward_pass()
        first_after = torch_rng(device)
        first_norm = None if first_loss is None else _unscale_and_norm(model, scale)
        if first_norm is None:
            skip_reason = "first_pass_nonfinite"
        elif rho > 0:
            snapshot = parameter_snapshot(model)
            with torch.no_grad():
                squares = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    perturbation = parameter.grad * (rho / (first_norm + 1e-12))
                    parameter.copy_(snapshot[name] + perturbation)
                    squares.append((parameter.detach() - snapshot[name]).float().norm())
                perturbation_norm = float(torch.stack(squares).norm())
            optimizer.zero_grad(set_to_none=True)
            restore_torch_rng(first_before, device)
            second_loss, _ = backward_pass()
            second_norm = None if second_loss is None else _unscale_and_norm(model, scale)
            if second_norm is None:
                skip_reason = "second_pass_nonfinite"
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if snapshot is not None:
            restore_parameters(model, snapshot)
        if first_after is not None:
            restore_torch_rng(first_after, device)
        if skip_reason is not None:
            optimizer.zero_grad(set_to_none=True)

    if skip_reason is not None:
        scaler.update(False)
        return {"success": False, "seen": count, "loss_sum": first_loss,
                "correct": correct, "microbatches": microbatches, "second_loss_sum": second_loss,
                "first_grad_norm": first_norm, "second_grad_norm": second_norm,
                "perturbation_norm": perturbation_norm, "skip_reason": skip_reason, "scale": scaler.get_scale()}
    # Original parameters have already been restored. AdamW decay acts at w, not w+epsilon.
    norm = second_norm if rho > 0 else first_norm
    factor = min(1., SETTINGS["gradient_clip"] / (norm + 1e-12))
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all().item():
        raise RuntimeError("AdamW produced nonfinite parameters; abort uncommitted epoch and resume the previous one")
    if ema is not None:
        ema.update(model)
    scaler.update(True)
    return {"success": True, "seen": count, "loss_sum": first_loss,
            "correct": int(correct), "microbatches": microbatches, "second_loss_sum": second_loss,
            "first_grad_norm": first_norm, "second_grad_norm": second_norm,
            "perturbation_norm": perturbation_norm, "skip_reason": None, "scale": scaler.get_scale()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_copy(source, destination):
    import shutil
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def checked_plan(root, mode):
    import prepare_hse_sam_v7 as preparation
    root = Path(root).resolve()
    if mode not in MODES:
        raise ValueError("Unknown HSE V7 mode")
    plan = preparation.verify_plan(root)
    fields = ("classes", "settings", "mode_settings", "source_pins", "input_receipt",
              "original_train_indices", "excluded_train_indices", "train_indices", "val_indices",
              "filtered_membership_sha256", "audit_summary_sha256", "parent_checkpoint",
              "parent_checkpoint_sha256", "baseline_submission_sha256", "baseline_correct_count",
              "model_provenance")
    config = {"format": FORMAT + "_config", "mode": mode,
              **{key: plan[key] for key in fields}, "plan_sha256": h.sha256(root/"logs/hse_sam_v7_plan.json"),
              "rho": MODE_SETTINGS[mode]["rho"], "validation_weighted": False,
              "effective_train_rows": 21246, "train_sample_weights": "uniform1",
              "validation_batch_size": SETTINGS["validation_batch_size"],
              "preprocessing": h.FIXED["preprocessing"], "augmentation": h.FIXED["augmentation"],
              "loss_definition": "standard CE(label_smoothing=0), sum per microbatch / actual logical group count",
              "amp": "float16 forward; manually scaled backward; FP32 parameter gradients",
              "ema_buffer_policy": "copy buffers only on successful update; BN running stats frozen",
              "internal_test_read": False, "external_answers_used": False,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    return plan, config


def load_rows(root, config):
    rows, receipt = h.load_trainval(root)
    if receipt != config["input_receipt"]:
        raise ValueError("Original train/validation content receipt differs from the locked plan")
    if [row["manifest_index"] for row in rows["train"]] != config["original_train_indices"]:
        raise ValueError("Original training membership changed")
    excluded = set(config["excluded_train_indices"])
    rows["train"] = [row for row in rows["train"] if row["manifest_index"] not in excluded]
    if (h.signature(rows["train"]) != config["filtered_membership_sha256"]
            or len(rows["train"]) != 21246 or len(rows["val"]) != 2656):
        raise ValueError("Filtered training membership differs")
    for split in ("train", "val"):
        if [row["manifest_index"] for row in rows[split]] != config[split + "_indices"]:
            raise ValueError("Train/validation row order differs")
    return rows


def model_payload(model, config, epoch, source):
    fields = ("classes", "mode", "source_pins", "original_train_indices", "excluded_train_indices",
              "train_indices", "val_indices", "audit_summary_sha256", "model_provenance")
    return {"format": FORMAT, "config": config, **{key: config[key] for key in fields},
            "epoch": epoch, "weight_source": source, "model_state": h.cpu_tree(model.state_dict()),
            "parent_sha": config["parent_checkpoint_sha256"], "plan_sha256": config["plan_sha256"],
            "manifest_sha256": config["input_receipt"]["manifest_sha256"]}


def _verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("Unsupported HSE V7 checkpoint/class order")
    _, expected = checked_plan(root, payload.get("mode"))
    if payload.get("config") != expected or (config is not None and expected != config):
        raise ValueError("Checkpoint configuration differs from the locked V7 plan")
    for key in ("source_pins", "original_train_indices", "excluded_train_indices", "train_indices",
                "val_indices", "audit_summary_sha256", "model_provenance", "plan_sha256"):
        if payload.get(key) != expected[key]:
            raise ValueError("Checkpoint/config binding differs: " + key)
    if payload.get("parent_sha") != expected["parent_checkpoint_sha256"]:
        raise ValueError("Checkpoint parent differs")
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError("Checkpoint has no complete model_state")
    return expected


def load_checkpoint(root, path, device="cpu"):
    root = Path(root).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _verify_payload(root, payload)
    model = h.model_api().load_trained_model(root, payload, device=device)
    excluded = {"model_state", "optimizer", "scaler", "rng", "ema_state"}
    return model, {key: value for key, value in payload.items() if key not in excluded}


def make_loader(rows, training, epoch, settings):
    augmentation_epoch = settings["augmentation_epoch_offset"] + epoch
    seed = settings["seed"] + 1009 * augmentation_epoch + (0 if training else 1000003)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(h.ExpressionImages(rows, training),
                      batch_size=settings["effective_batch_size"] if training else settings["validation_batch_size"],
                      shuffle=training, num_workers=0, pin_memory=True, generator=generator, drop_last=False)


def set_learning_rate(optimizer, successful_steps, total_steps, settings):
    warmup = settings["warmup_successful_steps"]
    if successful_steps < warmup:
        factor = (successful_steps + 1) / warmup
    else:
        progress = min(1., max(0., (successful_steps - warmup) / max(1, total_steps - warmup - 1)))
        factor = settings["min_lr_ratio"] + (1 - settings["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = settings[group["name"] + "_lr"] * factor


def make_optimizer(model, settings):
    head = h.classifier_parameters(model)
    ids = {id(parameter) for parameter in head}
    return torch.optim.AdamW([
        {"params": [parameter for parameter in model.parameters() if id(parameter) not in ids],
         "lr": settings["backbone_lr"], "name": "backbone"},
        {"params": head, "lr": settings["head_lr"], "name": "head"}],
        weight_decay=settings["weight_decay"])


@torch.inference_mode()
def validate_views(model, rows, epoch, device, settings):
    import validate_cosine_fusion as c
    model.eval()
    original, flipped = [], []
    amp = torch.device(device).type == "cuda"
    for images, _ in make_loader(rows, False, epoch, settings):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            first, second = model(images), model(images.flip(-1))
        original.append(first.float().softmax(1).cpu().numpy())
        flipped.append(second.float().softmax(1).cpu().numpy())
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    for values in (original, flipped):
        if values.dtype != np.float32 or values.shape != (len(rows), 6):
            raise ValueError("Validation probabilities must be float32 Nx6")
        c.check_probability(values, len(rows))
    targets = np.asarray([row["target"] for row in rows])
    return c.metrics(targets, original), c.metrics(targets, (original + flipped) * .5), original, flipped


def evaluate_candidate(model, rows, device, output, config, epoch, source):
    name = f"epoch_{epoch:02d}_{source}"
    original_metrics, tta_metrics, original, flipped = validate_views(model, rows, epoch, device, config["settings"])
    paths = {"model": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy",
             "flip": f"candidates/{name}_flip.npy"}
    payload = model_payload(model, config, epoch, source)
    payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics)
    h.save_torch(output/paths["model"], payload)
    h.save_array(output/paths["original"], original)
    h.save_array(output/paths["flip"], flipped)
    fields = ("classes", "mode", "source_pins", "original_train_indices", "excluded_train_indices",
              "train_indices", "val_indices", "audit_summary_sha256", "plan_sha256")
    candidate = {"name": name, "epoch": epoch, "weight_source": source,
                 **{key: config[key] for key in fields}, **paths, "path": paths["model"],
                 "candidate_json": f"candidates/{name}.json",
                 "artifact_sha256": {key: h.sha256(output/path) for key, path in paths.items()},
                 "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
                 "parent_sha": config["parent_checkpoint_sha256"], "validation_weighted": False}
    h.write_json(output/candidate["candidate_json"], candidate)
    print(json.dumps({"stage": "candidate_complete", "mode": config["mode"], "name": name,
                      "single_original_metrics": {key: original_metrics[key] for key in ("accuracy", "macro_f1", "correct_count")}}), flush=True)
    return candidate


def verify_candidate(root, output, candidate, config):
    import validate_cosine_fusion as c
    for key in ("classes", "mode", "source_pins", "train_indices", "val_indices", "original_train_indices",
                "excluded_train_indices", "audit_summary_sha256", "plan_sha256"):
        if candidate.get(key) != config[key]:
            raise ValueError("Candidate metadata binding differs: " + key)
    if candidate.get("parent_sha") != config["parent_checkpoint_sha256"] or candidate.get("validation_weighted") is not False:
        raise ValueError("Candidate parent or validation weighting differs")
    if read_json(output/candidate["candidate_json"]) != candidate:
        raise ValueError("Candidate JSON differs from committed index")
    for key in ("model", "original", "flip"):
        h.check_hash(output/candidate[key], candidate["artifact_sha256"][key])
    original = np.load(output/candidate["original"], allow_pickle=False)
    flipped = np.load(output/candidate["flip"], allow_pickle=False)
    if original.dtype != np.float32 or flipped.dtype != np.float32:
        raise ValueError("Candidate probabilities are not float32")
    c.check_probability(original, 2656); c.check_probability(flipped, 2656)


def commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates):
    payload = {**model_payload(model, config, epoch, "raw"),
               "optimizer": h.cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
               "ema_state": ema.state_dict(), "rng": h.rng_state(), "counters": dict(counters),
               "successful_steps": counters["successful_steps"],
               "cumulative_skips": counters["amp_skipped_steps"], "zero_weight_groups": 0,
               "history": list(history), "candidates": list(candidates)}
    relative = f"epochs/epoch_{epoch:02d}.pt"
    h.save_torch(output/relative, payload)
    # last.json is the only commit marker. A partial next epoch never advances it.
    atomic_copy(output/relative, output/"last.pt")
    h.write_json(output/"last.json", {"epoch": epoch, "checkpoint": relative,
                 "sha256": h.sha256(output/relative), "plan_sha256": config["plan_sha256"], "mode": config["mode"]})
    h.write_json(output/"history.json", history)
    h.write_json(output/"candidates.json", candidates)


def resume_checkpoint(root, output, config):
    pointer = read_json(output/"last.json")
    epoch = pointer.get("epoch")
    if (not isinstance(epoch, int) or not 0 <= epoch <= 4 or pointer.get("mode") != config["mode"]
            or pointer.get("plan_sha256") != config["plan_sha256"]
            or pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt"):
        raise ValueError("Invalid complete-epoch resume pointer")
    h.check_hash(output/pointer["checkpoint"], pointer["sha256"])
    payload = torch.load(output/pointer["checkpoint"], map_location="cpu", weights_only=True)
    _verify_payload(root, payload, config)
    expected = ["epoch_00_baseline"] + [f"epoch_{index:02d}_{kind}" for index in range(1, epoch+1) for kind in ("raw", "ema")]
    if payload["epoch"] != epoch or len(payload["history"]) != epoch or [value["name"] for value in payload["candidates"]] != expected:
        raise ValueError("Resume candidate/history sequence differs")
    if payload["ema_state"]["updates"] != payload["counters"]["successful_steps"]:
        raise ValueError("Resume optimizer and EMA counts differ")
    for candidate in payload["candidates"]:
        verify_candidate(root, output, candidate, config)
    if not (output/"last.pt").is_file() or h.sha256(output/"last.pt") != pointer["sha256"]:
        atomic_copy(output/pointer["checkpoint"], output/"last.pt")
    # A crash after advancing last.json can leave these derived indexes stale.
    h.write_json(output/"history.json", payload["history"])
    h.write_json(output/"candidates.json", payload["candidates"])
    return payload


def write_summary(root, output, config, history, candidates, counters, ema_updates):
    if len(history) != 4 or len(candidates) != 9 or ema_updates != counters["successful_steps"]:
        raise ValueError("Cannot summarize an incomplete V7 arm")
    if (sum(value["successful_optimizer_steps"] for value in history) != counters["successful_steps"]
            or sum(value["amp_skipped_steps"] for value in history) != counters["amp_skipped_steps"]
            or any(value["seen"] != 21246 or value["successful_optimizer_steps"] <= 0 for value in history)):
        raise ValueError("Training membership/update counters differ")
    for candidate in candidates:
        verify_candidate(root, output, candidate, config)
    pointer = read_json(output/"last.json")
    paths = ["config.json", "history.json", "candidates.json", "last.json", "last.pt"]
    paths += [f"epochs/epoch_{index:02d}.pt" for index in range(5)]
    for candidate in candidates:
        paths += [candidate[key] for key in ("candidate_json", "model", "original", "flip")]
    summary = {"format": FORMAT + "_summary", "state": "complete", "classes": CLASSES,
               "mode": config["mode"], "epochs_completed": 4, "candidate_count": 9,
               "config": config, "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
               "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
               "audit_summary_sha256": config["audit_summary_sha256"],
               "effective_train_rows": 21246, "original_train_rows": 21248,
               **{key: config[key] for key in ("original_train_indices", "excluded_train_indices", "train_indices", "val_indices")},
               "mode_settings": MODE_SETTINGS, "rho": config["rho"], "counters": counters,
               "successful_optimizer_steps": counters["successful_steps"], "ema_updates": ema_updates,
               "amp_skipped_steps": counters["amp_skipped_steps"], "zero_weight_groups": 0,
               "validation_weighted": False, "train_sample_weights": "uniform1", "class_normalization": False,
               "internal_test_read": False, "external_answers_used": False,
               "candidates": candidates, "artifact_sha256": {relative: h.sha256(output/relative) for relative in paths}}
    h.check_hash(output/pointer["checkpoint"], pointer["sha256"])
    h.write_json(output/"summary.json", summary)
    return summary


def _run_training(root, mode, device):
    plan, config = checked_plan(root, mode)
    rows = load_rows(root, config)
    output = root/"experiments/hse_sam_v7"/mode
    output.mkdir(parents=True, exist_ok=True)
    for name in ("epochs", "candidates"):
        (output/name).mkdir(exist_ok=True)
    if (output/"config.json").exists() and read_json(output/"config.json") != config:
        raise ValueError("Existing V7 configuration differs; refusing to overwrite")
    h.write_json(output/"config.json", config)
    model = h.model_api().load_trained_model(root, config["parent_checkpoint"], device=device)
    model.requires_grad_(True)
    optimizer = make_optimizer(model, config["settings"])
    scaler = ManualGradScaler(torch.device(device).type == "cuda", config["settings"]["amp_initial_scale"],
                              config["settings"]["amp_growth_interval"])
    ema = FullModelEMA(model, config["settings"]["ema_decay"])
    history, candidates = [], []
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    epoch = 0
    # Apply backend determinism on both entry paths; resume then restores saved RNG.
    h.seed_everything(config["settings"]["seed"])
    if (output/"last.json").exists():
        saved = resume_checkpoint(root, output, config)
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        ema.load_state_dict(saved["ema_state"])
        epoch, counters, history, candidates = saved["epoch"], saved["counters"], saved["history"], saved["candidates"]
        h.restore_rng(saved["rng"])
        print(json.dumps({"stage": "resume", "mode": mode, "committed_epoch": epoch}), flush=True)
    else:
        candidates.append(evaluate_candidate(model, rows["val"], device, output, config, 0, "baseline"))
        commit_epoch(output, model, optimizer, scaler, ema, config, 0, counters, history, candidates)
    total_steps = math.ceil(len(rows["train"]) / 32) * 4
    for epoch in range(epoch+1, 5):
        model.train(); freeze_batchnorm(model)
        before = dict(counters)
        seen = correct = loss_count = 0
        loss_sum = 0.
        start = last_progress = time.monotonic()
        loader = make_loader(rows["train"], True, epoch, config["settings"])
        for group_index, (images, targets) in enumerate(loader, 1):
            set_learning_rate(optimizer, counters["successful_steps"], total_steps, config["settings"])
            result = group_step(model, optimizer, scaler, ema, images, targets, device=device,
                                batch_size=config["settings"]["batch_size"], rho=config["rho"])
            seen += result["seen"]
            if result["loss_sum"] is not None:
                loss_sum += result["loss_sum"]; loss_count += result["seen"]
                correct += result["correct"]
            counters["successful_steps"] += int(result["success"])
            counters["amp_skipped_steps"] += int(not result["success"])
            if time.monotonic()-last_progress >= 45 or group_index == len(loader):
                print(json.dumps({"stage": "train", "mode": mode, "epoch": epoch, "group": group_index,
                                  "groups": len(loader), "seen": seen, "successful_steps": counters["successful_steps"],
                                  "amp_skipped_steps": counters["amp_skipped_steps"], "scale": scaler.get_scale(),
                                  "elapsed_seconds": time.monotonic()-start}), flush=True)
                last_progress = time.monotonic()
        successes = counters["successful_steps"] - before["successful_steps"]
        if not successes or seen != 21246:
            raise RuntimeError("An epoch missed training members or had no finite optimizer update; epoch not committed")
        raw = evaluate_candidate(model, rows["val"], device, output, config, epoch, "raw")
        with ema.apply(model):
            averaged = evaluate_candidate(model, rows["val"], device, output, config, epoch, "ema")
        candidates.extend([raw, averaged])
        record = {"epoch": epoch, "seen": seen, "train_loss": loss_sum/loss_count,
                  "train_ce": loss_sum/loss_count, "train_accuracy": correct/loss_count,
                  "successful_optimizer_steps": successes,
                  "amp_skipped_steps": counters["amp_skipped_steps"]-before["amp_skipped_steps"],
                  "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"],
                  "ema_updates": ema.updates, "augmentation_epoch": config["settings"]["augmentation_epoch_offset"]+epoch,
                  "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                  "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-start,
                  "raw_original_metrics": raw["single_original_metrics"], "raw_tta_metrics": raw["single_tta_metrics"],
                  "ema_original_metrics": averaged["single_original_metrics"], "ema_tta_metrics": averaged["single_tta_metrics"]}
        history.append(record)
        checked_plan(root, mode)
        commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates)
    summary = write_summary(root, output, config, history, candidates, counters, ema.updates)
    print(json.dumps({"stage": "arm_complete", "mode": mode, "epochs_completed": 4,
                      "successful_steps": counters["successful_steps"], "candidate_count": 9}), flush=True)
    return summary


def run_training(root, mode, device="cuda"):
    """A shared arm lock also prevents accidental concurrent manual arm launches."""
    import msvcrt
    root = Path(root).resolve()
    lock = (root/"logs/hse_sam_v7_training.lock").open("a+b")
    lock.seek(0, 2)
    if lock.tell() == 0:
        lock.write(b"0"); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError("Another V7 arm is using the training device")
    try:
        return _run_training(root, mode, device)
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no implicit training-device fallback")
    run_training(args.project_root, args.mode, args.device)


if __name__ == "__main__":
    main()
