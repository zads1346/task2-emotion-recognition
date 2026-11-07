"""Four-epoch low-LR refinement of a saved DINOv2 LoRA model, train/val only.

Validation selects among the parent checkpoint and each epoch's raw/EMA weights
using original/flip probability averaging. No test images or labels are used.
The original model source and this refinement source have separate provenance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

import train_lora as base

FORMAT = "dinov2_lora_refinement_v1"
CLASSES = base.CLASSES
ROOT = Path(__file__).resolve().parents[1]


class AdapterEMA:
    """EMA of trainable adapter/head parameters; never copies the full backbone."""
    def __init__(self, model, decay=0.995):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0,1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model, successful):
        if not successful:
            return
        named = dict(model.named_parameters())
        for name, value in self.shadow.items():
            value.mul_(self.decay).add_(named[name].detach(), alpha=1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates,
                "shadow": {name: p.detach().cpu().clone() for name, p in self.shadow.items()}}

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA configuration/parameter names changed")
        for name, value in state["shadow"].items():
            if value.shape != self.shadow[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid EMA state: {name}")
            self.shadow[name].copy_(value)
        self.updates = int(state["updates"])


def save_array(path, array):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.save(f, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def load_refinement_checkpoint(path, root, device="cpu"):
    """Return (model, metadata), with official logits order from CLASSES."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != FORMAT or checkpoint.get("classes") != CLASSES:
        raise ValueError("Unsupported refinement checkpoint or class order")
    provenance = checkpoint["provenance"]
    if provenance["original_model_source_sha256"] != base.sha256(base.__file__):
        raise ValueError("Original model implementation changed")
    if provenance["refinement_source_sha256"] != base.sha256(__file__):
        raise ValueError("Refinement implementation changed")
    if provenance["official_weight_sha256"] != base.verify_official_weights(root):
        raise ValueError("Official pretrained backbone changed")
    for key, value in base.repository_info(root).items():
        if provenance[key] != value:
            raise ValueError("DINOv2 vendor implementation changed")
    model = base.make_model(root, checkpoint["config"]["architecture"], device)
    base.restore_adapter(model, checkpoint["adapter_state"])
    model.eval()
    return model, {key: value for key, value in checkpoint.items()
                   if key not in {"adapter_state", "optimizer", "scaler", "rng", "ema_state"}}


def make_eval_transform():
    return base.make_eval_transform()


@torch.inference_mode()
def validate_views(model, loader, device):
    model.eval()
    original, flipped, targets = [], [], []
    amp = torch.device(device).type == "cuda"
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            flip_logits = model(images.flip(-1))
        original.append(logits.float().softmax(1).cpu().numpy())
        flipped.append(flip_logits.float().softmax(1).cpu().numpy())
        targets.extend(target.tolist())
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    metrics = base.classification_metrics(targets, (original + flipped) * 0.5)
    return metrics, original, flipped


def common_checkpoint(model, config, provenance, records, epoch, weight_source):
    return {"format": FORMAT, "classes": CLASSES, "config": config, "provenance": provenance,
            "epoch": epoch, "weight_source": weight_source, "validation_tta": True,
            "adapter_state": base.adapter_state(model),
            "train_indices": [r["manifest_index"] for r in records["train"]],
            "val_indices": [r["manifest_index"] for r in records["val"]],
            "train_paths": [r["path"] for r in records["train"]],
            "val_paths": [r["path"] for r in records["val"]]}


def sync_best(output, candidate, records):
    for source_key, output_name in [("checkpoint", "best.pt"), ("original", "val_original.npy"),
                                    ("flip", "val_flip.npy")]:
        source = output / candidate[source_key]
        if base.sha256(source) != candidate[source_key + "_sha256"]:
            raise ValueError(f"Candidate {source_key} checksum mismatch")
        atomic_copy(source, output / output_name)
    original = np.load(output / "val_original.npy", allow_pickle=False)
    flip = np.load(output / "val_flip.npy", allow_pickle=False)
    base.write_predictions(output / "val_predictions.csv", records["val"], (original + flip) * 0.5)


def evaluate_candidate(model, loader, device, output, config, provenance, records,
                       epoch, weight_source, best, candidates):
    metrics, original, flipped = validate_views(model, loader, device)
    name = f"epoch_{epoch:02d}_{weight_source}"
    paths = {"checkpoint": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy",
             "flip": f"candidates/{name}_flip.npy"}
    checkpoint = common_checkpoint(model, config, provenance, records, epoch, weight_source)
    checkpoint["validation"] = metrics
    base.save_torch(output / paths["checkpoint"], checkpoint)
    save_array(output / paths["original"], original)
    save_array(output / paths["flip"], flipped)
    candidate = {"candidate_id": name, "epoch": epoch, "weight_source": weight_source,
                 "validation_tta": True, "validation": metrics, **paths,
                 **{key + "_sha256": base.sha256(output / value) for key, value in paths.items()}}
    candidates.append(candidate)
    if best is None or base.rank_metrics(metrics, "accuracy") > base.rank_metrics(best["validation"], "accuracy"):
        best = candidate
        sync_best(output, best, records)
    base.write_json(output / "candidates.json", candidates)
    print(json.dumps({"type": "validation_candidate", **candidate}, ensure_ascii=False), flush=True)
    return best, candidate


def train_epoch(model, ema, optimizer, scaler, loader, device, config, epoch, successful_steps, total_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp = torch.device(device).type == "cuda"
    loss_sum = correct = seen = skipped = updates = 0
    started = last_progress = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader):
        group_start = (batch_index // config["accumulation"]) * config["accumulation"] * config["batch_size"]
        group_images = min(config["batch_size"] * config["accumulation"], len(loader.dataset) - group_start)
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            raw_loss = F.cross_entropy(logits, targets, reduction="sum")
            loss = raw_loss / group_images
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite refinement loss at epoch {epoch}, batch {batch_index}")
        scaler.scale(loss).backward()
        loss_sum += raw_loss.detach().float().item()
        correct += (logits.detach().argmax(1) == targets).sum().item()
        seen += len(targets)
        if (batch_index + 1) % config["accumulation"] == 0 or batch_index + 1 == len(loader):
            base.set_learning_rate(optimizer, successful_steps, total_steps, warmup_steps=0)
            successful = base.optimizer_step(model, optimizer, scaler)
            ema.update(model, successful)
            successful_steps += int(successful)
            updates += int(successful)
            skipped += int(not successful)
        if time.monotonic() - last_progress >= 60:
            elapsed = time.monotonic() - started
            print(json.dumps({"type": "refinement_progress", "epoch": epoch, "images": seen,
                              "total_images": len(loader.dataset), "images_per_second": seen / elapsed,
                              "estimated_training_seconds_left": (len(loader.dataset) - seen) * elapsed / seen,
                              "successful_optimizer_steps": updates, "amp_skipped_steps": skipped}), flush=True)
            last_progress = time.monotonic()
    if updates == 0:
        raise RuntimeError("Refinement epoch had no successful optimizer updates")
    return successful_steps, {"train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                              "successful_optimizer_steps": updates, "amp_skipped_steps": skipped,
                              "amp_scale": scaler.get_scale(), "ema_updates": ema.updates,
                              "lr_lora": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"]}


def run_refinement(root, output, origin, config, provenance, records, device, resume=False):
    base.seed_everything(config["seed"])
    model, origin_metadata = base.load_lora_checkpoint(origin, root, device)
    optimizer_config = {**config, "lr_lora": config["lr_lora"], "lr_head": config["lr_head"]}
    optimizer = base.make_optimizer(model, optimizer_config)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=torch.device(device).type == "cuda")
    ema = AdapterEMA(model, config["ema_decay"])
    successful_steps, start_epoch = 0, 0
    history, candidates, best = [], [], None
    loader = base.make_loader(records["val"], config["batch_size"], config["workers"], config["seed"], device=device)
    total_steps = math.ceil(len(records["train"]) / (config["batch_size"] * config["accumulation"])) * config["epochs"]
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["format"] != FORMAT or saved["config"] != config or saved["provenance"] != provenance:
            raise ValueError("Refinement resume source/configuration/data changed")
        base.restore_adapter(model, saved["adapter_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        ema.load_state_dict(saved["ema_state"])
        base.restore_rng(saved["rng"])
        successful_steps, start_epoch = saved["successful_steps"], saved["epoch"]
        history, candidates, best = saved["history"], saved["candidates"], saved["best_candidate"]
        sync_best(output, best, records)
    else:
        best, _ = evaluate_candidate(model, loader, device, output, config, provenance, records,
                                      0, "baseline", None, candidates)
    def save_last(epoch):
        state = common_checkpoint(model, config, provenance, records, epoch, "raw")
        state.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), ema_state=ema.state_dict(),
                     rng=base.rng_state(), successful_steps=successful_steps, history=history,
                     candidates=candidates, best_candidate=best)
        base.save_torch(output / "last.pt", state)
    if not resume:
        save_last(0)
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        started = time.monotonic()
        train_loader = base.make_loader(records["train"], config["batch_size"], config["workers"], config["seed"],
                                         epoch=config["augmentation_epoch_offset"] + epoch, training=True, device=device)
        successful_steps, training = train_epoch(model, ema, optimizer, scaler, train_loader, device,
                                                   config, epoch, successful_steps, total_steps)
        best, raw_result = evaluate_candidate(model, loader, device, output, config, provenance, records,
                                               epoch, "raw", best, candidates)
        raw_state = base.adapter_state(model)
        base.restore_adapter(model, ema.shadow)
        try:
            best, ema_result = evaluate_candidate(model, loader, device, output, config, provenance, records,
                                                   epoch, "ema", best, candidates)
        finally:
            base.restore_adapter(model, raw_state)
        entry = {"epoch": epoch, **training, "cumulative_optimizer_steps": successful_steps,
                 "raw_validation": raw_result["validation"], "ema_validation": ema_result["validation"],
                 "best_candidate": best["candidate_id"], "epoch_seconds": time.monotonic() - started}
        history.append(entry)
        save_last(epoch)
        base.write_json(output / "history.json", history)
        print(json.dumps({"type": "refinement_epoch", **entry}, ensure_ascii=False), flush=True)
    sync_best(output, best, records)
    summary = {"format": FORMAT, "state": "complete", "classes": CLASSES, "epochs_completed": len(history),
               "best_epoch": best["epoch"], "weight_source": best["weight_source"], "validation_tta": True,
               "best_validation": best["validation"], "selected": best, "selection_metric": "accuracy",
               "secondary_metric": "macro_f1", "tertiary_metric": "lower_cross_entropy", "candidate_count": len(candidates),
               "best_checkpoint_sha256": base.sha256(output / "best.pt"),
               "val_original_sha256": base.sha256(output / "val_original.npy"),
               "val_flip_sha256": base.sha256(output / "val_flip.npy"),
               "train_indices": [r["manifest_index"] for r in records["train"]],
               "val_indices": [r["manifest_index"] for r in records["val"]],
               "provenance": provenance, "test_images_read": 0, "test_labels_used": False,
               "external_answers_used": False}
    base.write_json(output / "summary.json", summary)
    print(json.dumps({"type": "refinement_complete", "best_candidate": best["candidate_id"],
                      "best_validation": best["validation"]}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--origin-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr-lora", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1 or args.workers < 0:
        raise ValueError("Invalid epoch, batch or worker count")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr_lora, args.lr_head)):
        raise ValueError("Learning rates must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("Invalid regularization settings")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU training fallback")
    torch.set_num_threads(4)
    root = args.project_root.resolve()
    output = (args.output or root / "experiments" / "lora_refinement").resolve()
    origin = (args.origin_checkpoint or root / "experiments" / "lora" / "best.pt").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidates").mkdir(exist_ok=True)
    if args.resume and not (output / "last.pt").exists():
        raise ValueError("--resume requires an existing last.pt")
    if not args.resume and any((output / name).exists() for name in ("best.pt", "last.pt", "summary.json")):
        raise ValueError("Refinement results already exist: use --resume or another output directory")
    print("Checking parent checkpoint and train/validation data...", flush=True)
    parent = torch.load(origin, map_location="cpu", weights_only=True)
    if parent["classes"] != CLASSES or parent["provenance"]["source_sha256"] != base.sha256(base.__file__):
        raise ValueError("Original model source or class order changed")
    records, data_provenance = base.load_train_val(root)
    for split in ("train", "val"):
        if [r["manifest_index"] for r in records[split]] != parent[split + "_indices"]:
            raise ValueError("Parent training/validation membership changed")
        if [r["path"] for r in records[split]] != parent[split + "_paths"]:
            raise ValueError("Parent training/validation image paths changed")
    for key in ("manifest_sha256", "train_val_image_content_sha256"):
        if data_provenance[key] != parent["provenance"][key]:
            raise ValueError("Parent training/validation data changed")
    config = {"architecture": parent["config"]["architecture"], "epochs": args.epochs,
              "batch_size": args.batch_size, "accumulation": args.accumulation, "workers": args.workers,
              "seed": args.seed, "lr_lora": args.lr_lora, "lr_head": args.lr_head,
              "weight_decay": args.weight_decay, "ema_decay": args.ema_decay,
              "image_size": 224, "amp": "float16", "amp_initial_scale": 1024.0,
              "scheduler": "cosine_successful_steps", "warmup_steps": 0, "min_lr_ratio": 0.01,
              "selection_metric": "accuracy", "validation_tta": True, "early_stopping": False,
              "augmentation": parent["config"]["augmentation"], "augmentation_epoch_offset": int(parent["epoch"]),
              "preprocessing": parent["config"]["preprocessing"], "label_smoothing": 0.0,
              "original_parent_epoch": int(parent["epoch"])}
    provenance = {k: v for k, v in parent["provenance"].items() if k != "source_sha256"}
    provenance.update(data_provenance, original_model_source_sha256=base.sha256(base.__file__),
                      refinement_source_sha256=base.sha256(__file__), origin_checkpoint_sha256=base.sha256(origin),
                      origin_checkpoint_path=str(origin), original_training_provenance=parent["provenance"],
                      refinement_torch_version=str(torch.__version__), refinement_gpu=torch.cuda.get_device_name())
    if args.resume:
        previous = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if previous["config"] != config or previous["provenance"] != provenance:
            raise ValueError("Resume source/configuration/data changed; existing files preserved")
    base.write_json(output / "config.json", config)
    base.write_json(output / "provenance.json", provenance)
    run_refinement(root, output, origin, config, provenance, records, "cuda", args.resume)


if __name__ == "__main__":
    main()
