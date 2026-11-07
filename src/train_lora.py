"""Local DINOv2-B/14 LoRA training; train/val only, no test-set evaluation.

Only qkv LoRA adapters in the last blocks and a CLS+mean-patch classifier are
trainable. Official backbone weights remain external to the small checkpoints.
The main command probes two successful optimizer steps before actual training.
"""
from __future__ import annotations

import os
os.environ.setdefault("XFORMERS_DISABLED", "1")
import argparse
import csv
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF, InterpolationMode
from torchvision import transforms

CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path, label, split")
        return list(reader)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires an nn.Linear and positive rank")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling


class DinoLoRAClassifier(nn.Module):
    def __init__(self, backbone, last_blocks=4, rank=8, alpha=16.0, dropout=0.05):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        if not 1 <= last_blocks <= len(backbone.blocks):
            raise ValueError("Invalid number of adapted blocks")
        self.adapted_blocks = list(range(len(backbone.blocks) - last_blocks, len(backbone.blocks)))
        for index in self.adapted_blocks:
            attn = backbone.blocks[index].attn
            attn.qkv = LoRALinear(attn.qkv, rank, alpha, dropout)
        self.head = nn.Linear(int(backbone.embed_dim) * 2, len(CLASSES))
        self.train(False)
        self.assert_trainable()

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen attention/drop-path/etc. in eval, enabling only LoRA dropout.
        self.backbone.eval()
        self.head.train(mode)
        for module in self.backbone.modules():
            if isinstance(module, LoRALinear):
                module.training = mode
                module.dropout.train(mode)
        return self

    def assert_trainable(self):
        for name, parameter in self.named_parameters():
            expected = name.startswith("head.") or name.endswith(".lora_A") or name.endswith(".lora_B")
            if parameter.requires_grad != expected:
                raise RuntimeError(f"Unexpected requires_grad: {name}")

    def forward(self, images):
        z = self.backbone.forward_features(images)
        features = torch.cat((z["x_norm_clstoken"], z["x_norm_patchtokens"].mean(1)), dim=1)
        return self.head(features)


def repository_info(project_root):
    repo = Path(project_root) / "vendor" / "dinov2"
    digest = hashlib.sha256()
    for path in sorted((repo / "dinov2").rglob("*.py")):
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path)))
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    return {"repository_commit": commit, "repository_python_sha256": digest.hexdigest()}


def verify_official_weights(project_root):
    path = Path(project_root) / "models" / "dinov2_vitb14_pretrain.pth"
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    actual = sha256(path)
    if receipt.get("sha256") != actual:
        raise ValueError("Official pretrained weight SHA256 does not match local download receipt")
    return actual


def make_model(project_root, architecture, device="cpu"):
    repo = str(Path(project_root) / "vendor" / "dinov2")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dinov2.hub.backbones import dinov2_vitb14
    backbone = dinov2_vitb14(pretrained=False)
    weight = torch.load(Path(project_root) / "models" / "dinov2_vitb14_pretrain.pth",
                        map_location="cpu", weights_only=True)
    backbone.load_state_dict(weight, strict=True)
    del weight
    if int(backbone.embed_dim) != 768 or len(backbone.blocks) != 12:
        raise ValueError("Expected official ViT-B/14: 768 embedding, 12 blocks")
    model = DinoLoRAClassifier(backbone, last_blocks=architecture["last_blocks"],
                               rank=architecture["rank"], alpha=architecture["alpha"],
                               dropout=architecture["lora_dropout"])
    return model.to(device)


def adapter_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def restore_adapter(model, state):
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint adapter/head parameter names differ from this architecture")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            if tensor.shape != named[name].shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            named[name].copy_(tensor)


def load_lora_checkpoint(path, project_root, device="cpu"):
    """Return (model, checkpoint metadata), verifying local source and backbone."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != VERSION or checkpoint.get("classes") != CLASSES:
        raise ValueError("Unsupported checkpoint version or official class order")
    provenance = checkpoint["provenance"]
    if provenance.get("source_sha256") != sha256(__file__):
        raise ValueError("LoRA implementation source differs from training")
    if verify_official_weights(project_root) != provenance["official_weight_sha256"]:
        raise ValueError("Backbone differs from the one used during training")
    actual_repo = repository_info(project_root)
    if any(actual_repo[key] != provenance[key] for key in actual_repo):
        raise ValueError("DINOv2 vendor source differs from training")
    model = make_model(project_root, checkpoint["config"]["architecture"], device)
    restore_adapter(model, checkpoint["adapter_state"])
    model.eval()
    return model, {key: value for key, value in checkpoint.items() if key != "adapter_state"}


def make_eval_transform():
    """Caller should convert a PIL image to RGB before applying this transform."""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def load_train_val(project_root):
    manifest = Path(project_root) / "features" / "manifest.csv"
    manifest_sha = sha256(manifest)
    audit = json.loads((Path(project_root) / "data" / "split_audit.json").read_text(encoding="utf-8"))
    if audit.get("manifest_sha256") != manifest_sha or audit.get("classes") != CLASSES:
        raise ValueError("Manifest/classes differ from the audited data split")
    rows = read_csv(manifest)
    partitions = {split: [] for split in ("train", "val")}
    image_digest = hashlib.sha256()
    path_sets, hash_sets, people = {}, {}, {}
    for split in partitions:
        path_sets[split], hash_sets[split] = set(), set()
        for index, row in enumerate(rows):
            if row["split"].strip().lower() != split:
                continue  # Never read or evaluate test/competition image pixels or labels.
            label = row["label"].strip()
            if label not in CLASSES:
                raise ValueError(f"Unknown train/val class: {label}")
            path = Path(row["path"]).resolve()
            canonical = os.path.normcase(str(path))
            if canonical in path_sets[split]:
                raise ValueError(f"Repeated image path in {split}: {path}")
            path_sets[split].add(canonical)
            content_sha = sha256(path)
            hash_sets[split].add(content_sha)
            record = {"manifest_index": index, "path": str(path), "label": label,
                      "target": CLASSES.index(label), "image_sha256": content_sha}
            partitions[split].append(record)
            image_digest.update(json.dumps([split, index, canonical, label, content_sha],
                                           ensure_ascii=False).encode())
            person = row.get("subject_id", "").strip()
            if person:
                people.setdefault(person, set()).add(split)
        if {r["target"] for r in partitions[split]} != set(range(6)):
            raise ValueError(f"{split} must have all six classes")
    if path_sets["train"] & path_sets["val"] or hash_sets["train"] & hash_sets["val"]:
        raise ValueError("Train/validation image leakage detected")
    if any(len(splits) > 1 for splits in people.values()):
        raise ValueError("Train/validation subject leakage detected")
    return partitions, {"manifest_sha256": manifest_sha,
                        "train_val_image_content_sha256": image_digest.hexdigest(),
                        "train_count": len(partitions["train"]), "val_count": len(partitions["val"]),
                        "test_images_read": 0, "test_labels_used": False}


class FaceImages(Dataset):
    def __init__(self, records, training=False, seed=42, epoch=0):
        self.records, self.training, self.seed, self.epoch = records, training, seed, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]["path"]) as source:
            im = source.convert("RGB")
        im = TF.resize(im, [224, 224], interpolation=InterpolationMode.BICUBIC)
        if self.training:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + index)
            draws = torch.rand(3, generator=generator).tolist()
            if draws[0] < 0.5:
                im = TF.hflip(im)
            im = TF.rotate(im, angle=(draws[1] * 2 - 1) * 5,
                           interpolation=InterpolationMode.BILINEAR, fill=[127, 127, 127])
            im = TF.adjust_brightness(im, 0.95 + 0.10 * draws[2])
        return TF.normalize(TF.to_tensor(im), MEAN, STD), self.records[index]["target"]


def make_loader(records, batch_size, workers, seed, epoch=0, training=False, device="cuda"):
    dataset = FaceImages(records, training=training, seed=seed, epoch=epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
                      pin_memory=str(device).startswith("cuda"), drop_last=False,
                      generator=torch.Generator().manual_seed(seed + epoch), persistent_workers=False)


def classification_metrics(targets, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.shape != (len(targets), 6) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid validation probabilities")
    predicted = probabilities.argmax(1)
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (targets, predicted), 1)
    tp = matrix.diagonal().astype(float)
    f1 = np.divide(2 * tp, matrix.sum(0) + matrix.sum(1), out=np.zeros(6),
                   where=(matrix.sum(0) + matrix.sum(1)) > 0)
    return {"accuracy": float(np.mean(targets == predicted)), "macro_f1": float(f1.mean()),
            "loss": float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1)).mean()),
            "per_class_f1": dict(zip(CLASSES, f1.tolist())), "confusion_matrix": matrix.tolist()}


@torch.inference_mode()
def validate(model, loader, device, amp):
    model.eval()
    probabilities, targets = [], []
    for images, target in loader:
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        targets.extend(target.tolist())
    probabilities = np.concatenate(probabilities)
    return classification_metrics(targets, probabilities), probabilities


def make_optimizer(model, config):
    adapters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("head.")]
    return torch.optim.AdamW([{"params": adapters, "lr": config["lr_lora"], "base_lr": config["lr_lora"]},
                              {"params": model.head.parameters(), "lr": config["lr_head"], "base_lr": config["lr_head"]}],
                             weight_decay=config["weight_decay"])


def set_learning_rate(optimizer, successful_steps, total_steps, warmup_steps):
    if successful_steps < warmup_steps:
        factor = (successful_steps + 1) / max(1, warmup_steps)
    else:
        fraction = min(1.0, (successful_steps - warmup_steps) / max(1, total_steps - warmup_steps - 1))
        factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def optimizer_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    successful = scaler.get_scale() >= old_scale
    optimizer.zero_grad(set_to_none=True)
    return successful


def _probe_attempt(project_root, config, batch_size, accumulation, device):
    seed_everything(config["seed"])
    model = make_model(project_root, config["architecture"], device).train()
    frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = make_optimizer(model, config)
    amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    if amp:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    successes = skipped = 0
    while successes < 2 and skipped < 10:
        for _ in range(accumulation):
            images = torch.randn(batch_size, 3, 224, 224, device=device)
            targets = torch.randint(0, 6, (batch_size,), device=device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(images), targets) / accumulation
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite probe loss")
            scaler.scale(loss).backward()
        okay = optimizer_step(model, optimizer, scaler)
        successes += int(okay)
        skipped += int(not okay)
    if successes != 2:
        raise RuntimeError("GPU probe did not complete two successful optimizer steps")
    if any(p.grad is not None or p._version != frozen_versions[name]
           for name, p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError("Frozen backbone changed or acquired gradients during the probe")
    if amp:
        torch.cuda.synchronize()
    report = {"batch_size": batch_size, "accumulation": accumulation,
              "effective_batch_size": batch_size * accumulation, "successful_optimizer_steps": successes,
              "amp_skipped_steps": skipped, "amp_scale": scaler.get_scale(),
              "frozen_backbone_unchanged": True, "frozen_backbone_gradients": False,
              "seconds": time.monotonic() - started,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3 if amp else None,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3 if amp else None,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    del images, targets, loss, optimizer, scaler, model
    return report


def probe(project_root, config, device):
    batch = config["batch_size"]
    target_batch = batch * config["accumulation"]
    while True:
        accumulation = max(1, math.ceil(target_batch / batch))
        try:
            result = _probe_attempt(project_root, config, batch, accumulation, device)
            break
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
        # Retry outside except: release traceback-held CUDA tensors first.
        gc.collect()
        torch.cuda.empty_cache()
        batch //= 2
        print(f"Probe memory pressure: microbatch={batch}, target effective batch={target_batch}", flush=True)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def rank_metrics(metrics, primary):
    secondary = "macro_f1" if primary == "accuracy" else "accuracy"
    return (metrics[primary], metrics[secondary], -metrics["loss"])


def write_predictions(path, records, probabilities):
    fields = ["manifest_index", "path", "label", "target", "prediction", "prediction_label"] + [f"p_{c}" for c in CLASSES]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, probs in zip(records, probabilities):
            prediction = int(probs.argmax())
            writer.writerow({**{k: row[k] for k in fields[:4]}, "prediction": prediction,
                             "prediction_label": CLASSES[prediction],
                             **{f"p_{c}": float(p) for c, p in zip(CLASSES, probs)}})
    os.replace(tmp, path)


def run_training(project_root, output, config, provenance, records, device, resume):
    seed_everything(config["seed"])
    model = make_model(project_root, config["architecture"], device)
    optimizer = make_optimizer(model, config)
    amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=amp)
    steps_per_epoch = math.ceil(len(records["train"]) / (config["batch_size"] * config["accumulation"]))
    total_steps = steps_per_epoch * config["epochs"]
    warmup_steps = steps_per_epoch
    successful_steps, start_epoch, stale, history = 0, 0, 0, []
    best_metrics, best_epoch = None, None
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["provenance"] != provenance:
            raise ValueError("Resume source, data, architecture or training configuration changed")
        restore_adapter(model, checkpoint["adapter_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        successful_steps, start_epoch = checkpoint["successful_steps"], checkpoint["epoch"]
        stale, history = checkpoint["stale"], checkpoint["history"]
        best_metrics, best_epoch = checkpoint["best_metrics"], checkpoint["best_epoch"]
        if not (output / "best.pt").exists() and best_metrics is not None:
            raise ValueError("Resume requires the existing best.pt checkpoint")
    val_loader = make_loader(records["val"], config["batch_size"], config["workers"],
                             config["seed"], device=device)
    session_started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        if stale >= config["patience"]:
            break
        epoch_started = time.monotonic()
        loader = make_loader(records["train"], config["batch_size"], config["workers"],
                             config["seed"], epoch=epoch, training=True, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = correct = seen = skipped = epoch_steps = 0
        last_progress = time.monotonic()
        for batch_index, (images, targets) in enumerate(loader):
            group_start = (batch_index // config["accumulation"]) * config["accumulation"] * config["batch_size"]
            group_images = min(config["batch_size"] * config["accumulation"], len(loader.dataset) - group_start)
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                raw_loss = F.cross_entropy(logits, targets, reduction="sum")
                loss = raw_loss / group_images
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}, batch {batch_index}")
            scaler.scale(loss).backward()
            loss_sum += raw_loss.detach().float().item()
            correct += (logits.detach().argmax(1) == targets).sum().item()
            seen += len(targets)
            if (batch_index + 1) % config["accumulation"] == 0 or batch_index + 1 == len(loader):
                set_learning_rate(optimizer, successful_steps, total_steps, warmup_steps)
                success = optimizer_step(model, optimizer, scaler)
                successful_steps += int(success)
                epoch_steps += int(success)
                skipped += int(not success)
            if time.monotonic() - last_progress >= 60:
                elapsed = time.monotonic() - epoch_started
                print(json.dumps({"type": "training_progress", "epoch": epoch + 1,
                                  "images": seen, "total_images": len(loader.dataset),
                                  "images_per_second": seen / max(elapsed, 1e-6),
                                  "estimated_training_seconds_left": (len(loader.dataset) - seen) * elapsed / seen,
                                  "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": skipped}), flush=True)
                last_progress = time.monotonic()
        metrics, probabilities = validate(model, val_loader, device, amp)
        entry = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                 "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
                 "epoch_seconds": time.monotonic() - epoch_started, "session_seconds": time.monotonic() - session_started,
                 "successful_optimizer_steps": epoch_steps, "cumulative_optimizer_steps": successful_steps,
                 "amp_skipped_steps": skipped, "amp_scale": scaler.get_scale(),
                 "lr_lora": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"]}
        history.append(entry)
        improved = epoch_steps > 0 and (best_metrics is None or rank_metrics(metrics, config["selection_metric"]) > rank_metrics(best_metrics, config["selection_metric"]))
        if improved:
            best_metrics, best_epoch, stale = metrics, epoch + 1, 0
        elif epoch_steps:
            stale += 1
        common = {"format_version": VERSION, "classes": CLASSES, "config": config, "provenance": provenance,
                  "epoch": epoch + 1, "adapter_state": adapter_state(model), "validation": metrics,
                  "best_metrics": best_metrics, "best_epoch": best_epoch,
                  "train_indices": [r["manifest_index"] for r in records["train"]],
                  "val_indices": [r["manifest_index"] for r in records["val"]],
                  "train_paths": [r["path"] for r in records["train"]],
                  "val_paths": [r["path"] for r in records["val"]]}
        if improved:
            save_torch(output / "best.pt", common)
            np.save(output / "val_probabilities.npy", probabilities, allow_pickle=False)
            write_predictions(output / "val_predictions.csv", records["val"], probabilities)
        save_torch(output / "last.pt", {**common, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                                        "rng": rng_state(), "successful_steps": successful_steps, "stale": stale, "history": history})
        write_json(output / "history.json", history)
        print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)
        if epoch_steps == 0 and len(history) >= 2 and history[-2]["successful_optimizer_steps"] == 0:
            raise RuntimeError("No successful optimizer updates in two epochs; inspect AMP rather than accepting an untrained checkpoint")
    if best_metrics is None:
        raise RuntimeError("Training completed without a successful checkpoint")
    summary = {"model": "dinov2_vitb14_lora", "classes": CLASSES, "selection_metric": config["selection_metric"],
               "best_epoch": best_epoch, "best_validation": best_metrics, "epochs_completed": len(history),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
               "validation_tta": False, "test_evaluated": False, "test_labels_used": False,
               "best_checkpoint_sha256": sha256(output / "best.pt"),
               "val_probabilities_sha256": sha256(output / "val_probabilities.npy"), "provenance": provenance}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--last-blocks", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.accumulation, args.rank) < 1 or args.workers < 0:
        raise ValueError("Invalid training sizes")
    if not 1 <= args.last_blocks <= 12 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Invalid LoRA architecture")
    if any(not math.isfinite(x) or x <= 0 for x in (args.lr_lora, args.lr_head, args.alpha)) or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Learning rates, alpha and weight decay must be valid")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training command; no silent CPU fallback")
    torch.set_num_threads(4)
    root, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and not (output / "last.pt").exists():
        raise ValueError("--resume requires last.pt in the output directory")
    if not args.resume and any((output / name).exists() for name in ("last.pt", "best.pt", "summary.json")):
        raise ValueError("Existing training results: use --resume or another output directory")
    config = {"architecture": {"model": "dinov2_vitb14", "last_blocks": args.last_blocks, "rank": args.rank,
                               "alpha": args.alpha, "lora_dropout": args.lora_dropout,
                               "pooling": "CLS+mean_patch", "feature_dim": 1536, "classes": CLASSES},
              "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
              "accumulation": args.accumulation, "workers": args.workers, "seed": args.seed,
              "selection_metric": args.selection_metric, "lr_lora": args.lr_lora,
              "lr_head": args.lr_head, "weight_decay": args.weight_decay, "image_size": 224,
              "amp": "float16", "amp_initial_scale": 1024.0, "label_smoothing": 0.0,
              "augmentation": "horizontal flip 0.5; rotation +/-5 deg; brightness [0.95,1.05]",
              "preprocessing": "RGB; bicubic resize 224; ImageNet mean/std", "validation_tta": False}
    print("Checking official weights and train/validation provenance...", flush=True)
    provenance = {"source_sha256": sha256(__file__), "official_weight_sha256": verify_official_weights(root),
                  **repository_info(root), "torch_version": str(torch.__version__),
                  "gpu": torch.cuda.get_device_name(), "cuda_version": torch.version.cuda}
    if not args.probe_only:
        records, data_provenance = load_train_val(root)
        provenance.update(data_provenance)
    print("Probing two synthetic optimizer steps on a throwaway model...", flush=True)
    report = probe(root, config, "cuda")
    report.update(gpu=torch.cuda.get_device_name(), source_sha256=sha256(__file__),
                  official_weight_sha256=provenance["official_weight_sha256"], real_training=False)
    write_json(output / "gpu_probe.json", report)
    print(json.dumps({"gpu_probe": report}, ensure_ascii=False), flush=True)
    if args.probe_only:
        return
    config.update(batch_size=report["batch_size"], accumulation=report["accumulation"])
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["provenance"] != provenance:
            raise ValueError("Resume configuration/provenance changed; existing results are preserved")
    write_json(output / "config.json", config)
    write_json(output / "provenance.json", provenance)
    run_training(root, output, config, provenance, records, "cuda", args.resume)


if __name__ == "__main__":
    main()
