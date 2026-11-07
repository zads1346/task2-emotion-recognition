"""Six-epoch AffectNet-initialized DDAMFN++ transfer, train/validation only."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import train_hse_specialist as h
import train_hse_sam_v7 as math_v7
import validate_cosine_fusion as c
import ddamfn_v8_model as model_api

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "ddamfn_v8"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SETTINGS = {"epochs": 6, "seed": 42, "augmentation_epoch_offset": 80,
    "effective_batch_size": 32, "workers": 0, "cpu_threads": 4, "image_size": 112,
    "validation_batch_size": 32, "backbone_lr": 1e-5, "head_lr": 1e-4,
    "weight_decay": .01, "label_smoothing": 0., "gradient_clip": 1.,
    "warmup_successful_steps": 100, "min_lr_ratio": .1, "ema_decay": .995,
    "amp_initial_scale": 1024., "amp_growth_interval": 2000,
    "batchnorm_running_stats": "frozen_eval", "batchnorm_affine": "trainable",
    "sample_weights": "uniform1", "mixup": False, "cutmix": False,
    "rho": 0., "initialization": "official AffectNet7 only; mapped six-class head"}
ARCHITECTURE = {"model": "DDAMFN++", "backbone": "MixedFeatureNet", "num_head": 2,
                "num_class": 6, "feature_dim": 512, "input_size": 112,
                "native_class_indices": [6, 4, 1, 2, 3, 0]}


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def training_transform():
    return transforms.Compose([
        transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomAffine(6, translate=(.03, .03), scale=(.95, 1.05), interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=.1, contrast=.1), transforms.ToTensor(),
        transforms.Normalize(h.MEAN, h.STD)])


def evaluation_transform():
    return transforms.Compose([transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR),
                               transforms.ToTensor(), transforms.Normalize(h.MEAN, h.STD)])


class Images(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        self.transform = training_transform() if training else evaluation_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            value = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return value, int(row.get("target", 0))


def make_loader(rows, training=False, epoch=0, batch_size=32, device="cuda"):
    generator = torch.Generator().manual_seed(42 + 1009*epoch + (0 if training else 1000003))
    return DataLoader(Images(rows, training), batch_size=batch_size, shuffle=training, num_workers=0,
                      pin_memory=torch.device(device).type == "cuda", drop_last=False, generator=generator)


def make_optimizer(model, settings):
    if not isinstance(model.fc, nn.Linear) or model.fc.out_features != 6:
        raise ValueError("DDAM must retain its mapped six-class Linear head")
    head = list(model.fc.parameters())
    head_ids = {id(value) for value in head}
    backbone = [value for value in model.parameters() if id(value) not in head_ids]
    return torch.optim.AdamW([
        {"name": "backbone", "params": backbone, "lr": settings["backbone_lr"], "base_lr": settings["backbone_lr"]},
        {"name": "head", "params": head, "lr": settings["head_lr"], "base_lr": settings["head_lr"]}],
        weight_decay=settings["weight_decay"])


def set_learning_rate(optimizer, successful_steps, total_steps, settings):
    warmup = settings["warmup_successful_steps"]
    if successful_steps < warmup:
        factor = (successful_steps + 1) / warmup
    else:
        position = min(1., (successful_steps - warmup) / max(1, total_steps - warmup - 1))
        floor = settings["min_lr_ratio"]
        factor = floor + (1-floor)*.5*(1 + math.cos(math.pi*position))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]*factor


def buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()}


def assert_frozen_buffers(model, reference):
    actual = buffers(model)
    if set(actual) != set(reference) or any(not torch.equal(actual[name], value) for name, value in reference.items()):
        raise ValueError("Frozen DDAM BatchNorm buffers changed")


def checked_plan(root):
    path = root / "logs/ddamfn_v8_plan.json"
    plan = c.read_json(path)
    if plan.get("format") != "ddamfn_v8_locked" or plan.get("classes") != CLASSES:
        raise ValueError("DDAM plan format/official classes differ")
    settings = plan["settings"]
    if any(settings.get(key) != value for key, value in SETTINGS.items()):
        raise ValueError("DDAM fixed settings differ from the locked protocol")
    batch = settings.get("batch_size")
    if batch not in (8, 16, 32) or settings.get("accumulation") != 32//batch:
        raise ValueError("GPU-probed microbatch must preserve effective32")
    c.verify_pins(root, plan["source_pins"])
    actual_receipt = model_api.source_receipt(root)
    if actual_receipt != plan["model_source_receipt"] or model_api.receipt_sha256(actual_receipt) != plan["model_source_receipt_sha256"]:
        raise ValueError("DDAM official source/weight/head conversion provenance changed")
    h.check_hash(plan["parent_checkpoint"], plan["parent_checkpoint_sha256"])
    if plan["excluded_train_indices"] != [7727, 10448] or len(plan["original_train_indices"]) != 21248 or len(plan["train_indices"]) != 21246:
        raise ValueError("The shared visually confirmed invalid-image exclusions changed")
    if plan["train_indices"] != [index for index in plan["original_train_indices"] if index not in set(plan["excluded_train_indices"])]:
        raise ValueError("Effective training membership is not the original set minus the two fixed exclusions")
    if Path(plan["baseline_directory"]).resolve() != (root / "hse_sam_v7_results").resolve():
        raise ValueError("DDAM must bind the completed V7 incumbent, not an earlier score")
    for relative, digest in plan["baseline_files"].items():
        location = Path(relative)
        if not location.is_absolute():
            location = root / location
        h.check_hash(location, digest)
    fields = ("classes", "settings", "source_pins", "input_receipt", "original_train_indices", "excluded_train_indices",
              "train_indices", "val_indices", "filtered_membership_sha256", "audit_summary_path", "audit_summary_sha256", "parent_checkpoint",
              "parent_checkpoint_sha256", "model_source_receipt", "model_source_receipt_sha256", "baseline_directory",
              "baseline_files", "baseline_correct_count", "baseline_submission_sha256")
    config = {"format": "ddamfn_v8_config", "architecture": ARCHITECTURE,
              **{key: plan[key] for key in fields}, "plan_sha256": h.sha256(path),
              "effective_train_rows": 21246, "original_train_rows": 21248,
              "preprocessing": "RGB; PIL bilinear112; ImageNet mean/std", "validation_weighted": False,
              "checkpoint_selection": "save every raw/EMA candidate; fixed CNN20% replacement selected separately",
              "internal_test_read": False, "external_answers_used": False,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    return plan, config


def payload(model, config, epoch, source):
    return {"format": FORMAT, "config": config, "classes": CLASSES, "epoch": epoch, "weight_source": source,
            "model_state": h.cpu_tree(model.state_dict()), "source_pins": config["source_pins"],
            "model_source_receipt": config["model_source_receipt"], "model_source_receipt_sha256": config["model_source_receipt_sha256"],
            "parent_sha": config["parent_checkpoint_sha256"], "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
            "original_train_indices": config["original_train_indices"], "excluded_train_indices": config["excluded_train_indices"],
            "train_indices": config["train_indices"], "val_indices": config["val_indices"],
            "filtered_membership_sha256": config["filtered_membership_sha256"], "audit_summary_sha256": config["audit_summary_sha256"]}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    originals, flips, targets = [], [], []
    stamp = started = time.monotonic()
    for index, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with h.autocast_for(device):
            original, flipped = model(images), model(images.flip(-1))
        originals.append(original.float().softmax(1).cpu().numpy())
        flips.append(flipped.float().softmax(1).cpu().numpy())
        targets.extend(labels.tolist())
        if time.monotonic()-stamp >= 45:
            emit("validate_ddamfn_v8", batch=index, batches=len(loader), seen=len(targets), elapsed_seconds=time.monotonic()-started)
            stamp = time.monotonic()
    a, b = np.concatenate(originals), np.concatenate(flips)
    c.check_probability(a, len(targets)); c.check_probability(b, len(targets))
    return c.metrics(targets, a), c.metrics(targets, (a+b)*.5), a, b


def evaluate_candidate(model, loader, device, output, config, epoch, source):
    name = f"epoch_{epoch:02d}_{source}"
    emit("validate_candidate_ddamfn_v8", name=name)
    original_metrics, tta_metrics, a, b = validate(model, loader, device)
    files = {"model": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy", "flip": f"candidates/{name}_flip.npy"}
    item_payload = payload(model, config, epoch, source)
    item_payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics)
    h.save_torch(output / files["model"], item_payload)
    h.save_array(output / files["original"], a); h.save_array(output / files["flip"], b)
    item = {"name": name, "epoch": epoch, "weight_source": source, **files,
            "path": files["model"], "candidate_json": f"candidates/{name}.json", "classes": CLASSES,
            "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
            "parent_sha": config["parent_checkpoint_sha256"], "model_source_receipt_sha256": config["model_source_receipt_sha256"],
            "original_train_indices": config["original_train_indices"], "excluded_train_indices": config["excluded_train_indices"],
            "train_indices": config["train_indices"], "val_indices": config["val_indices"], "validation_weighted": False,
            "artifact_sha256": {key: h.sha256(output / path) for key, path in files.items()},
            "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics}
    h.write_json(output / item["candidate_json"], item)
    emit("candidate_ddamfn_v8_complete", name=name, accuracy=tta_metrics["accuracy"], correct_count=tta_metrics["correct_count"])
    return item


def verify_candidate(root, output, item, config, targets):
    name = f"epoch_{item['epoch']:02d}_{item['weight_source']}"
    if item.get("name") != name or item.get("classes") != CLASSES or item.get("plan_sha256") != config["plan_sha256"] or item.get("validation_weighted") is not False:
        raise ValueError("Candidate identity/classes/plan differ")
    if item.get("model") != f"candidates/{name}.pt" or item.get("candidate_json") != f"candidates/{name}.json" or item.get("path") != item["model"]:
        raise ValueError("Candidate paths differ from the committed epoch/source")
    if c.read_json(output / item["candidate_json"]) != item:
        raise ValueError("Candidate JSON differs from committed metadata")
    for key in ("original_train_indices", "excluded_train_indices", "train_indices", "val_indices", "source_pins", "model_source_receipt_sha256"):
        if item.get(key) != config[key]:
            raise ValueError("Candidate member/source binding differs")
    if item.get("parent_sha") != config["parent_checkpoint_sha256"]:
        raise ValueError("Candidate official initialization changed")
    for key in ("model", "original", "flip"):
        location = Path(item[key])
        if location.is_absolute() or ".." in location.parts or not item[key].startswith("candidates/"):
            raise ValueError("Candidate file escapes its directory")
        h.check_hash(output / location, item["artifact_sha256"][key])
    saved = torch.load(output / item["model"], map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT or saved.get("config") != config or saved.get("epoch") != item["epoch"] or saved.get("weight_source") != item["weight_source"]:
        raise ValueError("Candidate full model is from another source/configuration")
    model_api.verify_state(root, saved["model_state"])
    a = np.load(output / item["original"], allow_pickle=False)
    b = np.load(output / item["flip"], allow_pickle=False)
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("Candidate validation probabilities must be float32")
    calculated = {"single_original_metrics": c.metrics(targets, a), "single_tta_metrics": c.metrics(targets, (a+b)*.5)}
    if any(item[key] != calculated[key] or saved[key] != calculated[key] for key in calculated):
        raise ValueError("Candidate validation probabilities do not reproduce their metrics")


def commit(output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates, frozen_buffers):
    relative = f"epochs/epoch_{epoch:02d}.pt"
    state = {**payload(model, config, epoch, "raw"), "optimizer": h.cpu_tree(optimizer.state_dict()),
             "scaler": scaler.state_dict(), "ema_state": ema.state_dict(), "rng": h.rng_state(),
             "counters": counters, "successful_steps": counters["successful_steps"], "cumulative_skips": counters["amp_skipped_steps"],
             "zero_weight_groups": 0, "history": history, "candidates": candidates, "frozen_buffers": frozen_buffers}
    h.save_torch(output / relative, state)
    temporary = output / "last.pt.tmp"
    import shutil
    shutil.copyfile(output / relative, temporary); os.replace(temporary, output / "last.pt")
    h.write_json(output / "last.json", {"epoch": epoch, "checkpoint": relative, "sha256": h.sha256(output / relative), "plan_sha256": config["plan_sha256"]})
    h.write_json(output / "history.json", history); h.write_json(output / "candidates.json", candidates)


def write_summary(root, output, config, history, candidates, counters, ema_updates, targets):
    if len(history) != 6 or len(candidates) != 13 or [row["epoch"] for row in history] != list(range(1, 7)):
        raise ValueError("Require six complete rounds and the fixed thirteen candidates")
    if any(row["seen"] != 21246 or row["successful_optimizer_steps"] + row["amp_skipped_steps"] != 664 for row in history):
        raise ValueError("An epoch did not cover every effective training member/group")
    if counters["successful_steps"] != sum(row["successful_optimizer_steps"] for row in history) or ema_updates != counters["successful_steps"]:
        raise ValueError("Successful optimizer/EMA counts differ")
    if counters["amp_skipped_steps"] != sum(row["amp_skipped_steps"] for row in history):
        raise ValueError("AMP skip counts differ")
    for item in candidates:
        verify_candidate(root, output, item, config, targets)
    paths = ["config.json", "last.json", "last.pt", "history.json", "candidates.json"] + [f"epochs/epoch_{epoch:02d}.pt" for epoch in range(7)]
    for item in candidates:
        paths.extend(item[key] for key in ("candidate_json", "model", "original", "flip"))
    summary = {"format": "ddamfn_v8_summary", "state": "complete", "classes": CLASSES, "epochs_completed": 6,
               "candidate_count": 13, "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
               "parent_checkpoint_sha256": config["parent_checkpoint_sha256"], "model_source_receipt_sha256": config["model_source_receipt_sha256"],
               "original_train_rows": 21248, "effective_train_rows": 21246,
               "original_train_indices": config["original_train_indices"], "excluded_train_indices": config["excluded_train_indices"],
               "train_indices": config["train_indices"], "val_indices": config["val_indices"],
               "counters": counters, "successful_optimizer_steps": counters["successful_steps"], "amp_skipped_steps": counters["amp_skipped_steps"],
               "zero_weight_groups": 0, "ema_updates": ema_updates, "validation_weighted": False,
               "batchnorm_buffers_frozen": True, "baseline_correct_count": config["baseline_correct_count"],
               "internal_test_read": False, "external_answers_used": False, "candidates": candidates,
               "artifact_sha256": {path: h.sha256(output / path) for path in paths}}
    h.write_json(output / "summary.json", summary)
    emit("train_ddamfn_v8_complete", successful_optimizer_steps=counters["successful_steps"], amp_skipped_steps=counters["amp_skipped_steps"], candidates=13)
    return summary


def run_training(root, device="cuda"):
    root = Path(root).resolve(); torch.set_num_threads(4)
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU training fallback")
    plan, config = checked_plan(root)
    rows, receipt = h.load_trainval(root)
    if receipt != plan["input_receipt"] or [row["manifest_index"] for row in rows["train"]] != plan["original_train_indices"]:
        raise ValueError("Original audited data/member receipt changed")
    rows["train"] = [row for row in rows["train"] if row["manifest_index"] not in set(plan["excluded_train_indices"])]
    if h.signature(rows["train"]) != plan["filtered_membership_sha256"] or [row["manifest_index"] for row in rows["val"]] != plan["val_indices"]:
        raise ValueError("Effective training or validation membership changed")
    output = root / "experiments/ddamfn_v8"
    output.mkdir(parents=True, exist_ok=True); (output / "candidates").mkdir(exist_ok=True); (output / "epochs").mkdir(exist_ok=True)
    if (output / "config.json").exists() and c.read_json(output / "config.json") != config:
        raise ValueError("Existing V8 configuration differs; preserve existing results")
    h.write_json(output / "config.json", config)
    targets = np.asarray([row["target"] for row in rows["val"]])
    if (output / "summary.json").exists():
        summary = c.read_json(output / "summary.json")
        if summary.get("state") != "complete" or summary.get("plan_sha256") != config["plan_sha256"]:
            raise ValueError("Existing summary is incomplete or belongs to another plan")
        for path, digest in summary["artifact_sha256"].items():
            h.check_hash(output / path, digest)
        return summary
    h.seed_everything(config["settings"]["seed"])
    model, initialization = model_api.load_initial_model(root, device)
    if initialization != config["model_source_receipt"]:
        raise ValueError("DDAM initialized from another official source")
    model.requires_grad_(True)
    settings = config["settings"]
    optimizer = make_optimizer(model, settings)
    scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", settings["amp_initial_scale"], settings["amp_growth_interval"])
    ema = math_v7.FullModelEMA(model, settings["ema_decay"])
    frozen = buffers(model)
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    history, candidates, done = [], [], 0
    val_loader = make_loader(rows["val"], batch_size=settings["validation_batch_size"], device=device)
    if (output / "last.json").exists():
        pointer = c.read_json(output / "last.json")
        done = pointer["epoch"]
        if not isinstance(done, int) or not 0 <= done <= 6 or pointer["checkpoint"] != f"epochs/epoch_{done:02d}.pt" or pointer["plan_sha256"] != config["plan_sha256"]:
            raise ValueError("Last pointer is not a complete epoch of the locked plan")
        h.check_hash(output / pointer["checkpoint"], pointer["sha256"])
        saved = torch.load(output / pointer["checkpoint"], map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["epoch"] != done or len(saved["history"]) != done:
            raise ValueError("Saved resume configuration/history differs")
        model_api.verify_state(root, saved["model_state"])
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); ema.load_state_dict(saved["ema_state"])
        h.restore_rng(saved["rng"])
        counters, history, candidates = saved["counters"], saved["history"], saved["candidates"]
        if set(saved["frozen_buffers"]) != set(frozen) or any(not torch.equal(value, saved["frozen_buffers"][name]) for name, value in frozen.items()):
            raise ValueError("Resume BatchNorm statistics differ from the official initializer")
        if counters["successful_steps"] != ema.updates:
            raise ValueError("Resume EMA and optimizer successful-step counts differ")
        expected_names = ["epoch_00_baseline"] + [f"epoch_{epoch:02d}_{source}" for epoch in range(1, done+1) for source in ("raw", "ema")]
        if [item["name"] for item in candidates] != expected_names:
            raise ValueError("Resume candidate set is incomplete")
        for item in candidates:
            verify_candidate(root, output, item, config, targets)
        import shutil
        shutil.copyfile(output / pointer["checkpoint"], output / "last.pt.tmp"); os.replace(output / "last.pt.tmp", output / "last.pt")
        # The immutable checkpoint/pointer commits first. Regenerate derivatives
        # after a crash between that pointer and its JSON exports, including epoch6.
        h.write_json(output / "history.json", history)
        h.write_json(output / "candidates.json", candidates)
        del saved
        emit("resume_ddamfn_v8", completed_epoch=done, successful_optimizer_steps=counters["successful_steps"])
    else:
        candidate = evaluate_candidate(model, val_loader, device, output, config, 0, "baseline")
        candidates.append(candidate)
        commit(output, model, optimizer, scaler, ema, config, 0, counters, history, candidates, frozen)
    total_steps = math.ceil(len(rows["train"])/32)*6
    for epoch in range(done+1, 7):
        data_epoch = settings["augmentation_epoch_offset"] + epoch - 1
        loader = make_loader(rows["train"], True, data_epoch, 32, device)
        model.train(); math_v7.freeze_batchnorm(model)
        seen = success = skips = right = 0
        loss_sum = 0.; diagnostic_seen = 0
        stamp = started = time.monotonic()
        for index, (images, labels) in enumerate(loader, 1):
            set_learning_rate(optimizer, counters["successful_steps"], total_steps, settings)
            result = math_v7.group_step(model, optimizer, scaler, ema, images, labels, device=device, batch_size=settings["batch_size"], rho=0.)
            seen += len(labels)
            good = bool(result["success"])
            success += int(good); skips += int(not good)
            counters["successful_steps"] += int(good); counters["amp_skipped_steps"] += int(not good)
            if result["loss_sum"] is not None:
                loss_sum += float(result["loss_sum"]); right += int(result["correct"]); diagnostic_seen += len(labels)
            if skips >= 25 and success == 0:
                raise RuntimeError("Twenty-five AMP skips without a successful update")
            if time.monotonic()-stamp >= 45 or index == len(loader):
                emit("train_ddamfn_v8", epoch=epoch, epochs=6, batch=index, batches=len(loader), seen=seen,
                     train_loss=loss_sum/max(1, diagnostic_seen), successful_optimizer_steps=success, amp_skipped_steps=skips,
                     cumulative_optimizer_steps=counters["successful_steps"], elapsed_seconds=time.monotonic()-started)
                stamp = time.monotonic()
        if seen != 21246 or success <= 0 or success+skips != 664:
            raise ValueError("Training epoch did not cover all effective members/groups")
        assert_frozen_buffers(model, frozen)
        raw = evaluate_candidate(model, val_loader, device, output, config, epoch, "raw")
        with ema.apply(model):
            assert_frozen_buffers(model, frozen)
            averaged = evaluate_candidate(model, val_loader, device, output, config, epoch, "ema")
        candidates.extend((raw, averaged))
        history.append({"epoch": epoch, "seen": seen, "augmentation_epoch": data_epoch,
                        "train_loss": loss_sum/max(1, diagnostic_seen), "train_accuracy": right/max(1, diagnostic_seen),
                        "diagnostic_samples": diagnostic_seen, "successful_optimizer_steps": success, "amp_skipped_steps": skips,
                        "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
                        "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-started,
                        "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                        "raw_original_metrics": raw["single_original_metrics"], "raw_tta_metrics": raw["single_tta_metrics"],
                        "ema_original_metrics": averaged["single_original_metrics"], "ema_tta_metrics": averaged["single_tta_metrics"]})
        assert_frozen_buffers(model, frozen)
        c.verify_pins(root, config["source_pins"])
        commit(output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates, frozen)
        emit("epoch_ddamfn_v8_committed", **history[-1])
    return write_summary(root, output, config, history, candidates, counters, ema.updates, targets)


def load_checkpoint(root, path, device="cpu"):
    return model_api.load_trained_model(root, path, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("Only the fixed project root is supported")
    run_training(args.project_root, args.device)


if __name__ == "__main__":
    main()
