"""Paired four-epoch LoRA continuation: exact1/.15/0 weights versus uniform control.

Each CPU loader group contains32 uniformly shuffled images. GPU microbatches
share that complete group's denominator. Zero-sum groups do not call forward,
backward, scaler, optimizer.step, EMA, or AdamW weight decay.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "4"
os.environ.setdefault("XFORMERS_DISABLED", "1")

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import train_task_lora_v4 as v4
import train_lora as base
import train_hse_specialist as hse
import prepare_zero_weights_v5 as prep
from refine_lora import AdapterEMA
from validate_cosine_fusion import metrics, val_rows

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "zero_weight_lora_v5"
CLASSES = list(v4.CLASSES)
ARCHITECTURE = dict(v4.ARCHITECTURE)
ALL_NAMES = set(v4.ALL_NAMES)
MODES = ("weighted", "uniform_control")
FIXED_SETTINGS = {"epochs": 4, "seed": 42, "augmentation_epoch_offset": 40,
    "label_smoothing": .02, "weight_decay": .001, "lr_new": 1e-5,
    "lr_inherited": 2.5e-6, "lr_head": 1e-5, "warmup_successful_steps": 50,
    "min_lr_ratio": .1, "ema_decay": .995, "batch_size": 8, "accumulation": 4,
    "effective_batch_size": 32, "workers": 0}

sha256, check_hash, read_json = v4.sha256, v4.check_hash, v4.read_json
save_array, atomic_copy, cpu_tree = v4.save_array, v4.atomic_copy, v4.cpu_tree
make_optimizer, set_learning_rate = v4.make_optimizer, v4.set_learning_rate
validate_views = v4.validate_views


def should_update(group_weight_sum):
    """A zero total means absolutely no forward/scaler/optimizer/EMA update."""
    if isinstance(group_weight_sum, torch.Tensor):
        if group_weight_sum.numel() != 1:
            raise ValueError("Group weight sum must be scalar")
        value = float(group_weight_sum.detach().cpu().item())
    else:
        value = float(group_weight_sum)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Group weight sum must be finite and nonnegative")
    return value > 0


def weighted_loss(logits, targets, weights, group_weight_sum):
    """Sum weighted microbatch CE divided by its complete32-image group's sum.

Caller must skip zero-sum groups via should_update before model.forward. This
helper deliberately rejects zero denominators instead of creating a fake step.
"""
    if not should_update(group_weight_sum):
        raise ValueError("Zero-weight groups must be skipped before model.forward")
    if logits.ndim != 2 or logits.shape[1] != 6 or targets.shape != (len(logits),):
        raise ValueError("Expected six-class logits and matching target rows")
    losses = F.cross_entropy(logits, targets, reduction="none", label_smoothing=.02)
    weights = torch.as_tensor(weights, device=logits.device, dtype=losses.dtype)
    if weights.shape != (len(logits),) or not torch.isfinite(weights).all().item() or torch.any(weights < 0).item():
        raise ValueError("Microbatch sample weights must be finite, nonnegative and row-aligned")
    divisor = float(group_weight_sum.detach().cpu().item()) if isinstance(group_weight_sum, torch.Tensor) else float(group_weight_sum)
    if float(weights.double().sum().detach().cpu().item()) > divisor + 2e-6:
        raise ValueError("Microbatch weights exceed their complete group's denominator")
    return (losses * weights).sum()/divisor


def train_group(model, optimizer, scaler, ema, images, targets, weights,
                device="cuda", batch_size=8):
    """Execute one complete CPU group's accumulation; useful for independent CPU tests.

Returns success=None for a zero group, never calling model/scaler/EMA there.
The caller sets a successful-step-indexed learning rate only for nonzero groups.
"""
    weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if weights.shape != (len(targets),) or len(images) != len(targets) or len(targets) > 32:
        raise ValueError("CPU group images/targets/weights differ or exceed32")
    if not torch.isfinite(weights).all().item() or torch.any(weights < 0).item():
        raise ValueError("Invalid complete-group weights")
    denominator = float(weights.sum().item())
    optimizer.zero_grad(set_to_none=True)
    if not should_update(denominator):
        return {"success": None, "zero_weight_group": True, "seen": len(targets),
                "weight_sum": 0., "weighted_loss_sum": 0., "unweighted_loss_sum": 0.,
                "correct": 0, "microbatches": 0, "forwarded_seen": 0}
    amp = torch.device(device).type == "cuda"
    numerator = plain_loss = correct = microbatches = 0
    for start in range(0, len(targets), batch_size):
        stop = min(start+batch_size, len(targets))
        micro_images = images[start:stop].to(device, non_blocking=True)
        micro_targets = targets[start:stop].to(device, non_blocking=True)
        micro_weights = weights[start:stop].to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            logits = model(micro_images)
            loss = weighted_loss(logits, micro_targets, micro_weights, denominator)
            diagnostic = F.cross_entropy(logits, micro_targets, reduction="sum", label_smoothing=.02)
        if not torch.isfinite(loss).item() or not torch.isfinite(diagnostic).item():
            raise RuntimeError("Nonfinite complete-group training loss")
        scaler.scale(loss).backward()
        numerator += loss.detach().float().item()*denominator
        plain_loss += diagnostic.detach().float().item()
        correct += (logits.detach().argmax(1) == micro_targets).sum().item()
        microbatches += 1
    successful = v4.optimizer_step(model, optimizer, scaler, ema)
    return {"success": bool(successful), "zero_weight_group": False, "seen": len(targets),
            "weight_sum": denominator, "weighted_loss_sum": numerator,
            "unweighted_loss_sum": plain_loss, "correct": correct, "microbatches": microbatches,
            "forwarded_seen": len(targets)}


class WeightedFaces(Dataset):
    def __init__(self, records, weights, seed, epoch):
        self.faces = base.FaceImages(records, training=True, seed=seed, epoch=epoch)
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(self.faces),):
            raise ValueError("Training sample weights/members differ")

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        image, target = self.faces[index]
        return image, target, float(self.weights[index])


def _verify_environment(root, config):
    if config.get("format") != "zero_weight_lora_v5_config" or config.get("classes") != CLASSES:
        raise ValueError("V5 configuration format/official class order differs")
    if config.get("mode") not in MODES or config.get("architecture") != ARCHITECTURE:
        raise ValueError("V5 mode/architecture differs")
    plan_path = root/"logs/zero_weight_v5_plan.json"
    check_hash(plan_path, config["plan_sha256"])
    plan = read_json(plan_path)
    fields = ("classes", "settings", "source_pins", "input_receipt", "train_indices", "val_indices",
              "weight_summary_sha256", "parent_checkpoint", "parent_checkpoint_sha256",
              "baseline_submission_sha256", "baseline_correct_count")
    if any(config.get(key) != plan.get(key) for key in fields):
        raise ValueError("V5 configuration differs from its locked plan")
    if any(config["settings"].get(key) != value for key, value in FIXED_SETTINGS.items()):
        raise ValueError("V5 fixed training settings differ")
    required = {Path(__file__).resolve(), Path(prep.__file__).resolve(), Path(v4.__file__).resolve(),
                Path(base.__file__).resolve(), Path(hse.__file__).resolve(),
                (root/"src/refine_lora.py").resolve(), (root/"src/validate_cosine_fusion.py").resolve()}
    if not required.issubset({Path(path).resolve() for path in config["source_pins"]}):
        raise ValueError("V5 source pins omit a required implementation")
    for path, digest in config["source_pins"].items():
        if not Path(path).is_absolute():
            raise ValueError("Source pins require absolute paths")
        check_hash(path, digest)
    expected_parent = (root/"experiments/lora_task_v4/candidates/epoch_05_ema.pt").resolve()
    if Path(config["parent_checkpoint"]).resolve() != expected_parent:
        raise ValueError("V5 must continue the complete V4 epoch5 EMA winner")
    check_hash(expected_parent, config["parent_checkpoint_sha256"])
    weight_summary_path = root/"experiments/lora_zero_weight_v5/weights/summary.json"
    check_hash(weight_summary_path, config["weight_summary_sha256"])
    prep.verify_existing(root, read_json(weight_summary_path))
    if config.get("dino_repository") != base.repository_info(root) or config.get("official_dino_sha256") != base.verify_official_weights(root):
        raise ValueError("V5 DINOv2 vendor/official weights changed")
    if config["baseline_correct_count"] != 1915:
        raise ValueError("V5 requires the1915-correct fixed baseline")


def checked_plan(root, mode):
    if mode not in MODES:
        raise ValueError("Unknown paired experiment mode")
    path = root/"logs/zero_weight_v5_plan.json"
    plan = read_json(path)
    if plan.get("classes") != CLASSES or any(plan.get("settings", {}).get(key) != value for key, value in FIXED_SETTINGS.items()):
        raise ValueError("V5 plan official classes/fixed settings differ")
    parent_payload = torch.load(plan["parent_checkpoint"], map_location="cpu", weights_only=True)
    # Reuse the original V4 loader's complete provenance verification, without
    # constructing another model or reinitializing any learned new-block adapter.
    v4._verify_payload(root, parent_payload)
    if parent_payload.get("epoch") != 5 or parent_payload.get("weight_source") != "ema":
        raise ValueError("V5 parent must be V4 epoch5 EMA")
    fields = ("classes", "settings", "source_pins", "input_receipt", "train_indices", "val_indices",
              "weight_summary_sha256", "parent_checkpoint", "parent_checkpoint_sha256",
              "baseline_submission_sha256", "baseline_correct_count")
    config = {"format": "zero_weight_lora_v5_config", "mode": mode,
        **{key: plan[key] for key in fields}, "architecture": ARCHITECTURE, "plan_sha256": sha256(path),
        "dino_repository": parent_payload["dino_repository"],
        "official_dino_sha256": parent_payload["official_dino_sha256"],
        "validation_batch_size": 4, "validation_weighted": False,
        "train_sample_weights": "exact normal1 uncertain.15 very_uncertain0" if mode == "weighted" else "all1 matched continuation control",
        "loss_normalization": "one complete CPU effective32 group sum(w), shared by every GPU microbatch",
        "class_normalization": False, "sampling": "uniform full-member shuffle; identical paired seeds/augmentation/order",
        "zero_group_behavior": "zero_grad only; no forward/backward/scaler/optimizer.step/EMA/weight_decay",
        "amp": "float16", "initial_grad_scale": 1024., "gradient_clip": 1.,
        "activation_checkpointing": "all adapted blocks4-11; non-reentrant; preserve RNG",
        "preprocessing": "original FaceImages RGB bicubic224 ImageNet normalization",
        "augmentation": "unchanged flip.5 rotation+/-5deg brightness[.95,1.05]; no Mixup/CutMix",
        "internal_test_read": False, "external_answers_used": False,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    _verify_environment(root, config)
    return plan, config


def model_payload(model, config, epoch, source):
    return {"format": FORMAT, "config": config, "classes": CLASSES, "mode": config["mode"],
            "epoch": epoch, "weight_source": source, "adapter_state": base.adapter_state(model),
            "source_pins": config["source_pins"], "dino_repository": config["dino_repository"],
            "official_dino_sha256": config["official_dino_sha256"],
            "parent_sha": config["parent_checkpoint_sha256"],
            "weight_summary_sha256": config["weight_summary_sha256"],
            "train_indices": config["train_indices"], "val_indices": config["val_indices"]}


def _verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("Unsupported zero-weight LoRA checkpoint/classes")
    saved = payload["config"]
    _verify_environment(root, saved)
    if config is not None and saved != config:
        raise ValueError("Resume V5 configuration changed")
    for key in ("source_pins", "dino_repository", "official_dino_sha256", "mode", "weight_summary_sha256"):
        if payload.get(key) != saved[key]:
            raise ValueError(f"V5 checkpoint/config binding differs: {key}")
    if payload.get("parent_sha") != saved["parent_checkpoint_sha256"]:
        raise ValueError("V5 parent checkpoint binding differs")
    if payload.get("train_indices") != saved["train_indices"] or payload.get("val_indices") != saved["val_indices"]:
        raise ValueError("V5 checkpoint train/validation members differ")
    v4._validate_adapter_tensors(payload.get("adapter_state"))
    return saved


def load_checkpoint(root, path, device="cpu"):
    root = Path(root).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = _verify_payload(root, payload)
    model = base.make_model(root, config["architecture"], "cpu")
    v4._strict_state(model, payload["adapter_state"], ALL_NAMES)
    model = v4._attach_checkpointing(model, True).to(device).eval()
    excluded = {"adapter_state", "optimizer", "scaler", "rng", "ema_state"}
    return model, {key: value for key, value in payload.items() if key not in excluded}


def evaluate_candidate(model, loader, device, output, config, epoch, source):
    name = f"epoch_{epoch:02d}_{source}"
    print(json.dumps({"stage": "validate_zero_weight_v5", "mode": config["mode"], "name": name}), flush=True)
    original_metrics, tta_metrics, original, flipped = validate_views(model, loader, device)
    paths = {"model": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy", "flip": f"candidates/{name}_flip.npy"}
    payload = model_payload(model, config, epoch, source)
    payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics)
    base.save_torch(output/paths["model"], payload)
    save_array(output/paths["original"], original)
    save_array(output/paths["flip"], flipped)
    candidate = {"name": name, "mode": config["mode"], "epoch": epoch, "weight_source": source,
        **paths, "path": paths["model"], "candidate_json": f"candidates/{name}.json",
        "val_indices": config["val_indices"], "classes": CLASSES,
        "artifact_sha256": {key: sha256(output/path) for key, path in paths.items()},
        "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
        "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
        "parent_sha": config["parent_checkpoint_sha256"], "weight_summary_sha256": config["weight_summary_sha256"],
        "validation_weighted": False}
    base.write_json(output/candidate["candidate_json"], candidate)
    print(json.dumps({"stage": "zero_weight_candidate_complete", "mode": config["mode"], "name": name,
        "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics}), flush=True)
    return candidate


def verify_candidate(root, output, candidate, config):
    expected_name = f"epoch_{candidate['epoch']:02d}_{candidate['weight_source']}"
    if candidate.get("name") != expected_name or candidate.get("classes") != CLASSES or candidate.get("mode") != config["mode"]:
        raise ValueError("Candidate identity/official class/mode differs")
    if candidate.get("val_indices") != config["val_indices"] or candidate.get("plan_sha256") != config["plan_sha256"] or candidate.get("validation_weighted") is not False:
        raise ValueError("Candidate validation member/plan/weighting differs")
    for key, expected in [("source_pins", config["source_pins"]), ("parent_sha", config["parent_checkpoint_sha256"]), ("weight_summary_sha256", config["weight_summary_sha256"])]:
        if candidate.get(key) != expected:
            raise ValueError("Candidate new source/parent/weight binding differs")
    expected_paths = {"model": f"candidates/{expected_name}.pt", "original": f"candidates/{expected_name}_original.npy", "flip": f"candidates/{expected_name}_flip.npy", "candidate_json": f"candidates/{expected_name}.json"}
    if any(candidate.get(key) != value for key, value in expected_paths.items()) or candidate.get("path") != candidate["model"]:
        raise ValueError("Candidate paths differ from its name")
    if read_json(output/candidate["candidate_json"]) != candidate:
        raise ValueError("Candidate JSON differs from its committed pointer")
    for key in ("model", "original", "flip"):
        check_hash(output/candidate[key], candidate["artifact_sha256"][key])
    payload = torch.load(output/candidate["model"], map_location="cpu", weights_only=True)
    _verify_payload(root, payload, config)
    if payload["epoch"] != candidate["epoch"] or payload["weight_source"] != candidate["weight_source"]:
        raise ValueError("Candidate model epoch/source differs")
    original = np.load(output/candidate["original"], allow_pickle=False)
    flipped = np.load(output/candidate["flip"], allow_pickle=False)
    for array in (original, flipped):
        if array.dtype != np.float32 or array.shape != (len(config["val_indices"]), 6) or not np.isfinite(array).all() or np.any(array < 0) or not np.allclose(array.sum(1), 1., atol=2e-6, rtol=0):
            raise ValueError("Candidate probability dtype/shape/range differs")
    records = val_rows(root)
    if [int(record["manifest_index"]) for record in records] != config["val_indices"]:
        raise ValueError("Candidate probability validation member order differs")
    targets = np.asarray([int(record["target"]) for record in records], dtype=np.int64)
    computed = {"single_original_metrics": metrics(targets, original), "single_tta_metrics": metrics(targets, (original+flipped)*.5)}
    if any(computed[key] != candidate[key] or payload[key] != candidate[key] for key in computed):
        raise ValueError("Candidate metrics do not reproduce from probabilities/model metadata")
    return candidate


def commit_epoch(root, output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates):
    _verify_environment(root, config)
    payload = {**model_payload(model, config, epoch, "raw"), "optimizer": cpu_tree(optimizer.state_dict()),
        "scaler": scaler.state_dict(), "ema_state": ema.state_dict(), "rng": base.rng_state(),
        "counters": counters, "successful_steps": counters["successful_steps"],
        "cumulative_skips": counters["amp_skipped_steps"],
        "zero_weight_groups": counters["zero_weight_groups"],
        "history": history, "candidates": candidates}
    relative = f"epochs/epoch_{epoch:02d}.pt"
    base.save_torch(output/relative, payload)
    atomic_copy(output/relative, output/"last.pt")
    base.write_json(output/"last.json", {"epoch": epoch, "checkpoint": relative,
        "sha256": sha256(output/relative), "plan_sha256": config["plan_sha256"], "mode": config["mode"]})
    base.write_json(output/"history.json", history)
    base.write_json(output/"candidates.json", candidates)


def resume_checkpoint(root, output, config):
    pointer = read_json(output/"last.json")
    epoch = pointer.get("epoch")
    if not isinstance(epoch, int) or not 0 <= epoch <= config["settings"]["epochs"]:
        raise ValueError("Resume completed epoch invalid")
    if pointer.get("plan_sha256") != config["plan_sha256"] or pointer.get("mode") != config["mode"] or pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt":
        raise ValueError("Resume pointer plan/mode/epoch filename differs")
    check_hash(output/pointer["checkpoint"], pointer["sha256"])
    payload = torch.load(output/pointer["checkpoint"], map_location="cpu", weights_only=True)
    _verify_payload(root, payload, config)
    if payload["epoch"] != epoch or len(payload["history"]) != epoch or payload["successful_steps"] != payload["counters"]["successful_steps"]:
        raise ValueError("Resume history/counter/epoch differs")
    expected = ["epoch_00_baseline"] + [f"epoch_{index:02d}_{source}" for index in range(1, epoch+1) for source in ("raw", "ema")]
    if [candidate["name"] for candidate in payload["candidates"]] != expected:
        raise ValueError("Resume committed candidate list incomplete")
    for candidate in payload["candidates"]:
        verify_candidate(root, output, candidate, config)
    if not (output/"last.pt").exists() or sha256(output/"last.pt") != pointer["sha256"]:
        atomic_copy(output/pointer["checkpoint"], output/"last.pt")
    return payload


def write_summary(root, output, config, history, candidates, counters, ema_updates):
    if len(history) != 4 or len(candidates) != 9:
        raise ValueError("Cannot finalize an incomplete four-epoch/nine-candidate arm")
    for candidate in candidates:
        verify_candidate(root, output, candidate, config)
    if any(row["seen"] != 21248 for row in history) or any(row["successful_optimizer_steps"] <= 0 for row in history):
        raise ValueError("A training epoch missed members or did not update")
    if sum(row["successful_optimizer_steps"] for row in history) != counters["successful_steps"] or ema_updates != counters["successful_steps"]:
        raise ValueError("Successful optimizer/EMA update counts differ")
    if sum(row["amp_skipped_steps"] for row in history) != counters["amp_skipped_steps"] or sum(row["zero_weight_groups"] for row in history) != counters["zero_weight_groups"]:
        raise ValueError("AMP/zero-weight-group counters differ")
    pointer = read_json(output/"last.json")
    check_hash(output/pointer["checkpoint"], pointer["sha256"])
    check_hash(output/"last.pt", pointer["sha256"])
    base.write_json(output/"history.json", history)
    base.write_json(output/"candidates.json", candidates)
    paths = ["config.json", "history.json", "candidates.json", "last.json", "last.pt", pointer["checkpoint"]]
    for candidate in candidates:
        paths.extend([candidate[key] for key in ("candidate_json", "model", "original", "flip")])
    summary = {"format": "zero_weight_lora_v5_summary", "state": "complete", "classes": CLASSES,
        "mode": config["mode"], "epochs_completed": 4, "candidate_count": 9,
        "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
        "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
        "weight_summary_sha256": config["weight_summary_sha256"],
        "counters": counters, "successful_optimizer_steps": counters["successful_steps"],
        "amp_skipped_steps": counters["amp_skipped_steps"], "zero_weight_groups": counters["zero_weight_groups"],
        "ema_updates": ema_updates, "validation_weighted": False,
        "train_sample_weights": config["train_sample_weights"], "class_normalization": False,
        "internal_test_read": False, "external_answers_used": False,
        "candidates": candidates, "artifact_sha256": {path: sha256(output/path) for path in paths}}
    base.write_json(output/"summary.json", summary)
    print(json.dumps({"stage": "zero_weight_v5_arm_complete", "mode": config["mode"], "epochs_completed": 4,
        "successful_optimizer_steps": counters["successful_steps"], "zero_weight_groups": counters["zero_weight_groups"]}), flush=True)
    return summary


def load_training_weights(root, records, config):
    folder = root/"experiments/lora_zero_weight_v5/weights"
    summary = prep.verify_existing(root, read_json(folder/"summary.json"))
    check_hash(folder/"summary.json", config["weight_summary_sha256"])
    entries = prep.original.csv_rows(folder/"train_weights.csv")
    for record, entry in zip(records, entries):
        if (int(entry["manifest_index"]) != record["manifest_index"] or entry["path"] != record["path"]
            or int(entry["target"]) != record["target"] or entry["label"] != record["label"]):
            raise ValueError("Training image/label/weight member order differs")
    if len(records) != len(entries) or len(records) != 21248:
        raise ValueError("Training image/weight count differs")
    weights = np.load(folder/"training_weights.npy", allow_pickle=False)
    return np.ones_like(weights) if config["mode"] == "uniform_control" else weights


def run_training(root, mode, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    plan, config = checked_plan(root, mode)
    records, receipt = hse.load_trainval(root)
    if receipt != config["input_receipt"]:
        raise ValueError("Locked train/validation data receipt differs")
    for split in ("train", "val"):
        if [record["manifest_index"] for record in records[split]] != config[split+"_indices"]:
            raise ValueError("Locked train/validation member order differs")
    weights = load_training_weights(root, records["train"], config)
    output = root/"experiments/lora_zero_weight_v5"/mode
    output.mkdir(parents=True, exist_ok=True)
    (output/"candidates").mkdir(exist_ok=True)
    (output/"epochs").mkdir(exist_ok=True)
    if (output/"config.json").exists():
        if read_json(output/"config.json") != config:
            raise ValueError("Existing arm configuration changed; its outputs are preserved")
    else:
        base.write_json(output/"config.json", config)
    base.seed_everything(config["settings"]["seed"])
    # Critical: load all learned blocks4-11/head from the V4 winner, rather than
    # build_model() which would discard its learned blocks4-7 by going backward.
    model, parent_metadata = v4.load_checkpoint(root, config["parent_checkpoint"], device)
    model = v4._attach_checkpointing(model, True)
    if parent_metadata["train_indices"] != config["train_indices"] or parent_metadata["val_indices"] != config["val_indices"]:
        raise ValueError("V4 parent and V5 train/validation members differ")
    parent_state = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=True)["adapter_state"]
    current = base.adapter_state(model)
    if set(current) != ALL_NAMES or any(not torch.equal(current[key], parent_state[key]) for key in ALL_NAMES):
        raise ValueError("V5 model initialization did not exactly preserve all eighteen parent tensors")
    del current
    settings = config["settings"]
    optimizer = make_optimizer(model, settings)
    amp = torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", init_scale=1024., enabled=amp)
    ema = AdapterEMA(model, settings["ema_decay"])
    val_loader = base.make_loader(records["val"], 4, 0, settings["seed"], device=device)
    total_steps = math.ceil(len(records["train"])/32)*settings["epochs"]
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    start_epoch, history, candidates = 0, [], []
    if (output/"last.json").exists():
        saved = resume_checkpoint(root, output, config)
        v4._strict_state(model, saved["adapter_state"], ALL_NAMES)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        ema.load_state_dict(saved["ema_state"])
        base.restore_rng(saved["rng"])
        start_epoch, counters = saved["epoch"], saved["counters"]
        history, candidates = saved["history"], saved["candidates"]
        if ema.updates != counters["successful_steps"]:
            raise ValueError("Resumed EMA/optimizer counts differ")
        del saved
        print(json.dumps({"stage": "resume_zero_weight_v5", "mode": mode, "completed_epoch": start_epoch}), flush=True)
    else:
        baseline_path = output/"candidates/epoch_00_baseline.json"
        if baseline_path.exists():
            candidate = verify_candidate(root, output, read_json(baseline_path), config)
            baseline_payload = torch.load(output/candidate["model"], map_location="cpu", weights_only=True)
            if any(not torch.equal(baseline_payload["adapter_state"][key], parent_state[key]) for key in ALL_NAMES):
                raise ValueError("Uncommitted baseline does not equal the full V4 parent")
            del baseline_payload
        else:
            candidate = evaluate_candidate(model, val_loader, device, output, config, 0, "baseline")
        candidates.append(candidate)
        commit_epoch(root, output, model, optimizer, scaler, ema, config, 0, counters, history, candidates)
    del parent_state
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if start_epoch == settings["epochs"]:
        return write_summary(root, output, config, history, candidates, counters, ema.updates)
    for epoch in range(start_epoch+1, settings["epochs"]+1):
        started = last_progress = time.monotonic()
        data_epoch = settings["augmentation_epoch_offset"]+epoch-1
        dataset = WeightedFaces(records["train"], weights, settings["seed"], data_epoch)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
            pin_memory=amp, drop_last=False, persistent_workers=False,
            generator=torch.Generator().manual_seed(settings["seed"]+data_epoch))
        model.train()
        seen = forwarded_seen = correct = weighted_sum = weight_sum = plain_sum = epoch_steps = skips = zeros = microbatches = 0
        for group_index, (images, targets, group_weights) in enumerate(loader, 1):
            denominator = float(group_weights.sum().item())
            if should_update(denominator):
                set_learning_rate(optimizer, counters["successful_steps"], total_steps, settings)
            result = train_group(model, optimizer, scaler, ema, images, targets, group_weights,
                                 device=device, batch_size=settings["batch_size"])
            seen += result["seen"]
            forwarded_seen += result["forwarded_seen"]
            correct += result["correct"]
            weighted_sum += result["weighted_loss_sum"]
            weight_sum += result["weight_sum"]
            plain_sum += result["unweighted_loss_sum"]
            microbatches += result["microbatches"]
            if result["zero_weight_group"]:
                zeros += 1
                counters["zero_weight_groups"] += 1
            else:
                successful = result["success"]
                epoch_steps += int(successful)
                skips += int(not successful)
                counters["successful_steps"] += int(successful)
                counters["amp_skipped_steps"] += int(not successful)
                if skips >= 25 and epoch_steps == 0:
                    raise RuntimeError("25AMP skips without a successful optimizer update")
            if time.monotonic()-last_progress >= 45:
                print(json.dumps({"stage": "train_zero_weight_v5", "mode": mode, "epoch": epoch,
                    "group": group_index, "groups": len(loader), "microbatch": microbatches,
                    "seen": seen, "train_loss": weighted_sum/max(weight_sum, 1e-12),
                    "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": skips,
                    "zero_weight_groups": zeros, "cumulative_optimizer_steps": counters["successful_steps"],
                    "elapsed_seconds": time.monotonic()-started}), flush=True)
                last_progress = time.monotonic()
        if seen != 21248 or epoch_steps <= 0:
            raise RuntimeError("Epoch missed training members or never updated")
        if any(parameter._version != frozen_versions[name] or parameter.grad is not None
               for name, parameter in model.named_parameters() if not parameter.requires_grad):
            raise RuntimeError("Frozen DINOv2 base changed or received gradients")
        raw_candidate = evaluate_candidate(model, val_loader, device, output, config, epoch, "raw")
        raw_state = base.adapter_state(model)
        try:
            v4._strict_state(model, ema.state_dict()["shadow"], ALL_NAMES)
            ema_candidate = evaluate_candidate(model, val_loader, device, output, config, epoch, "ema")
        finally:
            v4._strict_state(model, raw_state, ALL_NAMES)
        candidates.extend((raw_candidate, ema_candidate))
        entry = {"epoch": epoch, "mode": mode, "train_loss": weighted_sum/weight_sum,
            "unweighted_train_loss_forwarded_images": plain_sum/max(1, forwarded_seen),
            "train_accuracy_forwarded_images": correct/max(1, forwarded_seen), "seen": seen,
            "forwarded_seen": forwarded_seen,
            "zero_weight_groups": zeros, "forwarded_microbatches": microbatches,
            "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": skips,
            "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
            "augmentation_epoch": data_epoch, "amp_scale": scaler.get_scale(),
            "elapsed_seconds": time.monotonic()-started,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "raw_original_metrics": raw_candidate["single_original_metrics"], "raw_tta_metrics": raw_candidate["single_tta_metrics"],
            "ema_original_metrics": ema_candidate["single_original_metrics"], "ema_tta_metrics": ema_candidate["single_tta_metrics"]}
        history.append(entry)
        commit_epoch(root, output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates)
        print(json.dumps({"stage": "zero_weight_epoch_committed", **entry}), flush=True)
    return write_summary(root, output, config, history, candidates, counters, ema.updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    run_training(args.project_root, args.mode, args.device)


if __name__ == "__main__":
    main()
