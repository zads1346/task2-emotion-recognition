"""Matched standard-CE versus CE+flip-consistency LoRA continuation.

Both arms use identical uniformly weighted images, shuffle, paired-view forward,
augmentation, initialization and dropout/RNG order. Only the symmetric-KL
coefficient differs: control0 versus treatment.1. Two visually confirmed invalid
training images are omitted identically in both arms; source images/labels stay
unchanged. Validation stays unweighted and separate.
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
from torch.utils.data import DataLoader

import train_task_lora_v4 as v4
import train_lora as base
import train_hse_specialist as hse
from refine_lora import AdapterEMA
from validate_cosine_fusion import metrics, val_rows

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "flip_consistency_lora_v6"
CLASSES = list(v4.CLASSES)
ARCHITECTURE = dict(v4.ARCHITECTURE)
ALL_NAMES = set(v4.ALL_NAMES)
MODES = ("ce_control", "flip_consistency")
MODE_SETTINGS = {"ce_control": {"consistency_lambda": 0.}, "flip_consistency": {"consistency_lambda": .1}}
EXCLUDED_TRAIN_INDICES = [7727, 10448]
FIXED_SETTINGS = {"epochs": 4, "seed": 42, "augmentation_epoch_offset": 60,
    "label_smoothing": 0., "weight_decay": .001, "lr_new": 1e-5,
    "lr_inherited": 2.5e-6, "lr_head": 1e-5, "warmup_successful_steps": 50,
    "min_lr_ratio": .1, "ema_decay": .995, "batch_size": 4, "accumulation": 8,
    "effective_batch_size": 32, "workers": 0}

sha256, check_hash, read_json = v4.sha256, v4.check_hash, v4.read_json
save_array, atomic_copy, cpu_tree = v4.save_array, v4.atomic_copy, v4.cpu_tree
make_optimizer, set_learning_rate = v4.make_optimizer, v4.set_learning_rate
def paired_loss(logits_original, logits_flip, targets, consistency_lambda):
    """Return standard paired-view CE and fully differentiable symmetric KL(T=1).

The two arms always compute both terms in FP32, including lambda0. No branch is
detached and no pseudo-label or sample weight is used.
"""
    coefficient = float(consistency_lambda)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Consistency coefficient must be finite and nonnegative")
    if (logits_original.ndim != 2 or logits_original.shape[1] != 6
        or logits_flip.shape != logits_original.shape or targets.shape != (len(logits_original),)
        or len(targets) == 0 or logits_flip.device != logits_original.device):
        raise ValueError("Expected matching nonempty six-class original/flip logits and targets")
    with torch.autocast(device_type=logits_original.device.type, enabled=False):
        first, second = logits_original.float(), logits_flip.float()
        ce = .5*(F.cross_entropy(first, targets, reduction="mean", label_smoothing=0.)
                 + F.cross_entropy(second, targets, reduction="mean", label_smoothing=0.))
        log_first, log_second = F.log_softmax(first, dim=1), F.log_softmax(second, dim=1)
        probability_first, probability_second = log_first.exp(), log_second.exp()
        first_to_second = (probability_first*(log_first-log_second)).sum(1)
        second_to_first = (probability_second*(log_second-log_first)).sum(1)
        consistency = .5*(first_to_second+second_to_first).mean()
        total = ce+coefficient*consistency
    return {"total": total, "ce": ce, "consistency": consistency}


def train_group(model, optimizer, scaler, ema, images, targets,
                device="cuda", batch_size=4, consistency_lambda=.1):
    """One uniform effective32 group, physical batch2*logical microbatch.

Actual microbatch sample count/full-group count scales every backward, including
partial last groups. Both arms concatenate x then flip(x) in the same forward.
"""
    count = len(targets)
    if len(images) != count or not 1 <= count <= 32 or batch_size < 1:
        raise ValueError("CPU group images/targets must match and contain1..32 rows")
    optimizer.zero_grad(set_to_none=True)
    amp = torch.device(device).type == "cuda"
    ce_sum = consistency_sum = loss_sum = correct = microbatches = 0
    for start in range(0, count, batch_size):
        stop = min(start+batch_size, count)
        micro_images = images[start:stop].to(device, non_blocking=True)
        micro_targets = targets[start:stop].to(device, non_blocking=True)
        micro_count = len(micro_targets)
        paired = torch.cat((micro_images, micro_images.flip(-1)), dim=0)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            combined_logits = model(paired)
        first, second = combined_logits[:micro_count], combined_logits[micro_count:]
        losses = paired_loss(first, second, micro_targets, consistency_lambda)
        if any(not torch.isfinite(value).item() for value in losses.values()):
            raise RuntimeError("Nonfinite paired CE/consistency/total training loss")
        scaler.scale(losses["total"]*(micro_count/count)).backward()
        ce_sum += losses["ce"].detach().item()*micro_count
        consistency_sum += losses["consistency"].detach().item()*micro_count
        loss_sum += losses["total"].detach().item()*micro_count
        prediction = ((first.detach().float().softmax(1)+second.detach().float().softmax(1))*.5).argmax(1)
        correct += (prediction == micro_targets).sum().item()
        microbatches += 1
    successful = v4.optimizer_step(model, optimizer, scaler, ema)
    return {"success": bool(successful), "seen": count, "ce_sum": ce_sum,
            "consistency_sum": consistency_sum, "loss_sum": loss_sum,
            "correct": correct, "microbatches": microbatches}


def _verify_audit(root, config):
    folder = root/"experiments/lora_flip_consistency_v6/audit"
    check_hash(folder/"summary.json", config["audit_summary_sha256"])
    audit = read_json(folder/"summary.json")
    if (audit.get("state") != "complete" or audit.get("classes") != CLASSES
        or audit.get("train_rows") != 21248 or audit.get("sample_rows") != 192
        or audit.get("training_weights") != "uniform1"):
        raise ValueError("Train-only192-image audit state/classes/uniform training count differs")
    if any(audit.get(key) is not False for key in ("labels_changed", "images_changed", "weights_changed", "internal_test_read", "external_answers_used")):
        raise ValueError("Audit must not change labels/images/weights or use internal-test/external answers")
    if (audit.get("training_membership_changed") is not True
        or audit.get("excluded_train_indices") != EXCLUDED_TRAIN_INDICES
        or audit.get("original_train_indices") != config["original_train_indices"]
        or audit.get("effective_train_rows") != len(config["train_indices"])):
        raise ValueError("Audit must apply only the same two visually confirmed invalid-image exclusions")
    for split in ("train", "val"):
        if audit.get(split+"_indices") != config[split+"_indices"]:
            raise ValueError("Audit training/validation membership differs")
    if audit.get("train_membership_sha256") != config["input_receipt"]["train_rows_sha256"]:
        raise ValueError("Audit train membership signature differs from the verified input receipt")
    if "filtered_membership_sha256" in config and audit.get("filtered_membership_sha256") != config["filtered_membership_sha256"]:
        raise ValueError("Audit filtered training membership signature differs from the plan")
    if not isinstance(audit.get("source_pins"), dict) or not isinstance(audit.get("file_sha256"), dict):
        raise ValueError("Audit requires source and artifact checksum maps")
    for path, digest in audit["source_pins"].items():
        if not Path(path).is_absolute():
            raise ValueError("Audit source pins require absolute paths")
        check_hash(path, digest)
    for relative, digest in audit["file_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Audit artifact escapes its directory")
        check_hash(folder/path, digest)
    return audit


@torch.inference_mode()
def validate_views(model, loader, device, consistency_lambda=0.):
    model.eval()
    original, flipped, targets = [], [], []
    ce_sum = consistency_sum = total_sum = 0.
    amp = torch.device(device).type == "cuda"
    started = last_progress = time.monotonic()
    for batch_index, (images, target) in enumerate(loader, 1):
        images, gpu_targets = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=amp):
            first, second = model(images), model(images.flip(-1))
        losses = paired_loss(first, second, gpu_targets, consistency_lambda)
        original.append(first.float().softmax(1).cpu().numpy())
        flipped.append(second.float().softmax(1).cpu().numpy())
        targets.extend(target.tolist())
        ce_sum += losses["ce"].item()*len(target)
        consistency_sum += losses["consistency"].item()*len(target)
        total_sum += losses["total"].item()*len(target)
        if time.monotonic()-last_progress >= 45:
            print(json.dumps({"stage": "validate_flip_consistency_v6", "batch": batch_index,
                "batches": len(loader), "seen": len(targets), "elapsed_seconds": time.monotonic()-started}), flush=True)
            last_progress = time.monotonic()
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    for value in (original, flipped):
        if value.dtype != np.float32 or value.shape != (len(targets), 6) or not np.isfinite(value).all() or np.any(value < 0) or not np.allclose(value.sum(1), 1., atol=2e-6, rtol=0):
            raise ValueError("Invalid unweighted validation probabilities")
    losses = {"ce": ce_sum/len(targets), "consistency": consistency_sum/len(targets),
        "total": total_sum/len(targets), "consistency_lambda": float(consistency_lambda), "samples": len(targets)}
    return metrics(np.asarray(targets), original), metrics(np.asarray(targets), (original+flipped)*.5), original, flipped, losses


def _verify_environment(root, config):
    if config.get("format") != "flip_consistency_lora_v6_config" or config.get("classes") != CLASSES:
        raise ValueError("V6 configuration format/official class order differs")
    if config.get("mode") not in MODES or config.get("architecture") != ARCHITECTURE:
        raise ValueError("V6 mode/architecture differs")
    plan_path = root/"logs/flip_consistency_v6_plan.json"
    check_hash(plan_path, config["plan_sha256"])
    plan = read_json(plan_path)
    fields = ("classes", "settings", "mode_settings", "source_pins", "input_receipt", "train_indices", "val_indices",
              "original_train_indices", "excluded_train_indices",
              "audit_summary_sha256", "parent_checkpoint", "parent_checkpoint_sha256",
              "baseline_submission_sha256", "baseline_correct_count")
    if any(config.get(key) != plan.get(key) for key in fields):
        raise ValueError("V6 configuration differs from its locked plan")
    original_members, excluded = config["original_train_indices"], config["excluded_train_indices"]
    if (len(original_members) != 21248 or len(set(original_members)) != 21248 or excluded != EXCLUDED_TRAIN_INDICES
        or not set(excluded).issubset(original_members)
        or config["train_indices"] != [index for index in original_members if index not in excluded]):
        raise ValueError("Effective train membership must be the original21248 minus exactly the two audited invalid images")
    if config.get("effective_train_rows") != len(config["train_indices"]) or len(config["train_indices"]) != 21246:
        raise ValueError("V6 effective training count differs")
    if "filtered_membership_sha256" in plan and config.get("filtered_membership_sha256") != plan["filtered_membership_sha256"]:
        raise ValueError("V6 filtered training signature differs from its locked plan")
    if any(key in plan and config.get(key) != plan[key] for key in ("dino_repository", "official_dino_sha256")):
        raise ValueError("V6 official backbone provenance differs from the plan")
    if any(config["settings"].get(key) != value for key, value in FIXED_SETTINGS.items()):
        raise ValueError("V6 fixed training settings differ")
    if config.get("mode_settings") != MODE_SETTINGS or config.get("consistency_lambda") != MODE_SETTINGS[config["mode"]]["consistency_lambda"]:
        raise ValueError("V6 fixed mode consistency coefficients differ")
    required = {Path(__file__).resolve(), Path(v4.__file__).resolve(),
                Path(base.__file__).resolve(), Path(hse.__file__).resolve(),
                (root/"src/refine_lora.py").resolve(), (root/"src/validate_cosine_fusion.py").resolve()}
    if not required.issubset({Path(path).resolve() for path in config["source_pins"]}):
        raise ValueError("V6 source pins omit a required implementation")
    for path, digest in config["source_pins"].items():
        if not Path(path).is_absolute():
            raise ValueError("Source pins require absolute paths")
        check_hash(path, digest)
    expected_parent = (root/"experiments/lora_task_v4/candidates/epoch_05_ema.pt").resolve()
    if Path(config["parent_checkpoint"]).resolve() != expected_parent:
        raise ValueError("V6 must continue the complete V4 epoch5 EMA winner")
    check_hash(expected_parent, config["parent_checkpoint_sha256"])
    _verify_audit(root, config)
    if config.get("dino_repository") != base.repository_info(root) or config.get("official_dino_sha256") != base.verify_official_weights(root):
        raise ValueError("V6 DINOv2 vendor/official weights changed")
    if config["baseline_correct_count"] != 1915:
        raise ValueError("V6 requires the1915-correct fixed baseline")


def checked_plan(root, mode):
    if mode not in MODES:
        raise ValueError("Unknown paired experiment mode")
    path = root/"logs/flip_consistency_v6_plan.json"
    plan = read_json(path)
    if plan.get("classes") != CLASSES or any(plan.get("settings", {}).get(key) != value for key, value in FIXED_SETTINGS.items()):
        raise ValueError("V6 plan official classes/fixed settings differ")
    parent_payload = torch.load(plan["parent_checkpoint"], map_location="cpu", weights_only=True)
    # Reuse the original V4 loader's complete provenance verification, without
    # constructing another model or reinitializing any learned new-block adapter.
    v4._verify_payload(root, parent_payload)
    if parent_payload.get("epoch") != 5 or parent_payload.get("weight_source") != "ema":
        raise ValueError("V6 parent must be V4 epoch5 EMA")
    fields = ("classes", "settings", "mode_settings", "source_pins", "input_receipt", "train_indices", "val_indices",
              "original_train_indices", "excluded_train_indices",
              "audit_summary_sha256", "parent_checkpoint", "parent_checkpoint_sha256",
              "baseline_submission_sha256", "baseline_correct_count")
    config = {"format": "flip_consistency_lora_v6_config", "mode": mode,
        **{key: plan[key] for key in fields}, "architecture": ARCHITECTURE, "plan_sha256": sha256(path),
        "dino_repository": parent_payload["dino_repository"],
        "official_dino_sha256": parent_payload["official_dino_sha256"],
        "validation_batch_size": 4, "validation_weighted": False,
        "effective_train_rows": len(plan["train_indices"]),
        "consistency_lambda": MODE_SETTINGS[mode]["consistency_lambda"],
        "train_sample_weights": "uniform1; identical21246 train members after excluding two visually confirmed invalid images",
        "training_membership_changed": True,
        "loss_normalization": "microbatch actual image count / complete CPU group image count",
        "paired_view_forward": "one concat([x,flip(x)],dim=0) per logical microbatch4; physical batch8",
        "loss_definition": ".5*(CE(z1,y)+CE(z2,y)) + lambda*.5*(KL(p1||p2)+KL(p2||p1)); T1; no detach; no smoothing",
        "train_accuracy_definition": "argmax of mean original/flip softmax probabilities",
        "class_normalization": False, "sampling": "uniform full-member shuffle; identical paired seeds/augmentation/order",
        "amp": "float16", "initial_grad_scale": 1024., "gradient_clip": 1.,
        "activation_checkpointing": "all adapted blocks4-11; non-reentrant; preserve RNG",
        "preprocessing": "original FaceImages RGB bicubic224 ImageNet normalization",
        "augmentation": "unchanged flip.5 rotation+/-5deg brightness[.95,1.05]; no Mixup/CutMix",
        "internal_test_read": False, "external_answers_used": False,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    if "filtered_membership_sha256" in plan:
        config["filtered_membership_sha256"] = plan["filtered_membership_sha256"]
    _verify_environment(root, config)
    return plan, config


def model_payload(model, config, epoch, source):
    return {"format": FORMAT, "config": config, "classes": CLASSES, "mode": config["mode"],
            "epoch": epoch, "weight_source": source, "adapter_state": base.adapter_state(model),
            "source_pins": config["source_pins"], "dino_repository": config["dino_repository"],
            "official_dino_sha256": config["official_dino_sha256"],
            "parent_sha": config["parent_checkpoint_sha256"],
            "audit_summary_sha256": config["audit_summary_sha256"],
            "original_train_indices": config["original_train_indices"],
            "excluded_train_indices": config["excluded_train_indices"],
            "train_indices": config["train_indices"], "val_indices": config["val_indices"]}


def _verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("Unsupported flip-consistency LoRA checkpoint/classes")
    saved = payload["config"]
    _verify_environment(root, saved)
    if config is not None and saved != config:
        raise ValueError("Resume V6 configuration changed")
    for key in ("source_pins", "dino_repository", "official_dino_sha256", "mode", "audit_summary_sha256"):
        if payload.get(key) != saved[key]:
            raise ValueError(f"V6 checkpoint/config binding differs: {key}")
    if payload.get("parent_sha") != saved["parent_checkpoint_sha256"]:
        raise ValueError("V6 parent checkpoint binding differs")
    if payload.get("train_indices") != saved["train_indices"] or payload.get("val_indices") != saved["val_indices"]:
        raise ValueError("V6 checkpoint train/validation members differ")
    if payload.get("original_train_indices") != saved["original_train_indices"] or payload.get("excluded_train_indices") != saved["excluded_train_indices"]:
        raise ValueError("V6 checkpoint original/audited-excluded training members differ")
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
    print(json.dumps({"stage": "validate_flip_consistency_v6", "mode": config["mode"], "name": name}), flush=True)
    original_metrics, tta_metrics, original, flipped, validation_losses = validate_views(model, loader, device, config["consistency_lambda"])
    paths = {"model": f"candidates/{name}.pt", "original": f"candidates/{name}_original.npy", "flip": f"candidates/{name}_flip.npy"}
    payload = model_payload(model, config, epoch, source)
    payload.update(single_original_metrics=original_metrics, single_tta_metrics=tta_metrics,
                   validation_losses=validation_losses)
    base.save_torch(output/paths["model"], payload)
    save_array(output/paths["original"], original)
    save_array(output/paths["flip"], flipped)
    candidate = {"name": name, "mode": config["mode"], "epoch": epoch, "weight_source": source,
        **paths, "path": paths["model"], "candidate_json": f"candidates/{name}.json",
        "val_indices": config["val_indices"], "classes": CLASSES,
        "train_indices": config["train_indices"], "original_train_indices": config["original_train_indices"],
        "excluded_train_indices": config["excluded_train_indices"],
        "artifact_sha256": {key: sha256(output/path) for key, path in paths.items()},
        "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
        "validation_losses": validation_losses,
        "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
        "parent_sha": config["parent_checkpoint_sha256"], "audit_summary_sha256": config["audit_summary_sha256"],
        "validation_weighted": False}
    base.write_json(output/candidate["candidate_json"], candidate)
    print(json.dumps({"stage": "flip_consistency_candidate_complete", "mode": config["mode"], "name": name,
        "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics}), flush=True)
    return candidate


def verify_candidate(root, output, candidate, config):
    expected_name = f"epoch_{candidate['epoch']:02d}_{candidate['weight_source']}"
    if candidate.get("name") != expected_name or candidate.get("classes") != CLASSES or candidate.get("mode") != config["mode"]:
        raise ValueError("Candidate identity/official class/mode differs")
    if candidate.get("val_indices") != config["val_indices"] or candidate.get("plan_sha256") != config["plan_sha256"] or candidate.get("validation_weighted") is not False:
        raise ValueError("Candidate validation member/plan/weighting differs")
    if any(candidate.get(key) != config[key] for key in ("train_indices", "original_train_indices", "excluded_train_indices")):
        raise ValueError("Candidate effective/original/audited-excluded training members differ")
    for key, expected in [("source_pins", config["source_pins"]), ("parent_sha", config["parent_checkpoint_sha256"]), ("audit_summary_sha256", config["audit_summary_sha256"])]:
        if candidate.get(key) != expected:
            raise ValueError("Candidate new source/parent/audit binding differs")
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
    losses = candidate.get("validation_losses")
    if not isinstance(losses, dict) or payload.get("validation_losses") != losses:
        raise ValueError("Candidate validation CE/KL/total checkpoint binding differs")
    if losses.get("samples") != len(config["val_indices"]) or losses.get("consistency_lambda") != config["consistency_lambda"]:
        raise ValueError("Candidate validation loss sample count/coefficient differs")
    if any(not math.isfinite(losses.get(key, float("nan"))) for key in ("ce", "consistency", "total")):
        raise ValueError("Nonfinite validation CE/KL/total loss")
    if not math.isclose(losses["total"], losses["ce"]+config["consistency_lambda"]*losses["consistency"], rel_tol=2e-6, abs_tol=2e-7):
        raise ValueError("Candidate total loss does not match its fixed CE+lambda*KL objective")
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
    if any(row["seen"] != len(config["train_indices"]) for row in history) or any(row["successful_optimizer_steps"] <= 0 for row in history):
        raise ValueError("A training epoch missed members or did not update")
    if sum(row["successful_optimizer_steps"] for row in history) != counters["successful_steps"] or ema_updates != counters["successful_steps"]:
        raise ValueError("Successful optimizer/EMA update counts differ")
    if sum(row["amp_skipped_steps"] for row in history) != counters["amp_skipped_steps"] or counters["zero_weight_groups"] != 0 or any(row["zero_weight_groups"] != 0 for row in history):
        raise ValueError("AMP counter differs or uniform training recorded a zero-weight group")
    if any(not math.isclose(row["train_loss"], row["train_ce"]+config["consistency_lambda"]*row["train_consistency"], rel_tol=2e-6, abs_tol=2e-7) for row in history):
        raise ValueError("Training total loss differs from CE+fixed lambda*KL")
    pointer = read_json(output/"last.json")
    check_hash(output/pointer["checkpoint"], pointer["sha256"])
    check_hash(output/"last.pt", pointer["sha256"])
    base.write_json(output/"history.json", history)
    base.write_json(output/"candidates.json", candidates)
    paths = ["config.json", "history.json", "candidates.json", "last.json", "last.pt", pointer["checkpoint"]]
    for candidate in candidates:
        paths.extend([candidate[key] for key in ("candidate_json", "model", "original", "flip")])
    summary = {"format": "flip_consistency_lora_v6_summary", "state": "complete", "classes": CLASSES,
        "mode": config["mode"], "epochs_completed": 4, "candidate_count": 9,
        "plan_sha256": config["plan_sha256"], "source_pins": config["source_pins"],
        "parent_checkpoint_sha256": config["parent_checkpoint_sha256"],
        "audit_summary_sha256": config["audit_summary_sha256"],
        "effective_train_rows": len(config["train_indices"]), "original_train_rows": len(config["original_train_indices"]),
        "train_indices": config["train_indices"], "original_train_indices": config["original_train_indices"],
        "excluded_train_indices": config["excluded_train_indices"], "val_indices": config["val_indices"],
        "mode_settings": config["mode_settings"], "consistency_lambda": config["consistency_lambda"],
        "counters": counters, "successful_optimizer_steps": counters["successful_steps"],
        "amp_skipped_steps": counters["amp_skipped_steps"], "zero_weight_groups": counters["zero_weight_groups"],
        "ema_updates": ema_updates, "validation_weighted": False,
        "train_sample_weights": config["train_sample_weights"], "class_normalization": False,
        "internal_test_read": False, "external_answers_used": False,
        "candidates": candidates, "artifact_sha256": {path: sha256(output/path) for path in paths}}
    base.write_json(output/"summary.json", summary)
    print(json.dumps({"stage": "flip_consistency_v6_arm_complete", "mode": config["mode"], "epochs_completed": 4,
        "successful_optimizer_steps": counters["successful_steps"], "zero_weight_groups": counters["zero_weight_groups"]}), flush=True)
    return summary



def run_training(root, mode, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    plan, config = checked_plan(root, mode)
    records, receipt = hse.load_trainval(root)
    if receipt != config["input_receipt"]:
        raise ValueError("Locked train/validation data receipt differs")
    if [record["manifest_index"] for record in records["train"]] != config["original_train_indices"]:
        raise ValueError("Original raw training member order differs from the locked plan")
    records["train"] = [record for record in records["train"] if record["manifest_index"] not in set(config["excluded_train_indices"])]
    for split in ("train", "val"):
        if [record["manifest_index"] for record in records[split]] != config[split+"_indices"]:
            raise ValueError("Locked effective train/validation member order differs")
    if "filtered_membership_sha256" in config and hse.signature(records["train"]) != config["filtered_membership_sha256"]:
        raise ValueError("Filtered training membership signature differs")
    output = root/"experiments/lora_flip_consistency_v6"/mode
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
    if parent_metadata["train_indices"] != config["original_train_indices"] or parent_metadata["val_indices"] != config["val_indices"]:
        raise ValueError("V4 parent and V6 train/validation members differ")
    parent_state = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=True)["adapter_state"]
    current = base.adapter_state(model)
    if set(current) != ALL_NAMES or any(not torch.equal(current[key], parent_state[key]) for key in ALL_NAMES):
        raise ValueError("V6 model initialization did not exactly preserve all eighteen parent tensors")
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
        print(json.dumps({"stage": "resume_flip_consistency_v6", "mode": mode, "completed_epoch": start_epoch}), flush=True)
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
        dataset = base.FaceImages(records["train"], training=True, seed=settings["seed"], epoch=data_epoch)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0,
            pin_memory=amp, drop_last=False, persistent_workers=False,
            generator=torch.Generator().manual_seed(settings["seed"]+data_epoch))
        model.train()
        seen = correct = ce_sum = consistency_sum = loss_sum = epoch_steps = skips = microbatches = 0
        for group_index, (images, targets) in enumerate(loader, 1):
            set_learning_rate(optimizer, counters["successful_steps"], total_steps, settings)
            result = train_group(model, optimizer, scaler, ema, images, targets,
                device=device, batch_size=settings["batch_size"], consistency_lambda=config["consistency_lambda"])
            seen += result["seen"]
            correct += result["correct"]
            ce_sum += result["ce_sum"]
            consistency_sum += result["consistency_sum"]
            loss_sum += result["loss_sum"]
            microbatches += result["microbatches"]
            successful = result["success"]
            epoch_steps += int(successful)
            skips += int(not successful)
            counters["successful_steps"] += int(successful)
            counters["amp_skipped_steps"] += int(not successful)
            if skips >= 25 and epoch_steps == 0:
                raise RuntimeError("25AMP skips without a successful optimizer update")
            if time.monotonic()-last_progress >= 45:
                print(json.dumps({"stage": "train_flip_consistency_v6", "mode": mode, "epoch": epoch,
                    "group": group_index, "groups": len(loader), "microbatch": microbatches,
                    "seen": seen, "train_loss": loss_sum/seen, "train_ce": ce_sum/seen,
                    "train_consistency": consistency_sum/seen, "consistency_lambda": config["consistency_lambda"],
                    "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": skips,
                    "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"],
                    "elapsed_seconds": time.monotonic()-started}), flush=True)
                last_progress = time.monotonic()
        if seen != len(records["train"]) or epoch_steps <= 0:
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
        entry = {"epoch": epoch, "mode": mode, "train_loss": loss_sum/seen,
            "train_ce": ce_sum/seen, "train_consistency": consistency_sum/seen,
            "consistency_lambda": config["consistency_lambda"],
            "train_accuracy": correct/seen, "train_accuracy_forwarded_images": correct/seen,
            "accuracy_definition": config["train_accuracy_definition"], "seen": seen,
            "forwarded_seen": seen, "zero_weight_groups": 0, "forwarded_microbatches": microbatches,
            "successful_optimizer_steps": epoch_steps, "amp_skipped_steps": skips,
            "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
            "augmentation_epoch": data_epoch, "amp_scale": scaler.get_scale(),
            "elapsed_seconds": time.monotonic()-started,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "raw_original_metrics": raw_candidate["single_original_metrics"], "raw_tta_metrics": raw_candidate["single_tta_metrics"],
            "ema_original_metrics": ema_candidate["single_original_metrics"], "ema_tta_metrics": ema_candidate["single_tta_metrics"],
            "raw_validation_losses": raw_candidate["validation_losses"], "ema_validation_losses": ema_candidate["validation_losses"]}
        history.append(entry)
        commit_epoch(root, output, model, optimizer, scaler, ema, config, epoch, counters, history, candidates)
        print(json.dumps({"stage": "flip_consistency_epoch_committed", **entry}), flush=True)
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
