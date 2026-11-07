"""Official HSE initialization on the newly grouped expanded train/validation split.

The protocol is fixed: twelve uniform standard-CE epochs, final EMA only.
Epoch zero is an initialization diagnostic, never a checkpoint candidate.
Old models and their validation/fusion predictions are not loaded or compared.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import hse_model as model_api
import train_hse_sam_v7 as math_v7
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "hse_expanded_validation_v10"
CLASSES = list(h.CLASSES)
PLAN_NAME = "logs/hse_expanded_validation_v10_plan.json"
OUTPUT_NAME = "experiments/hse_expanded_validation_v10"
SETTINGS = {"epochs": 12, "seed": 42, "augmentation_epoch_offset": 100,
    "batch_size": 16, "effective_batch_size": 32, "accumulation": 2,
    "workers": 0, "cpu_threads": 4, "image_size": 224, "validation_batch_size": 16,
    "backbone_lr": 1e-5, "head_lr": 1e-4, "weight_decay": .01,
    "label_smoothing": 0., "gradient_clip": 1., "warmup_successful_steps": 100,
    "min_lr_ratio": .1, "ema_decay": .995, "amp_initial_scale": 1024.,
    "amp_growth_interval": 2000, "batchnorm_running_stats": "frozen_eval",
    "batchnorm_affine": "trainable", "sample_weights": "uniform1",
    "mixup": False, "cutmix": False, "rho": 0., "head_warmup_epochs": 0,
    "initialization": "official AffectNet7 only; mapped six-class head",
    "checkpoint_selection": "fixed epoch12 EMA; epoch0 diagnostic only"}
BINDINGS = ("classes", "source_pins", "plan_sha256", "parent_checkpoint_sha256",
    "model_source_receipt", "model_source_receipt_sha256", "original_pool_indices",
    "excluded_indices", "train_indices", "val_indices", "split_summary_sha256",
    "split_manifest_sha256")


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def _location(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def _indices(values):
    return (isinstance(values, list) and len(values) == len(set(values))
            and all(type(value) is int and value >= 0 for value in values))


def read_records(root, plan, verify_images=True):
    """Read only original train/val members; internal-test rows are filtered first."""
    root = Path(root).resolve()
    manifest = _location(root, plan["split_manifest_path"])
    h.check_hash(manifest, plan["split_manifest_sha256"])
    original = {}
    with (root/"features/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row.get("split") not in ("train", "val"):
                continue
            if row.get("label") not in CLASSES or not os.path.isabs(row.get("path", "")):
                raise ValueError("V10 original public train/val record is invalid")
            original[index] = {"manifest_index": index, "path": row["path"], "label": row["label"],
                               "target": CLASSES.index(row["label"])}
    if list(original) != plan["original_pool_indices"]:
        raise ValueError("V10 original pooled train/val member list differs")
    rows, group_sets, hashes = {"train": [], "val": []}, {"train": set(), "val": set()}, {}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"manifest_index", "path", "label", "target", "split", "group", "image_sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("V10 split manifest lacks its fixed member/group/image receipt fields")
        seen = set()
        for record in reader:
            split = record["split"]
            if split not in rows:
                raise ValueError("V10 split manifest must contain only effective train/val rows")
            index = int(record["manifest_index"])
            row = {"manifest_index": index, "path": record["path"], "label": record["label"], "target": int(record["target"])}
            if index in seen or index not in original or row != original[index] or index in plan["excluded_indices"]:
                raise ValueError("V10 split contains duplicate/excluded/modified original records")
            seen.add(index)
            digest = record["image_sha256"]
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("V10 split requires lowercase SHA256 image identities")
            if not record["group"]:
                raise ValueError("V10 split lacks a near-duplicate group identity")
            group_sets[split].add(record["group"])
            rows[split].append(row)
            hashes[index] = digest
    if group_sets["train"] & group_sets["val"]:
        raise ValueError("V10 near-duplicate groups cross the new train/val boundary")
    for split in rows:
        if ([row["manifest_index"] for row in rows[split]] != plan[split+"_indices"]
            or h.signature(rows[split]) != plan[split+"_rows_sha256"]
            or {row["target"] for row in rows[split]} != set(range(6))):
            raise ValueError("V10 split members/order/labels differ from the locked plan")
        digest = hashlib.sha256()
        started = time.monotonic()
        for number, row in enumerate(rows[split], 1):
            expected = hashes[row["manifest_index"]]
            if verify_images:
                h.check_hash(row["path"], expected)
            digest.update(expected.encode("ascii"))
            if verify_images and number % 4000 == 0:
                emit("verify_expanded_validation_v10_images", split=split, images=number,
                     total=len(rows[split]), elapsed_seconds=time.monotonic()-started)
        if digest.hexdigest() != plan[split+"_image_content_sha256"]:
            raise ValueError("V10 ordered image-content receipt differs")
    return rows


def checked_plan(root):
    """Return plan, exact plan-file SHA and bound rows without rehashing image bytes."""
    root = Path(root).resolve()
    plan = c.read_json(root/PLAN_NAME)
    if plan.get("format") != FORMAT+"_locked" or plan.get("classes") != CLASSES:
        raise ValueError("V10 plan format/classes differ")
    if any(plan.get("settings", {}).get(key) != value for key, value in SETTINGS.items()):
        raise ValueError("V10 fixed twelve-epoch final-EMA protocol differs")
    required = {Path(__file__).resolve(), Path(model_api.__file__).resolve(),
                Path(math_v7.__file__).resolve(), Path(h.__file__).resolve(), Path(c.__file__).resolve()}
    pins = plan.get("source_pins")
    if not isinstance(pins, dict) or not required.issubset({_location(root, value) for value in pins}):
        raise ValueError("V10 source pins omit a required model/training implementation")
    c.verify_pins(root, pins)
    for kind in ("manifest", "summary"):
        path = _location(root, plan["split_"+kind+"_path"])
        if path != (root/"data/expanded_validation_v10"/(kind+".csv" if kind == "manifest" else "summary.json")).resolve():
            raise ValueError("V10 split artifacts must stay in their independent data directory")
        h.check_hash(path, plan["split_"+kind+"_sha256"])
    split_summary = c.read_json(_location(root, plan["split_summary_path"]))
    if split_summary.get("state") != "complete" or split_summary.get("classes") != CLASSES:
        raise ValueError("V10 expanded-split receipt is incomplete or has another class order")
    if split_summary.get("internal_test_read", False) is not False or split_summary.get("external_answers_used", False) is not False:
        raise ValueError("V10 split receipt must exclude internal test and external answers")
    for key in ("original_pool_indices", "excluded_indices", "train_indices", "val_indices"):
        if not _indices(plan.get(key)):
            raise ValueError("V10 requires unique integer pooled/excluded/train/val members")
    pool, excluded, train, val = (plan[key] for key in ("original_pool_indices", "excluded_indices", "train_indices", "val_indices"))
    if (len(pool) != 23904 or excluded != [7727, 10448] or not set(excluded).issubset(pool)
        or set(train) & set(val) or set(train) | set(val) != set(pool)-set(excluded)
        or not .19 <= len(val)/(len(train)+len(val)) <= .21):
        raise ValueError("V10 must repartition the public pooled23904 minus the two fixed invalid images")
    if train != sorted(train) or val != sorted(val):
        raise ValueError("V10 members must retain original manifest order within each split")
    receipt = model_api.source_receipt(root)
    if (plan.get("model_source_receipt") != receipt
        or plan.get("model_source_receipt_sha256") != h.signature(receipt)):
        raise ValueError("V10 official HSE source receipt differs")
    parent = _location(root, plan["parent_checkpoint"])
    if parent != (root/"models/hsemotion/enet_b2_7.pt").resolve() or plan["parent_checkpoint_sha256"] != model_api.OFFICIAL_SHA:
        raise ValueError("V10 must initialize from the official AffectNet7 HSE weight, never an old trained model")
    h.check_hash(parent, plan["parent_checkpoint_sha256"])
    rows = read_records(root, plan, verify_images=False)
    return plan, h.sha256(root/PLAN_NAME), rows


def config_for(root, plan):
    config = {"format": FORMAT+"_config", "plan_sha256": h.sha256(Path(root)/PLAN_NAME),
        **{key: plan[key] for key in ("classes", "settings", "source_pins", "input_receipt",
            "split_summary_path", "split_summary_sha256", "split_manifest_path", "split_manifest_sha256",
            "original_pool_indices", "excluded_indices", "train_indices", "val_indices",
            "train_rows_sha256", "val_rows_sha256", "train_image_content_sha256", "val_image_content_sha256",
            "parent_checkpoint", "parent_checkpoint_sha256", "model_source_receipt", "model_source_receipt_sha256")},
        "model_provenance": plan["model_source_receipt"], "effective_train_rows": len(plan["train_indices"]),
        "evaluation_rows": len(plan["val_indices"]), "train_sample_weights": "uniform1",
        "validation_weighted": False, "checkpoint_selection": SETTINGS["checkpoint_selection"],
        "per_epoch_evaluation": False, "epoch0_purpose": "diagnostic_only",
        "old_model_validation_or_fusion": False, "internal_test_read": False, "external_answers_used": False,
        "preprocessing": "EXIF transpose, RGB, PIL bilinear224, ImageNet mean/std",
        "augmentation": "flip.5 affine6deg translate.03 scale.95..1.05 brightness/contrast.1",
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    return config


def configure_trainability(model, training=False):
    model.requires_grad_(True)
    model.train(bool(training))
    math_v7.freeze_batchnorm(model)
    head = model.get_classifier()
    if not isinstance(head, nn.Linear) or head.out_features != 6 or any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("V10 requires the full six-class HSE model, with every affine parameter trainable")
    return model


def frozen_reference(model):
    parameters = set(dict(model.named_parameters()))
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if key not in parameters}


def frozen_digest(reference):
    digest = hashlib.sha256()
    for key in sorted(reference):
        value = reference[key].detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model, reference):
    current = frozen_reference(model)
    if set(current) != set(reference) or any(not torch.equal(current[key], value) for key, value in reference.items()):
        raise ValueError("V10 official batchnorm/other frozen buffers changed")


def check_state(state, reference):
    """Prevent load_state_dict from silently accepting a cast tensor dtype."""
    if not isinstance(state, dict) or set(state) != set(reference):
        raise ValueError("V10 checkpoint must contain the complete native HSE state layout")
    for key, expected in reference.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
            or value.dtype != expected.dtype or not torch.isfinite(value).all().item()):
            raise ValueError("V10 native state shape/dtype/finite check failed: "+key)


def make_optimizer(model, settings):
    return math_v7.make_optimizer(model, settings)


def set_learning_rate(optimizer, successful_steps, total_steps, settings):
    return math_v7.set_learning_rate(optimizer, successful_steps, total_steps, settings)


def make_loader(rows, training, epoch, settings, device="cpu"):
    data_epoch = settings["augmentation_epoch_offset"]+epoch-1 if training else 0
    generator = torch.Generator().manual_seed(settings["seed"]+1009*data_epoch+(0 if training else 1000003))
    return DataLoader(h.ExpressionImages(rows, training),
        batch_size=settings["effective_batch_size"] if training else settings["validation_batch_size"],
        shuffle=training, num_workers=0, pin_memory=torch.device(device).type == "cuda",
        generator=generator, drop_last=False)


def metrics(probabilities, rows):
    targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    result = c.metrics(targets, probabilities)
    return {key: result[key] for key in ("accuracy", "macro_f1", "log_loss", "correct_count", "samples")}


@torch.inference_mode()
def validate_views(model, rows, device="cpu", batch_size=16):
    settings = {**SETTINGS, "validation_batch_size": batch_size}
    model.eval()
    original, flipped = [], []
    started = stamp = time.monotonic()
    for number, (images, _) in enumerate(make_loader(rows, False, 0, settings, device), 1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                            enabled=torch.device(device).type == "cuda"):
            first, second = model(images), model(images.flip(-1))
        original.append(first.float().softmax(1).cpu().numpy())
        flipped.append(second.float().softmax(1).cpu().numpy())
        if time.monotonic()-stamp >= 45:
            emit("validate_expanded_validation_v10", batch=number, samples=min(number*batch_size, len(rows)),
                 total=len(rows), elapsed_seconds=time.monotonic()-started)
            stamp = time.monotonic()
    original, flipped = np.concatenate(original), np.concatenate(flipped)
    for probability in (original, flipped):
        if probability.dtype != np.float32:
            raise ValueError("V10 probabilities must be persisted as float32")
        c.check_probability(probability, len(rows))
    return original, flipped


def model_payload(model, config, epoch, weight_source):
    return {"format": FORMAT, "config": config, **{key: config[key] for key in BINDINGS},
        "parent_sha": config["parent_checkpoint_sha256"], "model_provenance": config["model_source_receipt"],
        "epoch": epoch, "weight_source": weight_source, "model_state": h.cpu_tree(model.state_dict())}


def verify_payload(root, payload, config=None):
    if payload.get("format") != FORMAT or payload.get("classes") != CLASSES:
        raise ValueError("V10 requires its own complete HSE checkpoint format")
    plan, _, _ = checked_plan(root)
    expected = config_for(root, plan)
    if payload.get("config") != expected or (config is not None and expected != config):
        raise ValueError("V10 checkpoint configuration differs from the immutable split/training plan")
    for key in BINDINGS:
        if payload.get(key) != expected[key]:
            raise ValueError("V10 checkpoint source/member binding differs: "+key)
    if payload.get("parent_sha") != expected["parent_checkpoint_sha256"] or payload.get("model_provenance") != expected["model_source_receipt"]:
        raise ValueError("V10 official parent/model provenance differs")
    if type(payload.get("epoch")) is not int or not 0 <= payload["epoch"] <= SETTINGS["epochs"]:
        raise ValueError("V10 checkpoint epoch is invalid")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("V10 requires a finite tensor-only full model state")
    return expected


def load_checkpoint(root, path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True) if isinstance(path, (str, os.PathLike)) else path
    verify_payload(root, payload)
    official = model_api.load_initial_model(root, "cpu")
    check_state(payload["model_state"], official.state_dict())
    model = model_api.load_trained_model(root, payload, device)
    assert_frozen(model, frozen_reference(official))
    del official
    return model.eval(), payload


def _atomic_copy(source, destination):
    temporary = Path(str(destination)+".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, reference):
    relative = f"epochs/epoch_{epoch:02d}.pt"
    payload = {**model_payload(model, config, epoch, "raw"),
        "optimizer": h.cpu_tree(optimizer.state_dict()), "scaler": scaler.state_dict(),
        "ema_state": ema.state_dict(), "rng": h.rng_state(), "counters": dict(counters),
        "successful_steps": counters["successful_steps"], "cumulative_skips": counters["amp_skipped_steps"],
        "zero_weight_groups": 0, "history": list(history), "frozen_state_sha256": frozen_digest(reference)}
    h.save_torch(output/relative, payload)
    _atomic_copy(output/relative, output/"last.pt")
    h.write_json(output/"last.json", {"epoch": epoch, "checkpoint": relative,
        "sha256": h.sha256(output/relative), "plan_sha256": config["plan_sha256"]})
    h.write_json(output/"history.json", history)


def verify_history(history, counters, config, epoch):
    if len(history) != epoch or [row["epoch"] for row in history] != list(range(1, epoch+1)):
        raise ValueError("V10 history must contain exactly the committed complete epochs")
    count = len(config["train_indices"])
    groups = math.ceil(count/SETTINGS["effective_batch_size"])
    cumulative = skips = 0
    for row in history:
        if (row["seen"] != count or row["successful_optimizer_steps"] <= 0
            or row["successful_optimizer_steps"]+row["amp_skipped_steps"] != groups
            or row.get("zero_weight_groups") != 0
            or any(key in row for key in ("val_accuracy", "validation", "val_metrics", "raw_original_metrics", "ema_tta_metrics"))):
            raise ValueError("V10 epoch missed training members/groups or contains checkpoint-selection validation")
        cumulative += row["successful_optimizer_steps"]
        skips += row["amp_skipped_steps"]
        if row["cumulative_optimizer_steps"] != cumulative or row["ema_updates"] != cumulative:
            raise ValueError("V10 cumulative optimizer/EMA counts differ")
    if counters != {"successful_steps": cumulative, "amp_skipped_steps": skips, "zero_weight_groups": 0}:
        raise ValueError("V10 history and checkpoint counters differ")


def resume_epoch(root, output, config, model, optimizer, scaler, ema, reference):
    pointer = c.read_json(output/"last.json")
    epoch = pointer.get("epoch")
    if (type(epoch) is not int or not 0 <= epoch <= SETTINGS["epochs"]
        or pointer.get("checkpoint") != f"epochs/epoch_{epoch:02d}.pt" or pointer.get("plan_sha256") != config["plan_sha256"]):
        raise ValueError("V10 last pointer is not a complete epoch bound to this plan")
    path = output/pointer["checkpoint"]
    h.check_hash(path, pointer["sha256"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    verify_payload(root, saved, config)
    if saved.get("epoch") != epoch or saved.get("weight_source") != "raw" or saved.get("frozen_state_sha256") != frozen_digest(reference):
        raise ValueError("V10 resume epoch/source/frozen reference differs")
    counters = saved["counters"]
    verify_history(saved["history"], counters, config, epoch)
    if saved.get("successful_steps") != counters["successful_steps"] or saved.get("cumulative_skips") != counters["amp_skipped_steps"]:
        raise ValueError("V10 resume successful-step aliases differ")
    check_state(saved["model_state"], model.state_dict())
    check_state(saved["ema_state"]["shadow"], model.state_dict())
    if epoch == 0:
        initial = model.state_dict()
        if (counters != {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
            or any(not torch.equal(value.cpu(), initial[key].detach().cpu()) for key, value in saved["model_state"].items())
            or any(not torch.equal(value, saved["model_state"][key]) for key, value in saved["ema_state"]["shadow"].items())):
            raise ValueError("V10 epoch0 raw/EMA must exactly equal official initialization with zero updates")
    model.load_state_dict(saved["model_state"], strict=True)
    assert_frozen(model, reference)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    ema.load_state_dict(saved["ema_state"])
    if ema.updates != counters["successful_steps"]:
        raise ValueError("V10 resumed EMA updates differ from successful optimizer steps")
    with ema.apply(model):
        assert_frozen(model, reference)
    h.restore_rng(saved["rng"])
    if not (output/"last.pt").exists() or h.sha256(output/"last.pt") != pointer["sha256"]:
        _atomic_copy(path, output/"last.pt")
    h.write_json(output/"history.json", saved["history"])
    emit("resume_expanded_validation_v10", completed_epoch=epoch, successful_optimizer_steps=counters["successful_steps"])
    return epoch, counters, saved["history"]


def _result_bindings(config):
    return {key: config[key] for key in BINDINGS}


def _verify_result(root, output, config, rows, name, epoch, source, files):
    result = c.read_json(output/name)
    if (result.get("state") != "complete" or result.get("epoch") != epoch or result.get("weight_source") != source
        or any(result.get(key) != config[key] for key in BINDINGS)):
        raise ValueError("V10 diagnostic/final result source/member binding differs")
    for key, relative in files.items():
        if result.get(key) != relative:
            raise ValueError("V10 diagnostic/final artifact filename differs")
        h.check_hash(output/relative, result["artifact_sha256"][key])
    payload = torch.load(output/files["model"], map_location="cpu", weights_only=True)
    verify_payload(root, payload, config)
    if payload.get("epoch") != epoch or payload.get("weight_source") != source:
        raise ValueError("V10 diagnostic/final model epoch differs")
    original = np.load(output/files["original"], allow_pickle=False)
    flip = np.load(output/files["flip"], allow_pickle=False)
    for probability in (original, flip):
        if probability.dtype != np.float32:
            raise ValueError("V10 validation probabilities require float32 persistence")
        c.check_probability(probability, len(rows))
    calculated = {"single_original_metrics": metrics(original, rows), "single_tta_metrics": metrics((original+flip)*.5, rows)}
    for key, value in calculated.items():
        if result.get(key) != value or payload.get(key) != value:
            raise ValueError("V10 probabilities do not reproduce their unweighted scalar metrics")
    if epoch != 0:
        for alias, key in (("final_original_metrics", "single_original_metrics"), ("final_tta_metrics", "single_tta_metrics")):
            if result.get(alias) != calculated[key] or payload.get(alias) != calculated[key]:
                raise ValueError("V10 final metric aliases differ from the persisted probability metrics")
    if epoch == 0:
        if result.get("purpose") != "diagnostic_only" or payload.get("purpose") != "diagnostic_only":
            raise ValueError("V10 epoch0 may only be an official-initialization diagnostic")
        reference = torch.load(output/"epochs/epoch_00.pt", map_location="cpu", weights_only=True)["model_state"]
    else:
        last = torch.load(output/"epochs/epoch_12.pt", map_location="cpu", weights_only=True)
        reference = last["ema_state"]["shadow"]
    if set(payload["model_state"]) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in payload["model_state"].items()):
        raise ValueError("V10 model differs from official epoch0 or committed final EMA shadow")
    return result


def write_diagnostic(root, output, config, model, rows, reference):
    files = {"model": "diagnostic_epoch0.pt", "original": "diagnostic_epoch0_original.npy", "flip": "diagnostic_epoch0_flip.npy"}
    if (output/"epoch0_diagnostic.json").exists():
        return _verify_result(root, output, config, rows, "epoch0_diagnostic.json", 0, "official", files)
    state = h.rng_state()
    try:
        original, flip = validate_views(model, rows, next(model.parameters()).device, SETTINGS["validation_batch_size"])
        assert_frozen(model, reference)
        original_metrics, tta_metrics = metrics(original, rows), metrics((original+flip)*.5, rows)
        payload = {**model_payload(model, config, 0, "official"), "purpose": "diagnostic_only",
            "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
            "successful_steps": 0, "ema_updates": 0}
        h.save_torch(output/files["model"], payload)
        h.save_array(output/files["original"], original)
        h.save_array(output/files["flip"], flip)
        result = {"format": FORMAT+"_diagnostic", "state": "complete", "epoch": 0,
            "weight_source": "official", "purpose": "diagnostic_only", **_result_bindings(config), **files,
            "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
            "validation_weighted": False, "artifact_sha256": {key: h.sha256(output/value) for key, value in files.items()}}
        h.write_json(output/"epoch0_diagnostic.json", result)
        return _verify_result(root, output, config, rows, "epoch0_diagnostic.json", 0, "official", files)
    finally:
        h.restore_rng(state)


def finalize_run(root, output, config, model, ema, reference, rows, counters, history):
    verify_history(history, counters, config, SETTINGS["epochs"])
    if ema.updates != counters["successful_steps"]:
        raise ValueError("V10 finalization requires twelve complete epochs and aligned EMA updates")
    files = {"model": "final_ema.pt", "original": "final_original.npy", "flip": "final_flip.npy"}
    if (output/"final_result.json").exists():
        result = _verify_result(root, output, config, rows, "final_result.json", 12, "ema", files)
    else:
        with ema.apply(model):
            assert_frozen(model, reference)
            original, flip = validate_views(model, rows, next(model.parameters()).device, SETTINGS["validation_batch_size"])
            original_metrics, tta_metrics = metrics(original, rows), metrics((original+flip)*.5, rows)
            payload = {**model_payload(model, config, 12, "ema"), "purpose": "fixed_final_epoch",
                "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
                "final_original_metrics": original_metrics, "final_tta_metrics": tta_metrics,
                "successful_steps": counters["successful_steps"], "ema_updates": ema.updates,
                "frozen_state_sha256": frozen_digest(reference)}
            h.save_torch(output/files["model"], payload)
        h.save_array(output/files["original"], original)
        h.save_array(output/files["flip"], flip)
        result = {"format": FORMAT+"_final_result", "state": "complete", "epoch": 12, "weight_source": "ema",
            "purpose": "fixed_final_epoch", **_result_bindings(config), **files,
            "single_original_metrics": original_metrics, "single_tta_metrics": tta_metrics,
            "final_original_metrics": original_metrics, "final_tta_metrics": tta_metrics,
            "validation_weighted": False, "checkpoint_selection": False,
            "old_model_validation_or_fusion": False,
            "artifact_sha256": {key: h.sha256(output/value) for key, value in files.items()}}
        h.write_json(output/"final_result.json", result)
        _verify_result(root, output, config, rows, "final_result.json", 12, "ema", files)
    assert_frozen(model, reference)
    pointer = c.read_json(output/"last.json")
    if pointer.get("epoch") != 12:
        raise ValueError("V10 final result lacks twelve committed epochs")
    h.check_hash(output/"last.pt", pointer["sha256"])
    paths = ["config.json", "history.json", "last.json", "last.pt", "epoch0_diagnostic.json",
             "diagnostic_epoch0.pt", "diagnostic_epoch0_original.npy", "diagnostic_epoch0_flip.npy",
             "final_result.json", "final_ema.pt", "final_original.npy", "final_flip.npy"]
    paths.extend(f"epochs/epoch_{epoch:02d}.pt" for epoch in range(13))
    summary = {"format": FORMAT+"_summary", "state": "complete", "epochs_completed": 12,
        "weight_source": "ema", **_result_bindings(config), "effective_train_rows": len(config["train_indices"]),
        "evaluation_rows": len(rows), "single_original_metrics": result["single_original_metrics"],
        "single_tta_metrics": result["single_tta_metrics"], "final_original_metrics": result["single_original_metrics"],
        "final_tta_metrics": result["single_tta_metrics"], "counters": dict(counters), "history": history,
        "successful_optimizer_steps": counters["successful_steps"], "amp_skipped_steps": counters["amp_skipped_steps"],
        "ema_updates": ema.updates, "zero_weight_groups": 0, "frozen_state_sha256": frozen_digest(reference),
        "batchnorm_buffers_frozen": True, "frozen_state_unchanged": True, "validation_weighted": False,
        "per_epoch_evaluation": False, "epoch0_purpose": "diagnostic_only", "checkpoint_selection": False,
        "old_model_validation_or_fusion": False, "internal_test_read": False, "external_answers_used": False,
        "artifact_sha256": {relative: h.sha256(output/relative) for relative in paths}}
    if (output/"summary.json").exists():
        if c.read_json(output/"summary.json") != summary:
            raise ValueError("V10 completed summary does not reproduce its full committed run/artifact receipt")
    else:
        h.write_json(output/"summary.json", summary)
    emit("expanded_validation_v10_complete", correct_count=summary["single_tta_metrics"]["correct_count"],
         samples=len(rows), successful_optimizer_steps=counters["successful_steps"], amp_skipped_steps=counters["amp_skipped_steps"])
    return summary


def run_training(root, device="cuda"):
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V10 CUDA unavailable; never silently fall back to CPU training")
    torch.set_num_threads(4)
    plan, _, _ = checked_plan(root)
    rows = read_records(root, plan, verify_images=True)
    config = config_for(root, plan)
    output = root/OUTPUT_NAME
    output.mkdir(parents=True, exist_ok=True)
    (output/"epochs").mkdir(exist_ok=True)
    if (output/"config.json").exists() and c.read_json(output/"config.json") != config:
        raise ValueError("Existing V10 config differs; preserve the independent experiment")
    if not (output/"config.json").exists():
        h.write_json(output/"config.json", config)
    if (output/"summary.json").exists():
        existing = c.read_json(output/"summary.json")
        if existing.get("state") != "complete" or existing.get("plan_sha256") != config["plan_sha256"]:
            raise ValueError("Existing V10 summary is incomplete or belongs to another plan")
        for relative, digest in existing["artifact_sha256"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("V10 completed artifact escapes its independent output directory")
            h.check_hash(output/path, digest)
    h.seed_everything(SETTINGS["seed"])
    model = model_api.load_initial_model(root, device)
    configure_trainability(model)
    reference = frozen_reference(model)
    optimizer = make_optimizer(model, SETTINGS)
    scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", SETTINGS["amp_initial_scale"], SETTINGS["amp_growth_interval"])
    ema = math_v7.FullModelEMA(model, SETTINGS["ema_decay"])
    counters = {"successful_steps": 0, "amp_skipped_steps": 0, "zero_weight_groups": 0}
    history, done = [], 0
    if (output/"last.json").exists():
        done, counters, history = resume_epoch(root, output, config, model, optimizer, scaler, ema, reference)
    else:
        commit_epoch(output, model, optimizer, scaler, ema, config, 0, counters, history, reference)
    if done == 0:
        write_diagnostic(root, output, config, model, rows["val"], reference)
    elif not (output/"epoch0_diagnostic.json").is_file():
        raise ValueError("V10 resumed training lacks the committed official initialization diagnostic")
    else:
        _verify_result(root, output, config, rows["val"], "epoch0_diagnostic.json", 0, "official",
            {"model": "diagnostic_epoch0.pt", "original": "diagnostic_epoch0_original.npy", "flip": "diagnostic_epoch0_flip.npy"})
    groups = math.ceil(len(rows["train"])/SETTINGS["effective_batch_size"])
    total_steps = groups*SETTINGS["epochs"]
    for epoch in range(done+1, SETTINGS["epochs"]+1):
        loader = make_loader(rows["train"], True, epoch, SETTINGS, device)
        configure_trainability(model, True)
        seen = success = skips = diagnostic_seen = correct = 0
        loss_sum = 0.
        started = stamp = time.monotonic()
        for number, (images, targets) in enumerate(loader, 1):
            set_learning_rate(optimizer, counters["successful_steps"], total_steps, SETTINGS)
            result = math_v7.group_step(model, optimizer, scaler, ema, images, targets,
                                       device=device, batch_size=SETTINGS["batch_size"], rho=0.)
            seen += len(targets)
            okay = bool(result["success"])
            success += int(okay); skips += int(not okay)
            counters["successful_steps"] += int(okay); counters["amp_skipped_steps"] += int(not okay)
            if result["loss_sum"] is not None:
                diagnostic_seen += len(targets)
                loss_sum += float(result["loss_sum"])
                correct += int(result["correct"])
            if skips >= 25 and success == 0:
                raise RuntimeError("V10 hit25 AMP skips without a successful update")
            if time.monotonic()-stamp >= 45 or number == len(loader):
                emit("train_expanded_validation_v10", epoch=epoch, epochs=12, batch=number, batches=len(loader),
                    seen=seen, train_loss=loss_sum/max(1, diagnostic_seen), successful_optimizer_steps=success,
                    amp_skipped_steps=skips, cumulative_optimizer_steps=counters["successful_steps"],
                    elapsed_seconds=time.monotonic()-started)
                stamp = time.monotonic()
        if seen != len(rows["train"]) or success <= 0 or success+skips != groups:
            raise ValueError("V10 epoch missed training members/groups or never updated")
        assert_frozen(model, reference)
        history.append({"epoch": epoch, "seen": seen, "augmentation_epoch": SETTINGS["augmentation_epoch_offset"]+epoch-1,
            "train_loss": loss_sum/max(1, diagnostic_seen), "train_accuracy": correct/max(1, diagnostic_seen),
            "diagnostic_samples": diagnostic_seen, "successful_optimizer_steps": success, "amp_skipped_steps": skips,
            "zero_weight_groups": 0, "cumulative_optimizer_steps": counters["successful_steps"], "ema_updates": ema.updates,
            "amp_scale": scaler.get_scale(), "elapsed_seconds": time.monotonic()-started,
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups}})
        c.verify_pins(root, config["source_pins"])
        commit_epoch(output, model, optimizer, scaler, ema, config, epoch, counters, history, reference)
        emit("expanded_validation_v10_epoch_committed", **history[-1])
    return finalize_run(root, output, config, model, ema, reference, rows["val"], counters, history)


def smoke(root, device="cuda"):
    """Two real effective32 optimizer groups from official initialization; no weights saved."""
    root = Path(root).resolve()
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V10 CUDA unavailable for the explicitly requested smoke")
    torch.set_num_threads(4)
    plan, plan_sha, rows = checked_plan(root)
    selected = rows["train"][:64]
    for row in selected:
        if not Path(row["path"]).is_file():
            raise ValueError("V10 smoke training image missing")
    h.seed_everything(SETTINGS["seed"])
    model = configure_trainability(model_api.load_initial_model(root, device), True)
    before = h.cpu_tree(model.state_dict())
    reference = frozen_reference(model)
    optimizer = make_optimizer(model, SETTINGS)
    scaler = math_v7.ManualGradScaler(torch.device(device).type == "cuda", 1024., 2000)
    ema = math_v7.FullModelEMA(model, .995)
    if any(not torch.equal(before[key], ema.shadow[key].cpu()) for key in before):
        raise ValueError("V10 initial EMA differs from official initialization")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    success = skips = 0
    total_steps = math.ceil(len(rows["train"])/32)*12
    for images, targets in make_loader(selected, True, 1, SETTINGS, device):
        set_learning_rate(optimizer, success, total_steps, SETTINGS)
        result = math_v7.group_step(model, optimizer, scaler, ema, images, targets, device=device, batch_size=16, rho=0.)
        success += int(result["success"]); skips += int(not result["success"])
        assert_frozen(model, reference)
    if success != 2 or skips or ema.updates != 2:
        raise RuntimeError("V10 smoke requires exactly two successful non-skipped optimizer/EMA updates")
    after = h.cpu_tree(model.state_dict())
    changed = {key for key in before if not torch.equal(before[key], after[key])}
    parameters = dict(model.named_parameters())
    head_names = {name for name, value in parameters.items() if id(value) in {id(p) for p in h.classifier_parameters(model)}}
    if not head_names.issubset(changed) or not any(key not in head_names for key in changed):
        raise ValueError("V10 smoke must actually update both classifier and backbone")
    parameter_steps = {name: int(optimizer.state[value]["step"].item()) if "step" in optimizer.state[value] else 0 for name, value in parameters.items()}
    if any(step != 2 for step in parameter_steps.values()):
        raise ValueError("V10 smoke failed to update every trainable affine parameter twice")
    with ema.apply(model):
        assert_frozen(model, reference)
    return {"state": "passed", "classes": CLASSES, "plan_sha256": plan_sha, "source_pins": plan["source_pins"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"], "model_source_receipt_sha256": plan["model_source_receipt_sha256"],
        "split_summary_sha256": plan["split_summary_sha256"], "split_manifest_sha256": plan["split_manifest_sha256"],
        "successful_optimizer_steps": success, "amp_skipped_steps": skips, "ema_updates": ema.updates,
        "batch_size": 16, "effective_batch_size": 32, "uniform_CE": True, "weights_saved": False,
        "training_manifest_indices": [row["manifest_index"] for row in selected], "changed_state_keys": sorted(changed),
        "all_parameter_steps": parameter_steps, "batchnorm_buffers_frozen": True, "full_backbone_updated": True,
        "elapsed_seconds": time.monotonic()-started, "GPU_used": torch.device(device).type == "cuda",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None,
        "validation_evaluated": False, "internal_test_read": False, "external_answers_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("smoke", "train"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("V10 only supports the fixed project root")
    if args.stage == "smoke":
        report = smoke(args.project_root, args.device)
        kind = "gpu" if torch.device(args.device).type == "cuda" else "cpu"
        h.write_json(args.project_root/f"logs/hse_expanded_validation_v10_{kind}_smoke.json", report)
        emit("expanded_validation_v10_smoke_passed", plan_sha256=report["plan_sha256"], source_pins=report["source_pins"])
    else:
        run_training(args.project_root, args.device)


if __name__ == "__main__":
    main()
