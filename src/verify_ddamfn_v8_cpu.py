"""CPU contract smoke using an in-memory schema fixture, never a formal plan.

This test neither reads dataset labels nor writes checkpoints/plans/results. Its
synthetic membership and score are solely schema fixtures, not experiment data.
"""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ddamfn_v8_model as model_api
import train_ddamfn_v8 as trainer
import prepare_ddamfn_v8 as preparation
import finalize_ddamfn_v8 as finalizer
import train_hse_specialist as h
import train_hse_sam_v7 as math_v7

ROOT = Path(__file__).resolve().parents[1]


def roundtrip_schema():
    plan_path = ROOT / "logs/ddamfn_v8_plan.json"
    if plan_path.exists():
        raise RuntimeError("Run this pre-lock schema fixture before a formal V8 plan exists")
    model, receipt = model_api.load_initial_model(ROOT, "cpu")
    members = list(range(21248))
    audit_path = preparation.AUDIT
    # No fixture is ever saved to disk, and no actual validation score is claimed.
    fixture = {"format": "ddamfn_v8_locked", "classes": trainer.CLASSES,
        "settings": {**trainer.SETTINGS, "batch_size": 32, "accumulation": 1},
        "source_pins": {str(ROOT / "src" / name): h.sha256(ROOT / "src" / name)
                        for name in ("ddamfn_v8_model.py", "train_ddamfn_v8.py", "prepare_ddamfn_v8.py", "finalize_ddamfn_v8.py")},
        "input_receipt": {"internal_test_images_read": 0, "internal_test_labels_parsed": False},
        "original_train_indices": members, "excluded_train_indices": [7727, 10448],
        "train_indices": [index for index in members if index not in {7727, 10448}],
        "val_indices": list(range(21248, 23904)), "filtered_membership_sha256": "a"*64,
        "audit_summary_path": str(audit_path), "audit_summary_sha256": h.sha256(audit_path),
        "parent_checkpoint": str(preparation.WEIGHT), "parent_checkpoint_sha256": preparation.WEIGHT_SHA,
        "model_source_receipt": receipt, "model_source_receipt_sha256": model_api.receipt_sha256(receipt),
        "baseline_directory": str(ROOT / "hse_sam_v7_results"), "baseline_files": {},
        "baseline_correct_count": 0, "baseline_submission_sha256": "b"*64}
    digest = "f"*64
    real_json, real_hash = trainer.c.read_json, h.sha256
    real_api_json, real_api_hash = model_api._read_json, model_api._check_hash
    same_plan = lambda value: Path(value).resolve() == plan_path.resolve()
    def schema_json(path):
        return copy.deepcopy(fixture) if same_plan(path) else real_json(path)
    def schema_hash(path):
        return digest if same_plan(path) else real_hash(path)
    def api_json(path):
        return copy.deepcopy(fixture) if same_plan(path) else real_api_json(path)
    def api_hash(path, expected):
        if same_plan(path):
            if expected != digest:
                raise ValueError("The in-memory fixture SHA binding differs")
            return None
        return real_api_hash(path, expected)
    with patch.object(trainer.c, "read_json", side_effect=schema_json), \
         patch.object(h, "sha256", side_effect=schema_hash), \
         patch.object(model_api, "_read_json", side_effect=api_json), \
         patch.object(model_api, "_check_hash", side_effect=api_hash):
        plan, config = trainer.checked_plan(ROOT)
        if any(config.get(key) != plan[key] for key in finalizer.BINDINGS):
            raise AssertionError("Trainer config does not satisfy finalizer bindings")
        if config["audit_summary_path"] != str(audit_path):
            raise AssertionError("The audit absolute path was omitted from the trainer config")
        # Use production candidate creation; capture all artifacts in memory.
        probability = np.full((2656, 6), 1/6, dtype=np.float32)
        targets = np.arange(2656) % 6
        metrics = trainer.c.metrics(targets, probability)
        artifacts = {}
        def capture(path, value):
            artifacts[str(path)] = value
        def virtual_hash(path):
            if str(path) in artifacts:
                return "c"*64
            return schema_hash(path)
        output = ROOT / "research/ddamfn_v8_candidate/IN_MEMORY_ONLY"
        with patch.object(trainer, "validate", return_value=(metrics, metrics, probability, probability)), \
             patch.object(h, "save_torch", side_effect=capture), \
             patch.object(h, "save_array", side_effect=capture), \
             patch.object(h, "write_json", side_effect=capture), \
             patch.object(h, "sha256", side_effect=virtual_hash):
            candidate = trainer.evaluate_candidate(model, None, "cpu", output, config, 0, "baseline")
        saved = artifacts[str(output / candidate["model"])]
        memory = io.BytesIO()
        torch.save(saved, memory); memory.seek(0)
        safely_restored = torch.load(memory, map_location="cpu", weights_only=True)
        restored, metadata = model_api.load_trained_model(ROOT, safely_restored, "cpu")
        if metadata["single_tta_metrics"] != metrics or metadata["config"] != config:
            raise AssertionError("Safe roundtrip lost candidate metrics/configuration")
        finalizer.bound_payload(safely_restored, config, plan, model.state_dict())
        if not all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items()):
            raise AssertionError("The native six-class state did not restore exactly")
        broken = copy.deepcopy(safely_restored)
        broken["config"].pop("audit_summary_path")
        try:
            model_api.load_trained_model(ROOT, broken, "cpu")
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("Missing required audit path was not rejected")
        if set(candidate["artifact_sha256"]) != {"model", "original", "flip"} or candidate["val_indices"] != plan["val_indices"]:
            raise AssertionError("Candidate paths/probability/member schema differs")
    if plan_path.exists() or output.exists():
        raise AssertionError("The schema smoke wrote a formal plan or virtual artifact directory")
    return {"safe_candidate_roundtrip": True, "state_tensor_count": len(model.state_dict()),
            "model_state_bitwise_restored": True, "trainer_finalizer_bindings": True,
            "audit_summary_path_bound": True, "missing_required_audit_path_rejected": True,
            "formal_plan_written": False, "checkpoint_written": False, "dataset_labels_read": False}


def accumulation_math():
    torch.manual_seed(123)
    model = nn.Linear(5, 6)
    reference = copy.deepcopy(model)
    images, targets = torch.randn(30, 5), torch.arange(30) % 6
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    expected = torch.optim.AdamW(reference.parameters(), lr=1e-4, weight_decay=.01)
    ema = math_v7.FullModelEMA(model, .995)
    result = math_v7.group_step(model, optimizer, math_v7.ManualGradScaler(False), ema,
                               images, targets, device="cpu", batch_size=8, rho=0.)
    F.cross_entropy(reference(images), targets).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    expected.step()
    error = max(float((value-reference.state_dict()[name]).abs().max()) for name, value in model.state_dict().items())
    if not result["success"] or result["seen"] != 30 or result["microbatches"] != 4 or ema.updates != 1 or error > 1e-7:
        raise AssertionError("The actual30-sample tail CE accumulation/update is inconsistent")
    return {"tail30_accumulation": True, "tail30_full_group_max_parameter_error": error, "ema_updates": ema.updates}


def main():
    torch.set_num_threads(4)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU contract smoke must not initialize CUDA")):
        result = {**roundtrip_schema(), **accumulation_math(), "GPU_used": False, "state": "passed"}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
