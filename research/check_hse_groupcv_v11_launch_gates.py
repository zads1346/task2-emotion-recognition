"""Synthetic negative launch-gate checks; never launches a worker or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hse_groupcv_v11 as runner


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="v11_gate_", dir=ROOT / "research") as directory:
        base = Path(directory).resolve()
        assert base.parent == (ROOT / "research").resolve()
        runner.ROOT, runner.LOGS = base, base / "logs"
        runner.LOGS.mkdir()
        pins = {}
        for relative in runner.REQUIRED:
            path = base / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text("synthetic launch fixture", encoding="utf-8")
            pins[str(path)] = digest(path)
        plan = {"source_pins": pins, "parent_checkpoint_sha256": "a" * 64,
                "model_source_receipt_sha256": "b" * 64,
                "folds": [{"train_indices": list(range(96))}]}
        path = runner.LOGS / "hse_groupcv_v11_plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        common = {"state": "passed", "source_pins": pins, "plan_sha256": digest(path)}
        review = {**common, "implementation_review": True, "unresolved_blockers": []}
        head = ["classifier.weight", "classifier.bias"]
        def probe(names):
            return {"state": "passed", "successful_optimizer_steps": 2, "amp_skipped_steps": 0,
                    "ema_updates": 2, "head_updated": True, "frozen_state_unchanged": True,
                    "batchnorm_buffers_frozen": True, "trainable_parameter_names": names,
                    "all_parameter_steps": {name: 2 for name in names}, "changed_state_keys": names,
                    "dropout_configuration": {"drop_rate": 0, "drop_path": []}, "peak_allocated_bytes": 1024}
        smoke = {**common, "GPU_used": True, "weights_saved": False, "validation_evaluated": False,
                 "internal_test_read": False, "external_answers_used": False, "batch_size": 16,
                 "effective_batch_size": 32, "parent_checkpoint_sha256": "a" * 64,
                 "model_source_receipt_sha256": "b" * 64, "training_manifest_indices": list(range(64)),
                 "modes": {"full": probe(head + ["conv_stem.weight"]), "fc_only": probe(head)}}
        def write_and_check(r, s):
            for name, value in (("review", r), ("gpu_smoke", s)):
                (runner.LOGS / f"hse_groupcv_v11_{name}.json").write_text(json.dumps(value), encoding="utf-8")
            return runner.verify_review_and_smoke()
        assert write_and_check(review, smoke) == plan
        results.append({"case": "complete_bound_fixture", "passed": True})
        cases = [
            ("protocol_review_only", lambda r,s: r.update(implementation_review=False)),
            ("unresolved_review_blocker", lambda r,s: r.update(unresolved_blockers=["wrong gradients"])),
            ("stale_review_plan", lambda r,s: r.update(plan_sha256="0"*64)),
            ("cpu_smoke", lambda r,s: s.update(GPU_used=False)),
            ("outer_validation_in_smoke", lambda r,s: s.update(validation_evaluated=True)),
            ("wrong_training_members", lambda r,s: s.update(training_manifest_indices=list(range(1,65)))),
            ("wrong_effective_batch", lambda r,s: s.update(effective_batch_size=16)),
            ("one_arm_only", lambda r,s: s["modes"].pop("fc_only")),
            ("one_successful_step", lambda r,s: s["modes"]["full"].update(successful_optimizer_steps=1)),
            ("skipped_update", lambda r,s: s["modes"]["full"].update(amp_skipped_steps=1)),
            ("wrong_ema_count", lambda r,s: s["modes"]["fc_only"].update(ema_updates=1)),
            ("frozen_backbone_changed", lambda r,s: s["modes"]["fc_only"]["changed_state_keys"].append("conv_stem.weight")),
            ("optimizer_step_mismatch", lambda r,s: s["modes"]["full"]["all_parameter_steps"].update({"conv_stem.weight":1})),
            ("missing_dropout_evidence", lambda r,s: s["modes"]["full"].pop("dropout_configuration")),
            ("no_gpu_memory", lambda r,s: s["modes"]["full"].update(peak_allocated_bytes=0)),
        ]
        for name, mutate in cases:
            r, s = copy.deepcopy(review), copy.deepcopy(smoke)
            mutate(r, s)
            try:
                write_and_check(r, s)
            except (ValueError, KeyError):
                results.append({"case": name, "rejected": True})
            else:
                raise AssertionError("Unsafe launch fixture passed: " + name)
        next(iter(pins))
        changed = Path(next(iter(pins)))
        changed.write_text("changed source", encoding="utf-8")
        try:
            write_and_check(review, smoke)
        except ValueError:
            results.append({"case": "source_hash_tampered", "rejected": True})
        else:
            raise AssertionError("Tampered source passed")
    report = {"state": "passed", "GPU_used": False, "worker_launched": False,
              "runner_sha256": digest(ROOT / "src/run_hse_groupcv_v11.py"),
              "test_source_sha256": digest(Path(__file__)), "tests": results}
    (ROOT / "research/hse_groupcv_v11_launch_gate_tests.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"state": "passed", "tests": len(results), "GPU_used": False}))


if __name__ == "__main__":
    main()
