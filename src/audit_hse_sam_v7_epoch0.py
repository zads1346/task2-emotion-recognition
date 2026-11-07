"""Replay the same parent at the two already-used FP16 validation batch sizes.

This diagnostic changes neither training artifacts nor the locked candidate set.
It reads development validation images only, and requires no optimizer updates.
"""
from pathlib import Path
from datetime import datetime
import json
import os
import numpy as np
import torch
import train_hse_specialist as h
import train_hse_sam_v7 as t
import validate_cosine_fusion as c
from run_hse_sam_v7 import no_other_gpu_task

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "recovery/v7_epoch0_batch_audit"


def main():
    no_other_gpu_task()
    torch.set_num_threads(4)
    h.seed_everything(42)
    plan = c.read_json(ROOT / "logs/hse_sam_v7_plan.json")
    c.verify_pins(ROOT, plan["source_pins"])
    assert plan["settings"]["validation_batch_size"] == 16 and h.FIXED["batch_size"] == 8
    rows = c.val_rows(ROOT)
    for row in rows:
        row["target"] = int(row["target"])
        row["manifest_index"] = int(row["manifest_index"])
    assert [row["manifest_index"] for row in rows] == plan["val_indices"]
    parent_path = ROOT / "experiments/hse_specialist_v1/best.pt"
    h.check_hash(parent_path, plan["parent_checkpoint_sha256"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    pins = {str(ROOT/"logs/hse_sam_v7_plan.json"): h.sha256(ROOT/"logs/hse_sam_v7_plan.json"),
            str(parent_path): h.sha256(parent_path), str(Path(__file__)): h.sha256(__file__)}
    stored = {}
    for mode in ("adamw_control", "sam"):
        folder = ROOT / f"experiments/hse_sam_v7/{mode}"
        item = c.read_json(folder/"candidates/epoch_00_baseline.json")
        payload = torch.load(folder/item["model"], map_location="cpu", weights_only=True)
        assert payload["val_indices"] == plan["val_indices"]
        assert set(payload["model_state"]) == set(parent["model_state"])
        assert all(torch.equal(value, payload["model_state"][key]) for key,value in parent["model_state"].items())
        for key in ("model", "original", "flip"):
            path = folder/item[key]
            h.check_hash(path, item["artifact_sha256"][key]); pins[str(path)] = h.sha256(path)
        pins[str(folder/"candidates/epoch_00_baseline.json")] = h.sha256(folder/"candidates/epoch_00_baseline.json")
        stored[mode] = [np.load(folder/item[key], allow_pickle=False) for key in ("original", "flip")]
    assert all(np.array_equal(a,b) for a,b in zip(stored["adamw_control"],stored["sam"]))
    previous = []
    for view in ("original", "flip"):
        path=ROOT/f"experiments/hse_specialist_v1/val_{view}.npy"
        pins[str(path)]=h.sha256(path); previous.append(np.load(path, allow_pickle=False))
    model = h.model_api().load_trained_model(ROOT, parent, device="cuda")
    print(json.dumps({"stage":"replay_original_batch8","rows":len(rows)}),flush=True)
    replay8=h.predict_validation(model,rows,0,"cuda")
    print(json.dumps({"stage":"replay_v7_batch16","rows":len(rows)}),flush=True)
    _,_,a,b=t.validate_views(model,rows,0,"cuda",plan["settings"])
    replay16=[a,b]
    comparisons={}
    for label,actual,reference in (("old_batch8",replay8,previous),("v7_batch16",replay16,stored["adamw_control"])):
        comparisons[label]={}
        for view,first,second in zip(("original","flip"),actual,reference):
            comparisons[label][view]={"bitwise_equal":bool(np.array_equal(first,second)),
               "max_abs":float(np.abs(first-second).max()),"argmax_differences":int(np.sum(first.argmax(1)!=second.argmax(1)))}
            # Original materialize_best normalizes already-normalized float64
            # probabilities once more. Permit only float64 rounding (8 eps),
            # never the ~1e-3 cross-batch FP16 difference under investigation.
            tolerance = 8*np.finfo(np.float64).eps if label == "old_batch8" else 0.
            comparisons[label][view]["allowed_max_abs"] = float(tolerance)
            comparisons[label][view]["reproduced"] = bool(np.max(np.abs(first-second)) <= tolerance and np.array_equal(first.argmax(1),second.argmax(1)))
            if not comparisons[label][view]["reproduced"]:
                raise ValueError(f"{label}/{view} did not reproduce exactly: {comparisons[label][view]}")
    cross={view:{"max_abs":float(np.abs(x-y).max()),"mean_abs":float(np.abs(x-y).mean()),
                 "argmax_differences":int(np.sum(x.argmax(1)!=y.argmax(1)))}
           for view,x,y in zip(("original","flip"),replay8,replay16)}
    OUT.mkdir(parents=True,exist_ok=True)
    files={}
    for batch,arrays in ((8,replay8),(16,replay16)):
        for view,value in zip(("original","flip"),arrays):
            path=OUT/f"batch{batch}_{view}.npy";h.save_array(path,value);files[path.name]=h.sha256(path)
    report={"state":"passed","reason":"Different FP16 batch sizes reproduce their respective stored arrays; batch16 bitwise, batch8 within8 float64 eps from repeated normalization",
      "created_at":datetime.now().astimezone().isoformat(),"classes":t.CLASSES,"validation_rows":2656,
      "parent_state_bitwise_equal":True,"state_tensor_count":len(parent["model_state"]),
      "two_arms_epoch0_probabilities_bitwise_equal":True,"comparisons":comparisons,"cross_batch_difference":cross,
      "old_validation_batch_size":8,"new_validation_batch_size":16,"amp_dtype":"float16",
      "same_preprocessing":"h.ExpressionImages(training=False), EXIF RGB,224 bilinear,ImageNet,normalized tensor flip",
      "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
      "torch_version":str(torch.__version__),"device":torch.cuda.get_device_name(),
      "plan_sha256":h.sha256(ROOT/"logs/hse_sam_v7_plan.json"),"source_pins":plan["source_pins"],
      "producer_files":pins,"file_sha256":files,"val_indices":plan["val_indices"],
      "selection_policy_changed":False,"training_modified":False,"optimizer_updates":0,
      "internal_test_read":False,"external_answers_used":False}
    c.verify_pins(ROOT,pins);c.verify_pins(ROOT,plan["source_pins"])
    h.write_json(OUT/"report.json",report)
    print(json.dumps({"state":"passed","comparisons":comparisons,"cross_batch_difference":cross,
                      "report":str(OUT/"report.json")}),flush=True)


if __name__=="__main__":
    main()
