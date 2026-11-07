"""Audit six-epoch DDAMFN++ fine-tuning and replace only the incumbent20% CNN branch.

The incumbent is the completed V7 export pinned before V8 training. Validation
is repeatedly reused; no internal-test data or external answer is read.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[_key]="4"
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"logs/ddamfn_v8_plan.json"
TRAIN=ROOT/"experiments/ddamfn_v8"
OUT=ROOT/"ddamfn_v8_results"
BASELINE=ROOT/"hse_sam_v7_results"
CNN_FOLDER=ROOT/"experiments/cnn_reference"
CNN_FLIP=ROOT/"experiments/refinement_tta_cache/cnn_flip.npy"
CNN_CACHE=ROOT/"fusion_results/prediction_cache/cnn"
CNN_SHA="bc24cf78b30f87b098805b38ff0ecdd14c8797675e55774ddf36bff71294f33f"
CLASSES=["Angry","Fear","Happy","Sad","Surprise","Neutral"]
IMAGE_SHA="02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA="8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
PRESERVED={"lora_task_v4_results/submission.csv":"cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd",
 "lora_zero_weight_v5_results/submission.csv":"cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd",
 "lora_flip_consistency_v6_results/submission.csv":"cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd",
 "hse_results/submission.csv":"f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7",
 "fusion_results/submission.csv":"2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad"}
CAVEAT=("Repeated development validation, not Kaggle or an independent accuracy estimate. "
 "Six raw/EMA epochs plus the official mapped epoch0 are selected on one reused validation split and one seed. "
 "AffectNet pretraining overlap with this dataset has not been independently audited. "
 "No new same-architecture unpretrained control is run; gains cannot be assigned solely to fine-tuning or architecture.")
BINDINGS=("classes","settings","source_pins","input_receipt","original_train_indices","excluded_train_indices",
 "filtered_membership_sha256","train_indices","val_indices","audit_summary_sha256","parent_checkpoint",
 "parent_checkpoint_sha256","model_source_receipt","model_source_receipt_sha256","baseline_directory",
 "baseline_files","baseline_correct_count","baseline_submission_sha256")

def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)

def safe_child(folder, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("An artifact path escapes its dedicated directory")
    result = (folder / path).resolve()
    if not result.is_relative_to(folder.resolve()):
        raise ValueError("An artifact resolves outside its dedicated directory")
    return result

def pin(pins, path, expected=None):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else safe_child(ROOT, path)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Producer files must remain inside the project")
    digest = h.sha256(path) if expected is None else expected
    h.check_hash(path, digest)
    key = str(path)
    if key in pins and pins[key] != digest:
        raise ValueError("Conflicting hashes for the same producer file")
    pins[key] = digest

def merged_pins(*maps):
    result = {}
    for mapping in maps:
        if not isinstance(mapping, dict):
            raise ValueError("Producer pins must be a path-to-SHA256 mapping")
        for path, digest in mapping.items():
            pin(result, path, digest)
    return result

def verify_preserved():
    for relative, digest in PRESERVED.items():
        h.check_hash(ROOT / relative, digest)

def atomic_copy(source, destination):
    import shutil
    temporary = Path(str(destination) + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)

def verify_submission(rows, probability):
    c.submission_check(OUT / "submission.csv", rows, probability)
    with (OUT / "submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "Emotion"] or len(list(reader)) != 7065:
            raise ValueError("Final submission requires exactly7065 rows and ID,Emotion columns")

def write_metrics(entries):
    path = OUT / "candidate_metrics.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["name", "mode", "epoch", "weight_source", "single_accuracy", "single_macro_f1",
              "single_log_loss", "single_correct_count", "accuracy", "macro_f1", "log_loss", "correct_count"]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
    os.replace(temporary, path)

def metric_record(name, mode, item, single, fused):
    return {"name": name, "mode": mode, "epoch": item["epoch"], "weight_source": item["weight_source"],
            "single_accuracy": single["accuracy"], "single_macro_f1": single["macro_f1"],
            "single_log_loss": single["log_loss"], "single_correct_count": single["correct_count"],
            **fused}

def paired_errors(targets,baseline,candidate):
    first=c.check_probability(baseline,len(targets)).argmax(1)==targets
    second=c.check_probability(candidate,len(targets)).argmax(1)==targets
    return {"both_correct":int(np.sum(first&second)),"baseline_only_correct":int(np.sum(first&~second)),
      "candidate_only_correct":int(np.sum(~first&second)),"both_wrong":int(np.sum(~first&~second)),
      "net_correct_change":int(np.sum(second)-np.sum(first))}

def fixed_replacement(incumbent,old_views,new_views):
    incumbent=c.check_probability(incumbent)
    if len(old_views)!=2 or len(new_views)!=2:raise ValueError("Require original and flip views")
    old=[c.check_probability(a,len(incumbent)) for a in old_views]
    new=[c.check_probability(a,len(incumbent)) for a in new_views]
    value=incumbent-.1*old[0]-.1*old[1]+.1*new[0]+.1*new[1]
    if value.min() < -1e-10:raise ValueError("Fixed20% CNN replacement produced negative probabilities")
    return c.check_probability(np.maximum(value,0),len(incumbent))


def old_cnn_inputs(pins=None):
    """Verified saved CNN view probabilities only; no model inference/image reads."""
    pins={} if pins is None else pins
    rows=c.val_rows(ROOT);targets=np.asarray([int(row["target"]) for row in rows])
    indices=[int(row["manifest_index"]) for row in rows]
    summary=c.read_json(CNN_FOLDER/"summary.json")
    if (summary.get("classes")!=CLASSES or summary.get("val_indices")!=indices
        or summary.get("checkpoint_sha256")!=CNN_SHA or summary.get("model")!="resnet18"
        or summary.get("probability_column_permutation")!=[summary["checkpoint_classes"].index(label) for label in CLASSES]):
        raise ValueError("Old CNN class mapping/model/validation membership differs")
    pin(pins,CNN_FOLDER/"summary.json");pin(pins,ROOT/"models/cnn_resnet18_reference.pt",CNN_SHA)
    pin(pins,ROOT/"features/manifest.csv",summary["manifest_sha256"])
    original=CNN_FOLDER/"val_probabilities.npy";pin(pins,original,summary["val_probabilities_sha256"])
    flip_receipt=c.read_json(CNN_FLIP.with_suffix(".json"));signature=flip_receipt["signature"]
    if (flip_receipt.get("signature_sha256")!=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        or signature.get("classes")!=CLASSES or signature.get("manifest_indices")!=indices
        or signature.get("paths")!=[row["path"] for row in rows]
        or signature.get("checkpoints",{}).get("cnn",{}).get("sha256")!=CNN_SHA
        or signature.get("input_sha256",{}).get("manifest.csv")!=h.sha256(ROOT/"features/manifest.csv")):
        raise ValueError("Old CNN flip probability cache binding differs")
    pin(pins,CNN_FLIP,flip_receipt["probability_sha256"]);pin(pins,CNN_FLIP.with_suffix(".json"))
    views=[c.check_probability(np.load(path,allow_pickle=False),2656) for path in (original,CNN_FLIP)]
    if c.metrics(targets,(views[0]+views[1])*.5)["correct_count"]!=1728:raise ValueError("Old CNN TTA does not reproduce1728/2656")
    fusion_recipe_path=ROOT/"fusion_validation_results/selected_recipe.json"
    fusion=c.read_json(fusion_recipe_path);export=c.read_json(ROOT/"fusion_results/summary.json")
    if (fusion.get("classes")!=CLASSES or fusion.get("state")!="locked"
        or fusion.get("artifacts",{}).get("cnn",{}).get("sha256")!=CNN_SHA
        or export.get("state")!="complete" or export.get("recipe_sha256")!=h.sha256(fusion_recipe_path)):
        raise ValueError("Old CNN competition recipe binding differs")
    meta=c.read_json(CNN_CACHE/"metadata.json");binding=meta.get("binding",{})
    if (meta.get("complete") is not True or meta.get("signature")!=c.signature(binding)
        or binding.get("model")!="cnn" or binding.get("classes")!=CLASSES
        or binding.get("rows")!=7065 or binding.get("partition")!="unlabeled"
        or binding.get("checkpoint_sha256")!=CNN_SHA or binding.get("recipe_sha256")!=h.sha256(fusion_recipe_path)
        or binding.get("validator_sha256")!=fusion["validator_sha256"]
        or binding.get("source_producer_files_sha256")!=c.signature(fusion["producer_files"])
        or binding.get("row_order_sha256")!=ROW_SHA or binding.get("image_content_sha256")!=IMAGE_SHA):
        raise ValueError("Old CNN competition cache model/source/image/ID binding differs")
    cache_receipt=export["prediction_cache_receipts"]["cnn"]
    if cache_receipt.get("metadata_sha256")!=h.sha256(CNN_CACHE/"metadata.json") or cache_receipt.get("binding_sha256")!=c.signature(binding):raise ValueError("Old CNN cache differs from completed export receipt")
    for path in (CNN_CACHE/"metadata.json",fusion_recipe_path,ROOT/"fusion_results/summary.json"):pin(pins,path)
    competition=[]
    for view in ("original","flip"):
        path=CNN_CACHE/f"{view}.npy";pin(pins,path,meta["file_sha256"][f"{view}.npy"])
        competition.append(c.check_probability(np.load(path,allow_pickle=False),7065))
    for path in (original,CNN_FLIP,CNN_FOLDER/"summary.json",ROOT/"models/cnn_resnet18_reference.pt"):
        expected=fusion["producer_files"].get(str(path))
        if expected is None:raise ValueError("Old CNN producer recipe omits a reused artifact")
        pin(pins,path,expected)
    return rows,targets,views,competition,pins


def preflight_cnn(root=ROOT):
    if Path(root).resolve()!=ROOT.resolve():raise ValueError("Fixed project root required")
    _,targets,old,_,_=old_cnn_inputs()
    current=c.check_probability(np.load(ROOT/"lora_task_v4_results/selected_val_probabilities.npy",allow_pickle=False),2656)
    repeated=fixed_replacement(current,old,old)
    return {"passed":True,"old_cnn_tta_metrics":c.metrics(targets,(old[0]+old[1])*.5),
      "self_replacement_max_absolute_difference":float(np.max(np.abs(repeated-current))),
      "self_replacement_correct_count":c.metrics(targets,repeated)["correct_count"],"image_pixels_read":0,"GPU_used":False}


def load_plan():
    import ddamfn_v8_model as model_api
    plan=c.read_json(PLAN)
    if (plan.get("classes")!=CLASSES or plan.get("settings",{}).get("epochs")!=6
        or Path(plan["baseline_directory"]).resolve()!=BASELINE.resolve()
        or plan.get("internal_test_read",False) or plan.get("external_answers_used",False)):
        raise ValueError("V8 fixed classes/six epochs/V7 incumbent/data policy differ")
    if (len(plan["original_train_indices"])!=21248 or plan["excluded_train_indices"]!=[7727,10448]
        or plan["train_indices"]!=[i for i in plan["original_train_indices"] if i not in {7727,10448}]
        or len(plan["train_indices"])!=21246 or len(plan["val_indices"])!=2656):raise ValueError("V8 shared filtered membership differs")
    pins=merged_pins(plan["source_pins"],plan["baseline_files"])
    for name in ("finalize_ddamfn_v8.py","train_ddamfn_v8.py","ddamfn_v8_model.py"):
        if str((ROOT/"src"/name).resolve()) not in pins:raise ValueError("V8 source pins omit a required implementation")
    receipt=model_api.source_receipt(ROOT)
    if receipt!=plan["model_source_receipt"] or model_api.receipt_sha256(receipt)!=plan["model_source_receipt_sha256"]:raise ValueError("V8 official model/source receipt differs")
    pin(pins,Path(plan["parent_checkpoint"]),plan["parent_checkpoint_sha256"])
    audit_path=ROOT/"experiments/lora_flip_consistency_v6/audit/summary.json"
    pin(pins,audit_path,plan["audit_summary_sha256"]);audit=c.read_json(audit_path)
    if audit.get("state")!="complete" or audit.get("classes")!=CLASSES or audit.get("sample_rows")!=192 or audit.get("filtered_membership_sha256")!=plan["filtered_membership_sha256"]:raise ValueError("V8 audit binding differs")
    for key in ("original_train_indices","excluded_train_indices","train_indices","val_indices"):
        if audit.get(key)!=plan[key]:raise ValueError("V8 audit members differ")
    for relative,digest in audit["file_sha256"].items():pin(pins,safe_child(audit_path.parent,relative),digest)
    for path,digest in audit["source_pins"].items():pin(pins,path,digest)
    pin(pins,PLAN);verify_preserved();return plan,pins,audit


def baseline_inputs(plan,pins):
    rows,targets,old,competition,pins=old_cnn_inputs(pins)
    summary=c.read_json(BASELINE/"summary.json");recipe=c.read_json(BASELINE/"selected_recipe.json")
    if summary.get("state")!="complete" or summary.get("classes")!=CLASSES or summary.get("submission_rows")!=7065 or recipe.get("state")!="locked":raise ValueError("V7 must complete and lock before V8 baseline selection")
    if recipe.get("fusion_weights")!={"hse":.5,"lora":.2,"cnn":.2,"mlp":.1}:raise ValueError("V7 incumbent does not retain the fixed20% old CNN branch")
    if summary.get("competition_image_content_sha256")!=IMAGE_SHA or summary.get("competition_row_order_sha256")!=ROW_SHA:raise ValueError("V7 competition content/IDs differ")
    for relative,digest in summary["file_sha256"].items():pin(pins,safe_child(BASELINE,relative),digest)
    pin(pins,BASELINE/"submission.csv",plan["baseline_submission_sha256"])
    pin(pins,BASELINE/"selected_recipe.json",summary["recipe_sha256"]);pin(pins,BASELINE/"summary.json")
    for path,digest in recipe["producer_files"].items():pin(pins,path,digest)
    value=c.check_probability(np.load(BASELINE/"selected_val_probabilities.npy",allow_pickle=False),2656)
    score=c.metrics(targets,value)
    if score!=summary["selected_metrics"] or score["correct_count"]!=plan["baseline_correct_count"]:raise ValueError("V8 dynamic incumbent correct count differs from pinned V7 results")
    return rows,targets,value,score,old,competition,pins


def check_state(state,reference):
    if not isinstance(state,dict) or set(state)!=set(reference):raise ValueError("Require a complete six-class DDAMFN model state")
    for name,value in state.items():
        other=reference[name]
        if not isinstance(value,torch.Tensor) or value.dtype!=other.dtype or value.shape!=other.shape or not torch.isfinite(value).all().item():raise ValueError("Invalid DDAMFN tensor: "+name)
        if name.endswith(("running_mean","running_var","num_batches_tracked")) and not torch.equal(value.cpu(),other.cpu()):raise ValueError("Frozen DDAMFN BN buffers changed: "+name)


def same_state(first,second,reference):
    check_state(first,reference);check_state(second,reference)
    return all(torch.equal(first[key],second[key]) for key in reference)


def bound_payload(payload,config,plan,reference):
    if payload.get("format")!="ddamfn_v8" or payload.get("config")!=config or payload.get("classes")!=CLASSES:raise ValueError("DDAMFN checkpoint format/config/classes differ")
    for key in ("source_pins","original_train_indices","excluded_train_indices","train_indices","val_indices",
                "filtered_membership_sha256","audit_summary_sha256"):
        if payload.get(key)!=plan[key]:raise ValueError("DDAMFN checkpoint source/member binding differs: "+key)
    if (payload.get("parent_sha")!=plan["parent_checkpoint_sha256"]
        or payload.get("model_source_receipt")!=plan["model_source_receipt"]
        or payload.get("model_source_receipt_sha256")!=plan["model_source_receipt_sha256"]):raise ValueError("DDAMFN parent/official/source binding differs")
    check_state(payload.get("model_state"),reference)


def verify_training(plan,pins,reference):
    folder=TRAIN;summary=c.read_json(folder/"summary.json");config=c.read_json(folder/"config.json")
    if (summary.get("state")!="complete" or summary.get("classes")!=CLASSES or summary.get("epochs_completed")!=6
        or summary.get("candidate_count")!=13 or summary.get("validation_weighted") is not False):
        raise ValueError("V8 requires six complete epochs and thirteen unweighted validation candidates")
    for key in BINDINGS:
        if config.get(key)!=plan[key]:raise ValueError("DDAMFN config differs from plan: "+key)
    if config.get("plan_sha256")!=h.sha256(PLAN) or summary.get("plan_sha256")!=h.sha256(PLAN) or config.get("validation_weighted") is not False:raise ValueError("DDAMFN plan/unweighted validation binding differs")
    required={"config.json","history.json","candidates.json","last.json","last.pt","epochs/epoch_06.pt"}
    if not required.issubset(summary["artifact_sha256"]):raise ValueError("DDAMFN receipt omits a required committed artifact")
    for name,digest in summary["artifact_sha256"].items():pin(pins,safe_child(folder,name),digest)
    pin(pins,folder/"summary.json")
    pointer=c.read_json(folder/"last.json")
    if pointer.get("epoch")!=6 or pointer.get("checkpoint")!="epochs/epoch_06.pt" or pointer.get("plan_sha256")!=h.sha256(PLAN):raise ValueError("DDAMFN last pointer must identify complete epoch6")
    pin(pins,folder/pointer["checkpoint"],pointer["sha256"]);pin(pins,folder/"last.pt",pointer["sha256"])
    final=torch.load(folder/pointer["checkpoint"],map_location="cpu",weights_only=True);bound_payload(final,config,plan,reference)
    history=final["history"];candidates=c.read_json(folder/"candidates.json")
    expected=["epoch_00_baseline"]+[f"epoch_{epoch:02d}_{source}" for epoch in range(1,7) for source in ("raw","ema")]
    if len(history)!=6 or history!=c.read_json(folder/"history.json") or candidates!=final["candidates"] or candidates!=summary["candidates"] or [i["name"] for i in candidates]!=expected:raise ValueError("DDAMFN complete history/candidate set differs")
    steps=skips=0
    groups=math.ceil(21246/config["settings"]["effective_batch_size"])
    for epoch,row in enumerate(history,1):
        n,k,z=(row[key] for key in ("successful_optimizer_steps","amp_skipped_steps","zero_weight_groups"))
        if row["epoch"]!=epoch or row["seen"]!=21246 or n<=0 or k<0 or z!=0 or n+k!=groups:raise ValueError("DDAMFN epoch member/group counters differ")
        steps+=n;skips+=k
        if row["cumulative_optimizer_steps"]!=steps or row["ema_updates"]!=steps:raise ValueError("DDAMFN EMA/success counters differ")
    counters={"successful_steps":steps,"amp_skipped_steps":skips,"zero_weight_groups":0}
    if final.get("counters")!=counters or summary.get("counters")!=counters or summary["successful_optimizer_steps"]!=steps or summary["ema_updates"]!=steps or summary["amp_skipped_steps"]!=skips:raise ValueError("DDAMFN final counters differ")
    del final
    for epoch in range(7):
        path=folder/f"epochs/epoch_{epoch:02d}.pt";payload=torch.load(path,map_location="cpu",weights_only=True);bound_payload(payload,config,plan,reference)
        if payload.get("epoch")!=epoch or payload["history"]!=history[:epoch] or payload["candidates"]!=candidates[:1+2*epoch]:raise ValueError("DDAMFN complete epoch differs from final history")
        total=sum(row["successful_optimizer_steps"] for row in history[:epoch]);skip=sum(row["amp_skipped_steps"] for row in history[:epoch])
        if payload["counters"]!={"successful_steps":total,"amp_skipped_steps":skip,"zero_weight_groups":0} or payload["ema_state"]["updates"]!=total:raise ValueError("DDAMFN complete epoch update/EMA counters differ")
        check_state(payload["ema_state"]["shadow"],reference);pin(pins,path)
        optimizer_state=payload["optimizer"]["state"]
        if epoch==0:
            if optimizer_state or not same_state(payload["model_state"],reference,reference) or not same_state(payload["ema_state"]["shadow"],reference,reference):raise ValueError("DDAMFN epoch0 must use exact official mapped model and fresh optimizer/EMA")
            active=[candidates[0]]
        else:
            if not optimizer_state or any(int(s["step"].item() if isinstance(s["step"],torch.Tensor) else s["step"])!=total for s in optimizer_state.values()):raise ValueError("Actual AdamW per-parameter steps differ from successful updates")
            active=candidates[1+2*(epoch-1):1+2*epoch]
        for item in active:
            if item.get("classes")!=CLASSES or item.get("val_indices")!=plan["val_indices"] or item.get("plan_sha256")!=h.sha256(PLAN) or item.get("validation_weighted") is not False:raise ValueError("DDAMFN candidate class/validation/plan binding differs")
            if item!=c.read_json(safe_child(folder,item["candidate_json"])):raise ValueError("DDAMFN candidate JSON differs")
            for key in ("model","original","flip"):pin(pins,safe_child(folder,item[key]),item["artifact_sha256"][key])
            pin(pins,safe_child(folder,item["candidate_json"]))
            cp=torch.load(safe_child(folder,item["model"]),map_location="cpu",weights_only=True);bound_payload(cp,config,plan,reference)
            state=reference if epoch==0 else payload["model_state"] if item["weight_source"]=="raw" else payload["ema_state"]["shadow"]
            if cp.get("epoch")!=epoch or cp.get("weight_source")!=item["weight_source"] or not same_state(cp["model_state"],state,reference):raise ValueError("DDAMFN candidate model differs from complete raw/EMA state")
            for key in ("single_original_metrics","single_tta_metrics"):
                if cp.get(key)!=item[key]:raise ValueError("DDAMFN candidate metrics differ from model metadata")
            del cp
        del payload
    return candidates,summary,config,history


def candidate_views(item,targets):
    raw=[np.load(safe_child(TRAIN,item[key]),allow_pickle=False) for key in ("original","flip")]
    for array in raw:
        if array.dtype!=np.float32 or array.shape!=(2656,6):raise ValueError("DDAMFN validation probability dtype/shape differs")
    score=c.metrics(targets,(raw[0]+raw[1])*.5)
    if score!=item["single_tta_metrics"] or c.metrics(targets,raw[0])!=item["single_original_metrics"]:raise ValueError("DDAMFN metrics do not reproduce from actual probabilities")
    return [c.check_probability(a,2656) for a in raw],score


def verify_locked():
    plan,_,_=load_plan();recipe=c.read_json(OUT/"selected_recipe.json")
    if (recipe.get("state")!="locked" or recipe.get("classes")!=CLASSES
        or recipe.get("plan_sha256")!=h.sha256(PLAN) or recipe.get("finalizer_sha256")!=h.sha256(__file__)
        or recipe.get("candidate_count")!=13 or recipe.get("replacement_fraction")!=.2):raise ValueError("DDAMFN locked recipe differs")
    c.verify_pins(ROOT,recipe["producer_files"])
    h.check_hash(OUT/"selected_val_probabilities.npy",recipe["selected_val_sha256"])
    h.check_hash(OUT/"candidate_metrics.csv",recipe["candidate_metrics_sha256"])
    verify_preserved();return plan,recipe


def select():
    import ddamfn_v8_model as model_api
    plan,pins,audit=load_plan();OUT.mkdir(exist_ok=True)
    if (OUT/"selected_recipe.json").exists():return verify_locked()[1]
    rows,receipt=h.load_trainval(ROOT)
    if receipt!=plan["input_receipt"] or [r["manifest_index"] for r in rows["train"]]!=plan["original_train_indices"]:raise ValueError("DDAMFN original input binding differs")
    rows["train"]=[r for r in rows["train"] if r["manifest_index"] not in {7727,10448}]
    if h.signature(rows["train"])!=plan["filtered_membership_sha256"]:raise ValueError("DDAMFN filtered input membership differs")
    for split in ("train","val"):
        if [r["manifest_index"] for r in rows[split]]!=plan[split+"_indices"]:raise ValueError("DDAMFN effective train/validation order differs")
    _,targets,incumbent,baseline,old,_,pins=baseline_inputs(plan,pins)
    initial,source=model_api.load_initial_model(ROOT,"cpu")
    if source!=plan["model_source_receipt"]:raise ValueError("Official mapped six-class initialization differs")
    reference={key:value.detach().cpu().clone() for key,value in initial.state_dict().items()}
    del initial;gc.collect()
    candidates,training_summary,config,history=verify_training(plan,pins,reference)
    best_p,best,selected=incumbent,baseline,None
    best_single=best_fused=best_new_single=best_new_fused=None
    entries=[{"name":"previous_best","mode":"incumbent",**baseline}]
    old_single=c.metrics(targets,(old[0]+old[1])*.5)
    epoch0_single=None
    for item in candidates:
        views,single=candidate_views(item,targets)
        # Epoch0 is already a different pretrained DDAMFN classifier. It must
        # replace the old CNN like every candidate, never impersonate incumbent.
        probability=fixed_replacement(incumbent,old,views)
        fused=c.metrics(targets,probability);name=item["name"]
        entry=metric_record(name,"ddamfn",item,single,fused)
        entry["paired_vs_incumbent"]=paired_errors(targets,incumbent,probability)
        entries.append(entry)
        def info(metrics):return {"name":name,"epoch":item["epoch"],"weight_source":item["weight_source"],"metrics":metrics}
        if item["epoch"]==0:epoch0_single=single
        if best_single is None or single["correct_count"]>best_single["metrics"]["correct_count"]:best_single=info(single)
        if best_fused is None or fused["correct_count"]>best_fused["metrics"]["correct_count"]:best_fused=info(fused)
        if item["epoch"]>0:
            if best_new_single is None or single["correct_count"]>best_new_single["metrics"]["correct_count"]:best_new_single=info(single)
            if best_new_fused is None or fused["correct_count"]>best_new_fused["metrics"]["correct_count"]:best_new_fused=info(fused)
        if fused["correct_count"]>best["correct_count"]:selected,best,best_p=item,fused,probability
    comparison={"old_cnn_tta_metrics":old_single,"official_epoch0_ddamfn_tta_metrics":epoch0_single,
      "best_single":best_single,"best_fixed_fusion":best_fused,"best_new_single":best_new_single,"best_new_fused":best_new_fused,
      "new_candidate_definition":"epochs1-6 raw/EMA only; mapped official epoch0 excluded",
      "best_single_minus_old_cnn_correct":best_single["metrics"]["correct_count"]-old_single["correct_count"],
      "best_new_single_minus_epoch0_correct":best_new_single["metrics"]["correct_count"]-epoch0_single["correct_count"]}
    write_metrics(entries)
    if selected is None:atomic_copy(BASELINE/"selected_val_probabilities.npy",OUT/"selected_val_probabilities.npy")
    else:h.save_array(OUT/"selected_val_probabilities.npy",best_p)
    recipe={"format":"ddamfn_v8_fixed_fusion","state":"locked","classes":CLASSES,"selected":selected,
      "improved":selected is not None,"baseline_metrics":baseline,"selected_metrics":best,
      "baseline_directory":str(BASELINE),"baseline_correct_count":plan["baseline_correct_count"],
      "candidate_count":13,"candidate_order":[item["name"] for item in candidates],
      "selection_rule":"incumbent first; epoch0 then epoch1-6 raw/EMA; strict fused correct-count increase; every tie keeps earlier candidate",
      "replacement_fraction":.2,"replacement":"incumbent-.1*oldCNNoriginal-.1*oldCNNflip+.1*newDDAMoriginal+.1*newDDAMflip",
      "validation_candidates":entries,"model_comparison":comparison,"paired_selected_vs_incumbent":paired_errors(targets,incumbent,best_p),
      "producer_files":pins,"plan_sha256":h.sha256(PLAN),"parent_checkpoint_sha256":plan["parent_checkpoint_sha256"],
      "model_source_receipt_sha256":plan["model_source_receipt_sha256"],"audit_summary_sha256":plan["audit_summary_sha256"],
      "filtered_membership_sha256":plan["filtered_membership_sha256"],"effective_training_rows":21246,"excluded_train_indices":[7727,10448],
      "val_indices":plan["val_indices"],"val_rows_sha256":receipt["val_rows_sha256"],
      "selected_val_sha256":h.sha256(OUT/"selected_val_probabilities.npy"),"candidate_metrics_sha256":h.sha256(OUT/"candidate_metrics.csv"),
      "old_cnn_competition_cache":str(CNN_CACHE),"competition_image_content_sha256":IMAGE_SHA,"competition_row_order_sha256":ROW_SHA,
      "finalizer_sha256":h.sha256(__file__),"internal_test_read":False,"external_answers_used":False,
      "preserved_submission_sha256":PRESERVED,"validation_caveat":CAVEAT}
    c.verify_pins(ROOT,pins);h.write_json(OUT/"selected_recipe.json",recipe)
    emit("select_ddamfn_v8",improved=recipe["improved"],selected=selected["name"] if selected else "previous_best",correct_count=best["correct_count"],accuracy=best["accuracy"])
    return recipe


def predict_winner(recipe,rows,batch_size=16):
    import ddamfn_v8_model as model_api
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no silent CPU competition inference")
    item=recipe["selected"];checkpoint=safe_child(TRAIN,item["model"])
    model,_=model_api.load_trained_model(ROOT,checkpoint,"cuda");model.eval()
    transform=transforms.Compose([transforms.Resize((112,112),interpolation=InterpolationMode.BILINEAR),
      transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    recipe_sha=h.sha256(OUT/"selected_recipe.json");values=[[],[]];started=last=time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                h.check_hash(OUT/"selected_recipe.json",recipe_sha);h.check_hash(checkpoint,item["artifact_sha256"]["model"])
                tensors=[]
                for row in rows[start:start+batch_size]:
                    with Image.open(row["path"]) as image:tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                images=torch.stack(tensors).cuda()
                with h.autocast_for("cuda"):first,second=model(images),model(images.flip(-1))
                for view,logits in enumerate((first,second)):
                    value=logits.float().softmax(1).cpu().numpy();values[view].append(c.check_probability(value,len(images)))
                if time.monotonic()-last>=30 or start+batch_size>=len(rows):
                    emit("export_ddamfn_v8",seen=min(start+batch_size,len(rows)),total=len(rows),elapsed_seconds=time.monotonic()-started);last=time.monotonic()
        h.check_hash(OUT/"selected_recipe.json",recipe_sha)
        return [c.check_probability(np.concatenate(view),7065) for view in values]
    finally:
        del model;gc.collect();torch.cuda.empty_cache()


def export(batch_size=16):
    select();plan,recipe=verify_locked();recipe_sha=h.sha256(OUT/"selected_recipe.json")
    rows=c.unlabeled_rows(ROOT)
    if c.signature(rows)!=ROW_SHA:raise ValueError("DDAMFN competition IDs/order differ from reused old probabilities")
    images=c.image_receipt(ROOT,rows,verify_inventory=False)
    if images["image_content_sha256"]!=IMAGE_SHA:raise ValueError("DDAMFN competition bytes differ from reused old probabilities")
    if (OUT/"summary.json").exists():
        summary=c.read_json(OUT/"summary.json")
        if summary.get("state")!="complete" or summary.get("classes")!=CLASSES or summary.get("recipe_sha256")!=recipe_sha:raise ValueError("Completed DDAMFN export belongs to another recipe")
        for name,digest in summary["file_sha256"].items():h.check_hash(safe_child(OUT,name),digest)
        verify_submission(rows,c.check_probability(np.load(OUT/"probabilities.npy",allow_pickle=False),7065));verify_preserved();return summary
    if recipe["improved"]:
        new=predict_winner(recipe,rows,batch_size)
        for view,value in zip(("original","flip"),new):h.save_array(OUT/f"ddamfn_competition_{view}.npy",value)
        baseline=c.check_probability(np.load(BASELINE/"probabilities.npy",allow_pickle=False),7065)
        old=[c.check_probability(np.load(CNN_CACHE/f"{view}.npy",allow_pickle=False),7065) for view in ("original","flip")]
        probability=fixed_replacement(baseline,old,new);h.save_array(OUT/"probabilities.npy",probability)
        temp=OUT/"submission.csv.tmp"
        with temp.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["ID","Emotion"]);writer.writerows((row["ID"],int(label)) for row,label in zip(rows,probability.argmax(1)))
        os.replace(temp,OUT/"submission.csv")
    else:
        atomic_copy(BASELINE/"probabilities.npy",OUT/"probabilities.npy");atomic_copy(BASELINE/"submission.csv",OUT/"submission.csv")
        h.check_hash(OUT/"submission.csv",h.sha256(BASELINE/"submission.csv"));h.check_hash(OUT/"probabilities.npy",h.sha256(BASELINE/"probabilities.npy"))
        probability=c.check_probability(np.load(OUT/"probabilities.npy",allow_pickle=False),7065)
    verify_submission(rows,probability);h.check_hash(OUT/"selected_recipe.json",recipe_sha)
    c.verify_pins(ROOT,recipe["producer_files"]);load_plan();verify_preserved()
    if c.image_receipt(ROOT,rows,verify_inventory=False)["image_content_sha256"]!=IMAGE_SHA:raise ValueError("Competition bytes changed during DDAMFN export")
    files=["submission.csv","probabilities.npy","selected_val_probabilities.npy","selected_recipe.json","candidate_metrics.csv"]
    if recipe["improved"]:files += ["ddamfn_competition_original.npy","ddamfn_competition_flip.npy"]
    summary={"format":"ddamfn_v8_export","state":"complete","classes":CLASSES,"submission_rows":7065,"improved":recipe["improved"],
      "selected":recipe["selected"]["name"] if recipe["selected"] else "previous_best",
      "baseline_directory":str(BASELINE),"baseline_metrics":recipe["baseline_metrics"],"selected_metrics":recipe["selected_metrics"],
      "model_comparison":recipe["model_comparison"],"paired_selected_vs_incumbent":recipe["paired_selected_vs_incumbent"],
      "recipe_sha256":recipe_sha,"plan_sha256":recipe["plan_sha256"],"model_source_receipt_sha256":recipe["model_source_receipt_sha256"],
      "audit_summary_sha256":recipe["audit_summary_sha256"],"finalizer_sha256":h.sha256(__file__),
      "official_label_order_verified":True,"ID_template_order_verified":True,"probability_argmax_verified":True,
      "competition_image_content_sha256":IMAGE_SHA,"competition_row_order_sha256":ROW_SHA,
      "new_ddamfn_competition_inference":recipe["improved"],"other_model_inference_repeated":False,
      "effective_training_rows":21246,"excluded_train_indices":[7727,10448],"preserved_submission_sha256":PRESERVED,
      "internal_test_read":False,"external_answers_used":False,"validation_caveat":CAVEAT,"file_sha256":{name:h.sha256(OUT/name) for name in files}}
    h.write_json(OUT/"summary.json",summary)
    emit("complete_ddamfn_v8",improved=summary["improved"],selected=summary["selected"],correct_count=summary["selected_metrics"]["correct_count"],submission=str(OUT/"submission.csv"))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("preflight","select","export"),default="export")
    parser.add_argument("--batch-size",type=int,choices=(1,2,4,8,16),default=16);args=parser.parse_args()
    if args.project_root.resolve()!=ROOT.resolve():raise ValueError("Fixed project root required")
    torch.set_num_threads(4)
    if args.stage=="preflight":print(json.dumps(preflight_cnn(),ensure_ascii=False),flush=True)
    elif args.stage=="select":select()
    else:export(args.batch_size)


if __name__=="__main__":main()
