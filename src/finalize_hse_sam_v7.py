"""Audit paired HSE AdamW/SAM continuation and export a fixed50% HSE replacement.

Only the repeatedly reused development validation split selects candidates.
The incumbent wins ties; competition inference is gated on strict improvement.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_key]="4"
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import train_hse_specialist as h
import validate_cosine_fusion as c

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"logs/hse_sam_v7_plan.json"
TRAIN=ROOT/"experiments/hse_sam_v7"
OUT=ROOT/"hse_sam_v7_results"
BASELINE=ROOT/"lora_task_v4_results"
PARENT_FOLDER=ROOT/"experiments/hse_specialist_v1"
PARENT=PARENT_FOLDER/"best.pt"
HSE_EXPORT=ROOT/"hse_results"
HSE_CACHE=HSE_EXPORT/"prediction_cache/hse_unlabeled"
CLASSES=["Angry","Fear","Happy","Sad","Surprise","Neutral"]
MODES=("adamw_control","sam")
IMAGE_SHA="02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA="8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
PARENT_SHA="ed58a57d691943dc605053ac6182d6b54f05f4ae4133690d65ce5abeb10d20b7"
BASELINE_PROB_SHA="24b0781d3cb3bb75f5739bec3f40bd18bff83a1eafcaf26c2fb4807de7843239"
BASELINE_SUB_SHA="cb0ea39ee545ff4d9c229fb03a909e4bb1f28b5ddfef97b700a8974137aaf1fd"
FUSION_WEIGHTS={"hse":.5,"lora":.2,"cnn":.2,"mlp":.1}
PRESERVED={"lora_task_v4_results/submission.csv":BASELINE_SUB_SHA,
 "lora_zero_weight_v5_results/submission.csv":BASELINE_SUB_SHA,
 "lora_flip_consistency_v6_results/submission.csv":BASELINE_SUB_SHA,
 "hse_results/submission.csv":"f349f4f2069e342b15d5d42d2ca705d5fb9c98144eb8304b2ab7a9f9fe7225d7",
 "fusion_results/submission.csv":"2173c02318960c57f20fdb0f5c35d29a6d5436ee2bef3a37beb353169103c6ad"}
CAVEAT=("Repeated development validation, not Kaggle or an independent accuracy estimate. "
 "Both arms continue the same HSE epoch7 model, exclude the same two visually confirmed "
 "invalid training images, and retain21246 uniformly weighted training images. "
 "The matched AdamW control estimates the observed SAM increment for one seed. "
 "Selection on this same repeatedly reused validation set does not establish a general causal benefit.")

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

MODE_SETTINGS={"adamw_control":{"rho":0.},"sam":{"rho":.05}}
BINDINGS=("classes","settings","mode_settings","source_pins","input_receipt",
 "original_train_indices","excluded_train_indices","filtered_membership_sha256",
 "train_indices","val_indices","audit_summary_sha256","parent_checkpoint",
 "parent_checkpoint_sha256","baseline_submission_sha256","baseline_correct_count")


def fixed_replacement(incumbent,old_views,new_views):
    incumbent=c.check_probability(incumbent)
    old=[c.check_probability(value,len(incumbent)) for value in old_views]
    new=[c.check_probability(value,len(incumbent)) for value in new_views]
    if len(old)!=2 or len(new)!=2: raise ValueError("Exactly two original/flip views are required")
    raw=incumbent-.25*old[0]-.25*old[1]+.25*new[0]+.25*new[1]
    if raw.min() < -1e-10: raise ValueError("Fixed50% HSE replacement produced negative probabilities")
    return c.check_probability(np.maximum(raw,0),len(incumbent))


def baseline_inputs(pins=None):
    """Read verified saved probabilities and binding metadata; no image/model inference."""
    pins={} if pins is None else pins
    summary=c.read_json(BASELINE/"summary.json"); recipe=c.read_json(BASELINE/"selected_recipe.json")
    if (summary.get("state")!="complete" or summary.get("classes")!=CLASSES
        or summary.get("submission_rows")!=7065 or summary.get("selected")!="epoch_05_ema"
        or recipe.get("fusion_weights")!=FUSION_WEIGHTS):
        raise ValueError("The incumbent is not the verified V4 four-branch fusion")
    if summary.get("competition_image_content_sha256")!=IMAGE_SHA or summary.get("competition_row_order_sha256")!=ROW_SHA:
        raise ValueError("The incumbent competition content/ID binding differs")
    pin(pins,BASELINE/"submission.csv",BASELINE_SUB_SHA)
    pin(pins,BASELINE/"probabilities.npy",BASELINE_PROB_SHA)
    for relative,digest in summary["file_sha256"].items():pin(pins,safe_child(BASELINE,relative),digest)
    pin(pins,BASELINE/"selected_recipe.json",summary["recipe_sha256"])
    pin(pins,BASELINE/"summary.json")
    old_summary=c.read_json(HSE_EXPORT/"summary.json"); old_recipe=c.read_json(HSE_EXPORT/"selected_recipe.json")
    pin(pins,HSE_EXPORT/"selected_recipe.json",old_summary["recipe_sha256"])
    pin(pins,HSE_EXPORT/"summary.json")
    if (old_recipe.get("classes")!=CLASSES or old_recipe.get("state")!="locked"
        or old_recipe.get("selected",{}).get("name")!="hse_50_flip"
        or old_recipe["selected"].get("alpha")!=.5 or old_recipe["selected"].get("hse_mode")!="flip"
        or old_recipe.get("hse_checkpoint",{}).get("sha256")!=PARENT_SHA):
        raise ValueError("The original HSE half-branch is not the fixed epoch7 flip-TTA recipe")
    receipt=c.read_json(PARENT_FOLDER/"summary.json")
    if (receipt.get("state")!="complete" or receipt.get("classes")!=CLASSES
        or receipt.get("best_epoch")!=7 or receipt.get("best_checkpoint_sha256")!=PARENT_SHA):
        raise ValueError("HSE parent training receipt changed")
    pin(pins,PARENT,PARENT_SHA); pin(pins,PARENT_FOLDER/"summary.json")
    rows=c.val_rows(ROOT)
    if [int(row["manifest_index"]) for row in rows]!=receipt["val_indices"]:
        raise ValueError("The HSE parent and incumbent validation order differ")
    targets=np.asarray([int(row["target"]) for row in rows])
    current=c.check_probability(np.load(BASELINE/"selected_val_probabilities.npy",allow_pickle=False),2656)
    score=c.metrics(targets,current)
    if score!=summary["selected_metrics"] or score["correct_count"]!=1915:
        raise ValueError("The incumbent does not reproduce1915/2656")
    old_views=[]
    for view in ("original","flip"):
        p=PARENT_FOLDER/f"val_{view}.npy"
        pin(pins,p,receipt[f"val_{view}_sha256"])
        old_views.append(c.check_probability(np.load(p,allow_pickle=False),2656))
    meta=c.read_json(HSE_CACHE/"metadata.json"); binding=meta.get("binding",{})
    if (meta.get("complete") is not True or meta.get("signature")!=c.signature(binding)
        or binding.get("partition")!="unlabeled" or binding.get("rows")!=7065
        or binding.get("classes")!=CLASSES or binding.get("checkpoint_sha256")!=PARENT_SHA
        or binding.get("recipe_sha256")!=h.sha256(HSE_EXPORT/"selected_recipe.json")
        or binding.get("finalizer_sha256")!=old_recipe["finalizer_sha256"]
        or binding.get("producer_files_sha256")!=c.signature(old_recipe["producer_files"])
        or binding.get("row_order_sha256")!=ROW_SHA or binding.get("image_content_sha256")!=IMAGE_SHA
        or binding.get("hse_mode")!="flip"):
        raise ValueError("Old HSE competition cache source/model/image/ID binding differs")
    if (old_summary.get("prediction_receipt",{}).get("metadata_sha256")!=h.sha256(HSE_CACHE/"metadata.json")
        or old_summary["prediction_receipt"].get("binding_sha256")!=c.signature(binding)):
        raise ValueError("Old HSE competition cache does not match its export receipt")
    pin(pins,HSE_CACHE/"metadata.json")
    competition=[]
    for view in ("original","flip"):
        p=HSE_CACHE/f"{view}.npy";pin(pins,p,meta["file_sha256"][f"{view}.npy"])
        competition.append(c.check_probability(np.load(p,allow_pickle=False),7065))
    if np.max(np.abs(fixed_replacement(current,old_views,old_views)-current))>1e-12:
        raise ValueError("Replacing old HSE by itself does not reproduce the incumbent")
    return rows,targets,current,score,old_views,competition,receipt,recipe,pins


def preflight_baseline(root=ROOT):
    if Path(root).resolve()!=ROOT.resolve():raise ValueError("This finalizer uses its locked project root")
    _,targets,current,score,old,_,receipt,_,_=baseline_inputs()
    repeated=fixed_replacement(current,old,old)
    return {"passed":True,"incumbent_correct_count":score["correct_count"],
      "self_replacement_correct_count":c.metrics(targets,repeated)["correct_count"],
      "self_replacement_max_absolute_difference":float(np.max(np.abs(repeated-current))),
      "parent_epoch":receipt["best_epoch"],"parent_hse_tta_metrics":c.metrics(targets,(old[0]+old[1])*.5),
      "competition_hse_views":str(HSE_CACHE),"GPU_used":False,"image_pixels_read":0}


def load_plan():
    plan=c.read_json(PLAN)
    if (plan.get("classes")!=CLASSES or plan.get("baseline_correct_count")!=1915
        or plan.get("mode_settings")!=MODE_SETTINGS or plan.get("settings",{}).get("epochs")!=4):
        raise ValueError("V7 fixed classes, modes, four epochs or incumbent differ")
    if plan.get("baseline_submission_sha256")!=BASELINE_SUB_SHA or plan.get("parent_checkpoint_sha256")!=PARENT_SHA or Path(plan["parent_checkpoint"]).resolve()!=PARENT.resolve():
        raise ValueError("V7 fixed incumbent or HSE epoch7 parent binding differs")
    original=plan["original_train_indices"]
    if (len(original)!=21248 or len(set(original))!=21248 or plan["excluded_train_indices"]!=[7727,10448]
        or not {7727,10448}.issubset(original)
        or plan["train_indices"]!=[index for index in original if index not in {7727,10448}]
        or len(plan["train_indices"])!=21246 or len(plan["val_indices"])!=2656):
        raise ValueError("V7 requires only the shared two invalid-image exclusions")
    if plan.get("internal_test_read",False) or plan.get("external_answers_used",False):raise ValueError("Internal test/external answers forbidden")
    pins=merged_pins(plan["source_pins"],plan.get("baseline_files",{}))
    for name in ("finalize_hse_sam_v7.py","train_hse_sam_v7.py","hse_model.py"):
        if str((ROOT/"src"/name).resolve()) not in pins:raise ValueError("Missing required V7 implementation pin")
    audit_path=ROOT/"experiments/lora_flip_consistency_v6/audit/summary.json"
    pin(pins,audit_path,plan["audit_summary_sha256"]);audit=c.read_json(audit_path)
    if (audit.get("state")!="complete" or audit.get("classes")!=CLASSES or audit.get("sample_rows")!=192
        or audit.get("train_rows")!=21248 or audit.get("effective_train_rows")!=21246
        or audit.get("filtered_membership_sha256")!=plan["filtered_membership_sha256"]):
        raise ValueError("V7 shared192-image audit binding differs")
    for key in ("original_train_indices","excluded_train_indices","train_indices","val_indices"):
        if audit.get(key)!=plan[key]:raise ValueError("V7 audit training/validation members differ")
    for key in ("labels_changed","images_changed","weights_changed","internal_test_read","external_answers_used"):
        if audit.get(key) is not False:raise ValueError("Audit cannot relabel, edit or reweight images or use forbidden data")
    for relative,digest in audit["file_sha256"].items():pin(pins,safe_child(audit_path.parent,relative),digest)
    for p,digest in audit["source_pins"].items():pin(pins,p,digest)
    pin(pins,PLAN);verify_preserved()
    return plan,pins,audit


def check_state(state,expected):
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError("HSE checkpoint is not a complete six-class model state")
    for name,value in state.items():
        reference=expected[name]
        if (not isinstance(value,torch.Tensor) or value.dtype!=reference.dtype or value.shape!=reference.shape
            or not torch.isfinite(value).all().item()):raise ValueError("Invalid HSE tensor: "+name)
        if name.endswith(("running_mean","running_var","num_batches_tracked")) and not torch.equal(value.cpu(),reference.cpu()):
            raise ValueError("Frozen HSE BatchNorm running statistics changed: "+name)
    if state["classifier.weight"].shape[0]!=6:raise ValueError("HSE classifier must contain six official classes")


def same_state(first,second,parent):
    check_state(first,parent);check_state(second,parent)
    return all(torch.equal(first[name],second[name]) for name in parent)


def bound_payload(payload,config,mode,plan,parent):
    if payload.get("format")!="hse_sam_v7" or payload.get("config")!=config or payload.get("classes")!=CLASSES or payload.get("mode")!=mode:
        raise ValueError("V7 checkpoint format/config/classes/mode differ")
    for key in ("source_pins","original_train_indices","excluded_train_indices","train_indices","val_indices","audit_summary_sha256"):
        if payload.get(key)!=plan[key]:raise ValueError("V7 checkpoint source/member/audit binding differs: "+key)
    if payload.get("parent_sha")!=PARENT_SHA or payload.get("model_provenance")!=parent["model_provenance"]:
        raise ValueError("V7 checkpoint parent/official HSE provenance differs")
    check_state(payload.get("model_state"),parent["model_state"])


def verify_training(mode,plan,pins,parent):
    folder=TRAIN/mode
    summary=c.read_json(folder/"summary.json");config=c.read_json(folder/"config.json")
    if summary.get("state")!="complete" or summary.get("epochs_completed")!=4 or summary.get("candidate_count")!=9 or summary.get("mode")!=mode or summary.get("classes")!=CLASSES:
        raise ValueError("Each V7 arm requires four complete epochs and nine fixed candidates")
    for key in BINDINGS:
        if config.get(key)!=plan[key]:raise ValueError("Arm configuration differs from plan: "+key)
    if config.get("mode")!=mode or config.get("plan_sha256")!=h.sha256(PLAN) or summary.get("plan_sha256")!=h.sha256(PLAN):raise ValueError("Arm mode/plan SHA differs")
    if config.get("validation_weighted") is not False or config.get("model_provenance")!=parent["model_provenance"]:raise ValueError("HSE official provenance/unweighted validation differs")
    for rel,digest in summary["artifact_sha256"].items():pin(pins,safe_child(folder,rel),digest)
    required={"config.json","history.json","candidates.json","last.json","last.pt","epochs/epoch_04.pt"}
    if not required.issubset(summary["artifact_sha256"]):raise ValueError("Training receipt omits a required committed artifact")
    pin(pins,folder/"summary.json")
    pointer=c.read_json(folder/"last.json")
    if pointer.get("epoch")!=4 or pointer.get("checkpoint")!="epochs/epoch_04.pt" or pointer.get("mode")!=mode or pointer.get("plan_sha256")!=h.sha256(PLAN):raise ValueError("The last pointer is not complete epoch4")
    pin(pins,folder/pointer["checkpoint"],pointer["sha256"]);pin(pins,folder/"last.pt",pointer["sha256"])
    final=torch.load(folder/pointer["checkpoint"],map_location="cpu",weights_only=True)
    bound_payload(final,config,mode,plan,parent)
    history=final["history"];candidates=c.read_json(folder/"candidates.json")
    expected=["epoch_00_baseline"]+[f"epoch_{epoch:02d}_{source}" for epoch in range(1,5) for source in ("raw","ema")]
    if len(history)!=4 or history!=c.read_json(folder/"history.json") or candidates!=final["candidates"] or candidates!=summary["candidates"] or [item["name"] for item in candidates]!=expected:
        raise ValueError("The complete history/fixed candidate set differs")
    success=skips=0
    for epoch,row in enumerate(history,1):
        if row["epoch"]!=epoch or row.get("seen")!=21246:raise ValueError("V7 epoch missed effective training members")
        n,k,z=(row[key] for key in ("successful_optimizer_steps","amp_skipped_steps","zero_weight_groups"))
        if n<=0 or k<0 or z!=0 or n+k!=math.ceil(21246/32):raise ValueError("V7 epoch group counters do not cover664 groups")
        success+=n;skips+=k
        if row["cumulative_optimizer_steps"]!=success or row["ema_updates"]!=success:raise ValueError("EMA includes failed/skipped steps")
    counters={"successful_steps":success,"amp_skipped_steps":skips,"zero_weight_groups":0}
    if (final.get("counters")!=counters or summary.get("counters")!=counters
        or summary.get("successful_optimizer_steps")!=success or summary.get("ema_updates")!=success
        or final["ema_state"]["updates"]!=success or summary.get("amp_skipped_steps")!=skips):
        raise ValueError("V7 final success/AMP/EMA counters differ")
    del final
    for epoch in range(5):
        path=folder/f"epochs/epoch_{epoch:02d}.pt"
        payload=torch.load(path,map_location="cpu",weights_only=True);bound_payload(payload,config,mode,plan,parent)
        if payload.get("epoch")!=epoch or payload["history"]!=history[:epoch] or payload["candidates"]!=candidates[:1+2*epoch]:raise ValueError("An immutable complete epoch disagrees with final history")
        cumulative=sum(row["successful_optimizer_steps"] for row in history[:epoch])
        cumulative_skips=sum(row["amp_skipped_steps"] for row in history[:epoch])
        expected_counters={"successful_steps":cumulative,"amp_skipped_steps":cumulative_skips,"zero_weight_groups":0}
        if payload.get("counters")!=expected_counters or payload["ema_state"]["updates"]!=cumulative or payload["ema_state"]["decay"]!=.995:
            raise ValueError("An immutable epoch has inconsistent success/AMP/EMA counters")
        optimizer_state=payload["optimizer"]["state"]
        if epoch>0 and (not optimizer_state or any(int(state["step"].item() if isinstance(state["step"],torch.Tensor) else state["step"])!=cumulative for state in optimizer_state.values())):
            raise ValueError("Actual AdamW parameter update counts differ from successful group updates")
        pin(pins,path)
        check_state(payload["ema_state"]["shadow"],parent["model_state"])
        if epoch==0:
            if (payload["ema_state"]["updates"]!=0 or payload["optimizer"]["state"]
                or not same_state(payload["model_state"],parent["model_state"],parent["model_state"])
                or not same_state(payload["ema_state"]["shadow"],parent["model_state"],parent["model_state"])):
                raise ValueError("V7 must start the exact parent with fresh optimizer/EMA")
            active=[candidates[0]]
        else:active=candidates[1+2*(epoch-1):1+2*epoch]
        for item in active:
            if (item.get("classes")!=CLASSES or item.get("mode")!=mode or item.get("val_indices")!=plan["val_indices"]
                or item.get("plan_sha256")!=h.sha256(PLAN) or item.get("validation_weighted") is not False):raise ValueError("Candidate identity/validation binding differs")
            if item!=c.read_json(safe_child(folder,item["candidate_json"])):raise ValueError("Candidate JSON differs from committed pointer")
            files={key:safe_child(folder,item[key]) for key in ("model","original","flip")}
            for key,p in files.items():pin(pins,p,item["artifact_sha256"][key])
            pin(pins,safe_child(folder,item["candidate_json"]))
            cp=torch.load(files["model"],map_location="cpu",weights_only=True);bound_payload(cp,config,mode,plan,parent)
            source=item["weight_source"]
            state=parent["model_state"] if epoch==0 else payload["model_state"] if source=="raw" else payload["ema_state"]["shadow"]
            if cp.get("epoch")!=epoch or cp.get("weight_source")!=source or not same_state(cp["model_state"],state,parent["model_state"]):raise ValueError("Candidate model differs from its committed raw/EMA state")
            for key in ("single_original_metrics","single_tta_metrics"):
                if cp.get(key)!=item[key]:raise ValueError("Candidate metrics differ from its model metadata")
            del cp
        del payload
    return candidates,summary,config,history


def candidate_views(mode,item,targets):
    arrays=[np.load(safe_child(TRAIN/mode,item[key]),allow_pickle=False) for key in ("original","flip")]
    for a in arrays:
        if a.dtype!=np.float32 or a.shape!=(2656,6):raise ValueError("HSE validation probability dtype/shape differs")
    single=c.metrics(targets,(arrays[0]+arrays[1])*.5)
    if c.metrics(targets,arrays[0])!=item["single_original_metrics"] or single!=item["single_tta_metrics"]:raise ValueError("Candidate metrics do not reproduce from actual probability files")
    return [c.check_probability(a,2656) for a in arrays],single


def paired_errors(targets,baseline,candidate):
    first=c.check_probability(baseline,len(targets)).argmax(1)==targets
    second=c.check_probability(candidate,len(targets)).argmax(1)==targets
    return {"both_correct":int(np.sum(first&second)),"baseline_only_correct":int(np.sum(first&~second)),
      "candidate_only_correct":int(np.sum(~first&second)),"both_wrong":int(np.sum(~first&~second)),
      "net_correct_change":int(np.sum(second)-np.sum(first))}


def verify_locked():
    plan,_,_=load_plan();recipe=c.read_json(OUT/"selected_recipe.json")
    if (recipe.get("state")!="locked" or recipe.get("classes")!=CLASSES
        or recipe.get("plan_sha256")!=h.sha256(PLAN) or recipe.get("finalizer_sha256")!=h.sha256(__file__)
        or recipe.get("selection_order")!=["incumbent",*MODES] or recipe.get("fusion_weights")!=FUSION_WEIGHTS):
        raise ValueError("The V7 recipe differs from its fixed source/classes/plan/order/weights")
    c.verify_pins(ROOT,recipe["producer_files"])
    h.check_hash(OUT/"selected_val_probabilities.npy",recipe["selected_val_sha256"])
    h.check_hash(OUT/"candidate_metrics.csv",recipe["candidate_metrics_sha256"])
    verify_preserved();return plan,recipe


def select():
    plan,pins,audit=load_plan();OUT.mkdir(exist_ok=True)
    if (OUT/"selected_recipe.json").exists():return verify_locked()[1]
    rows,receipt=h.load_trainval(ROOT)
    if receipt!=plan["input_receipt"]:raise ValueError("Original training/validation input bytes differ")
    if [r["manifest_index"] for r in rows["train"]]!=plan["original_train_indices"]:raise ValueError("Original parent training member order differs")
    rows["train"]=[r for r in rows["train"] if r["manifest_index"] not in {7727,10448}]
    if h.signature(rows["train"])!=plan["filtered_membership_sha256"]:raise ValueError("Filtered training membership differs")
    for split in ("train","val"):
        if [r["manifest_index"] for r in rows[split]]!=plan[split+"_indices"]:raise ValueError("Effective member order differs")
    _,targets,incumbent,baseline,old_views,_,parent_receipt,v4_recipe,pins=baseline_inputs(pins)
    for mapping in (v4_recipe["producer_files"],c.read_json(HSE_EXPORT/"selected_recipe.json")["producer_files"]):
        for p,digest in mapping.items():pin(pins,p,digest)
    parent=torch.load(PARENT,map_location="cpu",weights_only=True)
    if (parent.get("classes")!=CLASSES or parent.get("best_epoch")!=7
        or parent["train_indices"]!=plan["original_train_indices"] or parent["val_indices"]!=plan["val_indices"]
        or parent["input_sha256"]!=plan["input_receipt"]):raise ValueError("Original HSE parent source/member/data binding differs")
    best_p,best,selected=incumbent,baseline,None
    entries=[{"name":"previous_best","mode":"incumbent",**baseline}]
    comparison={};configs={};histories={};paired={};paired_prob={}
    parent_single=c.metrics(targets,(old_views[0]+old_views[1])*.5)
    for mode in MODES:
        candidates,training_summary,config,history=verify_training(mode,plan,pins,parent)
        configs[mode],histories[mode]=config,history
        arm_best_single=arm_best_fused=best_new_single=best_new_fused=None
        arm_entries=[]
        for item in candidates:
            views,single=candidate_views(mode,item,targets)
            if item["epoch"]==0:
                if any(not np.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(views,old_views)):raise ValueError("V7 epoch0 differs from the original HSE parent predictions")
                probability=incumbent.copy()
            else:probability=fixed_replacement(incumbent,old_views,views)
            fused=c.metrics(targets,probability)
            name=f"{mode}/{item['name']}";entry=metric_record(name,mode,item,single,fused)
            entry["paired_vs_incumbent"]=paired_errors(targets,incumbent,probability)
            entries.append(entry);arm_entries.append(entry)
            paired[(mode,item["epoch"],item["weight_source"])]=entry
            paired_prob[(mode,item["epoch"],item["weight_source"])]=probability
            def info(m):return {"name":name,"epoch":item["epoch"],"weight_source":item["weight_source"],"metrics":m}
            if arm_best_single is None or single["correct_count"]>arm_best_single["metrics"]["correct_count"]:arm_best_single=info(single)
            if arm_best_fused is None or fused["correct_count"]>arm_best_fused["metrics"]["correct_count"]:arm_best_fused=info(fused)
            if item["epoch"]>0:
                if best_new_single is None or single["correct_count"]>best_new_single["metrics"]["correct_count"]:best_new_single=info(single)
                if best_new_fused is None or fused["correct_count"]>best_new_fused["metrics"]["correct_count"]:best_new_fused=info(fused)
            if fused["correct_count"]>best["correct_count"]:selected,best,best_p={**item,"mode":mode},fused,probability
        if best_new_single is None or best_new_fused is None:raise ValueError("Each arm must contain eight newly trained candidates")
        comparison[mode]={"best_single":arm_best_single,"best_fixed_fusion":arm_best_fused,
          "best_new_single":best_new_single,"best_new_fused":best_new_fused,
          "new_candidate_definition":"epochs1-4 raw/EMA only; epoch0 excluded","candidates":arm_entries,
          "training_counts":{key:training_summary[key] for key in ("successful_optimizer_steps","amp_skipped_steps","ema_updates")}}
    if configs[MODES[0]]["settings"]!=configs[MODES[1]]["settings"]:raise ValueError("The two arms have different shared hyperparameters")
    if [r["augmentation_epoch"] for r in histories[MODES[0]]]!=[r["augmentation_epoch"] for r in histories[MODES[1]]]:raise ValueError("Paired augmentation epochs differ")
    pairs=[]
    for epoch in range(1,5):
        for source in ("raw","ema"):
            first,second=(paired[(mode,epoch,source)] for mode in MODES)
            pairs.append({"epoch":epoch,"weight_source":source,
              "sam_minus_control_single_correct":second["single_correct_count"]-first["single_correct_count"],
              "sam_minus_control_fused_correct":second["correct_count"]-first["correct_count"],
              "paired_fusion_errors":paired_errors(targets,paired_prob[(MODES[0],epoch,source)],paired_prob[(MODES[1],epoch,source)])})
    comparison["paired_same_epoch_source"]=pairs
    for key,field in (("best_new_fused","sam_minus_control_best_new_fused_correct"),("best_new_single","sam_minus_control_best_new_single_correct")):
        comparison[field]=comparison["sam"][key]["metrics"]["correct_count"]-comparison["adamw_control"][key]["metrics"]["correct_count"]
    write_metrics(entries)
    if selected is None:atomic_copy(BASELINE/"selected_val_probabilities.npy",OUT/"selected_val_probabilities.npy")
    else:h.save_array(OUT/"selected_val_probabilities.npy",best_p)
    recipe={"format":"hse_sam_v7_fixed_fusion","state":"locked","classes":CLASSES,
      "selected":selected,"improved":selected is not None,"baseline_metrics":baseline,"selected_metrics":best,
      "parent_hse_tta_metrics":parent_single,"parent_hse_original_metrics":parent_receipt["best_validation"],
      "selection_order":["incumbent",*MODES],"candidate_count":18,
      "selection_rule":"strict correct-count increase; every tie retains the earlier incumbent/candidate",
      "fusion_weights":FUSION_WEIGHTS,"replacement":"current-.25*oldHSEoriginal-.25*oldHSEflip+.25*newHSEoriginal+.25*newHSEflip",
      "validation_candidates":entries,"arm_comparison":comparison,"paired_selected_vs_incumbent":paired_errors(targets,incumbent,best_p),
      "producer_files":pins,"plan_sha256":h.sha256(PLAN),"parent_checkpoint_sha256":PARENT_SHA,
      "audit_summary_sha256":plan["audit_summary_sha256"],"filtered_membership_sha256":plan["filtered_membership_sha256"],
      "effective_training_rows":21246,"excluded_train_indices":[7727,10448],"actual_training_weights":"uniform1",
      "val_indices":plan["val_indices"],"val_rows_sha256":receipt["val_rows_sha256"],
      "selected_val_sha256":h.sha256(OUT/"selected_val_probabilities.npy"),"candidate_metrics_sha256":h.sha256(OUT/"candidate_metrics.csv"),
      "old_hse_competition_cache":str(HSE_CACHE),"competition_image_content_sha256":IMAGE_SHA,"competition_row_order_sha256":ROW_SHA,
      "finalizer_sha256":h.sha256(__file__),"internal_test_read":False,"external_answers_used":False,
      "preserved_submission_sha256":PRESERVED,"validation_caveat":CAVEAT}
    c.verify_pins(ROOT,pins);h.write_json(OUT/"selected_recipe.json",recipe)
    emit("select_hse_sam_v7",improved=recipe["improved"],selected=(f"{selected['mode']}/{selected['name']}" if selected else "previous_best"),correct_count=best["correct_count"],accuracy=best["accuracy"])
    return recipe


def predict_winner(recipe,rows,batch_size=16):
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no silent CPU competition inference")
    from finalize_hse_specialist import predict_hse_views
    item=recipe["selected"];checkpoint=safe_child(TRAIN/item["mode"],item["model"])
    recipe_sha=h.sha256(OUT/"selected_recipe.json")
    def locked():
        h.check_hash(OUT/"selected_recipe.json",recipe_sha)
        h.check_hash(checkpoint,item["artifact_sha256"]["model"])
    # The V7 full model_state/model_provenance payload is deliberately compatible
    # with the existing strict official HSE model loader and preprocessing.
    return predict_hse_views(ROOT,checkpoint,rows,True,"cuda",batch_size,locked)


def export(batch_size=16):
    select();_,recipe=verify_locked();recipe_sha=h.sha256(OUT/"selected_recipe.json")
    rows=c.unlabeled_rows(ROOT)
    if c.signature(rows)!=ROW_SHA:raise ValueError("Competition template IDs/order differ")
    images=c.image_receipt(ROOT,rows,verify_inventory=False)
    if images["image_content_sha256"]!=IMAGE_SHA:raise ValueError("Competition content differs from the reused incumbent/cache")
    if (OUT/"summary.json").exists():
        summary=c.read_json(OUT/"summary.json")
        if summary.get("state")!="complete" or summary.get("classes")!=CLASSES or summary.get("recipe_sha256")!=recipe_sha:raise ValueError("Completed V7 export uses another recipe")
        for name,digest in summary["file_sha256"].items():h.check_hash(safe_child(OUT,name),digest)
        verify_submission(rows,c.check_probability(np.load(OUT/"probabilities.npy",allow_pickle=False),7065));verify_preserved();return summary
    if recipe["improved"]:
        new=predict_winner(recipe,rows,batch_size)
        for view,value in zip(("original","flip"),new):h.save_array(OUT/f"hse_competition_{view}.npy",value)
        baseline=c.check_probability(np.load(BASELINE/"probabilities.npy",allow_pickle=False),7065)
        old=[c.check_probability(np.load(HSE_CACHE/f"{view}.npy",allow_pickle=False),7065) for view in ("original","flip")]
        probability=fixed_replacement(baseline,old,new);h.save_array(OUT/"probabilities.npy",probability)
        temp=OUT/"submission.csv.tmp"
        with temp.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["ID","Emotion"])
            writer.writerows((row["ID"],int(label)) for row,label in zip(rows,probability.argmax(1)))
        os.replace(temp,OUT/"submission.csv")
    else:
        atomic_copy(BASELINE/"probabilities.npy",OUT/"probabilities.npy");atomic_copy(BASELINE/"submission.csv",OUT/"submission.csv")
        h.check_hash(OUT/"submission.csv",BASELINE_SUB_SHA);h.check_hash(OUT/"probabilities.npy",BASELINE_PROB_SHA)
        probability=c.check_probability(np.load(OUT/"probabilities.npy",allow_pickle=False),7065)
    verify_submission(rows,probability);h.check_hash(OUT/"selected_recipe.json",recipe_sha)
    c.verify_pins(ROOT,recipe["producer_files"]);load_plan();verify_preserved()
    if c.image_receipt(ROOT,rows,verify_inventory=False)["image_content_sha256"]!=IMAGE_SHA:raise ValueError("Competition content changed during inference")
    names=["submission.csv","probabilities.npy","selected_val_probabilities.npy","selected_recipe.json","candidate_metrics.csv"]
    if recipe["improved"]:names += ["hse_competition_original.npy","hse_competition_flip.npy"]
    summary={"format":"hse_sam_v7_export","state":"complete","classes":CLASSES,"submission_rows":7065,
      "improved":recipe["improved"],"selected":(f"{recipe['selected']['mode']}/{recipe['selected']['name']}" if recipe["selected"] else "previous_best"),
      "selected_mode":recipe["selected"]["mode"] if recipe["selected"] else "incumbent",
      "baseline_metrics":recipe["baseline_metrics"],"selected_metrics":recipe["selected_metrics"],
      "parent_hse_tta_metrics":recipe["parent_hse_tta_metrics"],"arm_comparison":recipe["arm_comparison"],
      "paired_selected_vs_incumbent":recipe["paired_selected_vs_incumbent"],
      "recipe_sha256":recipe_sha,"plan_sha256":recipe["plan_sha256"],"audit_summary_sha256":recipe["audit_summary_sha256"],
      "finalizer_sha256":h.sha256(__file__),"official_label_order_verified":True,"ID_template_order_verified":True,"probability_argmax_verified":True,
      "competition_image_content_sha256":IMAGE_SHA,"competition_row_order_sha256":ROW_SHA,
      "new_hse_competition_inference":recipe["improved"],"other_model_inference_repeated":False,
      "effective_training_rows":21246,"excluded_train_indices":[7727,10448],"preserved_submission_sha256":PRESERVED,
      "internal_test_read":False,"external_answers_used":False,"validation_caveat":CAVEAT,
      "file_sha256":{name:h.sha256(OUT/name) for name in names}}
    h.write_json(OUT/"summary.json",summary)
    emit("complete_hse_sam_v7",improved=summary["improved"],selected=summary["selected"],correct_count=summary["selected_metrics"]["correct_count"],submission=str(OUT/"submission.csv"))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("preflight","select","export"),default="export")
    parser.add_argument("--batch-size",type=int,choices=(1,2,4,8,16),default=16)
    args=parser.parse_args()
    if args.project_root.resolve()!=ROOT.resolve():raise ValueError("This finalizer uses its locked project root")
    torch.set_num_threads(4)
    if args.stage=="preflight":print(json.dumps(preflight_baseline(),ensure_ascii=False),flush=True)
    elif args.stage=="select":select()
    else:export(args.batch_size)


if __name__=="__main__":main()
