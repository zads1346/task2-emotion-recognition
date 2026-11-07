"""Select DDAM training mode using only grouped OOF, then audit one final refit.

No development score selects a fold checkpoint, epoch, mode or hyperparameter.
The sole development candidate is the official-restarted six-epoch final EMA.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[_name]="4"
import argparse
import csv
import gc
import math
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image,ImageOps
import train_hse_specialist as h
import validate_cosine_fusion as c
import prepare_ddamfn_groupcv_v9 as p
import train_ddamfn_groupcv_v9 as t
import ddamfn_v8_model as model_api

ROOT=p.ROOT;PLAN=p.PLAN;TRAIN=p.TRAIN;BASELINE=p.BASELINE
OUT=ROOT/"ddamfn_groupcv_v9_results"
SELECTION=TRAIN/"cv_selection.json"
CLASSES=p.CLASSES;MODES=p.MODES
CAVEAT=("Training-only near-duplicate-group OOF compares two prespecified modes using final epoch6 EMA. "
 "These folds are not person-independent because subject IDs are unavailable; official external pretraining overlap is unknown. "
 "The selected mode is restarted from official weights and checked once against the incumbent on the repeatedly reused development split. "
 "That development score is not Kaggle accuracy or an independent final-test estimate. "
 "Existing task-trained ensemble branches never enter the OOF selection.")


def emit(stage,**values):t.emit(stage,**values)


def child(folder,relative):
    rel=Path(relative)
    if rel.is_absolute() or ".." in rel.parts:raise ValueError("Artifact path escapes its experiment")
    result=(folder/rel).resolve()
    if not result.is_relative_to(folder.resolve()):raise ValueError("Resolved artifact escapes its experiment")
    return result


def same_state(first,second):
    return set(first)==set(second) and all(torch.equal(first[key],second[key]) for key in first)


def check_state(state,initial,mode,param_names):
    if not isinstance(state,dict) or set(state)!=set(initial):raise ValueError("Require all native DDAM model state tensors")
    for name,reference in initial.items():
        value=state[name]
        if not isinstance(value,torch.Tensor) or value.dtype!=reference.dtype or value.shape!=reference.shape or not torch.isfinite(value).all().item():
            raise ValueError("Invalid native DDAM tensor: "+name)
        frozen=name not in param_names or (mode=="fc_only" and name not in {"fc.weight","fc.bias"})
        if frozen and not torch.equal(value,reference):raise ValueError("Frozen DDAM backbone/attention/buffer changed: "+name)


def bound_payload(payload,config,initial,param_names):
    if payload.get("format")!=t.FORMAT or payload.get("config")!=config:
        raise ValueError("V9 checkpoint must match its exact phase/fold configuration")
    for key in ("classes","phase","mode","fold","source_pins","parent_checkpoint_sha256","model_source_receipt",
      "model_source_receipt_sha256","original_train_indices","excluded_train_indices","train_indices","val_indices",
      "run_train_indices","run_validation_indices","audit_summary_sha256","selection_sha256","plan_sha256"):
        if payload.get(key)!=config[key]:raise ValueError("Checkpoint phase/source/member binding differs: "+key)
    if payload.get("parent_sha")!=model_api.OFFICIAL_WEIGHT_SHA256:raise ValueError("V9 must start from official weights")
    check_state(payload.get("model_state"),initial,config["mode"],param_names)


def verify_run(plan,phase,mode,fold,pins,initial,param_names,targets):
    folder=TRAIN/"cv"/mode/f"fold_{fold}" if phase=="cv" else TRAIN/"refit"/mode
    config=t.config_for(ROOT,plan,phase,mode,fold)
    if c.read_json(folder/"config.json")!=config:raise ValueError("Saved run config differs from the locked fold/refit contract")
    summary=c.read_json(folder/"summary.json")
    required={"state":"complete","epochs_completed":6,"classes":CLASSES,"phase":phase,"mode":mode,"fold":fold,
      "plan_sha256":h.sha256(PLAN),"source_pins":plan["source_pins"],
      "model_source_receipt_sha256":plan["model_source_receipt_sha256"],"parent_checkpoint_sha256":model_api.OFFICIAL_WEIGHT_SHA256}
    for key,value in required.items():
        if summary.get(key)!=value:raise ValueError("Incomplete or unbound run summary: "+key)
    for key in ("original_train_indices","excluded_train_indices","train_indices","val_indices","run_train_indices","run_validation_indices"):
        if summary.get(key)!=config[key]:raise ValueError("Summary member identity differs: "+key)
    must={"config.json"}
    must|={"last.json","last.pt","history.json","final_ema.pt","original.npy","flip.npy","final_result.json"}
    must|={f"epochs/epoch_{epoch:02d}.pt" for epoch in range(7)}
    if not must.issubset(summary.get("artifact_sha256",{})):raise ValueError("Run summary omits immutable epochs or final EMA artifacts")
    for rel,digest in summary["artifact_sha256"].items():p.pin(pins,child(folder,rel),digest)
    p.pin(pins,folder/"summary.json")
    pointer=c.read_json(folder/"last.json")
    for key,value in {"epoch":6,"checkpoint":"epochs/epoch_06.pt","phase":phase,"mode":mode,"fold":fold,"plan_sha256":h.sha256(PLAN)}.items():
        if pointer.get(key)!=value:raise ValueError("The last pointer must identify committed epoch6: "+key)
    p.pin(pins,folder/"last.pt",pointer["sha256"]);p.pin(pins,folder/"epochs/epoch_06.pt",pointer["sha256"])
    history=c.read_json(folder/"history.json")
    if len(history)!=6 or [r["epoch"] for r in history]!=list(range(1,7)):raise ValueError("Need six fixed training epochs")
    success=skips=0;expected_counters=[]
    for row in history:
        n,k=row["successful_optimizer_steps"],row["amp_skipped_steps"]
        if row["seen"]!=len(config["run_train_indices"]) or n<=0 or k<0 or n+k!=math.ceil(len(config["run_train_indices"])/32):
            raise ValueError("Epoch failed to account for every fixed training group")
        if any(key.startswith(("val_","heldout_","dev_")) for key in row):raise ValueError("No per-epoch heldout metrics are permitted")
        if row.get("zero_weight_groups")!=0 or row.get("augmentation_epoch")!=plan["settings"]["augmentation_epoch_offset"]+row["epoch"]-1:
            raise ValueError("The paired augmentation epoch or uniform training groups differ")
        success+=n;skips+=k
        if row["cumulative_optimizer_steps"]!=success or row["ema_updates"]!=success:raise ValueError("EMA must count successful updates only")
        expected_counters.append({"successful_steps":success,"amp_skipped_steps":skips,"zero_weight_groups":0})
    if (summary.get("successful_optimizer_steps")!=success or summary.get("ema_updates")!=success
        or summary.get("amp_skipped_steps")!=skips or summary.get("counters")!=expected_counters[-1]):
        raise ValueError("Final summary success/AMP/EMA counters differ")
    frozen={name:value for name,value in initial.items() if name not in param_names or (mode=="fc_only" and name not in {"fc.weight","fc.bias"})}
    frozen_sha=t._frozen_digest(frozen)
    for key,value in {"frozen_state_sha256":frozen_sha,"frozen_state_unchanged":True,"batchnorm_buffers_frozen":True,
      "per_epoch_evaluation":False,"validation_weighted":False,"development_validation_evaluated":phase=="refit",
      "development_validation_used_for_cv":False,"internal_test_read":False,"external_answers_used":False}.items():
        if summary.get(key)!=value:raise ValueError("Run training/evaluation policy proof differs: "+key)
    for epoch in range(7):
        payload=torch.load(folder/f"epochs/epoch_{epoch:02d}.pt",map_location="cpu",weights_only=True)
        bound_payload(payload,config,initial,param_names)
        counters=expected_counters[epoch-1] if epoch else {"successful_steps":0,"amp_skipped_steps":0,"zero_weight_groups":0}
        if (payload.get("epoch")!=epoch or payload.get("weight_source")!="raw" or payload.get("history")!=history[:epoch]
            or payload.get("counters")!=counters or payload["ema_state"]["updates"]!=counters["successful_steps"]
            or payload["ema_state"]["decay"]!=.995):raise ValueError("Immutable epoch state/history/counters differ")
        if (payload.get("successful_steps")!=counters["successful_steps"] or payload.get("cumulative_skips")!=counters["amp_skipped_steps"]
            or payload.get("zero_weight_groups")!=0 or payload.get("frozen_state_sha256")!=frozen_sha):
            raise ValueError("Epoch success aliases or frozen source digest differ")
        check_state(payload["ema_state"]["shadow"],initial,mode,param_names)
        optimizer=payload["optimizer"]["state"]
        if epoch==0:
            if optimizer or not same_state(payload["model_state"],initial) or not same_state(payload["ema_state"]["shadow"],initial):
                raise ValueError("Every fold/arm/refit must start exact official state with fresh optimizer and EMA")
        elif not optimizer or any(int(value["step"].item() if isinstance(value["step"],torch.Tensor) else value["step"])!=counters["successful_steps"] for value in optimizer.values()):
            raise ValueError("Actual AdamW per-parameter steps differ from successful updates")
        if mode=="fc_only" and epoch>0 and len(optimizer)!=2:raise ValueError("fc_only optimizer must update only fc.weight and fc.bias")
        if epoch==6:final_shadow=payload["ema_state"]["shadow"]
        del payload
    final=torch.load(folder/"final_ema.pt",map_location="cpu",weights_only=True)
    bound_payload(final,config,initial,param_names)
    if final.get("epoch")!=6 or final.get("weight_source")!="ema" or not same_state(final["model_state"],final_shadow):
        raise ValueError("Only actual final epoch6 EMA may be evaluated")
    if final.get("ema_updates")!=success or final.get("successful_steps")!=success or final.get("frozen_state_sha256")!=frozen_sha:
        raise ValueError("Final EMA counters or frozen state digest differ")
    arrays=[np.load(folder/f"{view}.npy",allow_pickle=False) for view in ("original","flip")]
    for array in arrays:
        if array.shape!=(len(config["run_validation_indices"]),6) or array.dtype!=np.float32:raise ValueError("Final probability row/class/dtype differs")
        c.check_probability(array,len(targets))
    original=c.metrics(targets,arrays[0]);tta=c.metrics(targets,(arrays[0]+arrays[1])*.5)
    for key,metric in (("single_original_metrics",original),("single_tta_metrics",tta)):
        if summary.get(key)!=metric or final.get(key)!=metric:raise ValueError("Final summary/model metrics do not reproduce")
    result=c.read_json(folder/"final_result.json")
    for key in ("phase","mode","fold","classes","plan_sha256","source_pins","run_train_indices","run_validation_indices","selection_sha256"):
        if result.get(key)!=config[key]:raise ValueError("Final result phase/source/member binding differs: "+key)
    if result.get("state")!="complete" or result.get("epoch")!=6 or result.get("weight_source")!="ema" or result.get("validation_weighted") is not False or result.get("development_validation_evaluated")!=(phase=="refit"):
        raise ValueError("Final result does not describe the one allowed final EMA evaluation")
    for key,name in {"model":"final_ema.pt","original":"original.npy","flip":"flip.npy"}.items():
        if result.get(key)!=name:raise ValueError("Final result filenames differ")
        p.pin(pins,folder/name,result["artifact_sha256"][key])
    for key,metric in (("single_original_metrics",original),("single_tta_metrics",tta)):
        if result.get(key)!=metric:raise ValueError("Final result metrics do not reproduce")
    del final,final_shadow
    return arrays,tta,{"phase":phase,"mode":mode,"fold":fold,"metrics":tta,"summary_sha256":h.sha256(folder/"summary.json"),
      "final_model_sha256":h.sha256(folder/"final_ema.pt"),"successful_optimizer_steps":success,"amp_skipped_steps":skips,"ema_updates":success}


def initial_reference(plan):
    model,receipt=model_api.load_initial_model(ROOT,"cpu")
    if receipt!=plan["model_source_receipt"]:raise ValueError("Official source receipt changed")
    state=h.cpu_tree(model.state_dict());names=set(dict(model.named_parameters()));del model
    return state,names


def cv_decision(pooled,folds):
    wins=sum(folds["fc_only"][str(fold)]["correct_count"]>folds["full"][str(fold)]["correct_count"] for fold in range(3))
    return ("fc_only" if pooled["fc_only"]["correct_count"]>pooled["full"]["correct_count"] and wins>=2 else "full"),wins


def verify_cv_selection(plan=None):
    plan=plan or p.verify_plan(verify_baseline=False)
    selection=c.read_json(SELECTION)
    if (selection.get("state")!="locked" or selection.get("classes")!=CLASSES or selection.get("plan_sha256")!=h.sha256(PLAN)
        or selection.get("source_pins")!=plan["source_pins"] or selection.get("finalizer_sha256")!=h.sha256(__file__)
        or selection.get("selection_rule")!=p.CV_RULE or selection.get("development_validation_used") is not False):
        raise ValueError("CV selection binding or training-only decision differs")
    c.verify_pins(ROOT,selection["producer_files"])
    records=t.training_records(ROOT,plan,verify_images=False)
    targets=np.asarray([r["target"] for r in records]);indices=[r["manifest_index"] for r in records]
    if selection.get("oof_manifest_indices")!=indices:raise ValueError("OOF rows must cover the effective training set in its original order")
    pooled={};fold_scores={mode:{} for mode in MODES}
    for mode in MODES:
        path=TRAIN/f"oof_{mode}.npy";h.check_hash(path,selection["file_sha256"][path.name])
        array=np.load(path,allow_pickle=False);c.check_probability(array,21246)
        if array.dtype!=np.float32:raise ValueError("OOF persistence must retain its exact float32 view average")
        pooled[mode]=c.metrics(targets,array)
        for fold in range(3):
            members=set(plan["folds"][str(fold)]["validation_indices"]);mask=np.asarray([index in members for index in indices])
            fold_scores[mode][str(fold)]=c.metrics(targets[mask],array[mask])
    mode,wins=cv_decision(pooled,fold_scores)
    if pooled!=selection.get("pooled_oof_metrics") or fold_scores!=selection.get("fold_metrics") or mode!=selection.get("selected_mode") or wins!=selection.get("fc_only_fold_wins"):
        raise ValueError("Locked OOF decision does not reproduce exactly")
    return selection


def select_cv():
    plan=p.verify_plan(verify_baseline=False)
    if SELECTION.exists():return verify_cv_selection(plan)
    records=t.training_records(ROOT,plan,verify_images=True)
    targets=np.asarray([r["target"] for r in records]);indices=[r["manifest_index"] for r in records];positions={index:pos for pos,index in enumerate(indices)}
    initial,param_names=initial_reference(plan)
    pins=dict(plan["source_pins"]);p.pin(pins,PLAN);p.pin(pins,Path(__file__))
    pooled={};fold_scores={mode:{} for mode in MODES};runs=[]
    TRAIN.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        probability=np.zeros((21246,6),dtype=np.float32);coverage=np.zeros(21246,dtype=np.int64)
        for fold in range(3):
            heldout=plan["folds"][str(fold)]["validation_indices"];loc=np.array([positions[index] for index in heldout])
            views,_,run=verify_run(plan,"cv",mode,fold,pins,initial,param_names,targets[loc])
            # Preserve the evaluator's exact float32 mean; metrics normalize once on both reads.
            values=(views[0]+views[1])*.5;c.check_probability(values,len(loc))
            probability[loc]=values;coverage[loc]+=1
            fold_scores[mode][str(fold)]=c.metrics(targets[loc],values);runs.append(run)
        if not np.all(coverage==1):raise ValueError("Each training image must have exactly one genuine held-out prediction")
        h.save_array(TRAIN/f"oof_{mode}.npy",probability)
        pooled[mode]=c.metrics(targets,probability)
    selected,wins=cv_decision(pooled,fold_scores)
    selection={"format":"ddamfn_groupcv_v9_selection","state":"locked","classes":CLASSES,"selected_mode":selected,
      "selection_rule":p.CV_RULE,"pooled_oof_metrics":pooled,"fold_metrics":fold_scores,"fc_only_fold_wins":wins,"runs":runs,
      "oof_manifest_indices":indices,"plan_sha256":h.sha256(PLAN),"source_pins":plan["source_pins"],"producer_files":pins,
      "parent_checkpoint_sha256":model_api.OFFICIAL_WEIGHT_SHA256,"finalizer_sha256":h.sha256(__file__),
      "file_sha256":{f"oof_{mode}.npy":h.sha256(TRAIN/f"oof_{mode}.npy") for mode in MODES},
      "development_validation_used":False,"existing_task_weights_used":False,"internal_test_read":False,"external_answers_used":False,
      "validation_caveat":CAVEAT}
    c.verify_pins(ROOT,pins);h.write_json(SELECTION,selection);verify_cv_selection(plan)
    emit("v9_cv_selected",selected_mode=selected,fc_only_fold_wins=wins,pooled_correct={mode:pooled[mode]["correct_count"] for mode in MODES})
    return selection


def replacement(incumbent,old,new):
    base=c.check_probability(incumbent);old=[c.check_probability(a,len(base)) for a in old];new=[c.check_probability(a,len(base)) for a in new]
    raw=base-.1*old[0]-.1*old[1]+.1*new[0]+.1*new[1]
    if raw.min() < -1e-10:raise ValueError("Fixed20% DDAM replacement produced negative probabilities")
    return c.check_probability(np.maximum(raw,0),len(base))


def verify_recipe():
    plan=p.verify_plan();selection=verify_cv_selection(plan);recipe=c.read_json(OUT/"selected_recipe.json")
    if (recipe.get("state")!="locked" or recipe.get("classes")!=CLASSES or recipe.get("plan_sha256")!=h.sha256(PLAN)
        or recipe.get("cv_selection_sha256")!=h.sha256(SELECTION) or recipe.get("selected_mode")!=selection["selected_mode"]
        or recipe.get("finalizer_sha256")!=h.sha256(__file__) or recipe.get("replacement_fraction")!=.2
        or recipe.get("candidate_count")!=1 or recipe.get("selection_rule")!="one final EMA candidate; strict correct-count improvement; incumbent wins ties"):
        raise ValueError("V9 final refit recipe differs from its single fixed comparison")
    c.verify_pins(ROOT,recipe["producer_files"])
    for name,digest in recipe["selection_file_sha256"].items():h.check_hash(child(OUT,name),digest)
    return plan,recipe


def select_development():
    plan=p.verify_plan();selection=verify_cv_selection(plan)
    if (OUT/"selected_recipe.json").exists():return verify_recipe()[1]
    mode=selection["selected_mode"];records=t.development_records(ROOT,plan);targets=np.asarray([r["target"] for r in records])
    t.training_records(ROOT,plan,verify_images=True)
    initial,names=initial_reference(plan);pins={**plan["source_pins"],**plan["baseline_files"],**selection["producer_files"]}
    p.pin(pins,PLAN);p.pin(pins,SELECTION)
    for rel,digest in selection["file_sha256"].items():p.pin(pins,TRAIN/rel,digest)
    views,single,run=verify_run(plan,"refit",mode,None,pins,initial,names,targets)
    current=c.check_probability(np.load(BASELINE/"selected_val_probabilities.npy",allow_pickle=False),2656)
    baseline=c.metrics(targets,current)
    if baseline!=plan["baseline_metrics"] or baseline["correct_count"]!=1924:raise ValueError("V8 baseline does not reproduce1924/2656")
    old=[np.load(plan["old_ddamfn_val_views"][view],allow_pickle=False) for view in ("original","flip")]
    probability=replacement(current,old,views);candidate=c.metrics(targets,probability)
    improved=candidate["correct_count"]>baseline["correct_count"]
    OUT.mkdir(parents=True,exist_ok=True)
    h.save_array(OUT/"selected_val_probabilities.npy",probability if improved else np.load(BASELINE/"selected_val_probabilities.npy",allow_pickle=False))
    fields=["name","mode","accuracy","macro_f1","log_loss","correct_count"]
    with (OUT/"candidate_metrics.csv.tmp").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
        writer.writerows([{"name":"previous_best","mode":"incumbent",**baseline},{"name":"final_epoch06_ema","mode":mode,**candidate}])
    os.replace(OUT/"candidate_metrics.csv.tmp",OUT/"candidate_metrics.csv")
    recipe={"format":"ddamfn_groupcv_v9_fixed_fusion","state":"locked","classes":CLASSES,
      "improved":improved,"selected":"final_epoch06_ema" if improved else "previous_best","selected_mode":mode,
      "candidate_count":1,"selection_rule":"one final EMA candidate; strict correct-count improvement; incumbent wins ties",
      "baseline_metrics":baseline,"candidate_metrics":candidate,"selected_metrics":candidate if improved else baseline,"refit_single_metrics":single,
      "cv_comparison":{"pooled_oof_metrics":selection["pooled_oof_metrics"],"fold_metrics":selection["fold_metrics"],"fc_only_fold_wins":selection["fc_only_fold_wins"]},
      "refit_receipt":run,"replacement_fraction":.2,"replacement":plan["replacement"],"fusion_weights":plan["fusion_weights"],
      "plan_sha256":h.sha256(PLAN),"cv_selection_sha256":h.sha256(SELECTION),"parent_checkpoint_sha256":model_api.OFFICIAL_WEIGHT_SHA256,
      "finalizer_sha256":h.sha256(__file__),"producer_files":pins,
      "selection_file_sha256":{name:h.sha256(OUT/name) for name in ("selected_val_probabilities.npy","candidate_metrics.csv")},
      "competition_image_content_sha256":p.IMAGE_SHA,"competition_row_order_sha256":p.ROW_SHA,
      "development_validation_candidates":1,"internal_test_read":False,"external_answers_used":False,"validation_caveat":CAVEAT}
    c.verify_pins(ROOT,pins);h.write_json(OUT/"selected_recipe.json",recipe)
    emit("v9_refit_selected",mode=mode,improved=improved,candidate_correct=candidate["correct_count"],incumbent_correct=1924)
    return recipe


def predict_winner(recipe,rows,batch_size=32):
    if not torch.cuda.is_available():raise RuntimeError("No silent CPU competition prediction")
    path=TRAIN/"refit"/recipe["selected_mode"]/"final_ema.pt"
    model,_=t.load_checkpoint(ROOT,path,"cuda");model.eval();values=[[],[]];transform=t.previous.evaluation_transform()
    recipe_sha=h.sha256(OUT/"selected_recipe.json");last=started=time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0,len(rows),batch_size):
                h.check_hash(OUT/"selected_recipe.json",recipe_sha);h.check_hash(path,recipe["refit_receipt"]["final_model_sha256"])
                tensors=[]
                for row in rows[start:start+batch_size]:
                    with Image.open(row["path"]) as image:tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                images=torch.stack(tensors).cuda()
                with h.autocast_for("cuda"):a,b=model(images),model(images.flip(-1))
                for i,logits in enumerate((a,b)):values[i].append(logits.float().softmax(1).cpu().numpy())
                if time.monotonic()-last>=30 or start+batch_size>=len(rows):
                    emit("v9_competition_prediction",seen=min(start+batch_size,len(rows)),total=len(rows),elapsed_seconds=time.monotonic()-started);last=time.monotonic()
        return [c.check_probability(np.concatenate(view),7065) for view in values]
    finally:
        del model;gc.collect();torch.cuda.empty_cache()


def verify_submission(rows,probability):
    c.submission_check(OUT/"submission.csv",rows,probability)
    with (OUT/"submission.csv").open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=["ID","Emotion"] or len(list(reader))!=7065:raise ValueError("Final CSV requires7065 rows and ID,Emotion only")


def copy_atomic(source,destination):
    import shutil
    temp=Path(str(destination)+".tmp");shutil.copyfile(source,temp);os.replace(temp,destination)


def export():
    select_development();plan,recipe=verify_recipe();recipe_sha=h.sha256(OUT/"selected_recipe.json")
    rows=c.unlabeled_rows(ROOT)
    if c.signature(rows)!=p.ROW_SHA or c.image_receipt(ROOT,rows,verify_inventory=False)["image_content_sha256"]!=p.IMAGE_SHA:
        raise ValueError("Competition IDs/order/image bytes differ from reused V8 predictions")
    if (OUT/"summary.json").exists():
        summary=c.read_json(OUT/"summary.json")
        if summary.get("state")!="complete" or summary.get("recipe_sha256")!=recipe_sha or summary.get("classes")!=CLASSES:raise ValueError("Completed export belongs to another recipe")
        for rel,digest in summary["file_sha256"].items():h.check_hash(child(OUT,rel),digest)
        verify_submission(rows,np.load(OUT/"probabilities.npy",allow_pickle=False));return summary
    if recipe["improved"]:
        new=predict_winner(recipe,rows)
        for view,array in zip(("original","flip"),new):h.save_array(OUT/f"ddamfn_competition_{view}.npy",array)
        old=[np.load(BASELINE/f"ddamfn_competition_{view}.npy",allow_pickle=False) for view in ("original","flip")]
        probability=replacement(np.load(BASELINE/"probabilities.npy",allow_pickle=False),old,new);h.save_array(OUT/"probabilities.npy",probability)
        with (OUT/"submission.csv.tmp").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["ID","Emotion"]);writer.writerows((row["ID"],int(label)) for row,label in zip(rows,probability.argmax(1)))
        os.replace(OUT/"submission.csv.tmp",OUT/"submission.csv")
    else:
        copy_atomic(BASELINE/"probabilities.npy",OUT/"probabilities.npy");copy_atomic(BASELINE/"submission.csv",OUT/"submission.csv")
        h.check_hash(OUT/"submission.csv",plan["baseline_submission_sha256"]);h.check_hash(OUT/"probabilities.npy",plan["baseline_probability_sha256"])
        probability=np.load(OUT/"probabilities.npy",allow_pickle=False)
    verify_submission(rows,probability);h.check_hash(OUT/"selected_recipe.json",recipe_sha);c.verify_pins(ROOT,recipe["producer_files"])
    if c.image_receipt(ROOT,rows,verify_inventory=False)["image_content_sha256"]!=p.IMAGE_SHA:raise ValueError("Competition bytes changed during export")
    names=["submission.csv","probabilities.npy","selected_val_probabilities.npy","selected_recipe.json","candidate_metrics.csv"]
    if recipe["improved"]:names += ["ddamfn_competition_original.npy","ddamfn_competition_flip.npy"]
    summary={"format":"ddamfn_groupcv_v9_export","state":"complete","classes":CLASSES,"submission_rows":7065,
      "improved":recipe["improved"],"selected":recipe["selected"],"selected_mode":recipe["selected_mode"],
      "baseline_metrics":recipe["baseline_metrics"],"candidate_metrics":recipe["candidate_metrics"],"selected_metrics":recipe["selected_metrics"],
      "refit_single_metrics":recipe["refit_single_metrics"],"cv_comparison":recipe["cv_comparison"],
      "recipe_sha256":recipe_sha,"plan_sha256":h.sha256(PLAN),"cv_selection_sha256":h.sha256(SELECTION),"finalizer_sha256":h.sha256(__file__),
      "official_label_order_verified":True,"ID_template_order_verified":True,"probability_argmax_verified":True,
      "competition_image_content_sha256":p.IMAGE_SHA,"competition_row_order_sha256":p.ROW_SHA,
      "new_ddamfn_competition_inference":recipe["improved"],"other_model_inference_repeated":False,
      "internal_test_read":False,"external_answers_used":False,"validation_caveat":CAVEAT,"file_sha256":{name:h.sha256(OUT/name) for name in names}}
    h.write_json(OUT/"summary.json",summary);emit("v9_complete",improved=summary["improved"],correct_count=summary["selected_metrics"]["correct_count"],submission=str(OUT/"submission.csv"))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("select_cv","export","verify"),default="export");args=parser.parse_args()
    if args.project_root.resolve()!=ROOT:raise ValueError("Fixed project root required")
    torch.set_num_threads(4)
    if args.stage=="select_cv":select_cv()
    elif args.stage=="verify":verify_recipe();print("V9 recipe verified",flush=True)
    else:export()


if __name__=="__main__":main()
