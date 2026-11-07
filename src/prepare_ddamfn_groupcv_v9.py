"""Lock a training-only three-fold DDAMFN full-vs-frozen-backbone experiment.

The development split is bound for one final refit check, never for CV selection.
Every fold and the final refit starts from the official AffectNet checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import train_hse_specialist as h
import validate_cosine_fusion as c
import ddamfn_v8_model as model_api

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/"logs/ddamfn_groupcv_v9_plan.json"
TRAIN=ROOT/"experiments/ddamfn_groupcv_v9"
BASELINE=ROOT/"ddamfn_v8_results"
GROUPS=ROOT/"experiments/multilayer_groups"
AUDIT=ROOT/"experiments/lora_flip_consistency_v6/audit/summary.json"
WEIGHT=ROOT/"models/ddamfn_affectnet7_v8/affecnet7_epoch19_acc0.671.pth"
CLASSES=["Angry","Fear","Happy","Sad","Surprise","Neutral"]
MODES=["full","fc_only"]
IMAGE_SHA="02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
ROW_SHA="8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
GROUP_CSV_SHA="8f5a2cc7698feb78d3e088fb3ed5f8d300c1395e7bbebdbd419fb3a5ca288831"
GROUP_SOURCE_SHA="c5679f1eb9a6f01c91eeb1dd605cd35aef242806dc7b16ef9ac2c29edfaa8b60"
CV_RULE="fc_only only if pooled OOF correct_count strictly exceeds full and fc_only strictly wins at least2 of3 folds; otherwise full"
SOURCE_FILES=["src/prepare_ddamfn_groupcv_v9.py","src/train_ddamfn_groupcv_v9.py",
 "src/finalize_ddamfn_groupcv_v9.py","src/run_ddamfn_groupcv_v9.py","src/ddamfn_groupcv_v9_worker.py",
 "start_ddamfn_groupcv_v9.ps1","run_ddamfn_groupcv_v9_worker.ps1",
 "src/ddamfn_v8_model.py","src/train_ddamfn_v8.py","src/train_hse_sam_v7.py",
 "src/train_hse_specialist.py","src/validate_cosine_fusion.py","src/group_training_images.py","src/run_hse_sam_v7.py"]


def pin(pins,path,expected=None):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError("V9 pins must remain within the project")
    digest=expected or h.sha256(path);h.check_hash(path,digest)
    if str(path) in pins and pins[str(path)]!=digest:raise ValueError("Conflicting V9 producer pin")
    pins[str(path)]=digest


def read_training_rows(plan=None):
    """No non-training path or label is inspected after the split filter."""
    rows=[]
    with (ROOT/"features/manifest.csv").open(encoding="utf-8-sig",newline="") as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            if row.get("split")!="train":continue
            if row.get("label") not in CLASSES or not Path(row["path"]).is_absolute():
                raise ValueError("Invalid official training row")
            rows.append({"manifest_index":index,"path":row["path"],"label":row["label"],"target":CLASSES.index(row["label"])})
    if len(rows)!=21248:raise ValueError("Original training membership must contain21248 rows")
    if plan is not None:
        h.check_hash(ROOT/"features/manifest.csv",plan["input_receipt"]["manifest_sha256"])
        if [r["manifest_index"] for r in rows]!=plan["original_train_indices"] or h.signature(rows)!=plan["input_receipt"]["train_rows_sha256"]:
            raise ValueError("Original training membership or row values changed")
    return rows


def verify_groups(original_rows,receipt):
    summary=c.read_json(GROUPS/"groups_summary.json")
    h.check_hash(GROUPS/"train_groups.csv",GROUP_CSV_SHA)
    h.check_hash(ROOT/"src/group_training_images.py",GROUP_SOURCE_SHA)
    if (summary.get("complete") is not True or summary.get("train_rows")!=21248
        or summary.get("train_groups_sha256")!=GROUP_CSV_SHA or summary.get("grouping_script_sha256")!=GROUP_SOURCE_SHA
        or summary.get("source_manifest_sha256")!=receipt["manifest_sha256"]
        or summary.get("train_image_content_sha256")!=receipt["train_image_content_sha256"]
        or summary.get("train_manifest_sha256")!=c.signature(original_rows)
        or summary.get("parameters",{}).get("classes")!=CLASSES
        or summary.get("parameters",{}).get("folds")!=3
        or summary.get("no_group_cross_fold") is not True):
        raise ValueError("The existing training-only near-duplicate grouping receipt differs")
    with (GROUPS/"train_groups.csv").open(encoding="utf-8-sig",newline="") as stream:groups=list(csv.DictReader(stream))
    if len(groups)!=21248 or len({int(r["manifest_index"]) for r in groups})!=21248:
        raise ValueError("Grouping must cover the original training set exactly once")
    lookup={r["manifest_index"]:r for r in original_rows};group_folds={};assignment={}
    for row in groups:
        index=int(row["manifest_index"]);base=lookup.get(index)
        if base is None or row["path"]!=base["path"] or row["label"]!=base["label"] or int(row["target"])!=base["target"]:
            raise ValueError("A grouping row does not match the original training manifest")
        fold=int(row["fold"]);group=int(row["group"])
        if fold not in (0,1,2) or (group in group_folds and group_folds[group]!=fold):raise ValueError("Near-duplicate group crosses folds")
        group_folds[group]=fold;assignment[index]=(group,fold)
    effective=[r for r in original_rows if r["manifest_index"] not in {7727,10448}]
    folds={}
    for fold in range(3):
        train=[r for r in effective if assignment[r["manifest_index"]][1]!=fold]
        val=[r for r in effective if assignment[r["manifest_index"]][1]==fold]
        train_groups=sorted({assignment[r["manifest_index"]][0] for r in train})
        val_groups=sorted({assignment[r["manifest_index"]][0] for r in val})
        if len(train)!=14164 or len(val)!=7082 or set(train_groups)&set(val_groups):raise ValueError("Fixed three-fold members differ")
        if {r["target"] for r in train}!={0,1,2,3,4,5} or {r["target"] for r in val}!={0,1,2,3,4,5}:raise ValueError("A fold misses an official class")
        folds[str(fold)]={"fold":fold,"train_indices":[r["manifest_index"] for r in train],
          "validation_indices":[r["manifest_index"] for r in val],"train_groups":train_groups,"validation_groups":val_groups,
          "train_rows_sha256":h.signature(train),"validation_rows_sha256":h.signature(val)}
    return summary,folds


def verify_plan(root=ROOT,verify_baseline=True):
    if Path(root).resolve()!=ROOT:raise ValueError("Fixed project root required")
    import train_ddamfn_groupcv_v9 as trainer
    plan=c.read_json(PLAN)
    if (plan.get("format")!="ddamfn_groupcv_v9_locked" or plan.get("classes")!=CLASSES
        or plan.get("modes")!=MODES or plan.get("settings")!=trainer.SETTINGS
        or plan.get("cv_selection_rule")!=CV_RULE or plan.get("baseline_correct_count")!=1924
        or plan.get("parent_checkpoint_sha256")!=model_api.OFFICIAL_WEIGHT_SHA256
        or plan.get("internal_test_read") is not False or plan.get("external_answers_used") is not False):
        raise ValueError("V9 fixed protocol differs")
    c.verify_pins(ROOT,plan["source_pins"])
    if verify_baseline:c.verify_pins(ROOT,plan["baseline_files"])
    if model_api.source_receipt(ROOT)!=plan["model_source_receipt"]:raise ValueError("Official DDAM model provenance differs")
    original=read_training_rows(plan);grouping,folds=verify_groups(original,plan["input_receipt"])
    effective=[r for r in original if r["manifest_index"] not in {7727,10448}]
    if (folds!=plan["folds"] or grouping!=plan["grouping_receipt"] or plan["excluded_train_indices"]!=[7727,10448]
        or plan["train_indices"]!=[r["manifest_index"] for r in effective]
        or plan["filtered_membership_sha256"]!=h.signature(effective)):
        raise ValueError("V9 effective/grouped membership differs")
    return plan


def prepare(root=ROOT):
    if Path(root).resolve()!=ROOT:raise ValueError("Fixed project root required")
    if PLAN.exists():return verify_plan(root)
    import train_ddamfn_groupcv_v9 as trainer
    if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError("V9 requires at least4GiB for immutable folds/refit artifacts")
    summary=c.read_json(BASELINE/"summary.json");recipe=c.read_json(BASELINE/"selected_recipe.json")
    if (summary.get("state")!="complete" or summary.get("classes")!=CLASSES or summary.get("submission_rows")!=7065
        or recipe.get("state")!="locked" or recipe.get("selected",{}).get("name")!="epoch_00_baseline"
        or recipe.get("replacement_fraction")!=.2
        or recipe.get("replacement")!="incumbent-.1*oldCNNoriginal-.1*oldCNNflip+.1*newDDAMoriginal+.1*newDDAMflip"
        or Path(recipe.get("baseline_directory","")).resolve()!=(ROOT/"hse_sam_v7_results").resolve()
        or summary.get("competition_image_content_sha256")!=IMAGE_SHA or summary.get("competition_row_order_sha256")!=ROW_SHA):
        raise ValueError("The completed V8 official-DDAM20% incumbent is required")
    baseline_files={}
    for rel,digest in summary["file_sha256"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:raise ValueError("Baseline artifact escapes result folder")
        pin(baseline_files,BASELINE/rel,digest)
    pin(baseline_files,BASELINE/"summary.json")
    pin(baseline_files,BASELINE/"selected_recipe.json",summary["recipe_sha256"])
    for path,digest in recipe["producer_files"].items():pin(baseline_files,path,digest)
    v7_recipe_path=ROOT/"hse_sam_v7_results/selected_recipe.json"
    v7_summary_path=ROOT/"hse_sam_v7_results/summary.json"
    if str(v7_recipe_path) not in baseline_files or str(v7_summary_path) not in baseline_files:
        raise ValueError("V8 must bind its V7 source fusion recipe and export")
    v7_recipe=c.read_json(v7_recipe_path);v7_summary=c.read_json(v7_summary_path)
    if (v7_recipe.get("fusion_weights")!={"hse":.5,"lora":.2,"cnn":.2,"mlp":.1}
        or v7_recipe.get("state")!="locked" or v7_summary.get("state")!="complete"
        or v7_summary.get("recipe_sha256")!=h.sha256(v7_recipe_path)):
        raise ValueError("The locked V7-to-V8 branch-weight chain differs")
    rows,receipt=h.load_trainval(ROOT)
    audit=c.read_json(AUDIT);effective=[r for r in rows["train"] if r["manifest_index"] not in {7727,10448}]
    members={"original_train_indices":[r["manifest_index"] for r in rows["train"]],"excluded_train_indices":[7727,10448],
      "train_indices":[r["manifest_index"] for r in effective],"val_indices":[r["manifest_index"] for r in rows["val"]]}
    if audit.get("state")!="complete" or audit.get("classes")!=CLASSES or any(audit.get(k)!=v for k,v in members.items()) or audit["filtered_membership_sha256"]!=h.signature(effective):
        raise ValueError("Shared training-only image audit differs")
    grouping,folds=verify_groups(rows["train"],receipt)
    targets=np.array([r["target"] for r in rows["val"]])
    probability=c.check_probability(np.load(BASELINE/"selected_val_probabilities.npy",allow_pickle=False),2656)
    score=c.metrics(targets,probability)
    if score!=summary["selected_metrics"] or score["correct_count"]!=1924:raise ValueError("V8 incumbent does not reproduce1924/2656")
    selected=recipe["selected"];old_views={}
    for view in ("original","flip"):
        path=ROOT/"experiments/ddamfn_v8"/selected[view]
        pin(baseline_files,path,selected["artifact_sha256"][view]);old_views[view]=str(path)
        pin(baseline_files,BASELINE/f"ddamfn_competition_{view}.npy",summary["file_sha256"][f"ddamfn_competition_{view}.npy"])
    pins={}
    for relative in SOURCE_FILES:pin(pins,ROOT/relative)
    for path in (AUDIT,GROUPS/"groups_summary.json",GROUPS/"train_groups.csv",WEIGHT,
                 ROOT/"features/manifest.csv",ROOT/"vendor/ddamfn_v8/source_receipt.json",ROOT/"models/ddamfn_affectnet7_v8/source_receipt.json"):
        pin(pins,path)
    for path,digest in audit["source_pins"].items():pin(pins,path,digest)
    for relative,digest in audit["file_sha256"].items():pin(pins,AUDIT.parent/relative,digest)
    for relative,digest in model_api.SOURCE_SHA256.items():pin(pins,ROOT/"vendor/ddamfn_v8"/relative,digest)
    provenance=model_api.source_receipt(ROOT)
    plan={"format":"ddamfn_groupcv_v9_locked","classes":CLASSES,"settings":trainer.SETTINGS,"modes":MODES,
      "source_pins":pins,"input_receipt":receipt,**members,"filtered_membership_sha256":h.signature(effective),
      "audit_summary_path":str(AUDIT),"audit_summary_sha256":h.sha256(AUDIT),
      "parent_checkpoint":str(WEIGHT),"parent_checkpoint_sha256":model_api.OFFICIAL_WEIGHT_SHA256,
      "model_source_receipt":provenance,"model_source_receipt_sha256":model_api.receipt_sha256(provenance),
      "baseline_directory":str(BASELINE),"baseline_files":baseline_files,"baseline_correct_count":1924,"baseline_metrics":score,
      "baseline_submission_sha256":h.sha256(BASELINE/"submission.csv"),"baseline_probability_sha256":h.sha256(BASELINE/"probabilities.npy"),
      "baseline_val_probability_sha256":h.sha256(BASELINE/"selected_val_probabilities.npy"),"old_ddamfn_val_views":old_views,
      "grouping_receipt":grouping,"grouping_sha256":h.sha256(GROUPS/"groups_summary.json"),"train_groups_sha256":GROUP_CSV_SHA,"folds":folds,
      "cv_selection_path":str(TRAIN/"cv_selection.json"),"cv_selection_rule":CV_RULE,"cv_validation":"training-only near-duplicate-group OOF; not person-independent",
      "fold_initialization":"official AffectNet weights independently for every fold and arm; no task checkpoint",
      "checkpoint_selection":"six fixed epochs; final EMA only; no intermediate held-out candidates",
      "refit_initialization":"official AffectNet weights, never a fold/task checkpoint","development_validation_candidates":1,
      "fusion_weights":{"hse":.5,"lora":.2,"cnn":.2,"mlp":.1},"replacement":"current-.1*oldDDAMoriginal-.1*oldDDAMflip+.1*newDDAMoriginal+.1*newDDAMflip",
      "competition_image_content_sha256":IMAGE_SHA,"competition_row_order_sha256":ROW_SHA,
      "external_pretrain_overlap_audited":False,"internal_test_read":False,"external_answers_used":False,"GPU_used_by_plan_generation":False}
    TRAIN.mkdir(parents=True,exist_ok=True);c.verify_pins(ROOT,pins);c.verify_pins(ROOT,baseline_files)
    h.write_json(PLAN,plan)
    print(json.dumps({"stage":"v9_plan_locked","folds":3,"modes":MODES,"epochs_per_run":6,"effective_train_rows":21246,"baseline_correct_count":1924}),flush=True)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--stage",choices=("prepare","verify"),default="prepare")
    args=parser.parse_args()
    if args.stage=="prepare":prepare(args.project_root)
    else:verify_plan(args.project_root);print(json.dumps({"state":"verified"}),flush=True)


if __name__=="__main__":main()
