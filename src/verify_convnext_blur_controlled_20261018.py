"""Read-only verification using saved arrays, without loading training code or a GPU."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/convnext_blur_controlled_20261018"
RUN = ROOT / "experiments/convnext_blur_controlled_20261018"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    receipt = read(OUT/"acceptance.json")
    for path, digest in receipt["artifacts"].items():
        assert sha(path) == digest, path
    plan = read(ROOT/"research/convnext_blur_controlled_20261018_plan.json")
    for path, digest in plan["source_and_input_pins"].items():
        assert sha(path) == digest, path
    with (ROOT/"data/expanded_validation_v10/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    val = [r for r in all_rows if r["split"] == "val"]
    train = [r for r in all_rows if r["split"] == "train"]
    assert len(train) == 19120 and len(val) == 4782
    assert not {r["group"] for r in train} & {r["group"] for r in val}
    recipe = read(ROOT/"hse_expanded_validation_v10_results/selected_recipe.json")
    assert recipe["val_indices"] == [int(r["manifest_index"]) for r in val]
    assert recipe["train_indices"] == [int(r["manifest_index"]) for r in train]
    target = np.array([int(r["target"]) for r in val])
    original = np.load(OUT/"convnext_clean_original.npy", allow_pickle=False)
    flip = np.load(OUT/"convnext_clean_flip.npy", allow_pickle=False)
    candidate = np.load(OUT/"convnext_clean_probabilities.npy", allow_pickle=False)
    assert np.array_equal(candidate, (original+flip)*np.float32(.5))
    baseline = np.load(ROOT/"hse_expanded_validation_v10_results/selected_val_probabilities.npy", allow_pickle=False)
    vote = (candidate + baseline)*np.float32(.5)
    candidate_blur = np.load(OUT/"convnext_blur_probabilities.npy", allow_pickle=False)
    baseline_blur = np.load(OUT/"hse_blur_probabilities.npy", allow_pickle=False)
    for prefix, mean in (("convnext_blur",candidate_blur),("hse_blur",baseline_blur)):
        ori = np.load(OUT/f"{prefix}_original.npy",allow_pickle=False)
        flp = np.load(OUT/f"{prefix}_flip.npy",allow_pickle=False)
        assert np.array_equal(mean,(ori+flp)*np.float32(.5))
    vote_blur = (candidate_blur+baseline_blur)*np.float32(.5)
    measurements = dict(baseline_clean=baseline, convnext_clean=candidate, fixed_vote_clean=vote,
        baseline_synthetic_blur=baseline_blur, convnext_synthetic_blur=candidate_blur, fixed_vote_synthetic_blur=vote_blur)
    counts = {}
    for key, arr in measurements.items():
        assert arr.shape == (4782,6) and np.isfinite(arr).all()
        assert arr.min() >= 0 and arr.max() <= 1 and np.abs(arr.sum(1)-1).max() < 1e-4
        counts[key] = int((arr.argmax(1)==target).sum())
        assert counts[key] == receipt["measured"][key]["correct"]
    adopted = counts["fixed_vote_clean"] > 3240 and counts["fixed_vote_synthetic_blur"] >= counts["baseline_synthetic_blur"]
    assert adopted == receipt["adopted"] == read(OUT/"selection.json")["adopted"]
    with (OUT/"submission.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["ID", "Emotion"]
        submitted = list(reader)
    with (ROOT/"data/raw/submission.csv").open(encoding="utf-8-sig", newline="") as stream:
        template = list(csv.DictReader(stream))
    with (ROOT/"features/unlabeled.csv").open(encoding="utf-8-sig", newline="") as stream:
        image_rows = list(csv.DictReader(stream))
    legacy_rows = [dict(inference_index=str(i),ID=r["ID"],path=r["path"]) for i,r in enumerate(image_rows)]
    row_sha = hashlib.sha256(json.dumps(legacy_rows,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert row_sha == "8ef47545eaa781b6abc114d833fa893157ccf0bca042341c9874df39b72b9f87"
    assert len(submitted) == len(template) == 7065
    assert [r["ID"] for r in submitted] == [r["ID"] for r in template]
    pred = np.array([int(r["Emotion"]) for r in submitted])
    assert all(str(int(r["Emotion"])) == r["Emotion"] for r in submitted)
    probabilities = np.load(OUT/"probabilities.npy", allow_pickle=False)
    assert probabilities.shape == (7065,6) and np.array_equal(probabilities.argmax(1), pred)
    assert np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1
    assert np.abs(probabilities.sum(1)-1).max() < 1e-4
    assert 0 <= pred.min() <= pred.max() <= 5
    baseline_test = np.load(ROOT/"hse_expanded_validation_v10_results/probabilities.npy", allow_pickle=False)
    base_folder = ROOT/"hse_expanded_validation_v10_results"
    base_receipt = read(base_folder/"competition_prediction_receipt.json")
    base_views = []
    for view in ("original","flip"):
        file = base_folder/base_receipt["files"][view]
        assert sha(file)==base_receipt["artifact_sha256"][view]
        base_views.append(np.load(file,allow_pickle=False))
    assert np.array_equal(baseline_test,(base_views[0]+base_views[1])*np.float32(.5))
    if adopted:
        ori = np.load(OUT/"convnext_competition_original.npy", allow_pickle=False)
        flp = np.load(OUT/"convnext_competition_flip.npy", allow_pickle=False)
        assert np.array_equal(probabilities, ((ori+flp)*np.float32(.5)+baseline_test)*np.float32(.5))
    else:
        assert np.array_equal(probabilities, baseline_test)
    history = read(RUN/"history.json")
    assert len(history) == 12 and all(r["samples"] == 19120 for r in history)
    assert history[-1]["attempts"] == 7176 == history[-1]["successful"] + history[-1]["skipped"]
    assert history[-1]["successful"] == history[-1]["ema_updates"]
    gain = (counts["fixed_vote_clean"]-3240)/4782*100
    old_right, new_right = baseline.argmax(1)==target, vote.argmax(1)==target
    result = dict(passed=True, verifier_source_sha256=sha(__file__), adopted=adopted, counts=counts,
        competition_row_order_sha256=row_sha,
        clean_gain_percentage_points=gain, corrected=int((~old_right & new_right).sum()),
        regressed=int((old_right & ~new_right).sum()), submission_sha256=sha(OUT/"submission.csv"),
        no_gpu_used=True, verification_type="separate read-only implementation; same agent, not independent external review")
    ranks = np.argsort(-probabilities, axis=1, kind="stable")
    entropy = -(probabilities*np.log(np.maximum(probabilities,1e-12))).sum(1)
    with (OUT/"prediction_uncertainty.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID","Emotion","second_class","max_probability_uncalibrated","top2_margin","entropy"])
        for i,row in enumerate(submitted):
            first,second = ranks[i,:2]
            assert int(first)==int(row["Emotion"])
            writer.writerow([row["ID"],int(first),int(second),float(probabilities[i,first]),
                             float(probabilities[i,first]-probabilities[i,second]),float(entropy[i])])
    result["uncertainty_csv_sha256"] = sha(OUT/"prediction_uncertainty.csv")
    (OUT/"verification_readonly.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = ["本次独立训练与固定投票实验", "", "训练：官方 ImageNet-1K ConvNeXt-Tiny，19120张，12轮最终EMA。",
        f"实际成功更新{history[-1]['successful']}次，非有限梯度跳步{history[-1]['skipped']}次，EMA更新{history[-1]['ema_updates']}次。",
        "增强：原有轻量几何与亮度增强，加25%概率的轻度原分辨率高斯模糊。",
        "投票：ConvNeXt与已验证HSE各50%；固定原图/翻转概率均值。未搜索权重或epoch。",
        "", f"原图开发集 HSE：{3240}/4782 = {3240/4782:.4%}",
        f"原图开发集 新ConvNeXt：{counts['convnext_clean']}/4782 = {counts['convnext_clean']/4782:.4%}",
        f"原图开发集 固定投票：{counts['fixed_vote_clean']}/4782 = {counts['fixed_vote_clean']/4782:.4%}",
        f"投票相对HSE变化：{gain:+.4f}个百分点；纠正{result['corrected']}，新增错误{result['regressed']}。",
        f"合成模糊(radius0.7) HSE：{counts['baseline_synthetic_blur']}/4782；固定投票：{counts['fixed_vote_synthetic_blur']}/4782。",
        "", "已采用固定投票新提交。" if adopted else "未达到预先规定的采用门槛，保留V10提交。",
        f"提交行数7065，列ID,Emotion，ID模板顺序、整数0..5与实际概率argmax均核验。SHA256：{result['submission_sha256']}",
        "prediction_uncertainty.csv含未经校准的最大概率、前两类差距、备选类别和熵，仅用于不确定性诊断。",
        "", "边界：这是反复使用的开发集确认结果，非Kaggle成绩；预训练图像重叠未知，也非人物独立验证。",
        "高斯模糊测试是合成腐蚀准确率，不能替代真实模糊照片准确率。",
        "用户提供的fold0权重训练成员不明，未用于本次比较或投票，无法据此声称超过该模型。",
        "所有旧提交、V11暂停状态和监控暂停状态均保留。未上传Kaggle。"]
    (OUT/"实验结果.txt").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
