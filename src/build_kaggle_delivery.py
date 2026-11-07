"""Build a real offline Kaggle inference bundle without reading labeled images."""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/kaggle_expression"
BUNDLE = OUT / "expression_bundle"
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def copy(source, relative):
    destination = BUNDLE / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / source, destination)


README = """# 表情识别：Kaggle 离线 Notebook 与真实权重

1. 在 Kaggle 新建 Notebook，使用 File → Import Notebook，导入 `expression_kaggle.ipynb`。
2. 将 `expression_pretrained_weights.zip` 上传为自己的私有 Kaggle Dataset，并通过 Add Input 加入 Notebook。Kaggle 自动解压或保留 ZIP 均支持。
3. Add Input 加入比赛图片数据集：需有 `test/*.jpg`，也支持原始 `2026task2.zip`。ZIP 只提取 test 图片，不读取训练图片/标签。模板 ID 已打包。
4. Settings 选择 GPU（T4 或 P100），Internet 可关闭。Run All，下载 `/kaggle/working/submission.csv`，也可 Save Version 后选择该输出提交。
5. 自动定位失败时，只需在配置单元填写 `BUNDLE_PATH` 和 `TEST_DIR`。不要解压后只上传某个模型；保留完整 ZIP/目录。

默认 `MODE='ensemble_v8'`：HSE 50% + DINOv2 LoRA 20% + DDAMFN++ 20% + DINOv2 多层 MLP 10%。每路先等权平均原图与水平翻转概率，再固定权重软投票。类别 0 愤怒、1 恐惧、2 快乐、3 悲伤、4 惊讶、5 中性。

这是本项目已有验证的融合方案。旧的、反复使用的开发集为 1924/2656=72.4398%，用户反馈 Kaggle 未改善；不能据此承诺排行榜提升，也不能称全球最优。公开更大/更新模型（例如 DINOv3）未在本任务完成训练验证，未冒充为可用分类器。DDAM 分支选中官方映射后的 epoch0，LoRA 为 epoch5 EMA，HSE 为旧 epoch7，多层 MLP 为固定12轮。

可选 `MODE='hse_v10'`：独立 HSE V10 最终 epoch12 EMA，较大的另一开发划分为 3240/4782=67.7541%；两套划分不同，两个百分比不能直接比较。V10 未与旧模型混合，所有旧权重和提交保留。

## 模糊图片

输入48×48灰度图会按训练时流程转RGB并缩放；多模型与原图/翻转投票同样适用于模糊输入。`prediction_diagnostics.csv` 提供模糊度启发指标、top-2、概率差、熵及各分支分歧。低清晰度阈值35、低置信度阈值0.6、概率差0.15仅作提示，未用有标签数据调阈值；低纹理/噪声也会影响模糊度。最大 softmax 概率不是校准后的正确率。

默认对前64张比赛图施加原始尺寸 GaussianBlur(radius=1)，输出 `blur_stability.json` 和对应概率，测量投票前后的预测一致性。它是无标签稳定性测试，不是模糊图片准确率；不参与选权重、改提交或训练。真实模糊图准确率尚未获得独立有标签验证。信息严重丢失的图像仍可能误判。

## 权重与复现

ZIP 包含实际 FP32 张量权重、DINOv2骨干、LoRA适配器、HSE两个版本、DDAM、MLP标准化缓冲及完整离线代码。没有用预存比赛概率代替推断。`manifest.json` 校验每个文件SHA256，记录源模型/源码SHA和来源。torch.load 使用 weights_only=True；无 pip 安装、无联网下载、无远端执行、无上传。

运行依赖 Kaggle 自带 Python≥3.10、PyTorch≥2.1、torchvision、NumPy、Pillow、PyYAML。timm0.9.16、DINOv2、DDAM网络代码已本地打包；串行加载，单GPU推断。不同GPU/PyTorch/Pillow可能使临界样本数值略有差别，输出receipt保留版本。显存不足时将 BATCH_SIZE=8。

## 来源

- DINOv2: https://github.com/facebookresearch/dinov2 ，官方 ViT-B/14 权重；仓库 Apache-2.0 LICENSE 已包含。
- HSEmotion / EmotiEffLib: https://github.com/sb-ai-lab/EmotiEffLib ，官方 AffectNet7 EfficientNet-B2 初始化，固定6类映射后在本项目微调。官方模型来源记录在manifest；timm Apache-2.0 LICENSE 已包含。
- DDAMFN++: https://github.com/SainingZhang/DDAMFN ，固定commit cebfe5c5969b7d1fd64ecd775e318d3834ca76bb。此固定版本未找到许可证文件，不将其标记为开源许可已确认；保留官方README、来源与SHA，建议保持上传数据集私有并遵循原作者和比赛的外部模型规则。
- DINOv3官方候选调研: https://github.com/facebookresearch/dinov3 （未训练/未打包/不声称更优）。

外部预训练与比赛图片的重叠未知。评估属于反复使用的开发集，不是 Kaggle 分数，也不是从未使用的最终测试集。
"""


def notebook():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md("# 表情识别 · 离线四模型软投票\n\n将权重 ZIP 作为 Dataset 加入，再加入比赛数据，启用 GPU 后 **Run All**。输出 `submission.csv`，无需联网或重新训练。\n\n默认是项目已验证的 V8 融合；它不代表 Kaggle 最优。V10 是不同开发划分的独立候选，可在下格切换。"),
        code('''from pathlib import Path
import os

INPUT_ROOT = Path(os.environ.get("EXPRESSION_INPUT_ROOT", "/kaggle/input"))
WORKING = Path(os.environ.get("EXPRESSION_WORKING", "/kaggle/working"))
BUNDLE_PATH = os.environ.get("EXPRESSION_BUNDLE_PATH") or None  # ZIP 或 expression_bundle 目录；通常留空
TEST_DIR = os.environ.get("EXPRESSION_TEST_DIR") or None  # test 图片所在目录；通常留空
MODE = os.environ.get("EXPRESSION_MODE", "ensemble_v8")  # 或 "hse_v10"
DEVICE = "cuda"
BATCH_SIZE = 16
BLUR_PROBE_IMAGES = 64  # 无标签合成模糊稳定性检查；设0可跳过
WORKING.mkdir(parents=True, exist_ok=True)
'''),
        md("## 加载离线代码与权重\n\n支持 Kaggle 自动解压后的目录，或仍保留的 ZIP。逐文件 SHA256 校验后加载真实模型。"),
        code('''import importlib.util
import sys
import zipfile

if BUNDLE_PATH and Path(BUNDLE_PATH).is_dir():
    runtime_file = Path(BUNDLE_PATH) / "inference.py"
else:
    runtime_files = list(INPUT_ROOT.rglob("expression_bundle/inference.py")) if not BUNDLE_PATH else []
    if len(runtime_files) == 1:
        runtime_file = runtime_files[0]
    else:
        archives = [Path(BUNDLE_PATH)] if BUNDLE_PATH else list(INPUT_ROOT.rglob("expression_pretrained_weights.zip"))
        if len(archives) != 1:
            raise FileNotFoundError("Add Input: expression_pretrained_weights.zip（或填写 BUNDLE_PATH）")
        runtime_file = WORKING / "expression_inference.py"
        with zipfile.ZipFile(archives[0]) as z:
            runtime_file.write_bytes(z.read("expression_bundle/inference.py"))
spec = importlib.util.spec_from_file_location("expression_runtime", runtime_file)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)
BUNDLE = runtime.locate_bundle(INPUT_ROOT, WORKING, BUNDLE_PATH)
runtime.configure(BUNDLE, DEVICE)
manifest = runtime.read_json(BUNDLE / "manifest.json")
print("Mode:", MODE, "Classes:", runtime.CLASSES)
'''),
        md("## 定位比赛图片\n\n严格采用比赛模板的7065个 ID 与顺序。只读取测试图；不需要训练集或标签。"),
        code('''IDS = runtime.template_ids(BUNDLE / "submission_template.csv")
PATHS = runtime.locate_images(INPUT_ROOT, WORKING, IDS, TEST_DIR)
import hashlib
image_digest = hashlib.sha256()
for p in PATHS:
    image_digest.update(runtime.sha256(p).encode("ascii"))
if image_digest.hexdigest() != manifest["competition_image_content_sha256"]:
    raise ValueError("测试图内容与此比赛的冻结输入不同，请检查是否加入正确的比赛数据集")
print("Competition images verified:", len(PATHS))
'''),
        md("## 串行推断与软投票\n\n每路采用原图+水平翻转。前64张额外做合成模糊测试，仅报告稳定性，不改最终投票。可在日志中查看每路进度。"),
        code('''PROBABILITIES, VOTES, RUN_INFO = runtime.predict(
    BUNDLE, PATHS, WORKING, MODE, DEVICE, BATCH_SIZE, BLUR_PROBE_IMAGES)
receipt = runtime.write_outputs(BUNDLE, IDS, PATHS, PROBABILITIES, VOTES, WORKING, RUN_INFO)
print(receipt)
'''),
        md("## 模糊图和低置信度检查\n\n清晰度为启发式指标，softmax 概率未校准。真实模糊图准确率需要独立标注确认；合成模糊的预测一致性不能替代准确率。诊断表不改变 `ID,Emotion` 提交格式。"),
        code('''import csv
import html
from IPython.display import display, FileLink, HTML
with (WORKING / "prediction_diagnostics.csv").open(encoding="utf-8", newline="") as f:
    diagnostics = list(csv.DictReader(f))
diagnostics.sort(key=lambda row: (row["uncertain_heuristic"] != "True", float(row["margin"])))
columns = ["ID", "class_name", "top1_probability", "top2_class", "margin", "low_sharpness_heuristic"]
table = "<table><tr>" + "".join("<th>" + html.escape(c) + "</th>" for c in columns) + "</tr>"
for row in diagnostics[:20]:
    table += "<tr>" + "".join("<td>" + html.escape(row[c]) + "</td>" for c in columns) + "</tr>"
display(HTML(table + "</table>"))
display(FileLink(str(WORKING / "submission.csv")))
display(FileLink(str(WORKING / "prediction_diagnostics.csv")))
if (WORKING / "blur_stability.json").exists():
    print(runtime.read_json(WORKING / "blur_stability.json"))
'''),
        md("权重来源：[DINOv2](https://github.com/facebookresearch/dinov2)、[HSE/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)、[DDAMFN++](https://github.com/SainingZhang/DDAMFN)。默认融合旧开发集72.4398%；V10另一划分67.7541%，不作跨划分排名。开发集被反复使用、预训练重叠未知，不承诺 Kaggle 提分。详细来源和许可说明见 ZIP 内 README。")]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": False, "language": "python",
                   "sourceType": "notebook", "isGpuEnabled": True}})
    nbformat.validate(nb)
    nbformat.write(nb, OUT / "expression_kaggle.ipynb")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE.mkdir(exist_ok=True)
    provenance = {}
    sources = {"hse": ("experiments/hse_specialist_v1/best.pt", "model_state"),
               "lora": ("experiments/lora_task_v4/candidates/epoch_05_ema.pt", "adapter_state"),
               "ddam": ("experiments/ddamfn_v8/candidates/epoch_00_baseline.pt", "model_state"),
               "mlp": ("experiments/cosine_full_train/model.pt", "model_state"),
               "hse_v10": ("experiments/hse_expanded_validation_v10/final_ema.pt", "model_state")}
    v4, v8 = read("lora_task_v4_results/selected_recipe.json"), read("ddamfn_v8_results/selected_recipe.json")
    assert v4["selected"]["name"] == "epoch_05_ema" and v8["selected"]["name"] == "epoch_00_baseline"
    assert read("hse_results/selected_recipe.json")["selected"]["name"] == "hse_50_flip"
    assert v4["fusion_weights"] == {"hse": .5, "lora": .2, "cnn": .2, "mlp": .1}
    hard_pins = {"hse": "ed58a57d691943dc605053ac6182d6b54f05f4ae4133690d65ce5abeb10d20b7",
                 "hse_v10": "e872fff837d05c2bb319676bcb66f684242dcbc107d1a6b302fb88a418124091",
                 "mlp": "51c463457f8f70123d9d3e9f57c225afd56d74fd644be88941c1942cba19d8b9"}
    for branch, (relative, key) in sources.items():
        source = ROOT / relative
        digest = sha(source)
        pins = v4["producer_files"] if branch == "lora" else v8["producer_files"]
        expected = hard_pins.get(branch) or pins.get(str(source))
        if expected != digest:
            raise ValueError("Source checkpoint pin mismatch: " + branch)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        assert payload["classes"] == CLASSES
        tensors = {k: v.detach().cpu().contiguous() for k, v in payload[key].items()}
        assert all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in tensors.values())
        target = BUNDLE / "weights" / (branch + ".pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, target)
        actual = torch.load(target, map_location="cpu", weights_only=True)
        assert tensors.keys() == actual.keys() and all(torch.equal(v, actual[k]) for k, v in tensors.items())
        provenance[branch] = {"source_project_path": relative, "source_sha256": digest,
                              "tensor_count": len(tensors), "tensor_export_exact": True,
                              "epoch": payload.get("epoch", payload.get("best_epoch", payload.get("epochs_completed")))}
        print("Exported exact tensor state:", branch, len(tensors), flush=True)
    source = ROOT / "models/dinov2_vitb14_pretrain.pth"
    assert sha(source) == read("models/dinov2_vitb14_pretrain.json")["sha256"]
    copy(source, "weights/dinov2_vitb14.pth")
    provenance["dinov2"] = {"source_sha256": sha(source),
                            "url": read("models/dinov2_vitb14_pretrain.json")["url"],
                            "repository": "https://github.com/facebookresearch/dinov2",
                            "commit": subprocess.check_output(["git", "-C", str(ROOT/"vendor/dinov2"), "rev-parse", "HEAD"], text=True).strip()}
    # Copy only Python code / metadata / licenses, no caches, training data, or binaries.
    for base, destination in [(ROOT/"vendor/dinov2/dinov2", "vendor/dinov2/dinov2"),
                              (ROOT/"vendor/hse_runtime/timm", "vendor/timm"),
                              (ROOT/"vendor/hse_runtime/timm-0.9.16.dist-info", "vendor/timm-0.9.16.dist-info")]:
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and (p.suffix in (".py", ".json", ".txt", ".csv") or p.name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt")):
                copy(p, Path(destination) / p.relative_to(base))
    for name in ("LICENSE", "README.md", "MODEL_CARD.md"):
        copy(ROOT/"vendor/dinov2"/name, "vendor/dinov2/" + name)
    dd = ROOT / "vendor/ddamfn_v8/DDAMFN++/networks"
    copy(dd/"MixedFeatureNet.py", "vendor/ddam_networks/MixedFeatureNet.py")
    transformed = (dd/"DDAM.py").read_text(encoding="utf-8").replace("from networks import MixedFeatureNet", "from . import MixedFeatureNet")
    (BUNDLE/"vendor/ddam_networks/DDAM.py").write_text(transformed, encoding="utf-8")
    (BUNDLE/"vendor/ddam_networks/__init__.py").write_text("", encoding="utf-8")
    copy(ROOT/"vendor/ddamfn_v8/README.md", "vendor/ddam_networks/UPSTREAM_README.md")
    provenance["ddam_vendor"] = {"repository": "https://github.com/SainingZhang/DDAMFN", "commit": "cebfe5c5969b7d1fd64ecd775e318d3834ca76bb",
        "upstream_ddam_sha256": sha(dd/"DDAM.py"), "upstream_mixed_feature_sha256": sha(dd/"MixedFeatureNet.py"),
        "portable_change": "Only replace absolute networks import with relative import.", "license_found_at_pinned_revision": False,
        "official_weight_sha256": "edf731a458162eef6b2be12c3572bd10cbe92ce8c009da4331184a2943d7ed14"}
    provenance["hse_official"] = {"repository": "https://github.com/sb-ai-lab/EmotiEffLib",
        "weight_url": "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/520a051c64cd191521e5934655314e769a319684/models/affectnet_emotions/enet_b2_7.pt",
        "sha256": "e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f", "seven_to_six_rows": [0, 2, 3, 5, 6, 4]}
    copy(ROOT/"src/kaggle_expression_runtime.py", "inference.py")
    copy(ROOT/"data/raw/submission.csv", "submission_template.csv")
    (BUNDLE / "README_CN.md").write_text(README, encoding="utf-8")
    (OUT / "使用说明.txt").write_text(README, encoding="utf-8")
    files = {p.relative_to(BUNDLE).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(BUNDLE.rglob("*")) if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    manifest = {"format": "expression_offline_bundle_v1", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "classes": CLASSES, "files": files, "provenance": provenance,
                "recipe": {"ensemble_v8": {"hse": .5, "lora": .2, "ddam": .2, "mlp": .1}, "hse_v10": {"hse_v10": 1.}},
                "tta": "each branch: mean(original, normalized_tensor_horizontal_flip) probabilities",
                "competition_image_content_sha256": "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4",
                "reference_submissions": {"ensemble_v8": "55a037351e0677eadc82ebe81ce6a9c208da402f304cd9eeccba70e2121ce4b6",
                    "hse_v10": "214d6bec3b1cd40f204cc8db3ad6c91f2db7516b65cd89d4e582f3195e0bc683"},
                "limitations": "Repeated development validation; unknown external pretraining overlap; no global/Kaggle superiority claim; blur accuracy unmeasured."}
    (BUNDLE/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    notebook()
    target = OUT / "expression_pretrained_weights.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as z:
        for relative in [*files, "manifest.json"]:
            z.write(BUNDLE/relative, "expression_bundle/" + relative)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files) + 1
    receipt = {"state": "built", "notebook_sha256": sha(OUT/"expression_kaggle.ipynb"),
               "zip_sha256": sha(target), "zip_bytes": target.stat().st_size,
               "manifest_sha256": sha(BUNDLE/"manifest.json"), "files": len(files),
               "all_tensor_exports_equal": True, "zip_crc_ok": True, "nbformat_valid": True,
               "builder_sha256": sha(__file__), "runtime_sha256": sha(ROOT/"src/kaggle_expression_runtime.py")}
    (OUT/"build_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    build()
