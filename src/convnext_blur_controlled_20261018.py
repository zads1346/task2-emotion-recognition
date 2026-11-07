"""One preregistered ConvNeXt candidate; known training members; fixed HSE vote.

This experiment does not resume V11 or use the imported task-trained fold0 model.
Only the final epoch is evaluated. No outer-label-driven recipe/weight search.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
from contextlib import contextmanager
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models import convnext_tiny

ROOT = Path(__file__).resolve().parents[1]
NAME = "convnext_blur_controlled_20261018"
RUN = ROOT / "experiments" / NAME
OUT = ROOT / "deliverables" / NAME
PLAN = ROOT / "research" / (NAME + "_plan.json")
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
WEIGHT = ROOT / "models/convnext_imagenet_official/convnext_tiny-983f1562.pth"
WEIGHT_SHA = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
WEIGHT_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
TRAIN_MANIFEST = ROOT / "data/hse_groupcv_v11/train_manifest.csv"
TRAIN_SHA = "538ce7340600a43236cc5cdd22c144cf504ee99b7bab4e003e372c347bc9fc9c"
SPLIT = ROOT / "data/expanded_validation_v10/manifest.csv"
SPLIT_SHA = "b5a9d3cb15d6ce6f586a34194d669350a7f7d599265c73afdc1237ef02a98a8a"
HSE = ROOT / "experiments/hse_expanded_validation_v10/final_ema.pt"
HSE_SHA = "e872fff837d05c2bb319676bcb66f684242dcbc107d1a6b302fb88a418124091"
HSE_PROBS = ROOT / "hse_expanded_validation_v10_results/selected_val_probabilities.npy"
TEST_SHA = "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
SETTINGS = dict(epochs=12, seed=20261018, augmentation_epoch_offset=100,
    image_size=224, micro_batch=16, effective_batch=32, workers=0,
    backbone_lr=5e-5, head_lr=5e-4, weight_decay=.05,
    label_smoothing=.05, gradient_clip=1., warmup_successful_updates=100,
    cosine_min_ratio=.1, ema_decay=.995, amp="bfloat16_no_gradient_scaling",
    stochastic_depth_max=.1, blur_probability=.25, native_blur_radius=[.15, .65],
    eval_blur_radius=.7, hse_vote_weight=.5, convnext_vote_weight=.5,
    checkpoint_selection="fixed_final_epoch12_EMA_only",
    adoption="fixed_vote clean correct >3240 AND synthetic blur correct >= baseline",
    search_budget="one training recipe, one final fixed vote; no test-label search")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError("SHA256 differs: " + str(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@contextmanager
def gpu_lock():
    # An OS-backed lock is released even if this Python process crashes.
    import msvcrt
    path = ROOT / "logs/controlled_experiment_gpu.lock"
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False, allow_nan=False), flush=True)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)


def allowed_rows(outer=False, images=False):
    # Training never parses the outer manifest or its labels.
    check(TRAIN_MANIFEST, TRAIN_SHA)
    rows = list(csv.DictReader(TRAIN_MANIFEST.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 19120 or len({r["manifest_index"] for r in rows}) != 19120:
        raise ValueError("Training manifest coverage differs")
    if outer:
        check(SPLIT, SPLIT_SHA)
        with SPLIT.open(encoding="utf-8-sig", newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        training = [r for r in all_rows if r["split"] == "train"]
        for key in ("manifest_index", "path", "target", "group", "image_sha256", "label"):
            if [r[key] for r in training] != [r[key] for r in rows]:
                raise ValueError("Train-only manifest differs from frozen source")
        val = [r for r in all_rows if r["split"] == "val"]
        if len(val) != 4782 or {r["group"] for r in rows} & {r["group"] for r in val}:
            raise ValueError("Outer count/group separation differs")
        rows = val
    for r in rows:
        if int(r["manifest_index"]) in (7727, 10448) or CLASSES[int(r["target"])] != r["label"]:
            raise ValueError("Excluded member or class mismatch")
        if images:
            check(r["path"], r["image_sha256"])
    return rows


class Images(Dataset):
    def __init__(self, rows, training=False, blur=0.):
        self.rows, self.training, self.blur = rows, training, blur
        ops = [T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)]
        if training:
            ops += [T.RandomHorizontalFlip(.5),
                T.RandomAffine(6, translate=(.03, .03), scale=(.95, 1.05),
                               interpolation=T.InterpolationMode.BILINEAR),
                T.ColorJitter(brightness=.1, contrast=.1)]
        ops += [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            radius = self.blur
            if self.training and random.random() < SETTINGS["blur_probability"]:
                radius = random.uniform(*SETTINGS["native_blur_radius"])
            if radius:
                image = image.filter(ImageFilter.GaussianBlur(radius))
            tensor = self.transform(image)
        return tensor, int(row.get("target", -1))


def make_model(initial=True):
    # All pretrained feature tensors load strictly before replacing the head.
    model = convnext_tiny(weights=None, stochastic_depth_prob=SETTINGS["stochastic_depth_max"])
    if initial:
        check(WEIGHT, WEIGHT_SHA)
        model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    model.classifier[2] = nn.Linear(768, 6)
    return model


def runtime():
    return dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                numpy=np.__version__, cuda=torch.version.cuda)


def prepare():
    if PLAN.exists():
        verify_plan()
        emit("already_prepared", plan_sha256=sha(PLAN))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = allowed_rows(images=True)
    files = [Path(__file__), WEIGHT, TRAIN_MANIFEST, SPLIT, HSE, HSE_PROBS,
        ROOT / "hse_expanded_validation_v10_results/probabilities.npy",
        ROOT / "hse_expanded_validation_v10_results/submission.csv",
        ROOT / "src/hse_model.py", ROOT / "src/train_hse_specialist.py",
        ROOT / "src/train_hse_sam_v7.py", ROOT / "src/validate_cosine_fusion.py",
        ROOT / "hse_expanded_validation_v10_results/selected_recipe.json",
        ROOT / "hse_expanded_validation_v10_results/competition_prediction_receipt.json",
        ROOT / "models/hsemotion/enet_b2_7.pt", ROOT / "models/hsemotion/enet_b2_7_state.pt",
        ROOT / "models/hsemotion/official_weight.json", ROOT / "models/hsemotion/conversion_receipt.json",
        ROOT / "features/unlabeled.csv", ROOT / "data/raw/submission.csv",
        Path(inspect.getfile(convnext_tiny)), Path(inspect.getfile(T.RandomAffine)),
        Path(inspect.getfile(torch.optim.AdamW))]
    # Bind the complete local HSE inference vendor, without changing it.
    files += sorted((ROOT / "vendor/hse_runtime/timm").rglob("*.py"))
    files += sorted(Path(torchvision.__file__).parent.rglob("*.py"))
    files += sorted(Path(inspect.getfile(Image)).parent.glob("*.py"))
    files = sorted(set(p.resolve() for p in files))
    check(HSE, HSE_SHA)
    digest = hashlib.sha256("".join(r["image_sha256"] for r in rows).encode()).hexdigest()
    plan = dict(name=NAME, created_at=time.time(), settings=SETTINGS, classes=CLASSES,
        runtime=runtime(), source_and_input_pins={str(p.resolve()): sha(p) for p in files},
        train_rows=len(rows), train_content_sha256=digest,
        official_pretraining=dict(url=WEIGHT_URL, sha256=WEIGHT_SHA, dataset="ImageNet-1K",
            documentation="https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.convnext_tiny.html"),
        supplied_fold0_used=False, reason="training membership is unknown; excluded from fair comparison",
        validation_policy="one final fixed-vote confirmation on repeatedly used development4782; not untouched test",
        genuine_blur_accuracy="unknown; Gaussian corruption accuracy is only synthetic robustness",
        pretraining_overlap="unknown", person_independence=False,
        internal_holdout_read=False, external_answers_used=False, kaggle_upload=False,
        V11_resumed=False, review="local source/math/data audit and CPU checks; no independent-agent approval claimed")
    write_json(PLAN, plan)
    emit("prepared", plan_sha256=sha(PLAN), training_rows=len(rows), pins=len(files))


def verify_plan():
    p = read_json(PLAN)
    if p["settings"] != SETTINGS or p["runtime"] != runtime() or p["classes"] != CLASSES:
        raise ValueError("Locked configuration/runtime differs")
    for file, digest in p["source_and_input_pins"].items():
        check(file, digest)
    return p


def make_optimizer(model):
    head = list(model.classifier.parameters())
    head_ids = {id(x) for x in head}
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=SETTINGS["backbone_lr"], base_lr=SETTINGS["backbone_lr"]),
        dict(params=head, lr=SETTINGS["head_lr"], base_lr=SETTINGS["head_lr"])],
        weight_decay=SETTINGS["weight_decay"], foreach=True)


def lr_for(optimizer, successful, total):
    warm = SETTINGS["warmup_successful_updates"]
    if successful < warm:
        ratio = (successful + 1) / warm
    else:
        progress = min(1., (successful - warm) / max(1, total - warm))
        ratio = .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * ratio


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def one_update(model, optimizer, ema, images, targets, micro=16, device="cuda", amp=True):
    """Sum CE over microbatches, divide by actual logical-group size, including tail16."""
    optimizer.zero_grad(set_to_none=True)
    count = len(targets)
    if not 1 <= count <= 32 or len(images) != count:
        raise ValueError("Invalid logical batch")
    total_loss, right = 0., 0
    for start in range(0, count, micro):
        x = images[start:start+micro].to(device, memory_format=torch.channels_last)
        y = targets[start:start+micro].to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y, reduction="sum", label_smoothing=.05)
        if not torch.isfinite(loss).item():
            optimizer.zero_grad(set_to_none=True)
            return False, None, 0, None
        (loss / count).backward()
        total_loss += loss.item()
        right += (logits.argmax(1) == y).sum().item()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    if not math.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        return False, total_loss, right, None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        with torch.no_grad():
            for key, value in model.state_dict().items():
                ema[key].lerp_(value, 1 - SETTINGS["ema_decay"])
    return True, total_loss, right, norm


def smoke():
    verify_plan()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This locked run requires CUDA BF16")
    # Analytic full-batch versus microbatch-tail test with a tiny CPU network.
    seed(42)
    first = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
    second = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
    second.load_state_dict(first.state_dict())
    x, y = torch.randn(19, 3, 2, 2), torch.arange(19) % 6
    a = torch.optim.SGD(first.parameters(), lr=.01)
    b = torch.optim.SGD(second.parameters(), lr=.01)
    one_update(first, a, None, x, y, 16, "cpu", False)
    one_update(second, b, None, x, y, 19, "cpu", False)
    difference = max((v - second.state_dict()[k]).abs().max().item() for k, v in first.state_dict().items())
    assert difference < 1e-7, difference
    rows = allowed_rows()
    seed(SETTINGS["seed"])
    model = make_model().cuda().to(memory_format=torch.channels_last).train()
    optimizer = make_optimizer(model)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_head = model.classifier[2].weight.detach().clone()
    ds = Images(rows[:48], training=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    evidence = []
    started = time.monotonic()
    for i, (images, targets) in enumerate(loader):
        lr_for(optimizer, i, 7176)
        result = one_update(model, optimizer, ema, images, targets)
        assert result[0] and math.isfinite(result[1]), result
        evidence.append(dict(actual_group_size=len(targets), success=result[0], loss=result[1]/len(targets), grad_norm=result[3]))
    assert len(evidence) == 2 and not torch.equal(initial_head, model.classifier[2].weight)
    step_values = sorted({int(v["step"].item()) for v in optimizer.state.values()})
    assert step_values == [2]
    drop_path = [m.p for m in model.modules() if isinstance(m, torchvision.ops.StochasticDepth)]
    result = dict(passed=True, plan_sha256=sha(PLAN), gpu=torch.cuda.get_device_name(),
        cpu_tail_gradient_max_difference=difference, actual_gpu_updates=2, ema_updates=2,
        groups=evidence, optimizer_step_values=step_values, seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), trainable_parameters=sum(p.numel() for p in model.parameters()),
        stochastic_depth_probabilities=drop_path, dropout_modules=[m.p for m in model.modules() if isinstance(m, nn.Dropout)],
        batchnorm_modules=sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()),
        precision="BF16 autocast / FP32 CE and optimizer, no GradScaler needed", outer_labels_read=False,
        audit="tail normalization; exact train-image hashes; source/runtime pins; official-init strict restore; final-epoch-only fixed vote")
    write_json(RUN / "smoke.json", result)
    emit("smoke_passed", **{k: v for k, v in result.items() if k != "groups"})


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def save_pt(path, obj):
    temp = Path(str(path) + ".tmp")
    torch.save(cpu_tree(obj), temp)
    os.replace(temp, path)


def train():
    verify_plan()
    s = read_json(RUN / "smoke.json")
    assert s["passed"] and s["plan_sha256"] == sha(PLAN)
    seed(SETTINGS["seed"])
    rows = allowed_rows(images=True)
    model = make_model().cuda().to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    completed, successful, skipped, attempts, history = 0, 0, 0, 0, []
    latest = RUN / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, weights_only=True, map_location="cpu")
        assert payload["plan_sha256"] == sha(PLAN) and payload["epoch_complete"] is True
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        ema = {k: v.to("cuda") for k, v in payload["ema"].items()}
        completed, successful, skipped, attempts, history = [payload[k] for k in ("epoch", "successful", "skipped", "attempts", "history")]
        restore_rng(payload["rng"])
    total = math.ceil(len(rows)/32) * SETTINGS["epochs"]
    write_json(RUN / "runtime.json", dict(state="training", pid=os.getpid(), parent_pid=os.getppid(), plan_sha256=sha(PLAN), started_at=time.time()))
    for epoch in range(completed, SETTINGS["epochs"]):
        model.train()
        seed(SETTINGS["seed"] + SETTINGS["augmentation_epoch_offset"] + epoch)
        loader = DataLoader(Images(rows, training=True), batch_size=32, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(SETTINGS["seed"] + epoch))
        started = time.monotonic()
        seen, loss_sum, correct, epoch_skips = 0, 0., 0, 0
        for group, (images, targets) in enumerate(loader):
            lr_for(optimizer, successful, total)
            success, loss, right, norm = one_update(model, optimizer, ema, images, targets)
            attempts += 1
            successful += int(success)
            skipped += int(not success)
            epoch_skips += int(not success)
            seen += len(targets)
            loss_sum += 0. if loss is None else loss
            correct += right
            if group % 100 == 0:
                values = dict(state="training", pid=os.getpid(), parent_pid=os.getppid(),
                    epoch=epoch+1, group=group+1, groups=len(loader), seen=seen, successful=successful,
                    skipped=skipped, ema_updates=successful, updated_at=time.time(), plan_sha256=sha(PLAN))
                write_json(RUN / "runtime.json", values)
                emit("training_progress", **values)
            if epoch_skips > 5:
                raise RuntimeError("More than five nonfinite groups in an epoch; abort and audit")
        assert seen == 19120 and len(loader) == 598
        steps = sorted({int(v["step"].item()) for v in optimizer.state.values()})
        assert steps == [successful], (steps, successful)
        entry = dict(epoch=epoch+1, samples=seen, loss=loss_sum/seen, train_accuracy=correct/seen,
            successful=successful, skipped=skipped, ema_updates=successful, attempts=attempts,
            epoch_seconds=time.monotonic()-started, optimizer_step_values=steps, last_group_size=len(targets))
        history.append(entry)
        payload = dict(plan_sha256=sha(PLAN), epoch=epoch+1, epoch_complete=True,
            model=model.state_dict(), optimizer=optimizer.state_dict(), ema=ema,
            scaler={"enabled":False,"reason":"bfloat16"}, rng=rng_state(),
            successful=successful, skipped=skipped, attempts=attempts, history=history)
        epoch_path = RUN / f"epoch_{epoch+1:02d}.pt"
        save_pt(epoch_path, payload)
        # An atomic copy supports restoration only at a fully committed epoch.
        shutil.copyfile(epoch_path, RUN / "latest.pt.tmp")
        os.replace(RUN / "latest.pt.tmp", latest)
        write_json(RUN / "history.json", history)
        emit("epoch_complete", **entry)
    save_pt(RUN / "final_ema.pt", dict(plan_sha256=sha(PLAN), model=ema, epoch=12,
        successful=successful, ema_updates=successful, skipped=skipped))
    write_json(RUN / "runtime.json", dict(state="training_complete", pid=os.getpid(), parent_pid=os.getppid(),
        plan_sha256=sha(PLAN), epochs=12, successful=successful, skipped=skipped, updated_at=time.time()))


@torch.inference_mode()
def predict(model, rows, prefix, blur=0., batch=32):
    model.eval()
    views = [[], []]
    loader = DataLoader(Images(rows, blur=blur), batch_size=batch, shuffle=False, num_workers=0)
    for images, _ in loader:
        images = images.cuda().to(memory_format=torch.channels_last)
        for index, x in enumerate((images, images.flip(-1))):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            views[index].append(logits.float().softmax(1).cpu().numpy())
    original, flip = (np.concatenate(x).astype(np.float32) for x in views)
    probabilities = (original + flip) * np.float32(.5)
    for suffix, arr in (("original", original), ("flip", flip), ("probabilities", probabilities)):
        np.save(OUT / f"{prefix}_{suffix}.npy", arr, allow_pickle=False)
    return probabilities


def metrics(probs, targets):
    check_probs(probs, len(targets))
    prediction = probs.argmax(1)
    matrix = np.zeros((6,6), dtype=np.int64)
    np.add.at(matrix, (targets, prediction), 1)
    precision = np.diag(matrix)/np.maximum(1, matrix.sum(0))
    recall = np.diag(matrix)/np.maximum(1, matrix.sum(1))
    f1 = 2*precision*recall/np.maximum(1e-12, precision+recall)
    correct = int((prediction == targets).sum())
    return dict(correct=correct, total=len(targets), accuracy=correct/len(targets), macro_f1=float(f1.mean()),
        per_class_recall=recall.tolist(), confusion_matrix=matrix.tolist())


def check_probs(probs, count):
    assert probs.shape == (count,6) and probs.dtype == np.float32
    assert np.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1
    assert np.max(np.abs(probs.sum(1)-1)) < 1e-4


def test_rows():
    pins = read_json(PLAN)["source_and_input_pins"]
    manifest, template = ROOT/"features/unlabeled.csv", ROOT/"data/raw/submission.csv"
    check(manifest, pins[str(manifest)])
    check(template, pins[str(template)])
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    ids = [r["ID"] for r in csv.DictReader(template.open(encoding="utf-8-sig", newline=""))]
    assert [r["ID"] for r in rows] == ids and len(ids) == len(set(ids)) == 7065
    content = hashlib.sha256()
    for row in rows:
        content.update(sha(row["path"]).encode())
    assert content.hexdigest() == TEST_SHA
    return rows


def audit_training_final():
    verify_plan()
    history = read_json(RUN/"history.json")
    assert [x["epoch"] for x in history] == list(range(1,13))
    for number, entry in enumerate(history, 1):
        assert entry["samples"] == 19120 and entry["last_group_size"] == 16
        checkpoint = torch.load(RUN/f"epoch_{number:02d}.pt", weights_only=True, map_location="cpu")
        assert checkpoint["epoch_complete"] and checkpoint["plan_sha256"] == sha(PLAN)
        assert checkpoint["history"] == history[:number]
        assert checkpoint["attempts"] == 598*number == checkpoint["successful"] + checkpoint["skipped"]
        assert sorted({int(v["step"].item()) for v in checkpoint["optimizer"]["state"].values()}) == [checkpoint["successful"]]
        del checkpoint
    final = torch.load(RUN/"final_ema.pt", weights_only=True, map_location="cpu")
    assert final["epoch"] == 12 and final["successful"] == final["ema_updates"] == history[-1]["successful"]
    assert final["plan_sha256"] == sha(PLAN)
    latest = torch.load(RUN/"epoch_12.pt", weights_only=True, map_location="cpu")
    assert final["model"].keys() == latest["ema"].keys()
    assert all(torch.equal(v, latest["ema"][k]) and torch.isfinite(v).all() for k,v in final["model"].items())
    del latest
    return final, history


def final_confirmation(final):
    rows = allowed_rows(outer=True, images=True)
    targets = np.array([int(r["target"]) for r in rows])
    seed(SETTINGS["seed"])
    model = make_model(False).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(final["model"], strict=True)
    candidate = predict(model, rows, "convnext_clean")
    candidate_blur = predict(model, rows, "convnext_blur", blur=.7)
    del model, final
    torch.cuda.empty_cache()
    # Baseline clean probabilities were independently accepted with the same ordered members.
    baseline = np.load(HSE_PROBS, allow_pickle=False)
    assert metrics(baseline, targets)["correct"] == 3240
    import hse_model
    hse = hse_model.load_initial_model(ROOT, "cpu")
    payload = torch.load(HSE, map_location="cpu", weights_only=True)
    hse.load_state_dict(payload["model_state"], strict=True)
    del payload
    hse = hse.cuda().to(memory_format=torch.channels_last)
    baseline_blur = predict(hse, rows, "hse_blur", blur=.7, batch=16)
    vote = (candidate + baseline) * np.float32(.5)
    vote_blur = (candidate_blur + baseline_blur) * np.float32(.5)
    measured = dict(baseline_clean=metrics(baseline, targets), convnext_clean=metrics(candidate, targets),
        fixed_vote_clean=metrics(vote, targets), baseline_synthetic_blur=metrics(baseline_blur, targets),
        convnext_synthetic_blur=metrics(candidate_blur, targets), fixed_vote_synthetic_blur=metrics(vote_blur, targets))
    adopted = measured["fixed_vote_clean"]["correct"] > 3240 and measured["fixed_vote_synthetic_blur"]["correct"] >= measured["baseline_synthetic_blur"]["correct"]
    selection = dict(plan_sha256=sha(PLAN), measured=measured, adopted=adopted, decision_time=time.time(),
        rule=SETTINGS["adoption"], train_checkpoint_sha256=sha(RUN/"final_ema.pt"),
        caveat="Repeated development set, not Kaggle. Synthetic blur is not real-world blur accuracy. Unknown imported fold0 cannot be fairly ranked here.")
    write_json(OUT/"selection.json", selection)  # Lock decision before any competition inference.
    emit("final_confirmation", **selection)
    del hse
    torch.cuda.empty_cache()
    return selection


def finalize():
    verify_plan()
    if (OUT / "acceptance.json").exists():
        verify_delivery()
        return
    final, history = audit_training_final()
    if (OUT/"selection.json").exists():
        selection = read_json(OUT/"selection.json")
        assert selection["plan_sha256"] == sha(PLAN)
        assert selection["train_checkpoint_sha256"] == sha(RUN/"final_ema.pt")
    else:
        selection = final_confirmation(final)
    del final
    adopted, measured = selection["adopted"], selection["measured"]
    competition = test_rows()
    if adopted:
        # Reuse accepted V10 competition probabilities; no recomputation drift.
        hse_test = np.load(ROOT/"hse_expanded_validation_v10_results/probabilities.npy", allow_pickle=False)
        final = torch.load(RUN/"final_ema.pt", weights_only=True, map_location="cpu")
        model = make_model(False).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(final["model"], strict=True)
        conv_test = predict(model, competition, "convnext_competition")
        probabilities = (hse_test + conv_test) * np.float32(.5)
    else:
        probabilities = np.load(ROOT/"hse_expanded_validation_v10_results/probabilities.npy", allow_pickle=False)
    check_probs(probabilities, 7065)
    np.save(OUT/"probabilities.npy", probabilities, allow_pickle=False)
    with (OUT/"submission.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "Emotion"])
        writer.writerows((r["ID"], int(pred)) for r, pred in zip(competition, probabilities.argmax(1)))
    files = list(OUT.glob("*.npy")) + [OUT/"submission.csv", OUT/"selection.json", RUN/"final_ema.pt", RUN/"history.json", RUN/"smoke.json", PLAN]
    files += [RUN/f"epoch_{i:02d}.pt" for i in range(1,13)]
    receipt = dict(passed=True, plan_sha256=sha(PLAN), adopted=adopted, measured=measured,
        artifacts={str(p):sha(p) for p in files}, train_epochs=12, history=history,
        submitted_rows=7065, classes=CLASSES, test_images_sha256=TEST_SHA,
        imported_task_checkpoint_used=False, V11_resumed=False, external_answers_used=False,
        selection_sha256=sha(OUT/"selection.json"), no_kaggle_upload=True)
    write_json(OUT/"acceptance.json", receipt)
    verify_delivery()
    write_json(RUN/"runtime.json", dict(state="complete_verified", adopted=adopted, pid=os.getpid(),
        parent_pid=os.getppid(), completed_at=time.time(), submission=str(OUT/"submission.csv")))


def verify_delivery():
    verify_plan()
    receipt = read_json(OUT/"acceptance.json")
    assert receipt["plan_sha256"] == sha(PLAN) and receipt["passed"]
    for path, digest in receipt["artifacts"].items():
        check(path, digest)
    probs = np.load(OUT/"probabilities.npy", allow_pickle=False)
    check_probs(probs, 7065)
    source = list(csv.DictReader((ROOT/"data/raw/submission.csv").open(encoding="utf-8-sig", newline="")))
    with (OUT/"submission.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["ID", "Emotion"]
        rows = list(reader)
    assert len(rows) == 7065 and [r["ID"] for r in rows] == [r["ID"] for r in source]
    predictions = np.array([int(r["Emotion"]) for r in rows])
    assert all(str(int(r["Emotion"])) == r["Emotion"] for r in rows)
    assert np.array_equal(predictions, probs.argmax(1)) and predictions.min() >= 0 and predictions.max() <= 5
    if receipt["adopted"]:
        orig = np.load(OUT/"convnext_competition_original.npy", allow_pickle=False)
        flip = np.load(OUT/"convnext_competition_flip.npy", allow_pickle=False)
        base = np.load(ROOT/"hse_expanded_validation_v10_results/probabilities.npy", allow_pickle=False)
        assert np.array_equal(probs, (base + (orig+flip)*np.float32(.5))*np.float32(.5))
    else:
        check(OUT/"submission.csv", "214d6bec3b1cd40f204cc8db3ad6c91f2db7516b65cd89d4e582f3195e0bc683")
    emit("delivery_verified", adopted=receipt["adopted"], rows=7065, submission_sha256=sha(OUT/"submission.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "smoke", "train", "finalize", "verify", "run"])
    args = parser.parse_args()
    if args.stage in ("run", "train", "smoke", "finalize"):
        with gpu_lock():
            if args.stage == "run":
                train()
                finalize()
            else:
                dict(smoke=smoke, train=train, finalize=finalize)[args.stage]()
    else:
        dict(prepare=prepare, verify=verify_delivery)[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if RUN.exists():
            write_json(RUN/"fault.json", dict(type=type(exc).__name__, message=str(exc), pid=os.getpid(), time=time.time()))
        raise
