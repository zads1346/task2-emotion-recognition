"""Run reviewed package inference unchanged; capture probabilities for verification."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "imports/convnext_tiny_single_20261018/single_model_package"
OUT = ROOT / "deliverables/convnext_tiny_single_20261018"
sys.path.insert(0, str(ROOT / "vendor/convnext_single_runtime_20261018"))

import numpy as np
import torch
import torch.nn.functional as F
import timm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("Invalid inference output or batch size")
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "submission.csv").exists():
        raise FileExistsError("Preserve the existing submission; inspect its receipt before rerunning")
    assert timm.__version__ == "1.0.30"
    with (ROOT / "features/unlabeled.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7065 and len({r["ID"] for r in rows}) == 7065
    folder = Path(rows[0]["path"]).parent
    assert all(Path(r["path"]).parent == folder and Path(r["path"]).name == r["ID"] for r in rows)
    with (ROOT / "data/raw/submission.csv").open(encoding="utf-8-sig", newline="") as f:
        template = list(csv.DictReader(f))
    assert [r["ID"] for r in rows] == [r["ID"] for r in template]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(sha(row["path"]).encode("ascii"))
    assert digest.hexdigest() == "02f2044f158f457ab7384fd6d679ca63596a8cebbc945ca554a4cb2979819ad4"
    create = timm.create_model
    parts, model_info = [], {}

    def capture(module, inputs, logits):
        # Native code performs this same FP32 softmax inside its autocast context.
        parts.append(F.softmax(logits.float(), dim=1).detach().cpu().numpy())

    def instrumented_factory(*args, **kwargs):
        model = create(*args, **kwargs)
        model_info.update(parameters=sum(p.numel() for p in model.parameters()),
                          architecture=str(args[0]), num_classes=6)
        model.register_forward_hook(capture)
        return model

    timm.create_model = instrumented_factory
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    sys.argv = [str(PACKAGE / "predict.py"), "--test-dir", str(folder),
                "--checkpoint", str(PACKAGE / "convnext_tiny_fold0.pt"),
                "--template", str(ROOT / "data/raw/submission.csv"),
                "--out", str(OUT / "submission.csv"), "--device", "cuda",
                "--precision", "bf16", "--batch-size", str(args.batch_size)]
    start = time.perf_counter()
    try:
        runpy.run_path(str(PACKAGE / "predict.py"), run_name="__main__")
    finally:
        timm.create_model = create
    seconds = time.perf_counter() - start
    a, b = np.concatenate(parts[0::2]), np.concatenate(parts[1::2])
    probabilities = (a + b) / 2
    assert probabilities.shape == (7065, 6) and np.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
    with (OUT / "submission.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["ID", "Emotion"]
        exported = list(reader)
    assert [r["ID"] for r in exported] == [r["ID"] for r in rows]
    labels = [r["Emotion"] for r in exported]
    assert set(labels) <= set("012345") and list(map(int, labels)) == probabilities.argmax(1).tolist()
    for name, array in [("original.npy", a), ("flip.npy", b), ("probabilities.npy", probabilities)]:
        np.save(OUT / name, array, allow_pickle=False)
    # The author reference CSV is sorted by file name; compare in that order too.
    ordered = sorted(exported, key=lambda r: r["ID"])
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["ID", "Emotion"])
    writer.writerows((r["ID"], r["Emotion"]) for r in ordered)
    author_order_sha = hashlib.sha256(stream.getvalue().encode()).hexdigest()
    metadata = {"state": "verified", "rows": 7065, "template_order_exact": True,
                "probability_argmax_exact": True, "classes": ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                **model_info, "torch": torch.__version__, "timm": timm.__version__,
                "gpu": torch.cuda.get_device_name(0), "precision": "cuda_bfloat16", "batch_size": args.batch_size,
                "elapsed_seconds_including_model_load_and_probability_capture": seconds,
                "images_per_second": 7065 / seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_sha256": sha(PACKAGE / "convnext_tiny_fold0.pt"),
                "native_script_sha256": sha(PACKAGE / "predict.py"),
                "runner_sha256": sha(__file__), "submission_sha256": sha(OUT / "submission.csv"),
                "probabilities_sha256": sha(OUT / "probabilities.npy"),
                "competition_image_content_sha256": digest.hexdigest(),
                "package_archive_sha256": "3d80fdb35deb4920ef7bac67257a88f839115ca9ce11db864cedf0331a88c007",
                "author_reported_validation_accuracy": .7222,
                "author_accuracy_independently_reproduced": False,
                "author_reference_sorted_csv_sha256": "0d3f325b60777ee63ae51de7b365f9392c2a1a476496b26ffebaaa7862b35ac3",
                "local_sorted_csv_sha256": author_order_sha,
                "author_reference_byte_match_after_sorting": author_order_sha == "0d3f325b60777ee63ae51de7b365f9392c2a1a476496b26ffebaaa7862b35ac3",
                "training_performed": False, "V11_resumed": False, "internal_test_read": False,
                "external_answers_used": False, "kaggle_uploaded": False}
    (OUT / "verification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
