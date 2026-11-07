"""Train a small nonlinear head on frozen DINOv2 features; validation only.

The model accepts RAW cached features and embeds train-only normalization.
No held-out test metrics or competition predictions are generated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import train_classifier as linear

CLASSES = linear.CLASSES.copy()
VERSION = "1.0.0"
DEFAULT_PROJECT = Path(__file__).resolve().parents[1]


class FeatureMLP(nn.Module):
    """A six-class head with saved normalization; forward accepts raw features."""

    def __init__(self, feature_dim=1536, hidden_dim=256, dropout=0.3, mean=None, scale=None):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim) if mean is None
                             else torch.as_tensor(np.array(mean, copy=True), dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim) if scale is None
                             else torch.as_tensor(np.array(scale, copy=True), dtype=torch.float32))
        if self.feature_mean.shape != (self.feature_dim,) or self.feature_scale.shape != (self.feature_dim,):
            raise ValueError("Normalization dimension mismatch")
        if not torch.isfinite(self.feature_mean).all() or not torch.isfinite(self.feature_scale).all() or (self.feature_scale <= 0).any():
            raise ValueError("Normalization must be finite with positive scales")
        self.head = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
                                  nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden_dim, 6))

    def forward(self, raw_features):
        return self.head((raw_features.float() - self.feature_mean) / self.feature_scale)


def load_mlp_checkpoint(path, device="cpu"):
    """Return (model, checkpoint_metadata); model is in eval mode on device."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "dinov2_frozen_feature_mlp" or checkpoint.get("classes") != CLASSES:
        raise ValueError("Checkpoint type/classes do not match this six-class MLP")
    architecture = checkpoint["architecture"]
    model = FeatureMLP(**architecture)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not torch.isfinite(model.feature_mean).all() or not torch.isfinite(model.feature_scale).all() or (model.feature_scale <= 0).any():
        raise ValueError("Checkpoint contains invalid normalization")
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_mlp(model, features_original, indices=None, batch_size=1024):
    """Predict probabilities for raw feature rows in exactly the supplied order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if features_original.ndim != 2 or features_original.shape[1] != model.feature_dim:
        raise ValueError("Prediction feature dimension mismatch")
    if indices is None:
        indices = np.arange(len(features_original), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(features_original)):
        raise ValueError("Invalid feature indices")
    device = next(model.parameters()).device
    model.eval()
    probabilities = np.empty((len(indices), 6), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        x = torch.from_numpy(np.array(features_original[rows], dtype=np.float32, copy=True)).to(device)
        current = torch.softmax(model(x), dim=1).cpu().numpy()
        if not np.isfinite(current).all():
            raise RuntimeError("MLP produced nonfinite probabilities")
        probabilities[start:start + len(rows)] = current
    return probabilities


def fit_train_scaler(original, flipped, train_indices, chunk_size=2048):
    """Compute population statistics from both train views only, without writes."""
    scaler = StandardScaler()
    for array in (original, flipped):
        for start in range(0, len(train_indices), chunk_size):
            scaler.partial_fit(array[train_indices[start:start + chunk_size]])
    return scaler


def save_checkpoint(path, content):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


def ranked(metrics, primary="accuracy"):
    secondary = "macro_f1" if primary == "accuracy" else "accuracy"
    return metrics[primary], metrics[secondary], -metrics["log_loss"]


def signature_for(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def run(args):
    project = Path(args.project_root).resolve()
    output = Path(args.output).resolve()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, patience, batch-size and lr must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly set --device cpu for a CPU run")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    print("Verifying frozen feature cache for MLP training...", flush=True)
    metadata, hashes, manifest, _, y, splits, features = linear.load_inputs(project / "features")
    if metadata.get("frozen") is not True:
        raise ValueError("MLP requires a cache explicitly marked frozen=true")
    train_indices, val_indices = splits["train"], splits["val"]
    config = {"version": VERSION, "seed": 42, "architecture": {"feature_dim": int(metadata.get("feature_dim", 1536)),
              "hidden_dim": 256, "dropout": 0.3}, "epochs": args.epochs, "patience": args.patience,
              "batch_size": args.batch_size, "learning_rate": args.lr, "weight_decay": 1e-3,
              "optimizer": "AdamW", "gradient_clip": 5.0, "device": str(args.device),
              "selection_metric": args.selection_metric,
              "secondary_metric": "macro_f1" if args.selection_metric == "accuracy" else "accuracy",
              "final_tie_break": "lower_log_loss", "validation_tta": "original only",
              "train_views": "random original or flip independently for each train image each epoch",
              "normalization": "StandardScaler fit on original+flip TRAIN rows only",
              "training_splits": ["train"], "selection_splits": ["val"], "test_evaluation": False,
              "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(),
              "classes": CLASSES, "input_sha256": hashes,
              "source_sha256": linear.sha256(__file__), "loader_source_sha256": linear.sha256(linear.__file__),
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    signature = signature_for(config)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or summary.get("run_signature") != signature:
            raise RuntimeError("Existing output has a different configuration, source, or cache. Use a new --output directory.")
        for name, digest in summary["artifact_sha256"].items():
            if not (output / name).is_file() or linear.sha256(output / name) != digest:
                raise RuntimeError(f"Completed MLP artifact integrity check failed: {name}")
        print("Reused completed MLP run after input/config/source/artifact verification.", flush=True)
        return summary
    if (output / "config.json").exists() or (output / "best.pt").exists():
        raise RuntimeError("Incomplete MLP run exists; use a new output directory to restart. Optimizer resume is not implemented.")
    linear.write_json(output / "config.json", {**config, "run_signature": signature})
    started = time.perf_counter()
    original, flipped = features["labeled_original"], features["labeled_flip"]
    scaler = fit_train_scaler(original, flipped, train_indices)
    model = FeatureMLP(**config["architecture"], mean=scaler.mean_, scale=scaler.scale_).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(42)
    history, best, best_epoch, bad_epochs = [], None, 0, 0
    history_fields = ["epoch", "train_loss", "train_accuracy", "val_macro_f1", "val_accuracy", "val_loss",
                      "improved", "best_epoch", "epochs_without_improvement", "epoch_seconds", "elapsed_seconds"]
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(train_indices)
        loss_sum, correct, seen = 0.0, 0, 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            raw = np.array(original[indices], dtype=np.float32, copy=True)
            flip_mask = rng.random(len(indices)) < 0.5
            raw[flip_mask] = flipped[indices[flip_mask]]
            x = torch.from_numpy(raw).to(args.device)
            target = torch.from_numpy(y[indices].copy()).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite MLP training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            correct += int((logits.detach().argmax(dim=1) == target).sum())
            seen += len(indices)
        val_original = predict_mlp(model, original, val_indices, max(args.batch_size, 512))
        score = linear.metrics(y[val_indices], val_original)
        improved = best is None or ranked(score, args.selection_metric) > ranked(best, args.selection_metric)
        if improved:
            best, best_epoch, bad_epochs = dict(score), epoch, 0
            checkpoint = {"format_version": 1, "model_type": "dinov2_frozen_feature_mlp", "classes": CLASSES,
                          "architecture": config["architecture"], "model_state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}, "epoch": epoch, "val_metrics": best,
                          "config": config, "run_signature": signature,
                          "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(),
                          "train_paths": [manifest[i]["path"] for i in train_indices],
                          "val_paths": [manifest[i]["path"] for i in val_indices],
                          "manifest_sha256": hashes["manifest.csv"], "feature_metadata": metadata,
                          "backbone_frozen": True, "test_evaluated": False}
            save_checkpoint(output / "best.pt", checkpoint)
        else:
            bad_epochs += 1
        record = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                  "val_macro_f1": score["macro_f1"], "val_accuracy": score["accuracy"], "val_loss": score["log_loss"],
                  "improved": improved, "best_epoch": best_epoch, "epochs_without_improvement": bad_epochs,
                  "epoch_seconds": time.perf_counter() - epoch_started, "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        linear.write_csv(output / "history.csv", history, history_fields)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs; restoring epoch {best_epoch}.", flush=True)
            break
    del optimizer, model
    model, checkpoint = load_mlp_checkpoint(output / "best.pt", args.device)
    val_original = predict_mlp(model, original, val_indices, max(args.batch_size, 512))
    val_flip = predict_mlp(model, flipped, val_indices, max(args.batch_size, 512))
    probability = val_original
    restored_score = linear.metrics(y[val_indices], probability)
    if not all(np.isclose(restored_score[k], best[k], atol=1e-7, rtol=1e-6) for k in best):
        raise RuntimeError("Restored checkpoint validation metrics differ from selected best epoch")
    for name, values in [("val_original.npy", val_original), ("val_flip.npy", val_flip), ("val_predictions.npy", probability)]:
        np.save(output / name, values.astype(np.float32), allow_pickle=False)
    prediction_rows = [{"manifest_index": int(i), "path": manifest[i]["path"], "true_label": int(y[i]),
                        "predicted_label": int(p.argmax()), **{f"p_{name}": float(p[c]) for c, name in enumerate(CLASSES)}}
                       for i, p in zip(val_indices, probability)]
    linear.write_csv(output / "val_predictions.csv", prediction_rows,
                     ["manifest_index", "path", "true_label", "predicted_label"] + [f"p_{name}" for name in CLASSES])
    linear.write_json(output / "val_metadata.json", {"classes": CLASSES, "manifest_indices": val_indices.tolist(),
                      "paths": [manifest[i]["path"] for i in val_indices], "manifest_sha256": hashes["manifest.csv"],
                      "prediction_type": "original only; flip probabilities supplied separately", "best_epoch": best_epoch,
                      "best_checkpoint_sha256": linear.sha256(output / "best.pt"), "run_signature": signature})
    artifacts = ["config.json", "best.pt", "history.csv", "val_original.npy", "val_flip.npy",
                 "val_predictions.npy", "val_predictions.csv", "val_metadata.json"]
    summary = {"status": "complete", "version": VERSION, "run_signature": signature, "classes": CLASSES,
               "architecture": config["architecture"], "train_samples": len(train_indices), "val_samples": len(val_indices),
               "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(),
               "epochs_completed": len(history), "best_epoch": best_epoch, "best_val_metrics": restored_score,
               "early_stopped": bad_epochs >= args.patience, "selection_metric": args.selection_metric, "tta_flip": False,
               "test_evaluated": False, "competition_predictions_generated": False,
               "note": "Best validation score selected on val; it is not an independent generalization estimate.",
               "elapsed_seconds": time.perf_counter() - started, "checkpoint_path": str(output / "best.pt"),
               "artifact_sha256": {name: linear.sha256(output / name) for name in artifacts}}
    linear.write_json(summary_path, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("train_indices", "val_indices", "artifact_sha256")},
                     ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
