"""Fit a leakage-aware six-class linear probe on cached frozen DINOv2 features.

Example:
    python train_classifier.py --features-dir ../features --output ../results

The feature row order must match manifest.csv / unlabeled.csv exactly. The
extractor's metadata.json must declare classes and SHA256 hashes for every input.
No images, test labels, or unlabeled labels are used to select hyperparameters.
"""
from __future__ import annotations

import os

# Set these before importing NumPy/SciPy; threadpoolctl also limits loaded BLAS.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LABELS = np.arange(6)
INPUT_FILES = ["manifest.csv", "unlabeled.csv", "labeled_original.npy", "labeled_flip.npy",
               "unlabeled_original.npy", "unlabeled_flip.npy"]
VERSION = "1.0.0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path, required):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required CSV columns: {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def canonical_path(value):
    # pathlib resolves Windows drive paths on the intended Windows environment.
    return os.path.normcase(str(Path(value).expanduser().resolve()))


def parse_label(value):
    value = str(value).strip()
    if value in CLASSES:
        return CLASSES.index(value)
    if value in {str(i) for i in range(6)}:
        return int(value)
    raise ValueError(f"Unknown label {value!r}; use official class names or integers 0..5")


def load_inputs(directory):
    directory = Path(directory).resolve()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("classes") != CLASSES:
        raise ValueError(f"metadata.classes must use OFFICIAL order: {CLASSES}")
    hashes = metadata.get("file_sha256", {})
    actual_hashes = {}
    for name in INPUT_FILES:
        expected = hashes.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"metadata.file_sha256 lacks a valid SHA256 for {name}")
        actual_hashes[name] = sha256(directory / name)
        if actual_hashes[name] != expected.lower():
            raise ValueError(f"Feature cache integrity check failed: {name}")
    rows = read_csv(directory / "manifest.csv", ["path", "label", "split"])
    unlabeled = read_csv(directory / "unlabeled.csv", ["ID", "path"])
    labeled_paths = [canonical_path(row["path"]) for row in rows]
    unlabeled_paths = [canonical_path(row["path"]) for row in unlabeled]
    if any(not row["path"].strip() for row in rows + unlabeled):
        raise ValueError("Image paths must be nonempty")
    if len(set(labeled_paths)) != len(labeled_paths):
        raise ValueError("Duplicate labeled image paths; fix the manifest before training")
    if set(labeled_paths) & set(unlabeled_paths):
        raise ValueError("Labeled and competition-test image paths overlap")
    ids = [row["ID"] for row in unlabeled]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unlabeled submission IDs must be nonempty and unique")
    y = np.asarray([parse_label(row["label"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"].strip().lower() for row in rows])
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("manifest split must contain exactly nonempty train, val, test partitions")
    indices = {name: np.flatnonzero(splits == name) for name in ("train", "val", "test")}
    for name in ("train", "val"):
        if set(y[indices[name]]) != set(LABELS):
            raise ValueError(f"{name} must contain all six classes")
    person_splits = {}
    for row, split in zip(rows, splits):
        person = row.get("subject_id", "").strip()
        if person:
            person_splits.setdefault(person, set()).add(str(split))
    if any(len(partitions) > 1 for partitions in person_splits.values()):
        raise ValueError("A subject_id appears in different data splits")
    features = {}
    feature_dim = int(metadata.get("feature_dim", 1536))
    if feature_dim <= 0:
        raise ValueError("metadata.feature_dim must be positive")
    for name in INPUT_FILES[2:]:
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        count = len(unlabeled) if name.startswith("unlabeled_") else len(rows)
        if array.shape != (count, feature_dim) or array.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{name}: expected floating shape {(count, feature_dim)}, got {array.shape}/{array.dtype}")
        for start in range(0, count, 1024):
            if not np.isfinite(array[start:start + 1024]).all():
                raise ValueError(f"{name}: contains NaN/Inf")
        features[name.removesuffix(".npy")] = array
    actual_hashes["metadata.json"] = sha256(metadata_path)
    return metadata, actual_hashes, rows, unlabeled, y, indices, features


def augmented_scaled(features, y, indices):
    # Advanced indexing/concatenation creates writable arrays, never changes mmap.
    x = np.concatenate([features["labeled_original"][indices], features["labeled_flip"][indices]], axis=0)
    target = np.concatenate([y[indices], y[indices]])
    scaler = StandardScaler(copy=False)
    x = scaler.fit_transform(x)
    return scaler, x, target


def fit_linear(x, target, c, weight, max_iter, tol):
    classifier = LogisticRegression(C=c, class_weight=weight, solver="lbfgs", max_iter=max_iter,
                                    tol=tol, random_state=42)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x, target)
    messages = [str(item.message) for item in caught]
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    for message in messages:
        print("WARNING:", message, flush=True)
    if list(classifier.classes_) != list(LABELS):
        raise RuntimeError("Classifier class order does not match the official mapping")
    info = {"fit_seconds": time.perf_counter() - started,
            "n_iter": int(np.max(classifier.n_iter_)), "converged": not convergence_warnings,
            "warnings": messages}
    return classifier, info


def metrics(y, probabilities):
    prediction = probabilities.argmax(axis=1)
    return {"macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, prediction)),
            "log_loss": float(log_loss(y, probabilities, labels=LABELS))}


def rank_key(result, primary):
    secondary = "accuracy" if primary == "macro_f1" else "macro_f1"
    return result[primary], result[secondary], -result["log_loss"]


def predict_chunked(pipeline, original, flipped, indices=None, tta=False, chunk=1024):
    count = len(original) if indices is None else len(indices)
    probabilities = np.empty((count, 6), dtype=np.float64)
    for start in range(0, count, chunk):
        selector = slice(start, min(start + chunk, count)) if indices is None else indices[start:start + chunk]
        # Pipeline's scaler uses copy=False; give it an owned copy to protect cache.
        current = pipeline.predict_proba(np.array(original[selector], copy=True))
        if tta:
            current = (current + pipeline.predict_proba(np.array(flipped[selector], copy=True))) / 2
        probabilities[start:start + len(current)] = current
    return probabilities


def export_report(output, name, y, probabilities, source_rows):
    prediction = probabilities.argmax(axis=1)
    report = {"sample_count": len(y), "metrics": metrics(y, probabilities), "classes": CLASSES,
              "classification_report": classification_report(y, prediction, labels=LABELS,
                                                               target_names=CLASSES, output_dict=True,
                                                               zero_division=0)}
    write_json(output / f"{name}_report.json", report)
    matrix = confusion_matrix(y, prediction, labels=LABELS)
    write_csv(output / f"{name}_confusion.csv",
              [dict(true_label=CLASSES[i], **{CLASSES[j]: int(matrix[i, j]) for j in range(6)}) for i in range(6)],
              ["true_label"] + CLASSES)
    prediction_rows = []
    for row, truth, predicted, probability in zip(source_rows, y, prediction, probabilities):
        prediction_rows.append(dict(path=row["path"], true_label=int(truth), predicted_label=int(predicted),
                                    true_emotion=CLASSES[int(truth)], predicted_emotion=CLASSES[int(predicted)],
                                    **{f"p_{label}": float(probability[i]) for i, label in enumerate(CLASSES)}))
    write_csv(output / f"{name}_predictions.csv", prediction_rows,
              ["path", "true_label", "predicted_label", "true_emotion", "predicted_emotion"] +
              [f"p_{label}" for label in CLASSES])
    return report


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selected_config.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("This results folder already contains a locked/final run. Use a new --output folder.")
    print("Verifying feature cache and split integrity...", flush=True)
    metadata, hashes, rows, unlabeled, y, split, features = load_inputs(args.features_dir)
    write_json(output / "run_inputs.json", {"version": VERSION, "features_dir": str(Path(args.features_dir).resolve()),
               "input_sha256": hashes, "classes": CLASSES, "selection_metric": args.selection_metric,
               "max_iter": args.max_iter, "tol": args.tol, "feature_metadata": metadata,
               "split_counts": {name: len(indices) for name, indices in split.items()}})
    scaler, x_train, y_train = augmented_scaled(features, y, split["train"])
    # Copy inputs before transforming because scaler.copy=False.
    val_original = scaler.transform(np.array(features["labeled_original"][split["val"]], copy=True))
    val_flip = scaler.transform(np.array(features["labeled_flip"][split["val"]], copy=True))
    search = []
    best = None
    best_probabilities = None
    warnings_by_fit = []
    fields = ["C", "class_weight", "tta_flip", "macro_f1", "accuracy", "log_loss", "n_iter", "converged", "fit_seconds"]
    with threadpool_limits(limits=4):
        for fit_number, (c, weight) in enumerate([(c, weight) for c in (0.01, 0.1, 1.0, 10.0)
                                                 for weight in (None, "balanced")], start=1):
            print(f"Search fit {fit_number}/8: C={c}, class_weight={weight}; {len(y_train)} augmented rows", flush=True)
            classifier, fit_info = fit_linear(x_train, y_train, c, weight, args.max_iter, args.tol)
            original_probability = classifier.predict_proba(val_original)
            flip_probability = classifier.predict_proba(val_flip)
            warnings_by_fit.append({"C": c, "class_weight": weight, **fit_info})
            for tta in (False, True):
                probability = (original_probability + flip_probability) / 2 if tta else original_probability
                result = {"C": c, "class_weight": weight, "tta_flip": tta,
                          **metrics(y[split["val"]], probability),
                          **{key: fit_info[key] for key in ("n_iter", "converged", "fit_seconds")}}
                search.append(result)
                if best is None or rank_key(result, args.selection_metric) > rank_key(best, args.selection_metric):
                    best = dict(result)
                    best_probabilities = probability.copy()
            write_csv(output / "search.csv", search, fields)
            write_json(output / "search_fit_details.json", warnings_by_fit)
            print(json.dumps({"search_fit": fit_number, "best_so_far": best}, ensure_ascii=False), flush=True)
            del classifier
        assert best is not None and best_probabilities is not None
        val_report = export_report(output, "val", y[split["val"]], best_probabilities,
                                   [rows[i] for i in split["val"]])
        # Freeze every modeling choice BEFORE refitting or reading held-out metrics.
        locked = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "classes": CLASSES, "selection_metric": args.selection_metric, "selected": best,
                  "max_iter": args.max_iter, "tol": args.tol, "solver": "lbfgs", "threads": 4,
                  "training_augmentation": "original plus horizontal-flip frozen feature rows",
                  "final_training_splits": ["train", "val"], "heldout_split": "test",
                  "input_sha256": hashes, "test_used_for_selection": False,
                  "feature_backbone_updated": False}
        write_json(output / "selected_config.json", locked)
        del x_train, y_train, scaler, val_original, val_flip
        final_indices = np.sort(np.concatenate([split["train"], split["val"]]))
        print(f"Refitting locked configuration on train+val ({len(final_indices)} unique images)...", flush=True)
        scaler, x_final, y_final = augmented_scaled(features, y, final_indices)
        classifier, final_fit_info = fit_linear(x_final, y_final, best["C"], best["class_weight"],
                                               args.max_iter, args.tol)
        del x_final, y_final
        # Inference scaler copy=True allows normal read-only arrays in saved Pipeline.
        scaler.copy = True
        pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
        pipeline.emotion_classes_ = CLASSES.copy()
        pipeline.tta_flip_ = best["tta_flip"]
        pipeline.feature_metadata_ = metadata
        pipeline.selected_config_ = locked
        model_path = output / "classifier.joblib"
        temporary_model = output / "classifier.joblib.tmp"
        joblib.dump(pipeline, temporary_model, compress=3)
        os.replace(temporary_model, model_path)
        write_json(output / "final_fit.json", final_fit_info)
        print("Evaluating the locked final classifier on held-out test once...", flush=True)
        test_probability = predict_chunked(pipeline, features["labeled_original"], features["labeled_flip"],
                                           split["test"], best["tta_flip"])
        test_report = export_report(output, "test", y[split["test"]], test_probability,
                                    [rows[i] for i in split["test"]])
        print("Generating competition predictions in original template ID order...", flush=True)
        submission_probability = predict_chunked(pipeline, features["unlabeled_original"],
                                                 features["unlabeled_flip"], tta=best["tta_flip"])
        predictions = submission_probability.argmax(axis=1)
        submission_rows = [{"ID": row["ID"], "Emotion": int(label)} for row, label in zip(unlabeled, predictions)]
        write_csv(output / "submission.csv", submission_rows, ["ID", "Emotion"])
        np.save(output / "submission_probabilities.npy", submission_probability.astype(np.float32), allow_pickle=False)
    # Read the written CSV back to ensure columns, order, count and class encoding.
    written = read_csv(output / "submission.csv", ["ID", "Emotion"])
    if [row["ID"] for row in written] != [row["ID"] for row in unlabeled] or len(written) != len(unlabeled):
        raise RuntimeError("Submission ID order/count validation failed")
    if any(row["Emotion"] not in {str(i) for i in range(6)} for row in written):
        raise RuntimeError("Submission label encoding validation failed")
    summary = {"version": VERSION, "status": "complete", "classes": CLASSES, "selected": best,
               "selection_metric": args.selection_metric, "split_counts": {key: len(value) for key, value in split.items()},
               "final_training_images": len(final_indices), "final_training_augmented_rows": 2 * len(final_indices),
               "val_metrics": val_report["metrics"], "test_metrics": test_report["metrics"],
               "validation_note": "Hyperparameters/TTA selected on val; val metrics are a selection estimate.",
               "test_note": "One held-out evaluation after locking choices; submission uses this same train+val classifier.",
               "subject_independent": bool(all(row.get("subject_id", "").strip() for row in rows)),
               "final_fit": final_fit_info, "search_fits": 8, "search_variants_including_tta": 16,
               "submission_rows": len(written), "submission_path": str(output / "submission.csv"),
               "model_path": str(model_path), "model_sha256": sha256(model_path),
               "submission_sha256": sha256(output / "submission.csv"), "input_sha256": hashes,
               "elapsed_seconds": time.perf_counter() - started,
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.max_iter < 1 or not 0 < args.tol < 1:
        parser.error("--max-iter must be positive; --tol must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
