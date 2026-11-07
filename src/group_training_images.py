"""Label-blind near-duplicate groups and three folds, using TRAIN images only.

This script never opens images from val, test, or competition rows. Groups are
connected components of conservative pixel-verified pHash candidates, including
horizontal flips; they are not person identities. Cross-label groups are retained
without changing labels. Completed output is reused only after source/data hashes
and its CSV are verified. A mismatched or incomplete output is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
import scipy
import sklearn
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
FIELDS = ("manifest_index", "path", "label", "target", "group", "fold")
PARAMETERS = {
    "image_size": [48, 48], "pixel_conversion": "PIL convert L; no resize of source pixels",
    "phash_bits": 64, "phash_resize": [32, 32], "phash_resize_filter": "PIL LANCZOS",
    "phash_dct": "scipy.fft.dctn type=2 norm=ortho; top-left 8x8; threshold at all-64 median, including DC",
    "views": ["original", "horizontal_flip"], "hamming_max": 4,
    "pixel_mae_max": 8 / 255, "pixel_mae_max_255": 8,
    "centered_correlation_min": 0.995, "exact_pixels_bypass": True,
    "candidate_segments": [13, 13, 13, 13, 12],
    "grouping": "label-blind connected components of verified edges",
    "fold_method": "StratifiedGroupKFold", "folds": 3, "shuffle": True, "random_state": 42,
    "classes": CLASSES,
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self):
        self.last = 0.0
        self.stage = None

    def log(self, stage, done=None, total=None, detail="", force=False):
        now = time.monotonic()
        if force or stage != self.stage or now - self.last >= 30:
            count = "" if done is None else f" {done}/{total}"
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {stage}{count} {detail}".rstrip(), flush=True)
            self.last, self.stage = now, stage


def read_training_rows(root):
    manifest = root / "features" / "manifest.csv"
    records, seen = [], set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "label", "split"}.issubset(reader.fieldnames):
            raise ValueError("features/manifest.csv needs path,label,split columns")
        for manifest_index, row in enumerate(reader):
            # No path, label, or image processing is performed on other splits.
            if row.get("split") != "train":
                continue
            path, label = row.get("path") or "", row.get("label") or ""
            if not path or label not in CLASSES:
                raise ValueError(f"Invalid TRAIN path/label at manifest_index={manifest_index}")
            key = path.replace("\\", "/").casefold()
            if key in seen:
                raise ValueError(f"Repeated TRAIN path: {path}")
            seen.add(key)
            records.append({"manifest_index": manifest_index, "path": path, "label": label,
                            "target": CLASSES.index(label)})
    if not records or {row["label"] for row in records} != set(CLASSES):
        raise ValueError("TRAIN rows must contain all six established project classes")
    return records, file_sha(manifest), canonical_sha(records)


def training_content(records, root, decode, progress):
    """Hash only selected TRAIN files; fresh runs decode the same bytes once."""
    digest = hashlib.sha256()
    pixels = np.empty((len(records), 48, 48), dtype=np.uint8) if decode else None
    modes = Counter()
    stage = "TRAIN hash + pixel decode" if decode else "TRAIN content signature verification"
    progress.log(stage, 0, len(records))
    for i, row in enumerate(records):
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        # Manifest paths are already absolute in the project. No per-image
        # Path.resolve or directory scan is needed, including during reuse.
        raw = path.read_bytes()
        digest.update(hashlib.sha256(raw).hexdigest().encode("ascii"))
        if decode:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.size != (48, 48):
                    raise ValueError(f"TRAIN image must be 48x48: {row['path']} ({image.size})")
                modes[image.mode] += 1
                pixels[i] = np.asarray(image.convert("L"), dtype=np.uint8)
        progress.log(stage, i + 1, len(records))
    progress.log(stage, len(records), len(records), force=True)
    return digest.hexdigest(), pixels, dict(sorted(modes.items()))


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1
        return True


def phash64(pixels):
    resized = np.asarray(Image.fromarray(pixels).resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(resized, type=2, norm="ortho")[:8, :8]
    bits = low > np.median(low)
    return int.from_bytes(np.packbits(bits.reshape(-1), bitorder="big").tobytes(), "big")


def segment_keys(hashes):
    keys = set()
    for value in hashes:
        offset = 0
        for segment, width in enumerate(PARAMETERS["candidate_segments"]):
            keys.add((segment, (value >> offset) & ((1 << width) - 1)))
            offset += width
    return keys


def pixel_metrics(left, right):
    left_float, right_float = left.astype(np.float64), right.astype(np.float64)
    mae_255 = float(np.abs(left_float - right_float).mean())
    a, b = left_float - left_float.mean(), right_float - right_float.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    correlation = None if denominator == 0 else float(np.clip(np.vdot(a, b) / denominator, -1, 1))
    return mae_255, correlation


def verify_pair(left, right, left_hashes, right_hashes):
    """Purely pixel/hash-based decision; there are no labels in this function."""
    metrics = {}
    for flipped in (False, True):
        aligned = right[:, ::-1] if flipped else right
        if np.array_equal(left, aligned):
            right_view = int(flipped)
            _, correlation = pixel_metrics(left, aligned)
            return {"kind": "exact_pixels", "left_view": "original",
                    "right_view": "horizontal_flip" if flipped else "original",
                    "phash_hamming": (left_hashes[0] ^ right_hashes[right_view]).bit_count(),
                    "pixel_mae": 0.0, "pixel_mae_255": 0.0,
                    "centered_correlation": correlation, "exact_pixels_bypass": True}
    # Compare all view hashes. Flipping both images leaves pixel MAE/correlation
    # unchanged, so each relative alignment needs at most one metric evaluation.
    for left_view in (0, 1):
        for right_view in (0, 1):
            hamming = (left_hashes[left_view] ^ right_hashes[right_view]).bit_count()
            if hamming > PARAMETERS["hamming_max"]:
                continue
            flipped = left_view != right_view
            if flipped not in metrics:
                aligned = right[:, ::-1] if flipped else right
                metrics[flipped] = pixel_metrics(left, aligned)
            mae_255, correlation = metrics[flipped]
            if mae_255 <= PARAMETERS["pixel_mae_max_255"] and correlation is not None and correlation >= PARAMETERS["centered_correlation_min"]:
                return {"kind": "near_duplicate", "left_view": "horizontal_flip" if left_view else "original",
                        "right_view": "horizontal_flip" if right_view else "original", "phash_hamming": hamming,
                        "pixel_mae": mae_255 / 255, "pixel_mae_255": mae_255,
                        "centered_correlation": correlation, "exact_pixels_bypass": False}
    return None


def label_blind_groups(pixels, manifest_indices, progress):
    count = len(pixels)
    union = UnionFind(count)
    hashes, buckets, exact_representatives = [], defaultdict(list), {}
    evidence = []
    candidates = verified = skipped = exact_edges = near_edges = max_bucket = 0
    progress.log("Label-blind candidate grouping", 0, count)
    for i, image in enumerate(pixels):
        views = (phash64(image), phash64(image[:, ::-1]))
        hashes.append(views)
        canonical_pixels = min(image.tobytes(), image[:, ::-1].tobytes())
        exact_key = hashlib.sha256(canonical_pixels).digest()
        representative = exact_representatives.get(exact_key)
        if representative is not None:
            pair = verify_pair(pixels[representative], image, hashes[representative], views)
            if pair is None or pair["kind"] != "exact_pixels":
                raise RuntimeError("Canonical pixel hash collision or inconsistent exact-image comparison")
            changed = union.union(representative, i)
            evidence.append({"left_manifest_index": manifest_indices[representative], "right_manifest_index": manifest_indices[i],
                             **pair, "union_new_component": changed})
            exact_edges += 1
            # Equal images are interchangeable candidate representatives, which
            # avoids quadratic comparisons inside large exact-duplicate groups.
        else:
            keys = segment_keys(views)
            neighbors = set()
            for key in keys:
                neighbors.update(buckets[key])
            candidates += len(neighbors)
            for prior in sorted(neighbors):
                if union.find(prior) == union.find(i):
                    skipped += 1
                    continue
                if min((a ^ b).bit_count() for a in hashes[prior] for b in views) > PARAMETERS["hamming_max"]:
                    continue
                verified += 1
                pair = verify_pair(pixels[prior], image, hashes[prior], views)
                if pair is not None:
                    changed = union.union(prior, i)
                    evidence.append({"left_manifest_index": manifest_indices[prior], "right_manifest_index": manifest_indices[i],
                                     **pair, "union_new_component": changed})
                    exact_edges += pair["kind"] == "exact_pixels"
                    near_edges += pair["kind"] == "near_duplicate"
            exact_representatives[exact_key] = i
            for key in keys:
                buckets[key].append(i)
                max_bucket = max(max_bucket, len(buckets[key]))
        progress.log("Label-blind candidate grouping", i + 1, count,
                     f"candidate pairs={candidates}; accepted edges={len(evidence)}")
    components = defaultdict(list)
    for i in range(count):
        components[union.find(i)].append(i)
    ordered = sorted(components.values(), key=lambda members: min(manifest_indices[i] for i in members))
    groups = np.empty(count, dtype=np.int64)
    for group, members in enumerate(ordered):
        groups[members] = group
    progress.log("Label-blind candidate grouping", count, count,
                 f"groups={len(ordered)}; exact edges={exact_edges}; near edges={near_edges}", force=True)
    stats = {"candidate_image_pairs": candidates, "hamming_eligible_pairs_checked": verified,
             "already_connected_candidates_skipped": skipped, "exact_pixel_edges": int(exact_edges),
             "near_duplicate_edges": int(near_edges), "unique_canonical_pixel_images": len(exact_representatives),
             "largest_segment_bucket": max_bucket,
             "candidate_guarantee": "Hamming <=4 across five disjoint segments guarantees at least one matching segment. Both orientations are indexed.",
             "exact_duplicate_optimization": "Only one representative per exact pixel pattern up to horizontal flip is indexed."}
    return groups, ordered, evidence, stats


def make_folds(records, groups):
    targets = np.array([row["target"] for row in records], dtype=np.int64)
    for target, label in enumerate(CLASSES):
        if len(set(groups[targets == target])) < 3:
            raise ValueError(f"Class {label} has fewer than three distinct near-duplicate groups")
    fold = np.full(len(records), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for number, (train_ids, val_ids) in enumerate(splitter.split(np.zeros(len(records)), targets, groups)):
        if set(groups[train_ids]) & set(groups[val_ids]):
            raise RuntimeError("A near-duplicate group crosses a fold boundary")
        if set(targets[train_ids]) != set(range(6)) or set(targets[val_ids]) != set(range(6)):
            raise ValueError(f"Fixed StratifiedGroupKFold fold {number} does not contain all six classes")
        fold[val_ids] = number
    if np.any(fold < 0):
        raise RuntimeError("Unassigned fold")
    return fold


def verify_cached_csv(path, records, summary):
    if file_sha(path) != summary.get("train_groups_sha256"):
        raise ValueError("Completed train_groups.csv hash mismatch")
    with path.open(encoding="utf-8", newline="") as handle:
        cached = list(csv.DictReader(handle))
    if len(cached) != len(records):
        raise ValueError("Completed grouping row count mismatch")
    group_folds = defaultdict(set)
    class_counts = {str(fold): Counter() for fold in range(3)}
    for expected, row in zip(records, cached):
        if any(str(expected[key]) != row.get(key) for key in ("manifest_index", "path", "label", "target")):
            raise ValueError("Completed grouping manifest alignment mismatch")
        fold, group = int(row["fold"]), int(row["group"])
        if fold not in range(3) or group < 0:
            raise ValueError("Invalid cached group/fold")
        group_folds[group].add(fold)
        class_counts[str(fold)][row["label"]] += 1
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("Completed grouping contains cross-fold groups")
    actual_counts = {key: {label: counts[label] for label in CLASSES} for key, counts in class_counts.items()}
    if actual_counts != summary.get("fold_class_counts") or any(min(counts.values()) == 0 for counts in actual_counts.values()):
        raise ValueError("Completed grouping fold-class counts mismatch")


def execute(project_root, output):
    project_root, output = Path(project_root).resolve(), Path(output).resolve()
    if output == project_root or output.is_relative_to(project_root / "features") or output.is_relative_to(project_root / "data"):
        raise ValueError("Output must be separate from frozen input data/features")
    progress = Progress()
    records, source_manifest_sha, train_manifest_sha = read_training_rows(project_root)
    base = {"schema_version": 1, "source_manifest_sha256": source_manifest_sha,
            "train_manifest_sha256": train_manifest_sha, "grouping_script_sha256": file_sha(__file__),
            "parameters": PARAMETERS,
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pillow": PIL.__version__}}
    summary_path, csv_path = output / "groups_summary.json", output / "train_groups.csv"
    existing = output.exists() and any(output.iterdir())
    if existing:
        if not summary_path.is_file() or not csv_path.is_file():
            raise ValueError("Nonempty output lacks completed grouping; preserve it and choose another --output")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("complete") is not True:
            raise ValueError("Grouping output is incomplete; preserve it and choose another --output")
        for key, value in base.items():
            if previous.get(key) != value:
                raise ValueError(f"Completed grouping source/data/settings mismatch: {key}")
    content_sha, pixels, modes = training_content(records, project_root, decode=not existing, progress=progress)
    base["train_image_content_sha256"] = content_sha
    signature = canonical_sha(base)
    if existing:
        if previous.get("signature") != signature or previous.get("train_image_content_sha256") != content_sha:
            raise ValueError("TRAIN image bytes changed; completed grouping cannot be reused")
        verify_cached_csv(csv_path, records, previous)
        progress.log("Completed grouping reused after source/TRAIN-content/CSV verification", len(records), len(records), force=True)
        return previous
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        groups, components, evidence, stats = label_blind_groups(pixels, [r["manifest_index"] for r in records], progress)
        del pixels
        progress.log("Fixed three-fold stratified group allocation")
        folds = make_folds(records, groups)
        cross_label_groups = []
        for group, members in enumerate(components):
            counts = Counter(records[i]["label"] for i in members)
            if len(counts) > 1:
                cross_label_groups.append({"group": group, "size": len(members), "label_counts": dict(sorted(counts.items())),
                                           "manifest_indices": [records[i]["manifest_index"] for i in members]})
        output_rows = [{**row, "group": int(group), "fold": int(fold)} for row, group, fold in zip(records, groups, folds)]
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(csv_path)
        class_counts = {str(fold): {label: sum(row["fold"] == fold and row["label"] == label for row in output_rows)
                                   for label in CLASSES} for fold in range(3)}
        summary = {**base, "signature": signature, "complete": True, "train_rows": len(records),
                   "train_groups_sha256": file_sha(csv_path), "train_groups_csv": str(csv_path),
                   "manifest_index_definition": "Zero-based original manifest data-row index, excluding the header",
                   "train_manifest_sha256_definition": "SHA256 of compact UTF-8 canonical JSON train records (manifest_index,path,label,target), keys sorted, ensure_ascii=False",
                   "train_image_content_sha256_definition": "SHA256 of concatenated lowercase per-file SHA256 hex in original manifest TRAIN-row order",
                   "fold_definition": "fold 0/1/2 is the validation fold; the other two folds form training",
                   "source_image_mode_counts": modes, "group_count": len(components),
                   "group_size_histogram": dict(sorted(Counter(str(len(m)) for m in components).items(), key=lambda item: int(item[0]))),
                   "largest_groups": [{"group": group, "size": len(members), "manifest_indices": [records[i]["manifest_index"] for i in members]}
                                      for group, members in sorted(enumerate(components), key=lambda pair: (-len(pair[1]), pair[0]))[:20]],
                   "cross_label_group_count": len(cross_label_groups), "cross_label_groups": cross_label_groups,
                   "pair_evidence": evidence, "candidate_statistics": stats, "fold_class_counts": class_counts,
                   "fold_sizes": {str(fold): int(np.sum(folds == fold)) for fold in range(3)},
                   "fold_group_counts": {str(fold): len(set(groups[folds == fold])) for fold in range(3)},
                   "no_group_cross_fold": all(len(set(folds[members])) == 1 for members in components),
                   "label_changes": 0, "scope": "Only original manifest split=train images and labels",
                   "elapsed_grouping_seconds": time.monotonic() - started,
                   "limitations": ["No subject IDs exist; these are near-duplicate groups, not person-independent folds.",
                                   "Groups are transitive connected components; not every pair in a component must meet the edge thresholds.",
                                   "Conservative pHash/pixel thresholds can miss other related images or identities.",
                                   "Candidate hashing avoids all-image exhaustive pair enumeration; unusually large hash collisions can still increase candidate work."]}
        if not summary["no_group_cross_fold"]:
            raise RuntimeError("Internal group/fold isolation failure")
        write_json(summary_path, summary)
        verify_cached_csv(csv_path, records, summary)
        progress.log("TRAIN grouping complete", len(records), len(records),
                     f"groups={len(components)}; cross-label groups={len(cross_label_groups)}", force=True)
        return summary
    except Exception as error:
        write_json(summary_path, {**base, "signature": signature, "complete": False, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", help="Default: PROJECT/experiments/multilayer_groups")
    args = parser.parse_args()
    root = Path(args.project_root)
    output = Path(args.output) if args.output else root / "experiments" / "multilayer_groups"
    execute(root, output)


if __name__ == "__main__":
    main()
