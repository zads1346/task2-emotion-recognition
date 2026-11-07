#!/usr/bin/env python
"""Ingest a ZIP or read-only image directory without using template labels.

The source must contain one train/<class>/ image tree, its sibling test image
tree, and a submission.csv with ID,Emotion columns; nesting is allowed. ZIP mode
requires a new/empty extraction directory. Directory mode supports mounted Kaggle
datasets and never writes to source files. Output must always be new/empty.
Excluded labeled files are logically quarantined in quarantine.csv and never
enter labeled.csv.

Examples:
  python ingest_archive.py --archive task.zip --extract-dir raw --output audit
  python ingest_archive.py --data-dir /kaggle/input/task-data --output /kaggle/working/audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                    *[f"LPT{i}" for i in range(1, 10)]}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reserve_directory(path, option):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{option} must be a new or empty directory: {path}")
    return path


def safe_archive_target(info, extract_dir):
    name = info.filename.replace("\\", "/")
    relative = PurePosixPath(name)
    if not name or not relative.parts or relative.is_absolute() or PureWindowsPath(name).drive:
        raise ValueError(f"Unsafe absolute archive path: {info.filename!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe archive traversal path: {info.filename!r}")
    for part in relative.parts:
        if any(character in '<>:"|?*' or ord(character) < 32 for character in part) or part.rstrip(" .") != part:
            raise ValueError(f"Unsafe Windows archive path component: {info.filename!r}")
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED:
            raise ValueError(f"Reserved Windows archive path component: {info.filename!r}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Archive links and special files are not supported: {info.filename!r}")
    target = extract_dir.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(extract_dir):
        raise ValueError(f"Archive path escapes extraction directory: {info.filename!r}")
    return target


def extract_safely(archive, extract_dir, audit):
    audit["archive"] = {"path": str(archive), "bytes": archive.stat().st_size,
                        "sha256": sha256_file(archive)}
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        targets, seen = [], set()
        # Validate every entry before extracting any file, including otherwise
        # harmless non-image files. Case-folding catches Windows collisions.
        for info in entries:
            target = safe_archive_target(info, extract_dir)
            key = str(target).casefold()
            if key in seen:
                raise ValueError(f"Duplicate/colliding archive entries: {info.filename!r}")
            seen.add(key)
            targets.append((info, target))
        file_targets = {str(target).casefold() for info, target in targets if not info.is_dir()}
        for info, target in targets:
            for parent in target.parents:
                if parent == extract_dir:
                    break
                if str(parent).casefold() in file_targets:
                    raise ValueError(f"Archive file/directory path collision: {info.filename!r}")
        total_bytes = sum(info.file_size for info in entries if not info.is_dir())
        if total_bytes > shutil.disk_usage(extract_dir).free:
            raise ValueError("Insufficient free disk space for the archive's declared uncompressed size.")
        audit["archive"].update(entries=len(entries), files=sum(not info.is_dir() for info in entries),
                                uncompressed_bytes=total_bytes)
        for i, (info, target) in enumerate(targets, start=1):
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # 'xb' guarantees that neither earlier files nor archive entries
                # overwrite an existing file. ZipFile.open checks CRC on read.
                with bundle.open(info, "r") as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            if i % 10000 == 0:
                print(f"Extracted {i}/{len(targets)} archive entries", flush=True)
    audit["extraction_directory"] = str(extract_dir)


def discover_data(extract_dir, audit):
    train_roots = [p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() == "train"
                   and any(child.is_dir() for child in p.iterdir())]
    if len(train_roots) != 1:
        raise ValueError(f"Expected exactly one train/<class>/ tree; found {len(train_roots)}.")
    train_dir = train_roots[0]
    test_dirs = [p for p in train_dir.parent.iterdir() if p.is_dir() and p.name.lower() == "test"]
    templates = [p for p in extract_dir.rglob("*") if p.is_file() and p.name.lower() == "submission.csv"]
    if len(test_dirs) != 1 or len(templates) != 1:
        raise ValueError("Expected one test directory beside train and one submission.csv.")
    test_dir, template = test_dirs[0], templates[0]
    labeled = []
    class_dirs = sorted((p for p in train_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    for folder in class_dirs:
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No images found in class directory: {folder}")
        labeled.extend({"path": str(p.resolve()), "label": folder.name} for p in files)
    test_files = sorted(p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not labeled or not test_files:
        raise ValueError("Both labeled training images and unlabeled test images must be present.")
    by_name = {}
    for path in test_files:
        if path.name in by_name:
            raise ValueError(f"Ambiguous test image filename: {path.name}")
        by_name[path.name] = str(path.resolve())
    inference, template_values, ids = [], Counter(), set()
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"ID", "Emotion"}.issubset(reader.fieldnames):
            raise ValueError("submission.csv must contain ID and Emotion columns.")
        for line, row in enumerate(reader, start=2):
            identifier = (row.get("ID") or "").strip()
            if not identifier or identifier in ids:
                raise ValueError(f"submission.csv line {line}: empty or repeated ID.")
            if identifier not in by_name:
                raise ValueError(f"Submission image is unavailable: {identifier!r}")
            ids.add(identifier)
            inference.append({"ID": identifier, "path": by_name[identifier]})
            # Template values are counted for provenance only, never read as
            # image labels or used for model evaluation.
            template_values[(row.get("Emotion") or "").strip()] += 1
    if ids != set(by_name):
        raise ValueError(f"Test/template membership differs: {len(set(by_name) - ids)} test images have no submission ID.")
    audit["data_layout"] = {"train": str(train_dir), "test": str(test_dir), "template": str(template)}
    audit["input_labeled_samples"] = len(labeled)
    audit["input_labeled_class_counts"] = dict(sorted(Counter(r["label"] for r in labeled).items()))
    audit["input_unlabeled_test_samples"] = len(test_files)
    audit["submission_template"] = {"rows": len(inference), "column_names": ["ID", "Emotion"],
                                     "emotion_value_counts": dict(sorted(template_values.items())),
                                     "used_as_ground_truth": False,
                                     "id_order_sha256": hashlib.sha256("\n".join(r["ID"] for r in inference).encode("utf-8")).hexdigest()}
    return labeled, inference


def decode_and_hash(records, partition):
    valid, invalid = [], []
    for i, row in enumerate(records, start=1):
        path = Path(row["path"])
        if not path.is_file():
            raise ValueError(f"Extracted image is unavailable: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    raise ValueError("Image has no pixels.")
                width, height, mode = image.width, image.height, image.mode
                rgb = np.asarray(image.convert("RGB"))
                grayscale_pixels = bool(np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and
                                        np.array_equal(rgb[:, :, 1], rgb[:, :, 2]))
            valid.append({**row, "sha256": sha256_file(path), "width": width, "height": height,
                          "mode": mode, "grayscale_pixels": grayscale_pixels})
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            invalid.append({**row, "partition": partition, "reason": str(error)})
        if i % 5000 == 0:
            print(f"Decoded {i}/{len(records)} {partition} images", flush=True)
    return valid, invalid


def image_summary(records):
    if not records:
        return {"decodable_samples": 0}
    sizes = Counter(f"{r['width']}x{r['height']}" for r in records)
    grayscale = sum(r["grayscale_pixels"] for r in records)
    return {"decodable_samples": len(records), "unique_file_hashes": len({r["sha256"] for r in records}),
            "size_counts": dict(sorted(sizes.items())), "mode_counts": dict(sorted(Counter(r["mode"] for r in records).items())),
            "grayscale_by_pixels": grayscale, "grayscale_fraction": grayscale / len(records)}


def write_csv(path, records, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def ingest(args, data_root, output, audit):
    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        audit["data_source"] = {"kind": "archive", "path": str(archive),
                                "extraction_directory": str(data_root)}
        if not archive.is_file():
            raise ValueError(f"Archive is unavailable: {archive}")
        extract_safely(archive, data_root, audit)
    else:
        audit["data_source"] = {"kind": "directory", "path": str(data_root), "read_only_input": True}
    raw_labeled, inference = discover_data(data_root, audit)
    labeled, invalid_labeled = decode_and_hash(raw_labeled, "labeled")
    test, invalid_test = decode_and_hash(inference, "unlabeled_test")
    audit["invalid_images"] = {"labeled_count": len(invalid_labeled), "unlabeled_test_count": len(invalid_test),
                                "labeled_records": invalid_labeled, "unlabeled_test_records": invalid_test}
    audit["labeled_image_summary"] = image_summary(labeled)
    audit["unlabeled_test_image_summary"] = image_summary(test)
    if invalid_test:
        raise ValueError(f"{len(invalid_test)} submission images cannot be decoded. Prediction must cover every ID; correct these files first.")
    labeled_hashes, test_hashes = defaultdict(list), defaultdict(list)
    for row in labeled:
        labeled_hashes[row["sha256"]].append(row)
    for row in test:
        test_hashes[row["sha256"]].append(row)
    conflicts = {digest: rows for digest, rows in labeled_hashes.items() if len({r["label"] for r in rows}) > 1}
    overlap_hashes = set(labeled_hashes) & set(test_hashes)
    clean, quarantine = [], []
    for row in labeled:
        reasons = []
        if row["sha256"] in conflicts:
            reasons.append("conflicting_class_labels")
        if row["sha256"] in overlap_hashes:
            reasons.append("overlap_with_unlabeled_test")
        if reasons:
            quarantine.append({**row, "reasons": ";".join(reasons)})
        else:
            clean.append(row)
    quarantine.extend({**row, "sha256": "", "reasons": "invalid_image"} for row in invalid_labeled)
    missing_classes = set(audit["input_labeled_class_counts"]) - {r["label"] for r in clean}
    conflict_details = [{"sha256": digest, "files": [{"path": r["path"], "label": r["label"]} for r in rows]}
                        for digest, rows in sorted(conflicts.items())]
    overlap_details = [{"sha256": digest,
                        "labeled": [{"path": r["path"], "label": r["label"]} for r in labeled_hashes[digest]],
                        "unlabeled_test": [{"path": r["path"], "ID": r["ID"]} for r in test_hashes[digest]]}
                       for digest in sorted(overlap_hashes)]
    clean_hashes = defaultdict(list)
    for row in clean:
        clean_hashes[row["sha256"]].append(row)
    same_label_groups = {digest: rows for digest, rows in clean_hashes.items() if len(rows) > 1}
    audit["cleaning"] = {
        "policy": "Preserve raw source files. Exclude all conflicting-label hashes and all labeled hashes overlapping unlabeled test; never guess labels.",
        "conflicting_label_hash_groups": len(conflicts),
        "conflicting_label_files_excluded": sum(len(rows) for rows in conflicts.values()),
        "conflicting_label_details": conflict_details,
        "labeled_test_overlap_hash_groups": len(overlap_hashes),
        "labeled_test_overlap_files_excluded": sum(len(labeled_hashes[d]) for d in overlap_hashes),
        "labeled_test_overlap_details": overlap_details,
        "same_label_duplicate_groups_retained": len(same_label_groups),
        "same_label_duplicate_extra_files_retained": sum(len(rows) - 1 for rows in same_label_groups.values()),
        "duplicate_policy": "Same-label duplicates remain in labeled.csv; prepare_data.py removes them before splitting.",
        "same_label_duplicate_details": [{"sha256": digest, "label": rows[0]["label"], "paths": [r["path"] for r in rows]}
                                         for digest, rows in sorted(same_label_groups.items())],
        "test_duplicate_hash_groups": sum(len(rows) > 1 for rows in test_hashes.values()),
        "test_duplicate_extra_files": sum(max(0, len(rows) - 1) for rows in test_hashes.values()),
        "quarantined_labeled_files_total": len(quarantine),
        "usable_labeled_samples_before_same_label_deduplication": len(clean),
        "usable_unique_labeled_images": len(clean_hashes),
        "usable_labeled_class_counts": dict(sorted(Counter(r["label"] for r in clean).items())),
        "usable_unique_labeled_class_counts": dict(sorted(Counter(rows[0]["label"] for rows in clean_hashes.values()).items())),
        "clean_labeled_test_hash_overlap": len(set(clean_hashes) & set(test_hashes)),
        "clean_labeled_conflicting_hash_groups": sum(len({r["label"] for r in rows}) > 1 for rows in clean_hashes.values()),
    }
    if missing_classes:
        raise ValueError(f"Cleaning removes every valid image from these classes: {sorted(missing_classes)}")
    audit["limitations"] = [
        "Subject IDs are absent. Image splits cannot establish independence of people, cameras, or videos.",
        "Only byte-identical images are detected. Visually identical recompressed files and adjacent frames require additional review.",
        "submission.csv is an output template. Its Emotion values are not ground truth; the competition test cannot provide accuracy or F1.",
        "The numeric Emotion-to-class mapping must come from task instructions; this ingestion preserves string labels and does not guess that mapping.",
    ]
    destinations = {"labeled_csv": output / "labeled.csv", "inference_csv": output / "inference.csv",
                    "quarantine_csv": output / "quarantine.csv", "image_inventory_csv": output / "image_inventory.csv"}
    write_csv(destinations["labeled_csv"], clean, ("path", "label"))
    # The original submission order is preserved exactly, including duplicates
    # of image bytes under distinct valid IDs.
    write_csv(destinations["inference_csv"], inference, ("ID", "path"))
    write_csv(destinations["quarantine_csv"], quarantine, ("path", "label", "sha256", "reasons"))
    inventory = [{**r, "partition": "labeled"} for r in labeled] + [{**r, "partition": "unlabeled_test"} for r in test]
    write_csv(destinations["image_inventory_csv"], inventory,
              ("path", "partition", "label", "ID", "sha256", "width", "height", "mode", "grayscale_pixels"))
    audit["outputs"] = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in destinations.items()}
    audit["status"] = "success"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", help="ZIP archive to safely extract and ingest.")
    source.add_argument("--data-dir", help="Read-only dataset root, such as a mounted /kaggle/input dataset.")
    parser.add_argument("--extract-dir", help="New/empty extraction directory; required only with --archive.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.archive and not args.extract_dir:
        parser.error("--extract-dir is required with --archive.")
    if args.data_dir and args.extract_dir:
        parser.error("--extract-dir is only valid with --archive; --data-dir reads its source in place.")
    try:
        output = reserve_directory(args.output, "--output")
        if args.archive:
            data_root = reserve_directory(args.extract_dir, "--extract-dir")
        else:
            data_root = Path(args.data_dir).expanduser().resolve()
            if not data_root.is_dir():
                raise ValueError(f"Dataset directory is unavailable: {data_root}")
        if data_root == output or data_root.is_relative_to(output) or output.is_relative_to(data_root):
            raise ValueError("Source/extraction and output directories must be separate, without nesting.")
    except ValueError as error:
        parser.error(str(error))
    if args.archive:
        data_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "status": "running"}
    try:
        ingest(args, data_root, output, audit)
    except (OSError, ValueError, RuntimeError, csv.Error, zipfile.BadZipFile) as error:
        audit.update(status="error", error=str(error))
        (output / "archive_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        parser.exit(2, f"Archive ingestion failed: {error}\nSee {output / 'archive_audit.json'}\n")
    (output / "archive_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Labeled images: {audit['input_labeled_samples']}; unlabeled test: {audit['input_unlabeled_test_samples']}")
    print(f"Usable unique labeled images: {audit['cleaning']['usable_unique_labeled_images']}; "
          f"quarantined labeled files: {audit['cleaning']['quarantined_labeled_files_total']}")
    print(f"Wrote {output / 'archive_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
