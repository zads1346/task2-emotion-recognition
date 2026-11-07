"""Training-subset-only sample weights for fixed six-class expression targets.

Weights change loss contributions, not labels or sampling. The caller applies the
same returned image weight to that image's original and flipped views. Only the
entries selected by train_indices are inspected. Held-out values are deliberately
not validated or used; the vector dimensions/lengths are checked as metadata.
"""
from collections import Counter

import numpy as np


MODES = ("uniform", "class", "duplicate", "class_duplicate")
CLASS_COUNT = 6
CLASS_CLIP = (0.75, 1.5)
COMBINED_CLIP = (0.5, 1.5)


def _vector(value, name):
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional vector")
        return value
    if isinstance(value, (list, tuple)):
        # Do not coerce an entire vector: object/NaN values in held-out rows must
        # neither influence dtype inference nor enter any weight calculation.
        return value
    raise ValueError(f"{name} must be a one-dimensional numpy array, list, or tuple")


def _target(value, index):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Training target at index {index} must be an integer class ID 0..5")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"Training target at index {index} is invalid") from None
    if not np.isfinite(numeric) or not numeric.is_integer() or not 0 <= numeric < CLASS_COUNT:
        raise ValueError(f"Training target at index {index} must be a finite integer class ID 0..5")
    return int(numeric)


def _group(value, index):
    if isinstance(value, (str, np.str_)):
        if not str(value).strip():
            raise ValueError(f"Training group at index {index} must be nonempty")
        return str(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Training group at index {index} must be a finite numeric or nonempty string ID")
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"Training group at index {index} is invalid") from None
    if not np.isfinite(numeric):
        raise ValueError(f"Training group at index {index} must be finite")
    # Preserve exact integer identifiers rather than rounding through float64.
    return int(value) if isinstance(value, (int, np.integer)) else numeric


def _stats(values):
    return {"min": float(np.min(values)), "max": float(np.max(values)),
            "mean": float(np.mean(values, dtype=np.float64)), "std": float(np.std(values, dtype=np.float64))}


def build_sample_weights(targets, groups, train_indices, mode):
    """Return (float32 weights aligned with train_indices, JSON-safe receipt).

    Policies, all fitted solely on the current training subset:
      uniform:          raw w_i = 1
      class:            c_i = clip(sqrt(N / (6 * n_class_i)), .75, 1.5)
      duplicate:        d_i = 1 / sqrt(current-training group size_i)
      class_duplicate:  raw w_i = clip(c_i * d_i, .5, 1.5)
    Every mode finally uses w_i = raw w_i / mean(raw w over all train_indices).
    The final normalization can move weights outside the preceding clip range.

    All six targets (0..5) must occur in the selected subset. Duplicate indices,
    out-of-bounds indices, and invalid selected targets/groups are errors. This
    function does not inspect held-out labels/groups, open files, or resample.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    targets, groups = _vector(targets, "targets"), _vector(groups, "groups")
    if len(targets) != len(groups):
        raise ValueError("targets and groups must have equal lengths")
    indices = np.asarray(train_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("train_indices must be a one-dimensional integer index vector")
    if not len(indices):
        raise ValueError("train_indices must be nonempty and contain all six classes")
    if np.any(indices < 0) or np.any(indices >= len(targets)):
        raise ValueError("train_indices contains an out-of-bounds index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices must not contain repeated indices")
    indices = indices.astype(np.int64, copy=False)
    selected_targets = np.array([_target(targets[int(i)], int(i)) for i in indices], dtype=np.int64)
    selected_groups = [_group(groups[int(i)], int(i)) for i in indices]
    class_counts = np.bincount(selected_targets, minlength=CLASS_COUNT)
    if np.any(class_counts == 0):
        missing = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(f"The training subset must contain all six classes; missing {missing}")
    n = len(indices)
    group_counts = Counter(selected_groups)
    sample_group_sizes = np.array([group_counts[group] for group in selected_groups], dtype=np.float64)
    unclipped_class_factors = np.sqrt(n / (CLASS_COUNT * class_counts.astype(np.float64)))
    class_factors = np.clip(unclipped_class_factors, *CLASS_CLIP)
    duplicate_factors = 1 / np.sqrt(sample_group_sizes)
    formulas = {
        "uniform": "raw_w_i = 1",
        "class": "raw_w_i = clip(sqrt(N / (6 * n_class_i)), 0.75, 1.5)",
        "duplicate": "raw_w_i = 1 / sqrt(m_group_i_in_current_training_subset)",
        "class_duplicate": "raw_w_i = clip(clip(sqrt(N / (6 * n_class_i)), 0.75, 1.5) / sqrt(m_group_i_in_current_training_subset), 0.5, 1.5)",
    }
    if mode == "uniform":
        raw = np.ones(n, dtype=np.float64)
    elif mode == "class":
        raw = class_factors[selected_targets]
    elif mode == "duplicate":
        raw = duplicate_factors
    else:
        raw = np.clip(class_factors[selected_targets] * duplicate_factors, *COMBINED_CLIP)
    divisor = float(np.mean(raw, dtype=np.float64))
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Nonpositive or nonfinite normalization divisor")
    weights = (raw / divisor).astype(np.float32)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Sample weights must be finite and strictly positive")
    group_sizes = np.array(list(group_counts.values()), dtype=np.int64)
    final = _stats(weights)
    receipt = {
        "schema_version": 1, "mode": mode, "scope": "current_fold_train_indices_only",
        "heldout_values_used": False, "train_count": n, "classes": list(range(CLASS_COUNT)),
        "class_counts": {str(target): int(class_counts[target]) for target in range(CLASS_COUNT)},
        "formula": formulas[mode], "normalization_formula": "w_i = raw_w_i / mean(raw_w over all train_indices)",
        "configuration": {"class_factor_clip": list(CLASS_CLIP), "combined_product_clip": list(COMBINED_CLIP),
                          "class_factor_used": mode in ("class", "class_duplicate"),
                          "duplicate_factor_used": mode in ("duplicate", "class_duplicate"),
                          "combined_product_clip_used": mode == "class_duplicate",
                          "normalization_population": "all samples in the current training subset",
                          "output_dtype": "float32", "output_alignment": "same order as train_indices"},
        "class_factors_before_clip": {str(target): float(unclipped_class_factors[target]) for target in range(CLASS_COUNT)},
        "class_factors_after_clip": {str(target): float(class_factors[target]) for target in range(CLASS_COUNT)},
        "group_size_stats": {"unique_groups": len(group_counts), "samples": n,
                             "extra_samples_beyond_one_per_group": n - len(group_counts),
                             "min": int(group_sizes.min()), "max": int(group_sizes.max()),
                             "mean": float(group_sizes.mean()), "median": float(np.median(group_sizes)),
                             "size_histogram": {str(size): count for size, count in sorted(Counter(group_sizes.tolist()).items())}},
        "pre_normalization": _stats(raw), "normalization_divisor": divisor,
        "min": final["min"], "max": final["max"], "mean": final["mean"], "std": final["std"],
        "class_mean_weights": {str(target): float(np.mean(weights[selected_targets == target], dtype=np.float64))
                               for target in range(CLASS_COUNT)},
        "range_note": "Clipping occurs before mean normalization. Final min/max can fall outside the clipping interval.",
        "label_changes": 0, "resampling": False,
        "view_policy": "Apply each image's same returned weight to its original and horizontal-flip views.",
    }
    return weights, receipt
