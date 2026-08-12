"""Class-complete, purged chronological multiclass evaluation for CAFOAD.

For every original class independently, rows are ordered by source time and
assigned to the earliest 60% training, next 20% validation, and latest 20%
test blocks.  A causal sequence is retained only when every row in its context
belongs to the same block, class, and acquisition segment.  The rule is fixed
for every dataset and model and is intended as an event-level class-complete
diagnostic; it is not described as a single global future holdout.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CAFOAD_major_revision.py"
DATASETS = {
    "IGCPS": ROOT / "causal_grid_fusion" / "igcps_final_unified.csv",
    "WDT": ROOT / "causal_grid_fusion" / "wdt_final_unified.csv",
    "ICS-Flow": ROOT / "causal_grid_fusion" / "ics_flow_final_unified.csv",
}
METHODS = [
    "Dual-LSTM-AE", "MAD-GAN", "USAD", "GDN", "TranAD", "TV-DBN", "MGDN", "CAFOAD"
]


def load_revision():
    specification = importlib.util.spec_from_file_location("cafoad_multiclass", SOURCE)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def original_labels(bundle: Any) -> Any:
    converted = copy.copy(bundle)
    converted.labels = bundle.fine_labels.copy()
    converted.class_names = list(bundle.fine_class_names)
    return converted


def purged_class_blocks(bundle: Any, seq_len: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    assignment = np.full(len(bundle.labels), -1, dtype=np.int8)
    row_counts: dict[str, dict[str, int]] = {}
    for class_index, class_name in enumerate(bundle.class_names):
        rows = np.flatnonzero(bundle.labels == class_index)
        first = int(np.floor(0.60 * len(rows)))
        second = int(np.floor(0.80 * len(rows)))
        assignment[rows[:first]] = 0
        assignment[rows[first:second]] = 1
        assignment[rows[second:]] = 2
        row_counts[class_name] = {
            "train": first,
            "validation": second - first,
            "test": len(rows) - second,
        }
    candidates = np.arange(seq_len - 1, len(bundle.labels), dtype=np.int64)
    partitions: dict[str, np.ndarray] = {}
    for partition_index, name in enumerate(("train", "validation", "test")):
        valid = assignment[candidates] == partition_index
        for offset in range(1, seq_len):
            valid &= assignment[candidates - offset] == partition_index
            valid &= bundle.labels[candidates - offset] == bundle.labels[candidates]
            valid &= bundle.segment_ids[candidates - offset] == bundle.segment_ids[candidates]
        partitions[name] = candidates[valid]
    audit = {
        "rule": "within each original class: earliest 60% train, next 20% validation, latest 20% test",
        "sequence_length": seq_len,
        "window_constraint": "all context rows share partition, original class, and acquisition segment",
        "cross_partition_window_overlap": 0,
        "row_counts_by_class": row_counts,
        "target_counts": {k: int(len(v)) for k, v in partitions.items()},
        "target_counts_by_class": {
            part: {
                name: int(np.sum(bundle.labels[targets] == index))
                for index, name in enumerate(bundle.class_names)
            }
            for part, targets in partitions.items()
        },
    }
    for part, targets in partitions.items():
        missing = [
            bundle.class_names[index]
            for index in range(len(bundle.class_names))
            if not np.any(bundle.labels[targets] == index)
        ]
        if missing:
            raise RuntimeError(f"{bundle.name}/{part} lacks classes: {missing}")
    return partitions, audit


def stratified_event_blocks(
    bundle: Any, seq_len: int, block_rows: int = 8, split_seed: int = 202604
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assign whole, non-overlapping homogeneous event blocks to one split."""
    if block_rows < seq_len:
        raise ValueError("block_rows must be at least seq_len")
    run_break = np.r_[
        True,
        (bundle.labels[1:] != bundle.labels[:-1])
        | (bundle.segment_ids[1:] != bundle.segment_ids[:-1]),
    ]
    run_id = np.cumsum(run_break) - 1
    class_groups: dict[int, list[np.ndarray]] = {
        index: [] for index in range(len(bundle.class_names))
    }
    for current_run in np.unique(run_id):
        rows = np.flatnonzero(run_id == current_run)
        class_index = int(bundle.labels[rows[0]])
        for start in range(0, len(rows) - block_rows + 1, block_rows):
            class_groups[class_index].append(rows[start : start + block_rows])
    rng = np.random.default_rng(split_seed)
    grouped: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    group_counts: dict[str, dict[str, int]] = {}
    for class_index, class_name in enumerate(bundle.class_names):
        groups = class_groups[class_index]
        if len(groups) < 3:
            raise RuntimeError(
                f"{bundle.name}/{class_name} has only {len(groups)} full event blocks"
            )
        order = rng.permutation(len(groups))
        train_end = max(1, int(np.floor(0.60 * len(groups))))
        validation_end = max(train_end + 1, int(np.floor(0.80 * len(groups))))
        validation_end = min(validation_end, len(groups) - 1)
        allocation = {
            "train": order[:train_end],
            "validation": order[train_end:validation_end],
            "test": order[validation_end:],
        }
        group_counts[class_name] = {}
        for part, selected in allocation.items():
            grouped[part].extend(groups[int(index)] for index in selected)
            group_counts[class_name][part] = int(len(selected))
    partitions: dict[str, np.ndarray] = {}
    row_sets: dict[str, set[int]] = {}
    for part, groups in grouped.items():
        row_sets[part] = set(np.concatenate(groups).astype(int).tolist())
        targets = [
            np.arange(int(group[0]) + seq_len - 1, int(group[-1]) + 1, dtype=np.int64)
            for group in groups
        ]
        partitions[part] = np.sort(np.concatenate(targets))
    overlap = {
        "train_validation": len(row_sets["train"] & row_sets["validation"]),
        "train_test": len(row_sets["train"] & row_sets["test"]),
        "validation_test": len(row_sets["validation"] & row_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Event-block split overlap: {overlap}")
    audit = {
        "rule": "fixed-seed stratified assignment of whole homogeneous causal event blocks",
        "split_seed": split_seed,
        "block_rows": block_rows,
        "block_duration_seconds_nominal": block_rows * 0.2,
        "sequence_length": seq_len,
        "cross_partition_row_overlap": overlap,
        "groups_by_class": group_counts,
        "target_counts": {k: int(len(v)) for k, v in partitions.items()},
        "target_counts_by_class": {
            part: {
                name: int(np.sum(bundle.labels[targets] == index))
                for index, name in enumerate(bundle.class_names)
            }
            for part, targets in partitions.items()
        },
    }
    return partitions, audit


def context_rows(targets: np.ndarray, seq_len: int) -> np.ndarray:
    return np.unique(
        np.concatenate([np.arange(t - seq_len + 1, t + 1) for t in targets])
    ).astype(np.int64)


def loader(revision: Any, x: np.ndarray, y: np.ndarray, targets: np.ndarray,
           config: Any, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    sampler = None
    if shuffle:
        target_labels = y[targets]
        counts = np.bincount(target_labels, minlength=int(y.max()) + 1).astype(np.float64)
        sample_weights = 1.0 / np.maximum(counts[target_labels], 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(targets),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        revision.CausalWindowDataset(x, y, targets, config.seq_len),
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
    )


def run_one(revision: Any, bundle: Any, method: str, seed: int, config: Any,
            output: Path, device: torch.device) -> dict[str, Any]:
    parts, split_audit = stratified_event_blocks(bundle, config.seq_len)
    fitted_rows = context_rows(parts["train"], config.seq_len)
    reduced, pruning = revision.prune_features_from_training_rows(bundle, fitted_rows)
    prep = revision.TrainOnlyPreprocessor(config.clip_quantile_low, config.clip_quantile_high)
    prep.fit(reduced.features[fitted_rows], reduced.labels[fitted_rows])
    transformed = prep.transform(reduced.features)
    model_bundle = revision.with_normal_deviation_schema(
        copy.copy(reduced), transformed.shape[1]
    )
    train_loader = loader(revision, transformed, bundle.labels, parts["train"], config, True, seed)
    validation_loader = loader(
        revision, transformed, bundle.labels, parts["validation"], config, False, seed
    )
    test_loader = loader(revision, transformed, bundle.labels, parts["test"], config, False, seed)
    # The common training sampler is already class-balanced, so an additional
    # loss reweighting would count imbalance twice.
    weights = torch.ones(len(bundle.class_names), dtype=torch.float32)
    revision.set_deterministic(seed)
    model = revision.MODEL_FACTORIES[method](
        **revision.method_model_kwargs(method, model_bundle, config)
    )
    model, training = revision.train_model(
        method, model, train_loader, validation_loader, weights, config, device
    )
    truth, prediction, _, _ = revision.predict_loader(model, test_loader, device)
    labels = np.arange(len(bundle.class_names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(truth, prediction, labels=labels)
    destination = output / bundle.name / method / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=bundle.class_names, columns=bundle.class_names).to_csv(
        destination / "confusion_matrix.csv"
    )
    per_class = pd.DataFrame({
        "class": bundle.class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    per_class.to_csv(destination / "per_class_metrics.csv", index=False)
    (destination / "training_and_split_audit.json").write_text(
        json.dumps({
            "split": split_audit,
            "preprocessor": prep.to_dict(),
            "train_only_feature_pruning": pruning,
            "training": training,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "dataset": bundle.name,
        "method": method,
        "seed": seed,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "test_targets": int(len(truth)),
        "all_classes_evaluated": bool(np.all(support > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="IGCPS,WDT,ICS-Flow")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    revision = load_revision()
    config = revision.ExperimentConfig(
        seq_len=4,
        epochs=args.epochs,
        patience=args.epochs + 1,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=2e-4,
        max_train_windows=250_000,
        max_validation_windows=250_000,
        max_test_windows=250_000,
    )
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = original_labels(revision.load_dataset(dataset, DATASETS[dataset]))
        for seed in seeds:
            for method in methods:
                print(f"[{dataset}] seed={seed} method={method}", flush=True)
                rows.append(run_one(revision, bundle, method, seed, config, args.output, device))
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "all_seed_results.csv", index=False)
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary = raw.groupby(["dataset", "method"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary["rank_macro_f1"] = summary.groupby("dataset")["macro_f1_mean"].rank(
        ascending=False, method="min"
    )
    summary.to_csv(args.output / "summary_with_ranks.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
