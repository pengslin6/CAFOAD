"""Leakage-audited multiclass grouped cross-validation for CAFOAD.

The evaluator keeps every original class, forms non-overlapping homogeneous
causal event blocks, and discards a sequence-length purge gap between adjacent
blocks.  Five deterministic folds are then used in a rotating
train/validation/test design.  Every retained block is tested exactly once,
while preprocessing, feature pruning, model selection, and class balancing
are fitted only inside that fold's training/validation data.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch

from multiclass_eventwise_runner import (
    DATASETS,
    METHODS,
    context_rows,
    load_revision,
    loader,
    original_labels,
)


def build_folds(
    bundle: Any,
    seq_len: int,
    folds: int = 5,
    block_rows: int = 7,
    split_seed: int = 202604,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    if block_rows < seq_len:
        raise ValueError("block_rows must be at least seq_len")
    purge_rows = seq_len - 1
    stride = block_rows + purge_rows
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
        for start in range(0, len(rows) - block_rows + 1, stride):
            class_groups[class_index].append(rows[start : start + block_rows])

    rng = np.random.default_rng(split_seed)
    fold_groups: list[dict[int, list[np.ndarray]]] = [
        {index: [] for index in class_groups} for _ in range(folds)
    ]
    groups_by_class: dict[str, int] = {}
    for class_index, class_name in enumerate(bundle.class_names):
        groups = class_groups[class_index]
        if len(groups) < folds:
            raise RuntimeError(
                f"{bundle.name}/{class_name}: {len(groups)} groups cannot support "
                f"{folds}-fold class-complete evaluation"
            )
        order = rng.permutation(len(groups))
        for position, group_index in enumerate(order):
            fold_groups[position % folds][class_index].append(groups[int(group_index)])
        groups_by_class[class_name] = len(groups)

    fold_partitions: list[dict[str, np.ndarray]] = []
    fold_audits: list[dict[str, Any]] = []
    tested_group_ids: set[tuple[int, int]] = set()
    for fold in range(folds):
        test_fold = fold
        validation_fold = (fold + 1) % folds
        training_folds = [
            index for index in range(folds)
            if index not in (test_fold, validation_fold)
        ]
        grouped: dict[str, list[np.ndarray]] = {
            "train": [], "validation": [], "test": []
        }
        for class_index in class_groups:
            for source_fold in training_folds:
                grouped["train"].extend(fold_groups[source_fold][class_index])
            grouped["validation"].extend(fold_groups[validation_fold][class_index])
            grouped["test"].extend(fold_groups[test_fold][class_index])

        parts: dict[str, np.ndarray] = {}
        row_sets: dict[str, set[int]] = {}
        counts_by_class: dict[str, dict[str, int]] = {}
        for part, groups in grouped.items():
            row_sets[part] = set(
                np.concatenate(groups).astype(np.int64).tolist()
            )
            targets = [
                np.arange(
                    int(group[0]) + seq_len - 1,
                    int(group[-1]) + 1,
                    dtype=np.int64,
                )
                for group in groups
            ]
            parts[part] = np.sort(np.concatenate(targets))
            counts_by_class[part] = {
                class_name: int(np.sum(bundle.labels[parts[part]] == class_index))
                for class_index, class_name in enumerate(bundle.class_names)
            }
        overlap = {
            "train_validation": len(row_sets["train"] & row_sets["validation"]),
            "train_test": len(row_sets["train"] & row_sets["test"]),
            "validation_test": len(row_sets["validation"] & row_sets["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Fold {fold} row overlap: {overlap}")
        for class_index, groups in fold_groups[test_fold].items():
            for group in groups:
                group_key = (class_index, int(group[0]))
                if group_key in tested_group_ids:
                    raise RuntimeError(f"Group tested twice: {group_key}")
                tested_group_ids.add(group_key)
        fold_partitions.append(parts)
        fold_audits.append({
            "fold": fold,
            "training_folds": training_folds,
            "validation_fold": validation_fold,
            "test_fold": test_fold,
            "cross_partition_row_overlap": overlap,
            "target_counts": {part: int(len(rows)) for part, rows in parts.items()},
            "target_counts_by_class": counts_by_class,
        })

    expected_groups = sum(len(groups) for groups in class_groups.values())
    if len(tested_group_ids) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} once-tested groups, got {len(tested_group_ids)}"
        )
    audit = {
        "rule": (
            "five-fold stratified rotation of non-overlapping homogeneous causal "
            "event blocks; each retained block is test data exactly once"
        ),
        "split_seed": split_seed,
        "folds": folds,
        "block_rows": block_rows,
        "nominal_block_seconds": block_rows * 0.2,
        "purge_rows_between_adjacent_blocks": purge_rows,
        "nominal_purge_seconds": purge_rows * 0.2,
        "sequence_length": seq_len,
        "groups_by_class": groups_by_class,
        "test_group_coverage": "exactly once",
        "fold_audits": fold_audits,
    }
    return fold_partitions, audit


def run_fold(
    revision: Any,
    bundle: Any,
    method: str,
    model_seed: int,
    config: Any,
    parts: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fitted_rows = context_rows(parts["train"], config.seq_len)
    reduced, pruning = revision.prune_features_from_training_rows(bundle, fitted_rows)
    prep = revision.TrainOnlyPreprocessor(
        config.clip_quantile_low, config.clip_quantile_high
    )
    prep.fit(reduced.features[fitted_rows], reduced.labels[fitted_rows])
    transformed = prep.transform(reduced.features)
    model_bundle = revision.with_normal_deviation_schema(
        copy.copy(reduced), transformed.shape[1]
    )
    train_loader = loader(
        revision, transformed, bundle.labels, parts["train"], config, True, model_seed
    )
    validation_loader = loader(
        revision, transformed, bundle.labels, parts["validation"], config, False, model_seed
    )
    test_loader = loader(
        revision, transformed, bundle.labels, parts["test"], config, False, model_seed
    )
    weights = torch.ones(len(bundle.class_names), dtype=torch.float32)
    revision.set_deterministic(model_seed)
    model = revision.MODEL_FACTORIES[method](
        **revision.method_model_kwargs(method, model_bundle, config)
    )
    model, training = revision.train_model(
        method, model, train_loader, validation_loader, weights, config, device
    )
    truth, prediction, probabilities, _ = revision.predict_loader(
        model, test_loader, device
    )
    return truth, prediction, probabilities, {
        "preprocessor": prep.to_dict(),
        "train_only_feature_pruning": pruning,
        "training": training,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="IGCPS,WDT,ICS-Flow")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="13,29,47")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-rows", type=int, default=7)
    args = parser.parse_args()

    revision = load_revision()
    config = revision.ExperimentConfig(
        seq_len=4,
        epochs=args.epochs,
        patience=7,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=2e-4,
        max_train_windows=250_000,
        max_validation_windows=250_000,
        max_test_windows=250_000,
    )
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = original_labels(revision.load_dataset(dataset, DATASETS[dataset]))
        fold_parts, split_audit = build_folds(
            bundle, config.seq_len, args.folds, args.block_rows
        )
        dataset_dir = args.output / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "grouped_cv_protocol.json").write_text(
            json.dumps(split_audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        labels = np.arange(len(bundle.class_names), dtype=np.int64)
        for seed in seeds:
            for method in methods:
                all_truth: list[np.ndarray] = []
                all_prediction: list[np.ndarray] = []
                all_probabilities: list[np.ndarray] = []
                all_targets: list[np.ndarray] = []
                for fold, parts in enumerate(fold_parts):
                    model_seed = seed * 100 + fold
                    print(
                        f"[{dataset}] seed={seed} method={method} fold={fold + 1}/{args.folds}",
                        flush=True,
                    )
                    truth, prediction, probabilities, fold_audit = run_fold(
                        revision, bundle, method, model_seed, config, parts, device
                    )
                    all_truth.append(truth)
                    all_prediction.append(prediction)
                    all_probabilities.append(probabilities)
                    all_targets.append(parts["test"])
                    p, r, f, s = precision_recall_fscore_support(
                        truth, prediction, labels=labels, zero_division=0
                    )
                    fold_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "accuracy": float(accuracy_score(truth, prediction)),
                        "macro_precision": float(p.mean()),
                        "macro_recall": float(r.mean()),
                        "macro_f1": float(f.mean()),
                        "test_targets": int(len(truth)),
                    })
                    fold_dir = dataset_dir / method / f"seed_{seed}" / f"fold_{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    (fold_dir / "training_audit.json").write_text(
                        json.dumps(fold_audit, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                truth = np.concatenate(all_truth)
                prediction = np.concatenate(all_prediction)
                probabilities = np.concatenate(all_probabilities)
                targets = np.concatenate(all_targets)
                chronological_order = np.argsort(targets)
                truth = truth[chronological_order]
                prediction = prediction[chronological_order]
                probabilities = probabilities[chronological_order]
                targets = targets[chronological_order]
                precision, recall, f1, support = precision_recall_fscore_support(
                    truth, prediction, labels=labels, zero_division=0
                )
                matrix = confusion_matrix(truth, prediction, labels=labels)
                normal_index = next(
                    (
                        index for index, name in enumerate(bundle.class_names)
                        if name.lower() in revision.NORMAL_NAMES
                    ),
                    0,
                )
                normal_mask = truth == normal_index
                false_alarm_count = int(
                    np.sum(normal_mask & (prediction != normal_index))
                )
                false_alarm_rate = false_alarm_count / max(1, int(normal_mask.sum()))
                # All final fused files use the declared common 0.2-s grid.
                # Exposure uses only retained normal decisions, not elapsed
                # gaps occupied by training or validation blocks.
                normal_exposure_hours = max(
                    float(normal_mask.sum()) * 0.2 / 3600.0, 1.0e-12
                )
                seed_dir = dataset_dir / method / f"seed_{seed}"
                pd.DataFrame(
                    matrix, index=bundle.class_names, columns=bundle.class_names
                ).to_csv(seed_dir / "confusion_matrix_aggregated.csv")
                prediction_frame = pd.DataFrame({
                    "target_index": targets,
                    "timestamp": bundle.times[targets],
                    "true_class": [bundle.class_names[int(value)] for value in truth],
                    "predicted_class": [
                        bundle.class_names[int(value)] for value in prediction
                    ],
                })
                for class_index, class_name in enumerate(bundle.class_names):
                    prediction_frame[f"probability_{class_name}"] = probabilities[:, class_index]
                prediction_frame.to_csv(
                    seed_dir / "predictions_aggregated.csv", index=False
                )
                for index, class_name in enumerate(bundle.class_names):
                    per_class_rows.append({
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "class": class_name,
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(f1[index]),
                        "support": int(support[index]),
                    })
                result_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "accuracy": float(accuracy_score(truth, prediction)),
                    "macro_precision": float(precision.mean()),
                    "macro_recall": float(recall.mean()),
                    "macro_f1": float(f1.mean()),
                    "macro_roc_auc_ovr": revision.safe_multiclass_auc(
                        truth, probabilities
                    ),
                    "macro_pr_auc_ovr": revision.safe_macro_pr_auc(
                        truth, probabilities
                    ),
                    "false_alarm_rate": float(false_alarm_rate),
                    "false_alarm_count": false_alarm_count,
                    "false_alarms_per_retained_normal_hour": float(
                        false_alarm_count / normal_exposure_hours
                    ),
                    "test_targets": int(len(truth)),
                    "all_classes_evaluated": bool(np.all(support > 0)),
                })

    raw = pd.DataFrame(result_rows)
    raw.to_csv(args.output / "all_seed_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output / "fold_results.csv", index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(args.output / "per_class_by_seed.csv", index=False)
    class_summary = per_class.groupby(
        ["dataset", "method", "class"], sort=False
    )[["precision", "recall", "f1", "support"]].agg(["mean", "std"])
    class_summary.columns = ["_".join(column) for column in class_summary.columns]
    class_summary.reset_index().to_csv(
        args.output / "per_class_summary.csv", index=False
    )
    metrics = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "macro_roc_auc_ovr", "macro_pr_auc_ovr", "false_alarm_rate",
        "false_alarms_per_retained_normal_hour",
    ]
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
