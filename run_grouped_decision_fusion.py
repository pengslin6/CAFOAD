"""Five-fold, leakage-audited decision-fusion comparison for CAFOAD.

The split, train-only feature pruning, preprocessing, four-row causal windows,
and pooled reporting match ``multiclass_grouped_cv_runner.py``.  Three
independent ExtraTrees experts observe the cyber, physical, and complete
causal-statistical views.  Only validation predictions determine fusion
weights and the behavior-knowledge-space (BKS) lookup table.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import CAFOAD_major_revision as revision


DEFAULT_DATASETS = {
    "IGCPS": Path(r"C:\Users\pcsys\Desktop\论文\CAFOAD\igcps_final_unified.csv"),
    "WDT": Path(r"C:\Users\pcsys\Desktop\论文\CAFOAD\wdt_final_unified.csv"),
    "ICS-Flow": Path(r"C:\Users\pcsys\Desktop\论文\CAFOAD\ics_flow_final_unified.csv"),
}
METHODS = ("NB", "BKS", "Majority Voting", "Weighted Majority Voting")


def original_labels(bundle):
    converted = copy.copy(bundle)
    converted.labels = bundle.fine_labels.copy()
    converted.class_names = list(bundle.fine_class_names)
    return converted


def context_rows(targets: np.ndarray, seq_len: int) -> np.ndarray:
    return np.unique(
        np.concatenate([np.arange(t - seq_len + 1, t + 1) for t in targets])
    ).astype(np.int64)


def build_folds(bundle, seq_len: int, folds: int = 5, block_rows: int = 7,
                split_seed: int = 202604):
    """Reproduce the final grouped-CV event blocks exactly."""
    purge_rows = seq_len - 1
    stride = block_rows + purge_rows
    run_break = np.r_[
        True,
        (bundle.labels[1:] != bundle.labels[:-1])
        | (bundle.segment_ids[1:] != bundle.segment_ids[:-1]),
    ]
    run_id = np.cumsum(run_break) - 1
    class_groups = {index: [] for index in range(len(bundle.class_names))}
    for current_run in np.unique(run_id):
        rows = np.flatnonzero(run_id == current_run)
        class_index = int(bundle.labels[rows[0]])
        for start in range(0, len(rows) - block_rows + 1, stride):
            class_groups[class_index].append(rows[start:start + block_rows])

    rng = np.random.default_rng(split_seed)
    fold_groups = [
        {index: [] for index in class_groups} for _ in range(folds)
    ]
    groups_by_class = {}
    for class_index, class_name in enumerate(bundle.class_names):
        groups = class_groups[class_index]
        if len(groups) < folds:
            raise RuntimeError(
                f"{bundle.name}/{class_name}: only {len(groups)} event blocks"
            )
        order = rng.permutation(len(groups))
        for position, group_index in enumerate(order):
            fold_groups[position % folds][class_index].append(groups[int(group_index)])
        groups_by_class[class_name] = len(groups)

    partitions = []
    audits = []
    tested = set()
    for fold in range(folds):
        validation_fold = (fold + 1) % folds
        training_folds = [i for i in range(folds) if i not in (fold, validation_fold)]
        grouped = {"train": [], "validation": [], "test": []}
        for class_index in class_groups:
            for source_fold in training_folds:
                grouped["train"].extend(fold_groups[source_fold][class_index])
            grouped["validation"].extend(fold_groups[validation_fold][class_index])
            grouped["test"].extend(fold_groups[fold][class_index])

        parts = {}
        row_sets = {}
        for role, groups in grouped.items():
            row_sets[role] = set(np.concatenate(groups).astype(np.int64).tolist())
            targets = [
                np.arange(int(group[0]) + seq_len - 1, int(group[-1]) + 1,
                          dtype=np.int64)
                for group in groups
            ]
            parts[role] = np.sort(np.concatenate(targets))
        overlap = {
            "train_validation": len(row_sets["train"] & row_sets["validation"]),
            "train_test": len(row_sets["train"] & row_sets["test"]),
            "validation_test": len(row_sets["validation"] & row_sets["test"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Fold {fold} row overlap: {overlap}")
        for class_index, groups in fold_groups[fold].items():
            for group in groups:
                key = (class_index, int(group[0]))
                if key in tested:
                    raise RuntimeError(f"Event block tested twice: {key}")
                tested.add(key)
        partitions.append(parts)
        audits.append({
            "fold": fold,
            "training_folds": training_folds,
            "validation_fold": validation_fold,
            "test_fold": fold,
            "cross_partition_row_overlap": overlap,
            "target_counts": {key: int(len(value)) for key, value in parts.items()},
        })

    expected = sum(len(groups) for groups in class_groups.values())
    if len(tested) != expected:
        raise RuntimeError(f"Expected {expected} once-tested blocks, got {len(tested)}")
    return partitions, {
        "rule": "five-fold rotation of disjoint homogeneous causal event blocks",
        "split_seed": split_seed,
        "folds": folds,
        "block_rows": block_rows,
        "purge_rows": purge_rows,
        "sequence_length": seq_len,
        "groups_by_class": groups_by_class,
        "test_group_coverage": "exactly once",
        "fold_audits": audits,
    }


def aligned(model, features: np.ndarray, n_classes: int) -> np.ndarray:
    return revision.aligned_classifier_probabilities(model, features, n_classes)


def normalize(probability: np.ndarray) -> np.ndarray:
    return probability / np.maximum(probability.sum(axis=1, keepdims=True), 1.0e-12)


def bks_fit(votes: np.ndarray, truth: np.ndarray, n_classes: int):
    table = {}
    for key in np.unique(votes, axis=0):
        mask = np.all(votes == key, axis=1)
        counts = np.bincount(truth[mask], minlength=n_classes).astype(float) + 1.0
        table[tuple(key.tolist())] = counts / counts.sum()
    fallback = np.bincount(truth, minlength=n_classes).astype(float) + 1.0
    return table, fallback / fallback.sum()


def bks_predict(table, fallback: np.ndarray, votes: np.ndarray) -> np.ndarray:
    return np.stack([table.get(tuple(row.tolist()), fallback) for row in votes])


def fuse(probabilities: list[np.ndarray], method: str, prior: np.ndarray,
         weights: np.ndarray, bks_table, bks_fallback: np.ndarray) -> np.ndarray:
    if method == "NB":
        return normalize(probabilities[0] * probabilities[1] / prior[None, :])
    if method == "Weighted Majority Voting":
        return normalize(sum(w * probability for w, probability in zip(weights, probabilities)))
    votes = np.stack([probability.argmax(axis=1) for probability in probabilities], axis=1)
    if method == "BKS":
        return bks_predict(bks_table, bks_fallback, votes)
    n_classes = probabilities[0].shape[1]
    counts = np.eye(n_classes, dtype=float)[votes].sum(axis=1)
    return normalize(counts)


def score(truth: np.ndarray, probability: np.ndarray, labels: np.ndarray):
    prediction = probability.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "test_targets": int(len(truth)),
        "all_classes_evaluated": bool(np.all(support > 0)),
    }


def benchmark_one(models, feature_views, method: str, prior: np.ndarray,
                  weights: np.ndarray, bks_table, bks_fallback: np.ndarray,
                  seed: int = 20260806):
    for model in models:
        model.set_params(n_jobs=1)
    rng = np.random.default_rng(seed)
    sample_count = min(80, len(feature_views[0]))
    indices = rng.choice(len(feature_views[0]), size=sample_count, replace=False)

    def predict_index(index: int):
        probabilities = [
            aligned(model, features[index:index + 1], len(prior))
            for model, features in zip(models, feature_views)
        ]
        return fuse(probabilities, method, prior, weights, bks_table, bks_fallback)

    for index in indices[:10]:
        predict_index(int(index))
    samples = []
    for index in indices:
        started = time.perf_counter_ns()
        predict_index(int(index))
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    return float(np.median(samples)), float(np.quantile(samples, 0.95))


def run_dataset(bundle, config, output: Path, estimators: int):
    folds, protocol = build_folds(bundle, config.seq_len)
    dataset_dir = output / bundle.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "grouped_cv_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    labels = np.arange(len(bundle.class_names), dtype=np.int64)
    pooled_truth = {method: [] for method in METHODS}
    pooled_probability = {method: [] for method in METHODS}
    pooled_targets = {method: [] for method in METHODS}
    fold_rows = []
    latency_rows = []

    for fold, parts in enumerate(folds):
        print(f"[{bundle.name}] decision fusion fold={fold + 1}/5", flush=True)
        fitted_rows = context_rows(parts["train"], config.seq_len)
        reduced, pruning = revision.prune_features_from_training_rows(bundle, fitted_rows)
        preprocessor = revision.TrainOnlyPreprocessor(
            config.clip_quantile_low, config.clip_quantile_high
        )
        preprocessor.fit(
            reduced.features[fitted_rows], reduced.labels[fitted_rows]
        )
        transformed = preprocessor.transform(reduced.features)
        schema = revision.with_normal_deviation_schema(
            copy.copy(reduced), transformed.shape[1]
        )
        view_indices = [
            schema.cyber_indices,
            schema.physical_indices,
            np.arange(transformed.shape[1], dtype=np.int64),
        ]
        feature_views = []
        for indices in view_indices:
            feature_views.append({
                role: revision.causal_statistical_features(
                    transformed[:, indices], parts[role], config.seq_len
                )
                for role in ("train", "validation", "test")
            })

        models = []
        validation_probability = []
        test_probability = []
        model_seed = 1300 + fold
        for features in feature_views:
            model = ExtraTreesClassifier(
                n_estimators=estimators,
                max_features="sqrt",
                class_weight="balanced",
                random_state=model_seed,
                n_jobs=-1,
            )
            model.fit(features["train"], bundle.labels[parts["train"]])
            models.append(model)
            validation_probability.append(
                aligned(model, features["validation"], len(labels))
            )
            test_probability.append(aligned(model, features["test"], len(labels)))

        validation_truth = bundle.labels[parts["validation"]]
        test_truth = bundle.labels[parts["test"]]
        prior = np.bincount(
            bundle.labels[parts["train"]], minlength=len(labels)
        ).astype(float) + 1.0
        prior /= prior.sum()
        validation_scores = np.array([
            revision.macro_f1_present(validation_truth, probability.argmax(axis=1))
            for probability in validation_probability
        ])
        weights = np.maximum(validation_scores, 1.0e-6)
        weights /= weights.sum()
        validation_votes = np.stack(
            [probability.argmax(axis=1) for probability in validation_probability],
            axis=1,
        )
        bks_table, bks_fallback = bks_fit(
            validation_votes, validation_truth, len(labels)
        )

        for method in METHODS:
            probability = fuse(
                test_probability, method, prior, weights, bks_table, bks_fallback
            )
            fold_rows.append({
                "dataset": bundle.name,
                "method": method,
                "seed": 13,
                "fold": fold,
                **score(test_truth, probability, labels),
            })
            pooled_truth[method].append(test_truth)
            pooled_probability[method].append(probability)
            pooled_targets[method].append(parts["test"])
            if fold == 0:
                p50, p95 = benchmark_one(
                    models,
                    [features["test"] for features in feature_views],
                    method,
                    prior,
                    weights,
                    bks_table,
                    bks_fallback,
                )
                latency_rows.append({
                    "dataset": bundle.name,
                    "method": method,
                    "latency_p50_ms": p50,
                    "latency_p95_ms": p95,
                    "batch_size": 1,
                    "experts": 2 if method == "NB" else 3,
                    "trees_per_expert": estimators,
                    "device": "CPU; sklearn single-thread inference",
                })

        (dataset_dir / f"fold_{fold}_audit.json").write_text(
            json.dumps({
                "fold": fold,
                "train_only_feature_pruning": pruning,
                "preprocessor": preprocessor.to_dict(),
                "validation_expert_macro_f1": validation_scores.tolist(),
                "validation_fusion_weights": weights.tolist(),
                "test_labels_used_for_fusion_selection": False,
            }, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    pooled_rows = []
    for method in METHODS:
        truth = np.concatenate(pooled_truth[method])
        probability = np.concatenate(pooled_probability[method])
        targets = np.concatenate(pooled_targets[method])
        order = np.argsort(targets)
        pooled_rows.append({
            "dataset": bundle.name,
            "method": method,
            "seed": 13,
            **score(truth[order], probability[order], labels),
        })
    return fold_rows, pooled_rows, latency_rows


def append_deep_references(pooled: pd.DataFrame, latency: pd.DataFrame,
                           overall_path: Path, efficiency_path: Path):
    overall = pd.read_csv(overall_path)
    efficiency = pd.read_csv(efficiency_path)
    deep = overall[overall["method"].isin(["TV-DBN", "CAFOAD"])].copy()
    deep = deep[[
        "dataset", "method", "seed", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "test_targets", "all_classes_evaluated",
    ]]
    deep = deep.merge(
        efficiency[["dataset", "method", "latency_p50_ms", "latency_p95_ms"]],
        on=["dataset", "method"], how="left",
    )
    tree = pooled.merge(
        latency[["dataset", "method", "latency_p50_ms", "latency_p95_ms"]],
        on=["dataset", "method"], how="left",
    )
    tree["result_source"] = "new grouped decision-fusion run"
    deep["result_source"] = "same grouped five-fold main run"
    combined = pd.concat([tree, deep], ignore_index=True)
    method_order = {
        "NB": 0, "BKS": 1, "Majority Voting": 2,
        "Weighted Majority Voting": 3, "TV-DBN": 4, "CAFOAD": 5,
    }
    dataset_order = {"IGCPS": 0, "WDT": 1, "ICS-Flow": 2}
    combined["_dataset_order"] = combined["dataset"].map(dataset_order)
    combined["_method_order"] = combined["method"].map(method_order)
    return combined.sort_values(["_dataset_order", "_method_order"]).drop(
        columns=["_dataset_order", "_method_order"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("TMC_multiclass_results/decision_fusion_grouped"))
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--datasets", default="IGCPS,WDT,ICS-Flow")
    parser.add_argument("--overall", type=Path,
                        default=Path("TMC_multiclass_results/overall_multiclass_results.csv"))
    parser.add_argument("--efficiency", type=Path, default=Path("efficiency_metrics.csv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = revision.ExperimentConfig(seq_len=4)
    fold_rows = []
    pooled_rows = []
    latency_rows = []
    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        bundle = original_labels(revision.load_dataset(dataset, DEFAULT_DATASETS[dataset]))
        folds, pooled, latency = run_dataset(bundle, config, args.output, args.estimators)
        fold_rows.extend(folds)
        pooled_rows.extend(pooled)
        latency_rows.extend(latency)
        pd.DataFrame(fold_rows).to_csv(args.output / "fold_results.csv", index=False)
        pd.DataFrame(pooled_rows).to_csv(args.output / "pooled_results.csv", index=False)
        pd.DataFrame(latency_rows).to_csv(args.output / "latency_results.csv", index=False)

    fold_frame = pd.DataFrame(fold_rows)
    fold_summary = fold_frame.groupby(["dataset", "method"], sort=False)[
        ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    ].agg(["mean", "std"])
    fold_summary.columns = ["_".join(column) for column in fold_summary.columns]
    fold_summary.reset_index().to_csv(args.output / "fold_summary.csv", index=False)
    combined = append_deep_references(
        pd.DataFrame(pooled_rows), pd.DataFrame(latency_rows),
        args.overall, args.efficiency,
    )
    combined.to_csv(args.output.parent / "decision_fusion_multiclass_results.csv",
                    index=False)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
