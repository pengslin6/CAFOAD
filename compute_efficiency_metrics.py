"""Reproducible batch-one FLOP and latency audit for the revision tables.

The script reconstructs the fold-0 train-only feature schema used by the
grouped protocol, instantiates each architecture with the reported settings,
and measures one four-step window on the same single-thread CPU boundary as
the existing matched inference audit. FLOPs follow the common 2 x MAC rule.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from thop import profile

import CAFOAD_major_revision as revision


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "Desktop" / "论文" / "CAFOAD"
DATASETS = {
    "IGCPS": DATA_ROOT / "igcps_final_unified.csv",
    "WDT": DATA_ROOT / "wdt_final_unified.csv",
    "ICS-Flow": DATA_ROOT / "ics_flow_final_unified.csv",
}
METHODS = [
    "Dual-LSTM-AE",
    "MAD-GAN",
    "USAD",
    "GDN",
    "TranAD",
    "TV-DBN",
    "MGDN",
    "CAFOAD",
    "Aggregating-Network-Traffic",
    "Feature-Fusion",
    "CDRL",
]


def original_labels(bundle: revision.DatasetBundle) -> revision.DatasetBundle:
    revised = copy.copy(bundle)
    revised.labels = bundle.fine_labels.copy()
    revised.class_names = list(bundle.fine_class_names)
    return revised


def build_folds(
    bundle: revision.DatasetBundle,
    seq_len: int = 4,
    folds: int = 5,
    block_rows: int = 7,
    split_seed: int = 202604,
) -> list[dict[str, np.ndarray]]:
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
    fold_groups = [
        {index: [] for index in class_groups} for _ in range(folds)
    ]
    for class_index, groups in class_groups.items():
        order = rng.permutation(len(groups))
        for position, group_index in enumerate(order):
            fold_groups[position % folds][class_index].append(groups[int(group_index)])

    partitions: list[dict[str, np.ndarray]] = []
    for fold in range(folds):
        validation_fold = (fold + 1) % folds
        training_folds = [
            index for index in range(folds) if index not in (fold, validation_fold)
        ]
        sources = {
            "train": training_folds,
            "validation": [validation_fold],
            "test": [fold],
        }
        parts: dict[str, np.ndarray] = {}
        for role, source_folds in sources.items():
            groups: list[np.ndarray] = []
            for class_index in class_groups:
                for source_fold in source_folds:
                    groups.extend(fold_groups[source_fold][class_index])
            targets = [
                np.arange(int(group[0]) + seq_len - 1, int(group[-1]) + 1)
                for group in groups
            ]
            parts[role] = np.sort(np.concatenate(targets).astype(np.int64))
        partitions.append(parts)
    return partitions


def context_rows(targets: np.ndarray, seq_len: int) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [np.arange(int(end) - seq_len + 1, int(end) + 1) for end in targets]
        )
    )


def count_mha(module: nn.MultiheadAttention, inputs, output) -> None:
    query, key, value = inputs[:3]
    if module.batch_first:
        batch, query_length, embed_dim = query.shape
        key_length = key.shape[1]
        value_length = value.shape[1]
    else:
        query_length, batch, embed_dim = query.shape
        key_length = key.shape[0]
        value_length = value.shape[0]
    projection_macs = batch * (
        query_length * embed_dim * embed_dim
        + key_length * embed_dim * embed_dim
        + value_length * embed_dim * embed_dim
        + query_length * embed_dim * embed_dim
    )
    attention_macs = batch * (
        query_length * key_length * embed_dim
        + query_length * key_length * embed_dim
    )
    module.total_ops += torch.DoubleTensor([projection_macs + attention_macs])


def additional_functional_macs(model: nn.Module, x: torch.Tensor) -> float:
    batch, sequence_length, _ = x.shape
    if isinstance(model, revision.GDNReimplementation):
        feature_count = model.node_weight.shape[0]
        node_dim = model.node_weight.shape[1]
        return float(batch * (2 * feature_count * feature_count * node_dim + feature_count * node_dim))
    if isinstance(model, revision.TVDBNReimplementation):
        classes = model.transition.shape[0]
        return float(batch * max(0, sequence_length - 1) * classes * classes)
    if isinstance(model, revision.MGDNReimplementation):
        d_model = model.cyber_level.out_features
        return float(batch * 2 * d_model)
    if isinstance(model, revision.CAFOAD):
        classes, d_model = model.class_prototypes.shape
        return float(batch * classes * d_model)
    return 0.0


def measure(model: nn.Module, x: torch.Tensor) -> tuple[float, float, float]:
    model.eval()
    macs, _ = profile(
        model,
        inputs=(x,),
        custom_ops={nn.MultiheadAttention: count_mha},
        verbose=False,
    )
    macs += additional_functional_macs(model, x)
    with torch.inference_mode():
        for _ in range(40):
            model(x)
        timings = []
        for _ in range(300):
            start = perf_counter_ns()
            model(x)
            timings.append((perf_counter_ns() - start) / 1.0e6)
    values = np.asarray(timings, dtype=np.float64)
    return 2.0 * float(macs) / 1.0e6, float(np.median(values)), float(np.percentile(values, 95))


def main() -> None:
    torch.set_num_threads(1)
    revision.set_deterministic(13)
    config = revision.ExperimentConfig(
        seq_len=4,
        d_model=48,
        n_heads=4,
        n_layers=2,
        dropout=0.10,
        epochs=30,
        patience=7,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=2e-4,
    )
    rows: list[dict[str, object]] = []
    schemas: dict[str, object] = {}
    for dataset, path in DATASETS.items():
        bundle = original_labels(revision.load_dataset(dataset, path))
        fold_zero = build_folds(bundle, seq_len=config.seq_len)[0]
        fitted_rows = context_rows(fold_zero["train"], config.seq_len)
        reduced, pruning = revision.prune_features_from_training_rows(bundle, fitted_rows)
        model_bundle = revision.with_normal_deviation_schema(
            copy.copy(reduced), transformed_dim=2 * len(reduced.feature_names)
        )
        schemas[dataset] = {
            "original_features": len(bundle.feature_names),
            "retained_base_features": len(reduced.feature_names),
            "model_input_dim": model_bundle.model_input_dim,
            "cyber_features": len(model_bundle.cyber_indices),
            "physical_features": len(model_bundle.physical_indices),
            "classes": len(model_bundle.class_names),
            "train_only_pruning": pruning,
        }
        x = torch.zeros(1, config.seq_len, int(model_bundle.model_input_dim), dtype=torch.float32)
        for method in METHODS:
            revision.set_deterministic(13)
            torch.set_num_threads(1)
            model = revision.MODEL_FACTORIES[method](
                **revision.method_model_kwargs(method, model_bundle, config)
            )
            flops_m, p50_ms, p95_ms = measure(model, x)
            row = {
                "dataset": dataset,
                "method": method,
                "flops_m": flops_m,
                "latency_p50_ms": p50_ms,
                "latency_p95_ms": p95_ms,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "batch_size": 1,
                "sequence_length": config.seq_len,
                "device": "CPU; torch single-thread",
                "flop_convention": "2 x profiled MACs; one four-step window",
            }
            rows.append(row)
            print(row)
    output = ROOT / "TMC_multiclass_results" / "efficiency_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "scope": "architecture-only forward on a fold-0 train-fitted feature schema",
                "timing": "40 warmups, 300 batch-one repetitions, median and p95",
                "flops": "2 x THOP MAC count with explicit multi-head-attention and functional-matmul corrections",
                "schemas": schemas,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
