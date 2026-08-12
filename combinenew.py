"""Leakage-resistant cyber--physical fusion for the CAFOAD revision.

This script intentionally leaves ``combine.py`` and the legacy fused files
untouched.  It rebuilds all three datasets directly from their raw domain
streams with one label-blind 200-ms causal-grid rule.  A row emitted at grid
endpoint g contains the latest physical observation available no later than g
and summaries of network events satisfying g-200 ms < network_time <= g.

No interpolation, extrapolation, future/backward filling, clock-offset
optimisation, label-guided matching, random packet subsampling, or
whole-stream fitted statistic is used.  Missing-value imputation, clipping,
feature pruning, and scaling belong to the downstream experiment and must be
fitted on the chronological training partition only.

Outputs (in ``--root``):
    igcps_final_unified.csv     IGCPS: sr_features.csv + np.csv
    ics_flow_final_unified.csv  ICS-Flow: net_ics.csv + phy_ics.csv
    wdt_final_unified.csv       WDT: attack_1_features1.csv + phy.csv
    combinenew_manifest.json
    combinenew_runtime.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


DEFAULT_ROOT = Path(__file__).resolve().parent
NORMAL_NAMES = {"0", "benign", "normal"}
CAUSAL_SUBBINS = 10
CAUSAL_GRID_SECONDS = 0.2


@dataclass(frozen=True)
class FusionSpec:
    name: str
    cyber_file: str
    physical_file: str
    output_file: str
    clock_output_file: str
    label_source: str
    label_column: str
    cyber_label_columns: tuple[str, ...]
    physical_label_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    max_staleness_seconds: float = 2.0


SPECS = {
    "IGCPS": FusionSpec(
        name="IGCPS",
        cyber_file="sr_features.csv",
        physical_file="np.csv",
        output_file="sr_com_causal.csv",
        clock_output_file="igcps_final_unified.csv",
        label_source="physical",
        label_column="label",
        cyber_label_columns=(),
        physical_label_columns=("label",),
        categorical_columns=("Source IP", "Destination IP", "Protocol Type"),
    ),
    "ICS-Flow": FusionSpec(
        name="ICS-Flow",
        cyber_file="net_ics.csv",
        physical_file="phy_ics.csv",
        output_file="ics_com_causal.csv",
        clock_output_file="ics_flow_final_unified.csv",
        label_source="cyber",
        label_column="IT_M_Label",
        cyber_label_columns=("IT_B_Label", "IT_M_Label", "NST_B_Label", "NST_M_Label"),
        physical_label_columns=(),
        categorical_columns=(
            "sAddress", "rAddress", "sMACs", "rMACs", "sIPs", "rIPs", "protocol"
        ),
    ),
    "WDT": FusionSpec(
        name="WDT",
        cyber_file="attack_1_features1.csv",
        physical_file="phy.csv",
        output_file="wdt_com_causal.csv",
        clock_output_file="wdt_final_unified.csv",
        label_source="physical",
        label_column="Label",
        cyber_label_columns=(),
        physical_label_columns=("Label_n", "Label"),
        categorical_columns=(
            "Source IP", "Destination IP", "Protocol Type"
        ),
    ),
}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_times(series: pd.Series) -> pd.Series:
    """Return nanosecond-resolution UTC timestamps without locale dependence."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed


def stable_content_code(series: pd.Series) -> np.ndarray:
    """Encode a category from its own content, without fitting a future vocabulary."""
    text = series.fillna("<missing>").astype(str)
    values = pd.util.hash_pandas_object(
        text, index=False, hash_key="cafoad-revision!"
    ).to_numpy(dtype=np.uint64)
    return (values % np.uint64(100_003)).astype(np.float64)


def canonical_labels(series: pd.Series) -> pd.Series:
    values = series.fillna("<missing>").astype(str).str.strip()
    lowered = values.str.lower()
    values.loc[lowered.isin(NORMAL_NAMES)] = "normal"
    return values


def split_summary(labels: Iterable[str]) -> dict[str, object]:
    labels = pd.Series(list(labels), dtype="object")
    n_rows = len(labels)
    train_end = int(0.60 * n_rows)
    validation_end = int(0.80 * n_rows)
    ranges = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n_rows),
    }
    summary: dict[str, object] = {
        "rule": "strict chronological 60/20/20; boundaries applied before any fitted preprocessing",
        "row_boundaries_half_open": {key: list(value) for key, value in ranges.items()},
        "partitions": {},
    }
    for key, (start, end) in ranges.items():
        counts = labels.iloc[start:end].value_counts(dropna=False)
        summary["partitions"][key] = {
            "rows": int(end - start),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
        }
    return summary


def prepare_physical(path: Path, spec: FusionSpec) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Time" not in frame.columns:
        raise ValueError(f"{path.name} has no Time column")
    raw_rows = len(frame)
    frame["_phy_time"] = parse_times(frame["Time"])
    frame = frame.dropna(subset=["_phy_time"]).copy()
    frame = frame.sort_values("_phy_time", kind="stable")
    # At a duplicated timestamp, the final row is the latest observation
    # available at that instant.  This does not consult any later timestamp.
    frame = frame.drop_duplicates("_phy_time", keep="last").reset_index(drop=True)

    excluded = {"Time", *spec.physical_label_columns}
    physical_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and column != "_phy_time"
        and column.strip()
        and not column.lower().startswith("unnamed:")
    ]
    for column in physical_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    selected = frame[["_phy_time", *physical_columns]].copy()
    selected = selected.rename(columns={column: f"phy_{column}" for column in physical_columns})
    if spec.label_source == "physical":
        if spec.label_column not in frame.columns:
            raise ValueError(f"Physical label {spec.label_column!r} is absent from {path.name}")
        selected["_target_label"] = canonical_labels(frame[spec.label_column])
    stats = {
        "raw_rows": int(raw_rows),
        "valid_time_rows": int(frame.shape[0]),
        "duplicate_timestamps_removed": int(raw_rows - frame.shape[0]),
    }
    return selected, physical_columns, stats


def feature_schema(columns: list[str], spec: FusionSpec) -> tuple[list[str], list[str]]:
    excluded = {"Time", *spec.cyber_label_columns}
    raw_features = [
        column
        for column in columns
        if column not in excluded
        and column.strip()
        and not column.lower().startswith("unnamed:")
    ]
    categorical = [column for column in raw_features if column in spec.categorical_columns]
    numeric = [column for column in raw_features if column not in categorical]
    return numeric, categorical


def fuse_on_physical_clock(
    source_root: Path, output_root: Path, spec: FusionSpec, chunksize: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate non-overlapping network intervals on one causal 200-ms grid.

    The physical vector at endpoint ``g`` is the latest real observation whose
    timestamp is no later than ``g``.  Its age and change indicator are emitted
    explicitly.  Empty cyber intervals are discarded, so repeated physical
    values never create samples without independent network evidence.
    """
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss

    cyber_path = source_root / spec.cyber_file
    physical_path = source_root / spec.physical_file
    output_path = output_root / spec.clock_output_file
    if not cyber_path.exists() or not physical_path.exists():
        raise FileNotFoundError(f"Missing source pair: {cyber_path}, {physical_path}")

    physical, physical_columns, physical_stats = prepare_physical(physical_path, spec)
    phy_ns = physical["_phy_time"].astype("int64").to_numpy()
    if len(phy_ns) < 2:
        raise RuntimeError(f"{spec.name}: fewer than two valid physical timestamps")

    step_ns = int(round(CAUSAL_GRID_SECONDS * 1.0e9))
    physical_gaps = np.diff(phy_ns) / 1.0e9
    physical_segment = np.r_[0, np.cumsum(physical_gaps > spec.max_staleness_seconds)]
    grid_parts: list[np.ndarray] = []
    grid_segment_parts: list[np.ndarray] = []
    for segment in np.unique(physical_segment):
        segment_times = phy_ns[physical_segment == segment]
        if len(segment_times) < 2:
            continue
        # The first endpoint is strictly after the first real physical sample,
        # ensuring that the full left-open network interval is observable.
        first_grid = (int(segment_times[0]) // step_ns + 1) * step_ns
        last_grid = (int(segment_times[-1]) // step_ns) * step_ns
        if first_grid > last_grid:
            continue
        part = np.arange(first_grid, last_grid + 1, step_ns, dtype=np.int64)
        grid_parts.append(part)
        grid_segment_parts.append(np.full(len(part), int(segment), dtype=np.int64))
    if not grid_parts:
        raise RuntimeError(f"{spec.name}: no continuous physical segment supports the causal grid")
    grid_ns = np.concatenate(grid_parts)
    grid_source_segment = np.concatenate(grid_segment_parts)
    grid_phy_index = np.searchsorted(phy_ns, grid_ns, side="right") - 1
    grid_physical_age_s = (grid_ns - phy_ns[grid_phy_index]) / 1.0e9
    fresh = (grid_phy_index >= 0) & (grid_physical_age_s <= spec.max_staleness_seconds)
    grid_ns = grid_ns[fresh]
    grid_source_segment = grid_source_segment[fresh]
    grid_phy_index = grid_phy_index[fresh]
    grid_physical_age_s = grid_physical_age_s[fresh]

    source_header = pd.read_csv(cyber_path, nrows=0)
    source_columns = [str(column).strip() for column in source_header.columns]
    numeric_columns, categorical_columns = feature_schema(source_columns, spec)
    if spec.label_source == "cyber" and spec.label_column not in source_columns:
        raise ValueError(f"Cyber label {spec.label_column!r} is absent from {cyber_path.name}")

    n_anchor = len(grid_ns)
    n_numeric = len(numeric_columns)
    categorical_aggregate_names = list(categorical_columns)
    if categorical_columns:
        categorical_aggregate_names.append("joint_category")
    n_categorical = len(categorical_aggregate_names)
    event_count = np.zeros(n_anchor, dtype=np.int64)
    valid_count = np.zeros((n_anchor, n_numeric), dtype=np.int64)
    sums = np.zeros((n_anchor, n_numeric), dtype=np.float64)
    sumsq = np.zeros((n_anchor, n_numeric), dtype=np.float64)
    first = np.full((n_anchor, n_numeric), np.nan, dtype=np.float64)
    last = np.full((n_anchor, n_numeric), np.nan, dtype=np.float64)
    categorical_count = np.zeros((n_anchor, n_categorical), dtype=np.int64)
    categorical_sum = np.zeros((n_anchor, n_categorical), dtype=np.float64)
    categorical_sumsq = np.zeros((n_anchor, n_categorical), dtype=np.float64)
    categorical_last = np.full((n_anchor, n_categorical), np.nan, dtype=np.float64)
    categorical_bucket_mask = np.zeros((n_anchor, n_categorical), dtype=np.uint64)
    subbin_count = np.zeros((n_anchor, CAUSAL_SUBBINS), dtype=np.int64)
    target_last = np.full(n_anchor, None, dtype=object)

    total_rows = valid_time_rows = interval_rows = 0
    previous_chunk_last_ns: int | None = None
    globally_monotonic = True
    for chunk_index, chunk in enumerate(
        pd.read_csv(cyber_path, chunksize=chunksize, low_memory=False)
    ):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        total_rows += len(chunk)
        chunk_time = parse_times(chunk["Time"])
        valid_time = chunk_time.notna().to_numpy()
        if not valid_time.any():
            continue
        chunk = chunk.loc[valid_time].copy()
        chunk_time = chunk_time.loc[valid_time]
        order = np.argsort(chunk_time.astype("int64").to_numpy(), kind="stable")
        chunk = chunk.iloc[order].reset_index(drop=True)
        cyber_ns = chunk_time.iloc[order].astype("int64").to_numpy()
        valid_time_rows += len(chunk)
        if previous_chunk_last_ns is not None and cyber_ns[0] < previous_chunk_last_ns:
            globally_monotonic = False
        previous_chunk_last_ns = int(cyber_ns[-1])

        # Each event belongs to exactly one left-open/right-closed grid window.
        # Missing grid endpoints (physical gaps) are rejected below.
        event_endpoint = ((cyber_ns + step_ns - 1) // step_ns) * step_ns
        anchor = np.searchsorted(grid_ns, event_endpoint, side="left")
        inside = anchor < n_anchor
        inside_indices = np.flatnonzero(inside)
        if len(inside_indices):
            inside[inside_indices] &= grid_ns[anchor[inside_indices]] == event_endpoint[inside_indices]
        if not inside.any():
            continue
        chunk = chunk.loc[inside].reset_index(drop=True)
        anchor = anchor[inside]
        cyber_ns = cyber_ns[inside]
        endpoint = grid_ns[anchor]
        prev_endpoint = endpoint - step_ns
        causal = (cyber_ns > prev_endpoint) & (cyber_ns <= endpoint)
        chunk = chunk.loc[causal].reset_index(drop=True)
        anchor = anchor[causal]
        if len(chunk) == 0:
            continue
        interval_rows += len(chunk)
        np.add.at(event_count, anchor, 1)
        interval_width = np.full(np.sum(causal), step_ns, dtype=np.int64)
        relative_position = (cyber_ns[causal] - prev_endpoint[causal]) / interval_width
        subbin = np.minimum(
            (relative_position * CAUSAL_SUBBINS).astype(np.int64),
            CAUSAL_SUBBINS - 1,
        )
        np.add.at(subbin_count, (anchor, subbin), 1)

        for feature_index, column in enumerate(numeric_columns):
            values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            a = anchor[finite]
            v = values[finite]
            np.add.at(valid_count[:, feature_index], a, 1)
            np.add.at(sums[:, feature_index], a, v)
            np.add.at(sumsq[:, feature_index], a, v * v)
            # Input order is chronological.  Set the first observation only,
            # while repeated assignment deliberately retains the last one.
            unseen = np.isnan(first[a, feature_index])
            if unseen.any():
                for ai, vi in zip(a[unseen], v[unseen]):
                    if np.isnan(first[ai, feature_index]):
                        first[ai, feature_index] = vi
            last[a, feature_index] = v

        categorical_values = [stable_content_code(chunk[column]) for column in categorical_columns]
        if categorical_columns:
            joint_text = chunk[categorical_columns[0]].fillna("<missing>").astype(str)
            for column in categorical_columns[1:]:
                joint_text = joint_text.str.cat(
                    chunk[column].fillna("<missing>").astype(str), sep="|"
                )
            categorical_values.append(stable_content_code(joint_text))
        for feature_index, values in enumerate(categorical_values):
            values = np.asarray(values, dtype=np.float64)
            np.add.at(categorical_count[:, feature_index], anchor, 1)
            np.add.at(categorical_sum[:, feature_index], anchor, values)
            np.add.at(categorical_sumsq[:, feature_index], anchor, values * values)
            categorical_last[anchor, feature_index] = values
            bits = np.left_shift(
                np.uint64(1), (values.astype(np.uint64) % np.uint64(64))
            )
            np.bitwise_or.at(categorical_bucket_mask[:, feature_index], anchor, bits)
        if spec.label_source == "cyber":
            target_last[anchor] = canonical_labels(chunk[spec.label_column]).to_numpy()

        peak_rss = max(peak_rss, process.memory_info().rss)
        print(
            f"[{spec.name}] chunk={chunk_index + 1} input={total_rows:,} "
            f"causal-events={interval_rows:,}",
            flush=True,
        )

    if not globally_monotonic:
        raise RuntimeError(
            f"{spec.name}: cyber chunks overlap in time; sort the raw stream before fusion"
        )
    used = event_count > 0
    if spec.label_source == "cyber":
        used &= pd.notna(target_last)
    used_indices = np.flatnonzero(used)
    if len(used_indices) == 0:
        raise RuntimeError(f"{spec.name}: no non-empty causal grid intervals")

    output = pd.DataFrame({
        "Time": pd.to_datetime(grid_ns[used], utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "cyber_event_count": event_count[used].astype(np.float64),
        "causal_interval_s": np.full(np.sum(used), CAUSAL_GRID_SECONDS, dtype=np.float64),
        "physical_observation_age_s": grid_physical_age_s[used].astype(np.float64),
    })
    used_phy_index = grid_phy_index[used]
    used_source_segment = grid_source_segment[used]
    physical_changed = np.ones(np.sum(used), dtype=np.float64)
    if len(physical_changed) > 1:
        physical_changed[1:] = (
            (used_phy_index[1:] != used_phy_index[:-1])
            | (used_source_segment[1:] != used_source_segment[:-1])
        ).astype(np.float64)
    output["physical_observation_changed"] = physical_changed
    for subbin_index in range(CAUSAL_SUBBINS):
        output[f"cyber_subbin_{subbin_index:02d}_event_count"] = subbin_count[
            used, subbin_index
        ].astype(np.float64)
    for feature_index, column in enumerate(numeric_columns):
        count = valid_count[used, feature_index]
        denom = np.maximum(count, 1)
        mean = sums[used, feature_index] / denom
        variance = np.maximum(sumsq[used, feature_index] / denom - mean * mean, 0.0)
        output[f"cyber_{column}_mean"] = np.where(count > 0, mean, np.nan)
        output[f"cyber_{column}_std"] = np.where(count > 0, np.sqrt(variance), np.nan)
        output[f"cyber_{column}_last"] = last[used, feature_index]
        output[f"cyber_{column}_delta"] = last[used, feature_index] - first[used, feature_index]
    for feature_index, column in enumerate(categorical_aggregate_names):
        count = categorical_count[used, feature_index]
        denom = np.maximum(count, 1)
        mean = categorical_sum[used, feature_index] / denom
        variance = np.maximum(
            categorical_sumsq[used, feature_index] / denom - mean * mean, 0.0
        )
        # Exact constants (e.g. WDT is entirely TCP) can otherwise acquire a
        # tiny count-dependent variance through floating-point cancellation.
        # Zeroing numerical dust prevents a model from learning that artifact.
        variance[variance < 1.0e-3] = 0.0
        output[f"cyber_{column}_hash_mean"] = np.where(count > 0, mean, np.nan)
        output[f"cyber_{column}_hash_std"] = np.where(
            count > 0, np.sqrt(variance), np.nan
        )
        output[f"cyber_{column}_last_hash"] = categorical_last[used, feature_index]
        output[f"cyber_{column}_hash_bucket_coverage"] = np.asarray(
            [
                int(value).bit_count()
                for value in categorical_bucket_mask[used, feature_index]
            ],
            dtype=np.float64,
        )
    for column in physical_columns:
        physical_values = pd.to_numeric(
            physical[f"phy_{column}"], errors="coerce"
        ).to_numpy(np.float64)
        current_values = physical_values[used_phy_index]
        previous_values = physical_values[np.maximum(used_phy_index - 1, 0)]
        past_four_values = physical_values[np.maximum(used_phy_index - 4, 0)]
        output[f"phy_{column}"] = current_values
        output[f"phy_{column}_delta_observation_1"] = current_values - previous_values
        output[f"phy_{column}_delta_observation_4"] = current_values - past_four_values
    if spec.label_source == "cyber":
        output["label"] = canonical_labels(pd.Series(target_last[used])).to_numpy()
    else:
        output["label"] = canonical_labels(
            physical.iloc[used_phy_index]["_target_label"]
        ).to_numpy()

    # Temporal context is constructed later as non-crossing causal sequences.
    # Storing overlapping rolling summaries here would hide context across a
    # split boundary and unnecessarily multiply correlated features.
    output_time = pd.to_datetime(output["Time"], utc=True, errors="raise")
    positive_gaps = output_time.diff().dt.total_seconds().dropna()
    native_interval = float(positive_gaps.median())
    segment_threshold = float(max(5.0, 8.0 * native_interval))
    segment_id = (
        output_time.diff().dt.total_seconds().gt(segment_threshold).cumsum().astype(int)
    )
    output.insert(1, "_segment_id", segment_id)
    output.to_csv(output_path, index=False)

    labels = output["label"]
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    intervals = np.diff(phy_ns) / 1.0e9
    manifest = {
        "dataset": spec.name,
        "sources": {
            "cyber": {"file": spec.cyber_file, "sha256": sha256_file(cyber_path)},
            "physical": {"file": spec.physical_file, "sha256": sha256_file(physical_path)},
        },
        "output": {
            "file": spec.clock_output_file,
            "sha256": sha256_file(output_path),
            "rows": int(len(output)),
            "columns": int(len(output.columns)),
        },
        "causal_fusion": {
            "alignment": "fixed-grid causal interval aggregation with past-only physical as-of join",
            "invariant": "g-0.2 s < cyber_timestamp <= g; physical_timestamp <= g; output available at g",
            "interval_definition": "fixed 0.2-second non-overlapping windows (same rule for every dataset)",
            "network_statistics": [
                "event_count", "10 causal sub-bin counts", "mean", "std",
                "last", "last_minus_first", "categorical hash distribution",
                "categorical bucket coverage"
            ],
            "interpolation": "none",
            "future_fill": "none",
            "label_guided_alignment": "none",
            "physical_interval_seconds_median": float(np.median(intervals)),
            "physical_interval_seconds_p05": float(np.quantile(intervals, 0.05)),
            "physical_interval_seconds_p95": float(np.quantile(intervals, 0.95)),
            "future_match_count": 0,
            "grid_interval_seconds": CAUSAL_GRID_SECONDS,
            "empty_grid_intervals_discarded": int(len(grid_ns) - len(output)),
            "maximum_physical_age_seconds": spec.max_staleness_seconds,
            "physical_age_seconds_p95": float(np.quantile(grid_physical_age_s[used], 0.95)),
            "segment_gap_threshold_seconds": segment_threshold,
            "segments": int(segment_id.nunique()),
        },
        "rows": {
            "cyber_raw": int(total_rows),
            "cyber_valid_timestamp": int(valid_time_rows),
            "cyber_in_physical_intervals": int(interval_rows),
            "nonempty_physical_intervals": int(len(output)),
            "physical": physical_stats,
        },
        "feature_policy": {
            "numeric_cyber": numeric_columns,
            "categorical_cyber": categorical_columns,
            "categorical_encoding": "fixed content hash; no learned vocabulary",
            "physical": physical_columns,
            "physical_derived": [
                "current real observation",
                "current minus previous real observation",
                "current minus fourth previous real observation",
                "observation age and change indicator",
            ],
            "missing_value_imputation": "not performed here; train-only in experiment",
            "scaling": "not performed here; train-only in experiment",
        },
        "label_policy": {
            "source_domain": spec.label_source,
            "source_column": spec.label_column,
            "label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        },
        "chronological_split": split_summary(labels),
    }
    runtime = {
        "dataset": spec.name,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_mib_observed_at_chunk_boundaries": peak_rss / (1024.0**2),
        "input_rows_per_second": total_rows / wall_seconds,
        "output_rows_per_second": len(output) / wall_seconds,
        "chunksize": chunksize,
    }
    return manifest, runtime


def fuse_one(root: Path, spec: FusionSpec, chunksize: int) -> tuple[dict[str, object], dict[str, object]]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss

    cyber_path = root / spec.cyber_file
    physical_path = root / spec.physical_file
    output_path = root / spec.output_file
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if not cyber_path.exists() or not physical_path.exists():
        raise FileNotFoundError(f"Missing source pair: {cyber_path}, {physical_path}")
    # This partial file is owned exclusively by this script and is never a source file.
    temporary_path.unlink(missing_ok=True)

    physical, physical_columns, physical_stats = prepare_physical(physical_path, spec)
    peak_rss = max(peak_rss, process.memory_info().rss)
    source_header = pd.read_csv(cyber_path, nrows=0)
    source_columns = [str(column).strip() for column in source_header.columns]
    numeric_columns, categorical_columns = feature_schema(source_columns, spec)
    if spec.label_source == "cyber" and spec.label_column not in source_columns:
        raise ValueError(f"Cyber label {spec.label_column!r} is absent from {cyber_path.name}")

    total_rows = valid_time_rows = matched_rows = stale_rows = 0
    negative_age_rows = 0
    wrote_header = False
    previous_output_time: pd.Timestamp | None = None
    globally_monotonic = True
    label_parts: list[pd.Series] = []
    age_min = np.inf
    age_max = -np.inf

    reader = pd.read_csv(cyber_path, chunksize=chunksize, low_memory=False)
    for chunk_index, chunk in enumerate(reader):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        total_rows += len(chunk)
        chunk["_cyber_time"] = parse_times(chunk["Time"])
        chunk = chunk.dropna(subset=["_cyber_time"]).copy()
        valid_time_rows += len(chunk)
        chunk = chunk.sort_values("_cyber_time", kind="stable").reset_index(drop=True)
        if chunk.empty:
            continue
        if previous_output_time is not None and chunk["_cyber_time"].iloc[0] < previous_output_time:
            globally_monotonic = False

        left = chunk[["_cyber_time"]].copy()
        aligned = pd.merge_asof(
            left,
            physical,
            left_on="_cyber_time",
            right_on="_phy_time",
            direction="backward",
            allow_exact_matches=True,
        )
        age = (aligned["_cyber_time"] - aligned["_phy_time"]).dt.total_seconds()
        negative_age_rows += int((age < 0).sum())
        valid = age.notna() & age.ge(0.0) & age.le(spec.max_staleness_seconds)
        stale_rows += int((~valid).sum())
        if not valid.any():
            previous_output_time = chunk["_cyber_time"].iloc[-1]
            continue
        chunk = chunk.loc[valid.to_numpy()].reset_index(drop=True)
        aligned = aligned.loc[valid].reset_index(drop=True)
        age = age.loc[valid].reset_index(drop=True)
        matched_rows += len(chunk)
        age_min = min(age_min, float(age.min()))
        age_max = max(age_max, float(age.max()))

        output = pd.DataFrame({
            "Time": chunk["_cyber_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        })
        for column in numeric_columns:
            output[f"cyber_{column}"] = pd.to_numeric(chunk[column], errors="coerce")
        for column in categorical_columns:
            output[f"cyber_{column}_hash"] = stable_content_code(chunk[column])
        for column in physical_columns:
            output[f"phy_{column}"] = pd.to_numeric(aligned[f"phy_{column}"], errors="coerce")
        output["phy_alignment_age_s"] = age.to_numpy(np.float64)
        if spec.label_source == "cyber":
            output["label"] = canonical_labels(chunk[spec.label_column]).to_numpy()
        else:
            output["label"] = canonical_labels(aligned["_target_label"]).to_numpy()
        label_parts.append(output["label"].copy())
        output.to_csv(temporary_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        previous_output_time = chunk["_cyber_time"].iloc[-1]
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(
            f"[{spec.name}] chunk={chunk_index + 1} input={total_rows:,} "
            f"matched={matched_rows:,} stale/unmatched={stale_rows:,}",
            flush=True,
        )

    if not wrote_header:
        raise RuntimeError(f"{spec.name}: no rows survived causal alignment")
    if not globally_monotonic:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.name}: cyber input is not globally chronological across chunks; "
            "refusing to emit a potentially misordered stream"
        )
    if negative_age_rows:
        temporary_path.unlink(missing_ok=True)
        raise AssertionError(f"{spec.name}: {negative_age_rows} future physical matches detected")
    os.replace(temporary_path, output_path)

    labels = pd.concat(label_parts, ignore_index=True)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    output_hash = sha256_file(output_path)
    manifest = {
        "dataset": spec.name,
        "sources": {
            "cyber": {"file": spec.cyber_file, "sha256": sha256_file(cyber_path)},
            "physical": {"file": spec.physical_file, "sha256": sha256_file(physical_path)},
        },
        "output": {
            "file": spec.output_file,
            "sha256": output_hash,
            "rows": int(matched_rows),
            "columns": int(len(pd.read_csv(output_path, nrows=0).columns)),
        },
        "causal_fusion": {
            "alignment": "backward as-of / zero-order hold",
            "invariant": "physical_timestamp <= cyber_timestamp",
            "interpolation": "none",
            "future_fill": "none",
            "clock_offset_fitting": "none",
            "label_guided_alignment": "none",
            "max_staleness_seconds": spec.max_staleness_seconds,
            "alignment_age_seconds_min": float(age_min),
            "alignment_age_seconds_max": float(age_max),
            "future_match_count": int(negative_age_rows),
        },
        "rows": {
            "cyber_raw": int(total_rows),
            "cyber_valid_timestamp": int(valid_time_rows),
            "causally_matched": int(matched_rows),
            "unmatched_or_stale": int(stale_rows),
            "physical": physical_stats,
        },
        "feature_policy": {
            "numeric_cyber": numeric_columns,
            "categorical_cyber": categorical_columns,
            "categorical_encoding": "fixed content hash; no learned vocabulary",
            "physical": physical_columns,
            "missing_value_imputation": "not performed here; train-only in experiment",
            "scaling": "not performed here; train-only in experiment",
        },
        "label_policy": {
            "source_domain": spec.label_source,
            "source_column": spec.label_column,
            "label_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        },
        "chronological_split": split_summary(labels),
    }
    runtime = {
        "dataset": spec.name,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_mib_observed_at_chunk_boundaries": peak_rss / (1024.0**2),
        "input_rows_per_second": total_rows / wall_seconds,
        "output_rows_per_second": matched_rows / wall_seconds,
        "chunksize": chunksize,
    }
    return manifest, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_ROOT,
        help="Directory containing the six raw CSV streams",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_ROOT,
        help="Directory receiving fused CSVs and manifests",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(SPECS),
        default=list(SPECS),
        help="Datasets to rebuild (default: all three)",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    if args.chunksize < 10_000:
        parser.error("--chunksize must be at least 10000")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "combinenew_manifest.json"
    runtime_path = output_root / "combinenew_runtime.csv"
    manifests: dict[str, object] = {}
    retained_specs: dict[str, object] = {}
    runtimes: list[dict[str, object]] = []
    # A targeted rerun replaces only the requested datasets, so a corrected
    # small-dataset run does not discard the auditable WDT entry.
    if manifest_path.exists() and set(args.datasets) != set(SPECS):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.update(
            {
                key: value
                for key, value in previous.get("datasets", {}).items()
                if key not in args.datasets
            }
        )
        retained_specs.update(
            {
                key: value
                for key, value in previous.get("dataset_specs", {}).items()
                if key not in args.datasets
            }
        )
        if runtime_path.exists():
            old_runtime = pd.read_csv(runtime_path)
            old_runtime = old_runtime.loc[~old_runtime["dataset"].isin(args.datasets)]
            runtimes.extend(old_runtime.to_dict("records"))
    for name in args.datasets:
        manifest, runtime = fuse_on_physical_clock(
            source_root, output_root, SPECS[name], args.chunksize
        )
        manifests[name] = manifest
        runtimes.append(runtime)

    envelope = {
        "protocol_version": "CAFOAD unified 200-ms causal-grid fusion revision 3.0",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "psutil": psutil.__version__,
        },
        "fusion_script": {
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "dataset_specs": {
            **retained_specs,
            **{name: asdict(SPECS[name]) for name in args.datasets},
        },
        "datasets": manifests,
    }
    manifest_path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(runtimes).to_csv(runtime_path, index=False)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
