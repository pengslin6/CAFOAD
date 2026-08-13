"""Six-class prequential IGCPS adaptation and poisoning audit.

The offline checkpoint and every gate threshold are fitted without the held-out
stream.  Each online batch is predicted and logged before any pseudo-label
update.  Ground-truth labels are used only for post-hoc metrics and for
constructing the declared attack-only stress scenarios; they are never read by
the monitor, eligibility rule, or optimiser.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn.functional as F

from multiclass_eventwise_runner import DATASETS, context_rows, load_revision, loader, original_labels
from multiclass_grouped_cv_runner import build_folds


SCENARIOS = (
    "clean",
    "sustained_gradual_drift",
    "recurring_abrupt_shift",
    "stealth_low_rate_poisoning",
    "attack_induced_drift",
)
MODES = ("frozen", "ungated", "guarded")


def train_checkpoint(revision: Any, seed: int, epochs: int, device: torch.device):
    bundle = original_labels(revision.load_dataset("IGCPS", DATASETS["IGCPS"]))
    folds, split_audit = build_folds(bundle, seq_len=4, folds=5, block_rows=7)
    parts = folds[0]
    fitted_rows = context_rows(parts["train"], 4)
    reduced, pruning = revision.prune_features_from_training_rows(bundle, fitted_rows)
    prep = revision.TrainOnlyPreprocessor(0.005, 0.995)
    prep.fit(reduced.features[fitted_rows], reduced.labels[fitted_rows])
    transformed = prep.transform(reduced.features)
    model_bundle = revision.with_normal_deviation_schema(
        copy.copy(reduced), transformed.shape[1]
    )
    config = revision.ExperimentConfig(
        seq_len=4,
        epochs=epochs,
        patience=7,
        batch_size=256,
        learning_rate=8e-4,
        weight_decay=2e-4,
    )
    train_loader = loader(
        revision, transformed, bundle.labels, parts["train"], config, True, seed
    )
    validation_loader = loader(
        revision, transformed, bundle.labels, parts["validation"], config, False, seed
    )
    revision.set_deterministic(seed)
    model = revision.CAFOAD(
        **revision.method_model_kwargs("CAFOAD", model_bundle, config)
    )
    weights = torch.ones(len(bundle.class_names), dtype=torch.float32)
    model, training = revision.train_model(
        "CAFOAD", model, train_loader, validation_loader, weights, config, device
    )
    return (
        bundle,
        reduced,
        model_bundle,
        transformed,
        model,
        validation_loader,
        parts,
        {
            "grouped_cv": split_audit,
            "selected_online_fold": 0,
            "train_only_feature_pruning": pruning,
            "preprocessor": prep.to_dict(),
            "training": training,
        },
    )


def windows_from_targets(x: np.ndarray, targets: np.ndarray, seq_len: int = 4) -> np.ndarray:
    return np.stack([x[target - seq_len + 1 : target + 1] for target in targets]).astype(
        np.float32
    )


def apply_scenario(
    windows: np.ndarray,
    truth: np.ndarray,
    stream_positions: np.ndarray,
    stream_length: int,
    physical_indices: np.ndarray,
    normal_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    scenario = apply_scenario.current_scenario
    revised = windows.copy()
    contaminated = np.zeros(len(windows), dtype=bool)
    progress = stream_positions.astype(np.float32) / max(1, stream_length - 1)
    physical = np.asarray(physical_indices, dtype=np.int64)
    if scenario == "clean":
        return revised, contaminated
    if scenario == "sustained_gradual_drift":
        revised[:, :, physical] += (0.70 * progress)[:, None, None]
        contaminated[:] = True
    elif scenario == "recurring_abrupt_shift":
        phase = ((stream_positions // 32) % 2) * 2 - 1
        revised[:, :, physical] += (0.55 * phase.astype(np.float32))[:, None, None]
        contaminated[:] = True
    elif scenario == "stealth_low_rate_poisoning":
        contaminated = (truth != normal_index) & (stream_positions % 17 == 0)
        for index in np.flatnonzero(contaminated):
            revised[index, :, physical] *= 0.45
    elif scenario == "attack_induced_drift":
        contaminated = truth != normal_index
        for index in np.flatnonzero(contaminated):
            revised[index, :, physical] += 0.65 * progress[index]
    else:
        raise ValueError(scenario)
    return np.clip(revised, -20.0, 20.0), contaminated


apply_scenario.current_scenario = "clean"


@torch.no_grad()
def validation_gate_calibration(
    model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, device: torch.device
) -> dict[str, Any]:
    confidence: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    reconstruction: list[np.ndarray] = []
    normal_confidence: list[np.ndarray] = []
    normal_reconstruction: list[np.ndarray] = []
    batch_signals: list[list[float]] = []
    model.eval()
    for x, y in validation_loader:
        x = x.to(device)
        output = model(x)
        probability = torch.softmax(output["logits"], dim=-1)
        conf = probability.max(dim=-1).values
        ent = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=-1)
        recon = ((output["reconstruction"] - x[:, -1]) ** 2).mean(dim=-1)
        confidence.append(conf.cpu().numpy())
        entropy.append(ent.cpu().numpy())
        reconstruction.append(recon.cpu().numpy())
        normal = y.numpy() == 0
        if normal.any():
            normal_confidence.append(conf.cpu().numpy()[normal])
            normal_reconstruction.append(recon.cpu().numpy()[normal])
        batch_signals.append([
            float(recon.mean().cpu()),
            float(ent.mean().cpu()),
            float((1.0 - conf).mean().cpu()),
        ])
    signal_array = np.asarray(batch_signals, dtype=np.float64)
    center = np.median(signal_array, axis=0)
    mad = 1.4826 * np.median(np.abs(signal_array - center), axis=0)
    upper = center + 3.0 * np.maximum(mad, 1.0e-6)
    return {
        "confidence_min": float(
            np.clip(np.quantile(np.concatenate(normal_confidence), 0.05), 0.85, 0.95)
        ),
        "reconstruction_max": float(np.quantile(np.concatenate(normal_reconstruction), 0.99)),
        "entropy_max": float(max(0.10, np.quantile(np.concatenate(entropy), 0.90))),
        "signal_center": center.tolist(),
        "signal_upper": upper.tolist(),
        "fit_scope": "validation windows only",
        "test_labels_used": False,
    }


def update_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    windows: torch.Tensor,
    pseudo: torch.Tensor,
    eligible: np.ndarray,
    anchor: dict[str, torch.Tensor],
    mode: str,
) -> int:
    chosen = torch.as_tensor(eligible, dtype=torch.bool, device=windows.device)
    if not bool(chosen.any()):
        return 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(windows[chosen])
    loss = F.cross_entropy(output["logits"], pseudo[chosen])
    loss = loss + 0.02 * F.mse_loss(output["reconstruction"], windows[chosen, -1])
    if mode == "guarded":
        anchor_penalty = torch.zeros((), device=windows.device)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in anchor:
                anchor_penalty = anchor_penalty + (parameter - anchor[name]).square().mean()
        loss = loss + 0.02 * anchor_penalty
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # The projection prototypes already belong to the original FFAD path.
    # Updating them by a small EMA makes prototype contamination observable.
    with torch.no_grad():
        projection = F.normalize(output["projection"], dim=-1)
        selected_labels = pseudo[chosen]
        for class_index in torch.unique(selected_labels):
            mask = selected_labels == class_index
            candidate = F.normalize(projection[mask].mean(dim=0), dim=0)
            old = model.class_prototypes[class_index]
            model.class_prototypes[class_index].copy_(
                F.normalize(0.98 * old + 0.02 * candidate, dim=0)
            )
    return int(chosen.sum().item())


def evaluate_stream(
    revision: Any,
    base_model: torch.nn.Module,
    x: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    class_names: list[str],
    physical_indices: np.ndarray,
    calibration: dict[str, Any],
    scenario: str,
    mode: str,
    device: torch.device,
    batch_size: int = 16,
    threshold_multiplier: float = 3.0,
    action_min_signals: int = 2,
    persistence_batches: int = 2,
    evaluation_seed: int = 13,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    revision.set_deterministic(evaluation_seed)
    model = copy.deepcopy(base_model).to(device)
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-4,
        weight_decay=2.0e-4,
    )
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    consecutive_drift = 0
    update_batches = update_samples = admitted_contaminated = 0
    warning_batches = action_batches = 0
    normal_index = next(
        (index for index, name in enumerate(class_names) if name.lower() == "normal"), 0
    )
    for batch_index, start in enumerate(range(0, len(targets), batch_size)):
        current_targets = targets[start : start + batch_size]
        truth = labels[current_targets]
        positions = np.arange(start, start + len(current_targets), dtype=np.int64)
        windows = windows_from_targets(x, current_targets)
        apply_scenario.current_scenario = scenario
        windows, contaminated = apply_scenario(
            windows, truth, positions, len(targets), physical_indices, normal_index
        )
        tensor = torch.from_numpy(windows).to(device)
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            probability = torch.softmax(output["logits"], dim=-1)
            prediction = probability.argmax(dim=-1)
            confidence = probability.max(dim=-1).values
            entropy = -(
                probability * probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            reconstruction = (
                (output["reconstruction"] - tensor[:, -1]).square().mean(dim=-1)
            )
        # Strict prequential ordering: the following rows are stored before
        # either an eligibility decision or an optimiser step is made.
        truth_np = truth.copy()
        prediction_np = prediction.cpu().numpy()
        truths.append(truth_np)
        predictions.append(prediction_np)

        signals = np.array([
            float(reconstruction.mean().cpu()),
            float(entropy.mean().cpu()),
            float((1.0 - confidence).mean().cpu()),
        ])
        center = np.asarray(calibration["signal_center"], dtype=np.float64)
        fitted_three_mad = np.asarray(calibration["signal_upper"], dtype=np.float64) - center
        active_upper = center + (threshold_multiplier / 3.0) * fitted_three_mad
        exceeded = signals > active_upper
        warning = bool(np.sum(exceeded) >= 1)
        if warning:
            consecutive_drift += 1
            warning_batches += 1
        else:
            consecutive_drift = 0
        action = bool(
            np.sum(exceeded) >= action_min_signals
            and consecutive_drift >= persistence_batches
        )
        action_batches += int(action)
        if mode == "frozen":
            eligible = np.zeros(len(truth), dtype=bool)
        elif mode == "ungated":
            eligible = confidence.cpu().numpy() >= 0.60
        elif mode == "guarded":
            eligible = (
                action
                & (prediction_np == normal_index)
                & (confidence.cpu().numpy() >= calibration["confidence_min"])
                & (reconstruction.cpu().numpy() <= calibration["reconstruction_max"])
                & (entropy.cpu().numpy() <= calibration["entropy_max"])
            )
        else:
            raise ValueError(mode)
        changed = update_model(
            model, optimizer, tensor, prediction, eligible, anchor, mode
        )
        if changed:
            update_batches += 1
            update_samples += changed
        admitted_contaminated += int(np.sum(contaminated & eligible))
        batch_present = np.unique(truth_np)
        batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
            truth_np,
            prediction_np,
            labels=batch_present,
            zero_division=0,
        )
        batch_normal = truth_np == normal_index
        batch_attack = ~batch_normal
        false_alarm_count = int(np.sum(batch_normal & (prediction_np != normal_index)))
        false_negative_count = int(np.sum(batch_attack & (prediction_np == normal_index)))
        batch_far = float(
            false_alarm_count
            / max(1, int(batch_normal.sum()))
        )
        batch_fnr = float(false_negative_count / max(1, int(batch_attack.sum())))
        batch_rows.append({
            "scenario": scenario,
            "mode": mode,
            "batch": batch_index,
            "samples": len(truth),
            "accuracy": float(accuracy_score(truth_np, prediction_np)),
            "macro_precision": float(batch_precision.mean()),
            "macro_recall": float(batch_recall.mean()),
            "macro_f1": float(batch_f1.mean()),
            "false_alarm_rate": batch_far,
            "false_negative_rate": batch_fnr,
            "normal_windows": int(batch_normal.sum()),
            "attack_windows": int(batch_attack.sum()),
            "false_alarm_count": false_alarm_count,
            "false_negative_count": false_negative_count,
            "specificity": 1.0 - batch_far,
            "mean_reconstruction": signals[0],
            "mean_entropy": signals[1],
            "mean_one_minus_confidence": signals[2],
            "warning": warning,
            "action": action,
            "eligible_samples": int(np.sum(eligible)),
            "contaminated_samples": int(np.sum(contaminated)),
            "admitted_contaminated_samples": int(np.sum(contaminated & eligible)),
            "prediction_logged_before_update": True,
        })
        for local_index in range(len(current_targets)):
            prediction_rows.append({
                "scenario": scenario,
                "mode": mode,
                "record_type": "prediction",
                "stream_position": int(start + local_index),
                "batch": int(batch_index),
                "target": int(current_targets[local_index]),
                "truth": class_names[int(truth_np[local_index])],
                "prediction": class_names[int(prediction_np[local_index])],
                "admitted": bool(eligible[local_index]),
                "contaminated": bool(contaminated[local_index]),
                "prediction_logged_before_update": True,
            })

    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    class_ids = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=class_ids, zero_division=0
    )
    normal = truth == normal_index
    attack = ~normal
    third = max(1, len(truth) // 3)
    early_truth, early_prediction = truth[:third], prediction[:third]
    late_truth, late_prediction = truth[-third:], prediction[-third:]
    early_attack = early_truth != normal_index
    late_attack = late_truth != normal_index
    early_fnr = float(
        np.sum(early_attack & (early_prediction == normal_index))
        / max(1, int(early_attack.sum()))
    )
    late_fnr = float(
        np.sum(late_attack & (late_prediction == normal_index))
        / max(1, int(late_attack.sum()))
    )
    summary = {
        "scenario": scenario,
        "mode": mode,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "far": float(np.sum(normal & (prediction != normal_index)) / max(1, normal.sum())),
        "fnr": float(np.sum(attack & (prediction == normal_index)) / max(1, attack.sum())),
        "early_third_fnr": early_fnr,
        "late_third_fnr": late_fnr,
        "late_minus_early_fnr": late_fnr - early_fnr,
        "early_third_accuracy": float(accuracy_score(early_truth, early_prediction)),
        "late_third_accuracy": float(accuracy_score(late_truth, late_prediction)),
        "warning_batches": warning_batches,
        "action_batches": action_batches,
        "update_batches": update_batches,
        "update_samples": update_samples,
        "contaminated_samples": int(sum(row["contaminated_samples"] for row in batch_rows)),
        "admitted_contaminated_samples": admitted_contaminated,
        "stream_samples": int(len(truth)),
        "prediction_logged_before_update": True,
        "test_labels_used_by_update": False,
        "threshold_multiplier": threshold_multiplier,
        "action_min_signals": action_min_signals,
        "persistence_batches": persistence_batches,
    }
    class_rows = [
        {
            "scenario": scenario,
            "mode": mode,
            "class": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]
    for row in class_rows:
        row["record_type"] = "per_class_summary"
    return summary, batch_rows, class_rows + prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = load_revision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        bundle,
        reduced,
        model_bundle,
        transformed,
        model,
        validation_loader,
        parts,
        audit,
    ) = train_checkpoint(revision, args.seed, args.epochs, device)
    calibration = validation_gate_calibration(model, validation_loader, device)
    audit["gate_calibration"] = calibration
    audit["online_protocol"] = {
        "task": "six-class IGCPS",
        "source_fold": 0,
        "batch_size": 16,
        "prediction_update_order": "predict and log, then gate, then optional update",
        "ground_truth_use": "post-hoc scoring and declared stress-scenario construction only",
    }
    (args.output / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summaries: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    class_and_contamination: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for mode in MODES:
            print(f"[IGCPS-online] scenario={scenario} mode={mode}", flush=True)
            summary, batch_rows, detail_rows = evaluate_stream(
                revision,
                model,
                transformed,
                bundle.labels,
                parts["test"],
                bundle.class_names,
                model_bundle.physical_indices,
                calibration,
                scenario,
                mode,
                device,
            )
            summaries.append(summary)
            batches.extend(batch_rows)
            class_and_contamination.extend(detail_rows)
    pd.DataFrame(summaries).to_csv(args.output / "online_summary.csv", index=False)
    pd.DataFrame(batches).to_csv(args.output / "online_batch_trace.csv", index=False)
    detail = pd.DataFrame(class_and_contamination)
    detail.loc[detail["class"].notna()].to_csv(
        args.output / "online_per_class.csv", index=False
    )
    detail.to_csv(args.output / "online_detail.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
