from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_amp(device: torch.device, disable_amp: bool = False) -> tuple[bool, str]:
    if disable_amp:
        return False, device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    if device.type == "cuda":
        return True, "cuda"
    return False, "cpu"


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_init_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("student_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            f"Initialization checkpoint {checkpoint_path} does not contain student_model_state_dict or model_state_dict."
        )
    model_state = model.state_dict()
    compatible_state = {}
    skipped_shape_mismatch = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        compatible_state[key] = value
    load_result = model.load_state_dict(compatible_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def should_report_progress(current_step: int, total_steps: int, every_steps: int) -> bool:
    if current_step <= 1 or current_step >= total_steps:
        return True
    if every_steps > 0 and current_step % every_steps == 0:
        return True
    completed_percent = int((current_step * 100) / max(total_steps, 1))
    previous_percent = int(((current_step - 1) * 100) / max(total_steps, 1))
    return completed_percent != previous_percent and completed_percent % 10 == 0


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_stage(message: str) -> None:
    print(message, flush=True)


def safe_probability_metric(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    if np.unique(y_true).size < 2:
        return float("nan")
    if metric_name == "auroc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "auprc":
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(f"Unsupported metric: {metric_name}")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": safe_probability_metric("auroc", y_true, y_prob),
        "auprc": safe_probability_metric("auprc", y_true, y_prob),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "balanced_accuracy",
) -> float:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    quantile_grid = np.quantile(y_prob, np.linspace(0.0, 1.0, 501))
    dense_grid = np.linspace(0.001, 0.999, 999)
    candidate_thresholds = np.unique(np.clip(np.concatenate([dense_grid, quantile_grid]), 0.0, 1.0))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if objective == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif objective == "balanced_accuracy":
            score = sensitivity + specificity
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
