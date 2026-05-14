from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd


def create_split_masks(
    summary_df: pd.DataFrame,
    split_groups: dict[str, list[str]],
    group_column: str = "record_id",
) -> dict[str, np.ndarray]:
    group_values = summary_df[group_column].astype(str)
    return {
        split_name: group_values.isin(groups).to_numpy()
        for split_name, groups in split_groups.items()
    }


def stratified_record_split(
    summary_df: pd.DataFrame,
    val_record_count: int = 5,
    test_record_count: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    records = (
        summary_df.groupby("record_id", as_index=False)
        .agg(label=("label", "max"))
        .reset_index(drop=True)
    )
    pos_records = records.loc[records["label"] == 1, "record_id"].tolist()
    neg_records = records.loc[records["label"] == 0, "record_id"].tolist()

    rng = random.Random(seed)
    rng.shuffle(pos_records)
    rng.shuffle(neg_records)

    total_records = len(records)
    if total_records < 3:
        raise ValueError(
            "Patient-wise split requires at least 3 unique record_id values. "
            "Use --split-mode random_windows for single-record smoke tests."
        )

    if not pos_records or not neg_records:
        shuffled_records = records["record_id"].tolist()
        rng.shuffle(shuffled_records)
        test_records = shuffled_records[:test_record_count]
        val_records = shuffled_records[test_record_count : test_record_count + val_record_count]
        train_records = shuffled_records[test_record_count + val_record_count :]
        return {
            "train": sorted(train_records),
            "val": sorted(val_records),
            "test": sorted(test_records),
        }

    test_pos = round(test_record_count * len(pos_records) / total_records)
    test_neg = test_record_count - test_pos
    val_pos = round(val_record_count * len(pos_records) / total_records)
    val_neg = val_record_count - val_pos

    test_records = pos_records[:test_pos] + neg_records[:test_neg]
    val_records = pos_records[test_pos : test_pos + val_pos] + neg_records[test_neg : test_neg + val_neg]
    train_records = pos_records[test_pos + val_pos :] + neg_records[test_neg + val_neg :]

    return {
        "train": sorted(train_records),
        "val": sorted(val_records),
        "test": sorted(test_records),
    }


def create_random_window_split_masks(
    summary_df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("--val-fraction + --test-fraction must be smaller than 1.")

    labels = summary_df["label"].to_numpy(dtype=np.int64)
    indices = np.arange(labels.shape[0])
    rng = np.random.default_rng(seed)

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)

    for class_value in np.unique(labels):
        class_indices = indices[labels == class_value]
        rng.shuffle(class_indices)
        n_items = class_indices.shape[0]
        n_test = max(1, int(round(n_items * test_fraction)))
        n_val = max(1, int(round(n_items * val_fraction)))
        if n_test + n_val >= n_items:
            n_test = max(1, n_items // 5)
            n_val = max(1, n_items // 5)
        n_train = n_items - n_test - n_val
        if n_train <= 0:
            raise ValueError(
                "Not enough windows to create train/val/test splits for all classes. "
                "Reduce --val-fraction/--test-fraction or add more data."
            )

        test_idx = class_indices[:n_test]
        val_idx = class_indices[n_test : n_test + n_val]
        train_idx = class_indices[n_test + n_val :]

        test_mask[test_idx] = True
        val_mask[val_idx] = True
        train_mask[train_idx] = True

    return {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }


def _parse_fold_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        normalized = part.strip()
        if not normalized:
            continue
        values.append(int(normalized))
    if not values:
        raise ValueError("Fold list must contain at least one integer.")
    return values


def create_metadata_fold_split_masks(
    summary_df: pd.DataFrame,
    train_folds: list[int],
    val_folds: list[int],
    test_folds: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    if "strat_fold" not in summary_df.columns:
        raise ValueError("split_mode=metadata_folds requires a 'strat_fold' column in the summary CSV.")

    fold_values = pd.to_numeric(summary_df["strat_fold"], errors="raise").astype(int)
    split_masks = {
        "train": fold_values.isin(train_folds).to_numpy(),
        "val": fold_values.isin(val_folds).to_numpy(),
        "test": fold_values.isin(test_folds).to_numpy(),
    }
    split_groups = {
        "train": [str(value) for value in train_folds],
        "val": [str(value) for value in val_folds],
        "test": [str(value) for value in test_folds],
    }
    return split_masks, split_groups


def validate_split_masks(summary_df: pd.DataFrame, split_masks: dict[str, np.ndarray]) -> None:
    for split_name, mask in split_masks.items():
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Split '{split_name}' is empty.")
        labels = summary_df.loc[mask, "label"].to_numpy(dtype=np.int64)
        if np.unique(labels).size < 2:
            raise ValueError(
                f"Split '{split_name}' contains only one class. "
                "Use a different split configuration or add more data."
            )


def supports_record_level_metrics(summary_df: pd.DataFrame) -> bool:
    return bool(summary_df.groupby("record_id")["label"].nunique().max() <= 1)


def infer_record_grouping(summary_df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    if supports_record_level_metrics(summary_df):
        return "record_id", summary_df["record_id"].astype(str)

    if {"record_id", "event_id"}.issubset(summary_df.columns):
        event_uniqueness = summary_df.groupby(["record_id", "event_id"])["label"].nunique().max()
        if bool(event_uniqueness <= 1):
            event_values = pd.to_numeric(summary_df["event_id"], errors="coerce").astype("Int64").astype(str)
            group_ids = summary_df["record_id"].astype(str) + "::event_" + event_values
            return "record_id+event_id", group_ids

    if "signal_file_name" in summary_df.columns:
        signal_uniqueness = summary_df.groupby("signal_file_name")["label"].nunique().max()
        if bool(signal_uniqueness <= 1):
            return "signal_file_name", summary_df["signal_file_name"].astype(str)

    return None, None


def _allocate_group_counts(block_sizes: list[int], total_target: int) -> list[int]:
    if total_target <= 0 or not block_sizes:
        return [0] * len(block_sizes)

    total_available = sum(block_sizes)
    if total_target >= total_available:
        return block_sizes.copy()

    raw_counts = [total_target * size / total_available for size in block_sizes]
    base_counts = [min(size, int(math.floor(raw))) for size, raw in zip(block_sizes, raw_counts)]
    remainder = total_target - sum(base_counts)

    order = sorted(
        range(len(block_sizes)),
        key=lambda index: (raw_counts[index] - base_counts[index], block_sizes[index]),
        reverse=True,
    )
    for index in order:
        if remainder <= 0:
            break
        if base_counts[index] < block_sizes[index]:
            base_counts[index] += 1
            remainder -= 1
    return base_counts


def choose_split_group_column(summary_df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in summary_df.columns:
            raise ValueError(f"Requested split group column '{requested}' is not present in the summary CSV.")
        return requested

    if "subject_id" in summary_df.columns and "record_id" in summary_df.columns:
        if int(summary_df.groupby("record_id")["label"].nunique().max()) > 1:
            return "subject_id"
    if "record_id" in summary_df.columns:
        return "record_id"
    if "subject_id" in summary_df.columns:
        return "subject_id"
    raise ValueError("Unable to auto-select a split grouping column because neither record_id nor subject_id exists.")


def stratified_group_split(
    summary_df: pd.DataFrame,
    group_column: str,
    val_group_count: int = 5,
    test_group_count: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    group_summary = (
        summary_df.groupby(group_column, as_index=False)
        .agg(
            positive_rate=("label", "mean"),
            positive_count=("label", "sum"),
            segment_count=("label", "size"),
        )
        .sort_values(group_column)
        .reset_index(drop=True)
    )
    total_groups = int(group_summary.shape[0])
    if total_groups < 3:
        raise ValueError(f"Need at least 3 unique groups in '{group_column}' to create train/val/test splits.")
    if val_group_count + test_group_count >= total_groups:
        raise ValueError(
            f"Requested val/test group counts ({val_group_count}+{test_group_count}) leave no groups for training."
        )

    if group_summary["positive_rate"].nunique() <= 1:
        group_summary["stratum"] = 0
    else:
        group_summary["stratum"] = pd.qcut(
            group_summary["positive_rate"].rank(method="first"),
            q=min(5, total_groups),
            labels=False,
            duplicates="drop",
        )

    blocks = [block[group_column].astype(str).tolist() for _, block in group_summary.groupby("stratum", sort=True)]
    rng = random.Random(seed)
    for block in blocks:
        rng.shuffle(block)

    test_counts = _allocate_group_counts([len(block) for block in blocks], test_group_count)
    remaining_sizes = [len(block) - test_count for block, test_count in zip(blocks, test_counts)]
    val_counts = _allocate_group_counts(remaining_sizes, val_group_count)

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []
    for block, test_count, val_count in zip(blocks, test_counts, val_counts):
        test_groups.extend(block[:test_count])
        val_groups.extend(block[test_count : test_count + val_count])
        train_groups.extend(block[test_count + val_count :])

    return {
        "train": sorted(train_groups),
        "val": sorted(val_groups),
        "test": sorted(test_groups),
    }

