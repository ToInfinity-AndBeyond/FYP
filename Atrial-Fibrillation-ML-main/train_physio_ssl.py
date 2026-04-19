from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from physio_ssl_model import CrossModalSSLNet


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


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def stratified_group_split(
    summary_df: pd.DataFrame,
    group_column: str,
    val_group_count: int,
    test_group_count: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    group_summary = (
        summary_df.groupby(group_column, as_index=False)
        .agg(
            positive_rate=("label", "mean"),
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


def create_split_masks(
    summary_df: pd.DataFrame,
    split_groups: dict[str, list[str]],
    group_column: str,
) -> dict[str, np.ndarray]:
    group_values = summary_df[group_column].astype(str)
    return {
        split_name: group_values.isin(groups).to_numpy()
        for split_name, groups in split_groups.items()
    }


def load_and_concat_multimodal_datasets(
    segments_paths: list[Path],
    summary_paths: list[Path],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if len(segments_paths) != len(summary_paths):
        raise ValueError("--segments-path and --summary-path must be provided the same number of times.")

    array_blocks: dict[str, list[np.ndarray]] = {}
    summary_blocks = []
    expected_shapes: dict[str, tuple[int, ...]] = {}

    for segments_path, summary_path in zip(segments_paths, summary_paths):
        arrays = dict(np.load(segments_path))
        summary_df = pd.read_csv(summary_path)
        row_count = summary_df.shape[0]
        if arrays["ppg_segments"].shape[0] != row_count:
            raise ValueError(
                f"Accepted multimodal NPZ and summary CSV row counts do not match for {segments_path} and {summary_path}."
            )

        for key, values in arrays.items():
            trailing_shape = values.shape[1:]
            if key not in expected_shapes:
                expected_shapes[key] = trailing_shape
            elif expected_shapes[key] != trailing_shape:
                raise ValueError(
                    f"All multimodal datasets must agree on shape for {key}. "
                    f"Expected {expected_shapes[key]}, got {trailing_shape} from {segments_path}."
                )
            array_blocks.setdefault(key, []).append(values)
        summary_blocks.append(summary_df)

    merged_arrays = {key: np.concatenate(value_list, axis=0) for key, value_list in array_blocks.items()}
    return merged_arrays, pd.concat(summary_blocks, ignore_index=True)


class PhysioSSLDataset(Dataset):
    def __init__(
        self,
        ppg_waveforms: np.ndarray,
        ecg_waveforms: np.ndarray,
        quality_scores: np.ndarray,
        record_ids: np.ndarray,
        subject_ids: np.ndarray,
    ):
        self.ppg_waveforms = ppg_waveforms.astype(np.float32)
        self.ecg_waveforms = ecg_waveforms.astype(np.float32)
        self.quality_scores = quality_scores.astype(np.float32)
        self.record_ids = record_ids
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return self.ppg_waveforms.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "ppg_waveform": torch.from_numpy(self.ppg_waveforms[index]),
            "ecg_waveform": torch.from_numpy(self.ecg_waveforms[index]),
            "quality_score": torch.tensor(self.quality_scores[index], dtype=torch.float32),
            "record_id": self.record_ids[index],
            "subject_id": self.subject_ids[index],
        }


def _random_roll_batch(batch: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return batch
    shifted = batch.clone()
    shifts = torch.randint(-max_shift, max_shift + 1, (batch.size(0),), device=batch.device)
    for index, shift in enumerate(shifts.tolist()):
        if shift:
            shifted[index] = torch.roll(shifted[index], shifts=shift, dims=0)
    return shifted


def _random_time_mask_batch(batch: torch.Tensor, min_fraction: float, max_fraction: float) -> torch.Tensor:
    masked = batch.clone()
    signal_length = masked.size(1)
    min_width = max(1, int(signal_length * min_fraction))
    max_width = max(min_width, int(signal_length * max_fraction))
    widths = torch.randint(min_width, max_width + 1, (masked.size(0),), device=masked.device)
    for index, width in enumerate(widths.tolist()):
        if width >= signal_length:
            masked[index].zero_()
            continue
        start = torch.randint(0, signal_length - width + 1, (1,), device=masked.device).item()
        fill_value = float(masked[index].mean().item())
        masked[index, start : start + width] = fill_value
    return masked


def augment_ppg_batch(batch: torch.Tensor) -> torch.Tensor:
    augmented = batch.clone()
    scale = torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.92, 1.08)
    noise = torch.randn_like(augmented) * torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.003, 0.02)
    augmented = augmented * scale + noise
    augmented = _random_roll_batch(augmented, max_shift=32)
    if random.random() < 0.5:
        t = torch.linspace(0.0, 1.0, augmented.size(1), device=batch.device).unsqueeze(0)
        freq = torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.2, 1.0)
        phase = torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.0, 2 * math.pi)
        amp = torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.005, 0.035)
        drift = torch.sin(2.0 * math.pi * freq * t + phase) * amp
        augmented = augmented + drift
    if random.random() < 0.4:
        augmented = _random_time_mask_batch(augmented, min_fraction=0.02, max_fraction=0.06)
    return augmented


def augment_ecg_batch(batch: torch.Tensor) -> torch.Tensor:
    augmented = batch.clone()
    scale = torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.95, 1.05)
    noise = torch.randn_like(augmented) * torch.empty((batch.size(0), 1), device=batch.device).uniform_(0.001, 0.01)
    augmented = augmented * scale + noise
    augmented = _random_roll_batch(augmented, max_shift=24)
    if random.random() < 0.3:
        augmented = _random_time_mask_batch(augmented, min_fraction=0.01, max_fraction=0.04)
    return augmented


def symmetric_infonce(anchor: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = anchor @ target.T / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def evaluate_retrieval(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ppg_embeddings = []
    ecg_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            ppg_waveform = batch["ppg_waveform"].to(device)
            ecg_waveform = batch["ecg_waveform"].to(device)
            outputs = model(ppg_waveform, ecg_waveform)
            ppg_embeddings.append(F.normalize(outputs["ppg_embedding_a"], dim=1).cpu())
            ecg_embeddings.append(F.normalize(outputs["ecg_embedding_a"], dim=1).cpu())

    ppg = torch.cat(ppg_embeddings, dim=0)
    ecg = torch.cat(ecg_embeddings, dim=0)
    similarity = ppg @ ecg.T
    labels = torch.arange(similarity.size(0))
    ppg_to_ecg = (similarity.argmax(dim=1) == labels).float().mean().item()
    ecg_to_ppg = (similarity.argmax(dim=0) == labels).float().mean().item()
    diagonal = similarity.diag()
    negative_mean = (
        (similarity.sum() - diagonal.sum()) / max(similarity.numel() - diagonal.numel(), 1)
    ).item()
    return {
        "ppg_to_ecg_top1": float(ppg_to_ecg),
        "ecg_to_ppg_top1": float(ecg_to_ppg),
        "mean_top1": float(0.5 * (ppg_to_ecg + ecg_to_ppg)),
        "mean_positive_cosine": float(diagonal.mean().item()),
        "mean_negative_cosine": float(negative_mean),
    }


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
    temperature: float,
    alpha_cross: float,
    alpha_ppg: float,
    alpha_ecg: float,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_items = 0
    part_totals = {"cross": 0.0, "ppg_ssl": 0.0, "ecg_ssl": 0.0}
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        ppg_waveform = torch.nan_to_num(batch["ppg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)
        ecg_waveform = torch.nan_to_num(batch["ecg_waveform"].to(device), nan=0.0, posinf=0.0, neginf=0.0)

        ppg_view_a = augment_ppg_batch(ppg_waveform)
        ppg_view_b = augment_ppg_batch(ppg_waveform)
        ecg_view_a = augment_ecg_batch(ecg_waveform)
        ecg_view_b = augment_ecg_batch(ecg_waveform)

        with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
            outputs = model(
                ppg_view_a=ppg_view_a,
                ecg_view_a=ecg_view_a,
                ppg_view_b=ppg_view_b,
                ecg_view_b=ecg_view_b,
            )
            cross = symmetric_infonce(outputs["ppg_projection_a"], outputs["ecg_projection_a"], temperature)
            ppg_ssl = symmetric_infonce(outputs["ppg_projection_a"], outputs["ppg_projection_b"], temperature)
            ecg_ssl = symmetric_infonce(outputs["ecg_projection_a"], outputs["ecg_projection_b"], temperature)
            loss = alpha_cross * cross + alpha_ppg * ppg_ssl + alpha_ecg * ecg_ssl

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()

        batch_size = ppg_waveform.size(0)
        total_items += batch_size
        total_loss += float(loss.item()) * batch_size
        part_totals["cross"] += float(cross.detach().item()) * batch_size
        part_totals["ppg_ssl"] += float(ppg_ssl.detach().item()) * batch_size
        part_totals["ecg_ssl"] += float(ecg_ssl.detach().item()) * batch_size

    if total_items == 0:
        raise RuntimeError("All SSL training batches were skipped due to non-finite values.")
    averaged_parts = {key: value / total_items for key, value in part_totals.items()}
    return total_loss / total_items, averaged_parts


def maybe_limit_segments_per_subject(
    arrays: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    max_segments_per_subject: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if max_segments_per_subject <= 0:
        return arrays, summary_df
    rng = np.random.default_rng(seed)
    keep_indices: list[int] = []
    for _, block in summary_df.groupby("subject_id", sort=True):
        block_indices = block.index.to_numpy()
        if block_indices.size > max_segments_per_subject:
            block_indices = np.sort(rng.choice(block_indices, size=max_segments_per_subject, replace=False))
        keep_indices.extend(block_indices.tolist())
    keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    limited_summary = summary_df.iloc[keep_indices].reset_index(drop=True)
    limited_arrays = {key: values[keep_indices] for key, values in arrays.items()}
    return limited_arrays, limited_summary


def build_dataloaders(
    arrays: dict[str, np.ndarray],
    summary_df: pd.DataFrame,
    split_masks: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split_name, mask in split_masks.items():
        dataset = PhysioSSLDataset(
            ppg_waveforms=arrays["ppg_segments"][mask],
            ecg_waveforms=arrays["ecg_segments"][mask],
            quality_scores=np.clip(
                np.nan_to_num(summary_df.loc[mask, "ppg_quality_score"].to_numpy(dtype=np.float32), nan=0.5),
                0.0,
                1.0,
            ),
            record_ids=summary_df.loc[mask, "record_id"].to_numpy(),
            subject_ids=summary_df.loc[mask, "subject_id"].astype(str).to_numpy(),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=0,
        )
    return loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal ECG-PPG self-supervised pretraining on Zenodo bundles.")
    parser.add_argument(
        "--segments-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/physio_distill/physio_multimodal_accepted_segments.npz")],
        help="One or more accepted multimodal NPZ paths.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        nargs="+",
        default=[Path("artifacts/physio_distill/physio_multimodal_accepted_segment_summary.csv")],
        help="One or more accepted multimodal summary CSV paths.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/physio_ssl"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--alpha-cross", type=float, default=1.0)
    parser.add_argument("--alpha-ppg", type=float, default=0.25)
    parser.add_argument("--alpha-ecg", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subjects", type=int, default=4)
    parser.add_argument("--test-subjects", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--max-segments-per-subject",
        type=int,
        default=0,
        help="Optional cap used for smoke/debug runs. 0 keeps all available segments.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision for more stable training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp_enabled, amp_device_type = choose_amp(device, disable_amp=args.disable_amp)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, summary_df = load_and_concat_multimodal_datasets(args.segments_path, args.summary_path)
    arrays, summary_df = maybe_limit_segments_per_subject(
        arrays=arrays,
        summary_df=summary_df,
        max_segments_per_subject=args.max_segments_per_subject,
        seed=args.seed,
    )

    if "subject_id" not in summary_df.columns:
        raise ValueError("SSL pretraining expects subject_id in the multimodal summary CSV.")

    split_groups = stratified_group_split(
        summary_df=summary_df,
        group_column="subject_id",
        val_group_count=args.val_subjects,
        test_group_count=args.test_subjects,
        seed=args.seed,
    )
    split_masks = create_split_masks(summary_df, split_groups, group_column="subject_id")
    loaders = build_dataloaders(arrays, summary_df, split_masks, batch_size=args.batch_size)

    split_sizes = {split_name: int(mask.sum()) for split_name, mask in split_masks.items()}
    print(
        "dataset summary:",
        json.dumps(
            {
                "total_segments": int(summary_df.shape[0]),
                "subject_count": int(summary_df["subject_id"].nunique()),
                "record_count": int(summary_df["record_id"].nunique()),
                "split_subjects": {split_name: len(groups) for split_name, groups in split_groups.items()},
                "split_sizes": split_sizes,
                "max_segments_per_subject": args.max_segments_per_subject,
            }
        ),
        flush=True,
    )

    model = CrossModalSSLNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_epoch = 0
    best_val_score = -math.inf
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, loss_parts = run_training_epoch(
            model=model,
            dataloader=loaders["train"],
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            amp_device_type=amp_device_type,
            temperature=args.temperature,
            alpha_cross=args.alpha_cross,
            alpha_ppg=args.alpha_ppg,
            alpha_ecg=args.alpha_ecg,
        )
        scheduler.step()

        val_metrics = evaluate_retrieval(model, loaders["val"], device=device)
        score = val_metrics["mean_top1"] + val_metrics["mean_positive_cosine"] - val_metrics["mean_negative_cosine"]
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **loss_parts,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(history_row)

        elapsed = time.time() - start_time
        avg_epoch_seconds = elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(args.epochs - epoch, 0)
        print(
            f"epoch={epoch:02d} "
            f"loss={train_loss:.4f} "
            f"cross={loss_parts['cross']:.4f} "
            f"ppg_ssl={loss_parts['ppg_ssl']:.4f} "
            f"ecg_ssl={loss_parts['ecg_ssl']:.4f} "
            f"val_top1={val_metrics['mean_top1']:.4f} "
            f"val_pos_cos={val_metrics['mean_positive_cosine']:.4f} "
            f"val_neg_cos={val_metrics['mean_negative_cosine']:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta_seconds)}",
            flush=True,
        )

        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("SSL pretraining did not produce a valid model state.")

    model.load_state_dict(best_state)
    val_metrics = evaluate_retrieval(model, loaders["val"], device=device)
    test_metrics = evaluate_retrieval(model, loaders["test"], device=device)

    summary = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "split_groups": split_groups,
        "train_segments": split_sizes["train"],
        "val_segments": split_sizes["val"],
        "test_segments": split_sizes["test"],
        "val_retrieval": val_metrics,
        "test_retrieval": test_metrics,
        "runtime_seconds": time.time() - start_time,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ppg_encoder_state_dict": model.ppg_encoder.state_dict(),
            "ecg_encoder_state_dict": model.ecg_encoder.state_dict(),
            "split_groups": split_groups,
            "val_retrieval": val_metrics,
            "test_retrieval": test_metrics,
        },
        args.output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    save_json(summary, args.output_dir / "metrics.json")

    print("\nValidation retrieval:", json.dumps(val_metrics, indent=2), flush=True)
    print("Test retrieval:", json.dumps(test_metrics, indent=2), flush=True)
    print("Saved artifacts to:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
