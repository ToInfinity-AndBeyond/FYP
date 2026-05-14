from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from af_pipeline.data import PPGAugment, load_and_concat_signal_datasets
from af_pipeline.runtime import choose_amp, format_duration, get_device, log_stage, save_json, set_seed, should_report_progress
from physio_distill_model import WaveformEncoder1d


class PPGSimCLRDataset(Dataset):
    def __init__(self, signals: np.ndarray, augment: PPGAugment):
        self.signals = signals.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        signal = self.signals[index]
        return {
            "view_a": torch.from_numpy(self.augment(signal)),
            "view_b": torch.from_numpy(self.augment(signal)),
        }


class PPGSimCLRNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, projection_dim: int = 128):
        super().__init__()
        self.ppg_encoder = WaveformEncoder1d(in_channels=3, out_dim=embedding_dim, d_model=96)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, projection_dim),
        )

    @staticmethod
    def build_channels(waveform: torch.Tensor) -> torch.Tensor:
        vpg = torch.gradient(waveform, dim=1)[0]
        apg = torch.gradient(vpg, dim=1)[0]
        return torch.stack([waveform, vpg, apg], dim=1)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.ppg_encoder(self.build_channels(waveform))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(waveform)
        return F.normalize(self.projector(embedding), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0).float()
    logits = (z @ z.T / temperature).float()
    logits.fill_diagonal_(-1e4)
    targets = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    return F.cross_entropy(logits, targets)


def split_by_metadata_fold(
    signals: np.ndarray,
    summary_df: pd.DataFrame,
    train_folds: set[int],
    val_folds: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    if "strat_fold" not in summary_df.columns:
        raise ValueError("PPG SSL pretraining expects strat_fold in the summary CSV.")
    folds = pd.to_numeric(summary_df["strat_fold"], errors="coerce").fillna(-1).astype(int).to_numpy()
    train_mask = np.isin(folds, list(train_folds))
    val_mask = np.isin(folds, list(val_folds))
    return signals[train_mask], signals[val_mask]


def run_epoch(
    model: PPGSimCLRNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    amp_device_type: str,
    temperature: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader, start=1):
        view_a = batch["view_a"].to(device, non_blocking=True)
        view_b = batch["view_b"].to(device, non_blocking=True)
        batch_count = view_a.shape[0]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
                z1 = model(view_a)
                z2 = model(view_b)
                loss = nt_xent_loss(z1, z2, temperature)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.detach().item()) * batch_count
        total_count += batch_count

        if is_train and should_report_progress(step, len(dataloader), every_steps=100):
            elapsed = time.time() - start_time
            eta = elapsed / max(step, 1) * max(len(dataloader) - step, 0)
            log_stage(
                f"train_progress batch={step}/{len(dataloader)} "
                f"loss={total_loss / max(total_count, 1):.4f} "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )

    return total_loss / max(total_count, 1)


def parse_fold_set(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR-style self-supervised pretraining on MIMIC PPG windows.")
    parser.add_argument("--segments-path", type=Path, nargs="+", required=True)
    parser.add_argument("--summary-path", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-folds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--val-folds", default="8")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--max-train-segments",
        type=int,
        default=0,
        help="Optional cap for smoke/debug runs. 0 keeps all train segments.",
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
    signals, summary_df = load_and_concat_signal_datasets(args.segments_path, args.summary_path)
    train_signals, val_signals = split_by_metadata_fold(
        signals=signals,
        summary_df=summary_df,
        train_folds=parse_fold_set(args.train_folds),
        val_folds=parse_fold_set(args.val_folds),
    )
    if args.max_train_segments and train_signals.shape[0] > args.max_train_segments:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(train_signals.shape[0], size=args.max_train_segments, replace=False)
        train_signals = train_signals[np.sort(idx)]

    signal_length = int(signals.shape[1])
    train_loader = DataLoader(
        PPGSimCLRDataset(train_signals, PPGAugment(signal_length, enable_time_warp=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        PPGSimCLRDataset(val_signals, PPGAugment(signal_length, enable_time_warp=True)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=True,
    )

    log_stage(
        "dataset summary: "
        + json.dumps(
            {
                "total_segments": int(signals.shape[0]),
                "train_segments": int(train_signals.shape[0]),
                "val_segments": int(val_signals.shape[0]),
                "signal_length": signal_length,
            }
        )
    )

    model = PPGSimCLRNet(embedding_dim=args.embedding_dim, projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, Any]] = []
    best_state = None
    best_epoch = 0
    best_val_loss = math.inf
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, amp_enabled, amp_device_type, args.temperature)
        scheduler.step()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device, amp_enabled, amp_device_type, args.temperature)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        elapsed = time.time() - start_time
        log_stage(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"epoch_time={format_duration(time.time() - epoch_start)} elapsed={format_duration(elapsed)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_stage(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("SSL pretraining did not produce a checkpoint.")
    model.load_state_dict(best_state)

    metrics = {
        "device": str(device),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "train_segments": int(train_signals.shape[0]),
        "val_segments": int(val_signals.shape[0]),
        "runtime_seconds": time.time() - start_time,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ppg_encoder_state_dict": model.ppg_encoder.state_dict(),
            "embedding_dim": args.embedding_dim,
            "projection_dim": args.projection_dim,
            "metrics": metrics,
        },
        args.output_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    save_json(metrics, args.output_dir / "metrics.json")
    log_stage("Saved artifacts to: " + str(args.output_dir))


if __name__ == "__main__":
    main()
