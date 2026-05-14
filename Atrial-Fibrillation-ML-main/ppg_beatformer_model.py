from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class BeatMorphologyEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct1d(1, 32, kernel_size=7),
            nn.MaxPool1d(kernel_size=2),
            ConvNormAct1d(32, 64, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            ConvNormAct1d(64, 96, kernel_size=5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, beats: torch.Tensor) -> torch.Tensor:
        return self.net(beats)


class QualityAwareBeatFormer(nn.Module):
    """Beat-token transformer for PPG AF screening.

    The model keeps the public interface used by the existing training script:
    forward(waveform, handcrafted_features) -> logits.
    """

    def __init__(
        self,
        feature_dim: int,
        signal_length: int = 3750,
        sample_rate_hz: float = 125.0,
        max_beats: int = 64,
        beat_length: int = 128,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        peak_window: int = 31,
    ):
        super().__init__()
        if max_beats < 1:
            raise ValueError("max_beats must be at least 1")
        if beat_length < 8:
            raise ValueError("beat_length must be at least 8")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.feature_dim = int(feature_dim)
        self.signal_length = int(signal_length)
        self.sample_rate_hz = float(sample_rate_hz)
        self.max_beats = int(max_beats)
        self.beat_length = int(beat_length)
        self.d_model = int(d_model)
        self.peak_window = int(peak_window if peak_window % 2 == 1 else peak_window + 1)

        # Slightly asymmetric beat window: more context after the systolic peak.
        self.left_context = int(round(self.beat_length * 0.375))
        self.register_buffer(
            "beat_offsets",
            torch.arange(-self.left_context, self.beat_length - self.left_context, dtype=torch.long),
            persistent=False,
        )

        self.morphology_encoder = BeatMorphologyEncoder(out_dim=d_model)
        local_rhythm_dim = 6
        self.rhythm_quality_encoder = nn.Sequential(
            nn.Linear(feature_dim + local_rhythm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_beats, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.quality_pool_bias = nn.Sequential(
            nn.Linear(feature_dim + local_rhythm_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.segment_feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _smooth_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        x = F.avg_pool1d(x, kernel_size=5, stride=1, padding=2)
        return x.squeeze(1)

    def _select_peak_centres(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, signal_length = waveform.shape
        smoothed = self._smooth_waveform(waveform)
        pooled = F.max_pool1d(
            smoothed.unsqueeze(1),
            kernel_size=self.peak_window,
            stride=1,
            padding=self.peak_window // 2,
        ).squeeze(1)
        local_maxima = smoothed >= (pooled - 1e-6)
        threshold = smoothed.mean(dim=1, keepdim=True) + 0.05 * smoothed.std(dim=1, keepdim=True).clamp_min(1e-6)
        edge_valid = torch.ones_like(local_maxima, dtype=torch.bool)
        right_context = self.beat_length - self.left_context
        edge_valid[:, : self.left_context] = False
        edge_valid[:, max(signal_length - right_context, 0) :] = False

        peak_scores = torch.where(
            local_maxima & edge_valid & (smoothed > threshold),
            smoothed,
            smoothed.new_full(smoothed.shape, -1.0e6),
        )
        top_scores, top_indices = torch.topk(peak_scores, k=min(self.max_beats, signal_length), dim=1)
        if top_indices.shape[1] < self.max_beats:
            pad_count = self.max_beats - top_indices.shape[1]
            top_indices = F.pad(top_indices, (0, pad_count), value=signal_length // 2)
            top_scores = F.pad(top_scores, (0, pad_count), value=-1.0e6)

        valid = top_scores > -1.0e5
        no_peak = ~valid.any(dim=1)
        if no_peak.any():
            top_indices = top_indices.clone()
            valid = valid.clone()
            top_indices[no_peak, 0] = signal_length // 2
            valid[no_peak, 0] = True

        centres, order = torch.sort(top_indices, dim=1)
        valid = torch.gather(valid, dim=1, index=order)
        amplitudes = torch.gather(waveform, dim=1, index=centres.clamp(0, signal_length - 1))
        return centres, valid, amplitudes

    def _extract_beat_windows(self, waveform: torch.Tensor, centres: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, signal_length = waveform.shape
        gather_indices = centres.unsqueeze(-1) + self.beat_offsets.view(1, 1, -1)
        valid_windows = (gather_indices >= 0) & (gather_indices < signal_length) & valid.unsqueeze(-1)
        gather_indices = gather_indices.clamp(0, signal_length - 1)
        flat_indices = gather_indices.reshape(batch_size, -1)
        beats = torch.gather(waveform, dim=1, index=flat_indices).view(batch_size, self.max_beats, self.beat_length)
        beats = beats * valid_windows.to(beats.dtype)
        return beats

    def _build_local_rhythm_features(
        self,
        centres: torch.Tensor,
        valid: torch.Tensor,
        amplitudes: torch.Tensor,
        signal_length: int,
    ) -> torch.Tensor:
        centres_float = centres.to(dtype=amplitudes.dtype)
        valid_float = valid.to(dtype=amplitudes.dtype)
        prev_centres = torch.roll(centres_float, shifts=1, dims=1)
        next_centres = torch.roll(centres_float, shifts=-1, dims=1)
        prev_valid = valid & torch.roll(valid, shifts=1, dims=1)
        next_valid = valid & torch.roll(valid, shifts=-1, dims=1)
        prev_valid[:, 0] = False
        next_valid[:, -1] = False

        prev_ibi = torch.where(prev_valid, (centres_float - prev_centres) / self.sample_rate_hz, torch.zeros_like(centres_float))
        next_ibi = torch.where(next_valid, (next_centres - centres_float) / self.sample_rate_hz, torch.zeros_like(centres_float))
        ibi_delta = next_ibi - prev_ibi
        local_cv = torch.abs(ibi_delta) / (0.5 * (prev_ibi + next_ibi).abs() + 1e-3)
        position = centres_float / max(float(signal_length - 1), 1.0)
        local = torch.stack([prev_ibi, next_ibi, ibi_delta, local_cv, position, amplitudes], dim=-1)
        return local * valid_float.unsqueeze(-1)

    def forward(self, waveform: torch.Tensor, handcrafted_features: torch.Tensor) -> torch.Tensor:
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        handcrafted_features = torch.nan_to_num(handcrafted_features, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size, signal_length = waveform.shape

        centres, valid, amplitudes = self._select_peak_centres(waveform)
        beats = self._extract_beat_windows(waveform, centres, valid)
        beat_embeddings = self.morphology_encoder(beats.reshape(batch_size * self.max_beats, 1, self.beat_length))
        beat_embeddings = beat_embeddings.view(batch_size, self.max_beats, self.d_model)

        local_rhythm = self._build_local_rhythm_features(centres, valid, amplitudes, signal_length)
        feature_context = handcrafted_features.unsqueeze(1).expand(-1, self.max_beats, -1)
        rhythm_inputs = torch.cat([local_rhythm, feature_context], dim=-1)
        rhythm_embeddings = self.rhythm_quality_encoder(rhythm_inputs)

        tokens = beat_embeddings + rhythm_embeddings + self.position_embedding[:, : self.max_beats]
        tokens = self.token_norm(tokens)
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)

        pool_logits = self.pool_score(encoded).squeeze(-1) + self.quality_pool_bias(rhythm_inputs).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~valid, -1.0e4)
        pool_weights = torch.softmax(pool_logits, dim=1)
        beat_summary = torch.sum(encoded * pool_weights.unsqueeze(-1), dim=1)
        feature_summary = self.segment_feature_encoder(handcrafted_features)
        return self.head(torch.cat([beat_summary, feature_summary], dim=1)).squeeze(1)
