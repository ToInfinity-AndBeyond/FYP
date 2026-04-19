from __future__ import annotations

import torch
from torch import nn

from ppg_hybrid_model import ConvNormAct1d, MultiScaleResidualBlock1d


class SegmentWaveformEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct1d(1, 32, kernel_size=15, stride=2),
            MultiScaleResidualBlock1d(32, 64, stride=2),
            MultiScaleResidualBlock1d(64, 96, stride=2),
            MultiScaleResidualBlock1d(96, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        encoded = self.stem(waveform.unsqueeze(1))
        return self.proj(encoded)


class MultiScaleRhythmEncoder(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 128):
        super().__init__()
        hidden_dim = max(embedding_dim, 128)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class QualityAwareAttentionPool(nn.Module):
    def __init__(
        self,
        token_dim: int,
        quality_dim: int,
        quality_floor: float = 0.35,
        quality_power: float = 2.0,
    ):
        super().__init__()
        self.quality_floor = quality_floor
        self.quality_power = quality_power
        quality_hidden = max(token_dim // 4, 16)
        self.quality_proj = nn.Sequential(
            nn.Linear(quality_dim, quality_hidden),
            nn.LayerNorm(quality_hidden),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(token_dim + quality_hidden, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        quality_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quality_embedding = self.quality_proj(quality_features)
        quality_confidence = quality_features.mean(dim=-1).clamp(0.0, 1.0)
        if self.quality_floor > 0.0:
            quality_mask = quality_confidence >= self.quality_floor
            fallback_mask = quality_mask.any(dim=1, keepdim=True)
            pool_mask = torch.where(fallback_mask, quality_mask, mask)
            pool_mask = pool_mask & mask
        else:
            pool_mask = mask
        attn_logits = self.attention(torch.cat([tokens, quality_embedding], dim=-1)).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~pool_mask, -1e9)
        weights = torch.softmax(attn_logits, dim=1)
        quality_weights = quality_confidence.pow(self.quality_power)
        weights = weights * quality_weights * pool_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pooled_tokens = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        pooled_quality = torch.sum(quality_features * weights.unsqueeze(-1), dim=1)
        return pooled_tokens, pooled_quality, weights


class RecordMILPPGNet(nn.Module):
    def __init__(
        self,
        rhythm_feature_dim: int,
        quality_feature_dim: int,
        token_dim: int = 160,
        quality_floor: float = 0.35,
        quality_power: float = 2.0,
    ):
        super().__init__()
        self.waveform_encoder = SegmentWaveformEncoder(embedding_dim=128)
        self.rhythm_encoder = MultiScaleRhythmEncoder(rhythm_feature_dim, embedding_dim=128)
        self.segment_fuse = nn.Sequential(
            nn.Linear(256, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.sequence_encoder = nn.GRU(
            input_size=token_dim,
            hidden_size=token_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.segment_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )
        self.pool = QualityAwareAttentionPool(
            token_dim=token_dim,
            quality_dim=quality_feature_dim,
            quality_floor=quality_floor,
            quality_power=quality_power,
        )
        self.head = nn.Sequential(
            nn.Linear(token_dim * 2 + quality_feature_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        waveforms: torch.Tensor,
        rhythm_features: torch.Tensor,
        quality_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, segment_count, signal_length = waveforms.shape
        flat_waveforms = waveforms.reshape(batch_size * segment_count, signal_length)
        flat_rhythm = rhythm_features.reshape(batch_size * segment_count, rhythm_features.shape[-1])

        waveform_embedding = self.waveform_encoder(flat_waveforms).reshape(batch_size, segment_count, -1)
        rhythm_embedding = self.rhythm_encoder(flat_rhythm).reshape(batch_size, segment_count, -1)
        segment_tokens = self.segment_fuse(torch.cat([waveform_embedding, rhythm_embedding], dim=-1))
        segment_tokens = segment_tokens * mask.unsqueeze(-1).float()

        contextualized, _ = self.sequence_encoder(segment_tokens)
        contextualized = contextualized * mask.unsqueeze(-1).float()
        segment_logits = self.segment_head(contextualized).squeeze(-1)

        pooled_tokens, pooled_quality, attention_weights = self.pool(contextualized, quality_features, mask)
        masked_context = contextualized.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled_max = masked_context.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))

        record_features = torch.cat([pooled_tokens, pooled_max, pooled_quality], dim=-1)
        record_logits = self.head(record_features).squeeze(-1)

        return {
            "record_logits": record_logits,
            "segment_logits": segment_logits,
            "attention_weights": attention_weights,
            "pooled_quality": pooled_quality,
        }
