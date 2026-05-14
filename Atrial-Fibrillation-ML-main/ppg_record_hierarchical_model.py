from __future__ import annotations

import torch
from torch import nn

from ppg_record_mil_model import (
    MultiScaleRhythmEncoder,
    QualityAwareAttentionPool,
    SegmentWaveformEncoder,
)


class SegmentQualityEncoder(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int = 64):
        super().__init__()
        hidden_dim = max(embedding_dim * 2, 64)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, quality_features: torch.Tensor) -> torch.Tensor:
        return self.net(quality_features)


class HierarchicalRecordPPGNet(nn.Module):
    """Explicit rhythm-morphology-quality fusion with hierarchical record aggregation."""

    def __init__(
        self,
        rhythm_feature_dim: int,
        quality_feature_dim: int,
        token_dim: int = 192,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        quality_floor: float = 0.35,
        quality_power: float = 2.0,
    ):
        super().__init__()
        self.waveform_encoder = SegmentWaveformEncoder(embedding_dim=128)
        self.rhythm_encoder = MultiScaleRhythmEncoder(rhythm_feature_dim, embedding_dim=128)
        self.quality_encoder = SegmentQualityEncoder(quality_feature_dim, embedding_dim=64)

        self.segment_fuse = nn.Sequential(
            nn.Linear(128 + 128 + 64, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.segment_gate = nn.Sequential(
            nn.Linear(quality_feature_dim, token_dim),
            nn.Sigmoid(),
        )
        self.segment_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim // 2, 1),
        )

        self.pool = QualityAwareAttentionPool(
            token_dim=token_dim,
            quality_dim=quality_feature_dim,
            quality_floor=quality_floor,
            quality_power=quality_power,
        )
        self.record_head = nn.Sequential(
            nn.Linear(token_dim * 3 + quality_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
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
        flat_quality = quality_features.reshape(batch_size * segment_count, quality_features.shape[-1])

        waveform_embedding = self.waveform_encoder(flat_waveforms).reshape(batch_size, segment_count, -1)
        rhythm_embedding = self.rhythm_encoder(flat_rhythm).reshape(batch_size, segment_count, -1)
        quality_embedding = self.quality_encoder(flat_quality).reshape(batch_size, segment_count, -1)

        segment_tokens = self.segment_fuse(torch.cat([waveform_embedding, rhythm_embedding, quality_embedding], dim=-1))
        segment_tokens = segment_tokens * mask.unsqueeze(-1).float()

        key_padding_mask = ~mask
        contextualized = self.segment_context(segment_tokens, src_key_padding_mask=key_padding_mask)
        contextualized = contextualized * mask.unsqueeze(-1).float()

        quality_gate = self.segment_gate(quality_features)
        gated_tokens = contextualized * quality_gate * mask.unsqueeze(-1).float()
        segment_logits = self.segment_head(gated_tokens).squeeze(-1)

        pooled_tokens, pooled_quality, attention_weights = self.pool(gated_tokens, quality_features, mask)
        masked_tokens = gated_tokens.masked_fill(~mask.unsqueeze(-1), -1e9)
        pooled_max = masked_tokens.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        pooled_mean = (
            gated_tokens.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        )

        record_features = torch.cat([pooled_tokens, pooled_max, pooled_mean, pooled_quality], dim=-1)
        record_logits = self.record_head(record_features).squeeze(-1)

        return {
            "record_logits": record_logits,
            "segment_logits": segment_logits,
            "attention_weights": attention_weights,
            "pooled_quality": pooled_quality,
        }
