from __future__ import annotations

import torch
from torch import nn

from ppg_hybrid_model import ConvNormAct1d, MultiScaleResidualBlock1d


class AttentionPool1d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(tokens), dim=1)
        return torch.sum(weights * tokens, dim=1)


class WaveformEncoder1d(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 64, d_model: int = 96):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct1d(in_channels, 32, kernel_size=15, stride=2),
            MultiScaleResidualBlock1d(32, 64, stride=2),
            MultiScaleResidualBlock1d(64, 96, stride=2),
            MultiScaleResidualBlock1d(96, d_model, stride=2),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pool = AttentionPool1d(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(waveform).transpose(1, 2)
        tokens = self.encoder(tokens)
        pooled = self.pool(tokens)
        return self.proj(pooled)


class RhythmSequenceEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = WaveformEncoder1d(in_channels=1, out_dim=out_dim, d_model=64)

    def forward(self, ibi_sequence: torch.Tensor) -> torch.Tensor:
        return self.encoder(ibi_sequence.unsqueeze(1))


class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GatedFusion(nn.Module):
    def __init__(self, branch_count: int, branch_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        fused_dim = branch_count * branch_dim
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.branch_count = branch_count
        self.branch_dim = branch_dim

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        fused = torch.cat(list(embeddings), dim=1)
        gates = self.gate(fused).chunk(self.branch_count, dim=1)
        gated_embeddings = [embedding * gate for embedding, gate in zip(embeddings, gates)]
        return self.head(torch.cat(gated_embeddings, dim=1))


class PhysiologyAwareDistillationNet(nn.Module):
    def __init__(
        self,
        student_feature_dim: int,
        teacher_feature_dim: int,
        aux_target_dim: int,
    ):
        super().__init__()
        self.student_morph = WaveformEncoder1d(in_channels=3, out_dim=64, d_model=96)
        self.student_rhythm = RhythmSequenceEncoder(out_dim=64)
        self.student_feature = FeatureEncoder(student_feature_dim, out_dim=64)
        self.student_fusion = GatedFusion(branch_count=3, branch_dim=64, hidden_dim=128)
        self.student_classifier = nn.Linear(128, 1)
        self.aux_head = nn.Linear(128, aux_target_dim)

        self.teacher_ecg = WaveformEncoder1d(in_channels=1, out_dim=64, d_model=96)
        self.teacher_rhythm = RhythmSequenceEncoder(out_dim=64)
        self.teacher_resp = WaveformEncoder1d(in_channels=1, out_dim=64, d_model=64)
        self.teacher_feature = FeatureEncoder(teacher_feature_dim, out_dim=64)
        self.teacher_fusion = GatedFusion(branch_count=4, branch_dim=64, hidden_dim=128)
        self.teacher_classifier = nn.Linear(128, 1)

        self.student_distill_proj = nn.Linear(128, 128)
        self.teacher_distill_proj = nn.Linear(128, 128)

    @staticmethod
    def build_ppg_channels(ppg_waveform: torch.Tensor) -> torch.Tensor:
        vpg = torch.gradient(ppg_waveform, dim=1)[0]
        apg = torch.gradient(vpg, dim=1)[0]
        return torch.stack([ppg_waveform, vpg, apg], dim=1)

    def forward_student(
        self,
        ppg_waveform: torch.Tensor,
        ppg_ibi: torch.Tensor,
        student_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ppg_channels = self.build_ppg_channels(ppg_waveform)
        morph_embedding = self.student_morph(ppg_channels)
        rhythm_embedding = self.student_rhythm(ppg_ibi)
        feature_embedding = self.student_feature(student_features)
        student_embedding = self.student_fusion(morph_embedding, rhythm_embedding, feature_embedding)
        student_logit = self.student_classifier(student_embedding).squeeze(1)
        aux_prediction = self.aux_head(student_embedding)
        return student_logit, student_embedding, aux_prediction

    def forward_teacher(
        self,
        ecg_waveform: torch.Tensor,
        ecg_ibi: torch.Tensor,
        resp_waveform: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_embedding = self.teacher_ecg(ecg_waveform.unsqueeze(1))
        ecg_rhythm_embedding = self.teacher_rhythm(ecg_ibi)
        resp_embedding = self.teacher_resp(resp_waveform.unsqueeze(1))
        teacher_feature_embedding = self.teacher_feature(teacher_features)
        teacher_embedding = self.teacher_fusion(
            ecg_embedding,
            ecg_rhythm_embedding,
            resp_embedding,
            teacher_feature_embedding,
        )
        teacher_logit = self.teacher_classifier(teacher_embedding).squeeze(1)
        return teacher_logit, teacher_embedding

    def forward(
        self,
        ppg_waveform: torch.Tensor,
        ppg_ibi: torch.Tensor,
        student_features: torch.Tensor,
        ecg_waveform: torch.Tensor,
        ecg_ibi: torch.Tensor,
        resp_waveform: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        student_logit, student_embedding, aux_prediction = self.forward_student(
            ppg_waveform=ppg_waveform,
            ppg_ibi=ppg_ibi,
            student_features=student_features,
        )
        teacher_logit, teacher_embedding = self.forward_teacher(
            ecg_waveform=ecg_waveform,
            ecg_ibi=ecg_ibi,
            resp_waveform=resp_waveform,
            teacher_features=teacher_features,
        )
        return {
            "student_logit": student_logit,
            "teacher_logit": teacher_logit,
            "student_embedding": self.student_distill_proj(student_embedding),
            "teacher_embedding": self.teacher_distill_proj(teacher_embedding),
            "aux_prediction": aux_prediction,
        }
