from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from physio_distill_model import WaveformEncoder1d


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class CrossModalSSLNet(nn.Module):
    """TSTA-inspired cross-modal pretraining backbone.

    The goal is not to reproduce the paper exactly, but to learn a shared
    representation where paired PPG/ECG windows align while remaining stable
    under light temporal augmentations.
    """

    def __init__(self, embedding_dim: int = 64, projection_dim: int = 128):
        super().__init__()
        self.ppg_encoder = WaveformEncoder1d(in_channels=3, out_dim=embedding_dim, d_model=96)
        self.ecg_encoder = WaveformEncoder1d(in_channels=1, out_dim=embedding_dim, d_model=96)
        self.ppg_projection = ProjectionHead(embedding_dim, projection_dim)
        self.ecg_projection = ProjectionHead(embedding_dim, projection_dim)

    @staticmethod
    def build_ppg_channels(ppg_waveform: torch.Tensor) -> torch.Tensor:
        vpg = torch.gradient(ppg_waveform, dim=1)[0]
        apg = torch.gradient(vpg, dim=1)[0]
        return torch.stack([ppg_waveform, vpg, apg], dim=1)

    def encode_ppg(self, ppg_waveform: torch.Tensor) -> torch.Tensor:
        return self.ppg_encoder(self.build_ppg_channels(ppg_waveform))

    def encode_ecg(self, ecg_waveform: torch.Tensor) -> torch.Tensor:
        return self.ecg_encoder(ecg_waveform.unsqueeze(1))

    def project_ppg(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ppg_projection(embedding), dim=1)

    def project_ecg(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.ecg_projection(embedding), dim=1)

    def forward(
        self,
        ppg_view_a: torch.Tensor,
        ecg_view_a: torch.Tensor,
        ppg_view_b: torch.Tensor | None = None,
        ecg_view_b: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ppg_embedding_a = self.encode_ppg(ppg_view_a)
        ecg_embedding_a = self.encode_ecg(ecg_view_a)
        outputs = {
            "ppg_embedding_a": ppg_embedding_a,
            "ecg_embedding_a": ecg_embedding_a,
            "ppg_projection_a": self.project_ppg(ppg_embedding_a),
            "ecg_projection_a": self.project_ecg(ecg_embedding_a),
        }

        if ppg_view_b is not None:
            ppg_embedding_b = self.encode_ppg(ppg_view_b)
            outputs["ppg_embedding_b"] = ppg_embedding_b
            outputs["ppg_projection_b"] = self.project_ppg(ppg_embedding_b)

        if ecg_view_b is not None:
            ecg_embedding_b = self.encode_ecg(ecg_view_b)
            outputs["ecg_embedding_b"] = ecg_embedding_b
            outputs["ecg_projection_b"] = self.project_ecg(ecg_embedding_b)

        return outputs
