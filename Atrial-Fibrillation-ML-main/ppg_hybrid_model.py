from __future__ import annotations

import torch
from torch import nn


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConvNormAct2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: tuple[int, int] = (1, 1)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.net(inputs)


class MultiScaleResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        branch_channels = out_channels // 3
        branch3_channels = out_channels - (branch_channels * 2)

        self.branch1 = ConvNormAct1d(in_channels, branch_channels, kernel_size=3, stride=stride, dilation=1)
        self.branch2 = ConvNormAct1d(in_channels, branch_channels, kernel_size=7, stride=stride, dilation=2)
        self.branch3 = ConvNormAct1d(in_channels, branch3_channels, kernel_size=15, stride=stride, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(),
            SqueezeExcite1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.branch1(inputs), self.branch2(inputs), self.branch3(inputs)], dim=1)
        return self.fuse(combined) + self.skip(inputs)


class SpectralEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct2d(1, 16, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvNormAct2d(16, 32, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            ConvNormAct2d(32, 64, kernel_size=(3, 3)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.GELU(), nn.Dropout(0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.net(inputs)
        return self.proj(features)


class RhythmMorphologyFusionNet(nn.Module):
    def __init__(self, feature_dim: int, signal_length: int = 3750, d_model: int = 128):
        super().__init__()
        self.signal_length = signal_length

        self.time_stem = nn.Sequential(
            ConvNormAct1d(1, 32, kernel_size=15, stride=2),
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
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_attn = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.time_proj = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.1))

        self.spectral_encoder = SpectralEncoder(out_dim=64)

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(64 * 3, 64 * 3),
            nn.GELU(),
            nn.Linear(64 * 3, 64 * 3),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def _spectrogram(self, signals: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(128, device=signals.device, dtype=signals.dtype)
        spec = torch.stft(
            signals.squeeze(1),
            n_fft=128,
            hop_length=32,
            win_length=128,
            window=window,
            return_complex=True,
        )
        spec = torch.log1p(spec.abs())
        return spec.unsqueeze(1)

    def forward(self, waveform: torch.Tensor, handcrafted_features: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        time_tokens = self.time_stem(x).transpose(1, 2)
        time_tokens = self.temporal_encoder(time_tokens)
        attn_weights = torch.softmax(self.temporal_attn(time_tokens), dim=1)
        time_embedding = torch.sum(attn_weights * time_tokens, dim=1)
        time_embedding = self.time_proj(time_embedding)

        spectral_input = self._spectrogram(x)
        spectral_embedding = self.spectral_encoder(spectral_input)

        feature_embedding = self.feature_encoder(handcrafted_features)

        fused = torch.cat([time_embedding, spectral_embedding, feature_embedding], dim=1)
        gates = self.gate(fused).chunk(3, dim=1)
        gated = torch.cat(
            [
                time_embedding * gates[0],
                spectral_embedding * gates[1],
                feature_embedding * gates[2],
            ],
            dim=1,
        )
        return self.head(gated).squeeze(1)
