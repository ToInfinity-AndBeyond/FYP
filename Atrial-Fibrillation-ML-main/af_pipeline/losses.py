from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def mixup_batch(
    waveform: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    qualities: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        return waveform, features, labels, qualities

    lam = np.random.beta(alpha, alpha)
    permutation = torch.randperm(waveform.size(0), device=waveform.device)
    mixed_waveform = lam * waveform + (1.0 - lam) * waveform[permutation]
    mixed_features = lam * features + (1.0 - lam) * features[permutation]
    mixed_labels = lam * labels + (1.0 - lam) * labels[permutation]
    mixed_qualities = lam * qualities + (1.0 - lam) * qualities[permutation]
    return mixed_waveform, mixed_features, mixed_labels, mixed_qualities


class QualityAwareFocalLoss(nn.Module):
    def __init__(self, pos_weight: float, gamma: float = 1.5, label_smoothing: float = 0.02):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        quality_scores = quality_scores.clamp(0.0, 1.0)
        smooth_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smooth_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        sample_weights = 0.6 + 0.4 * quality_scores
        return (focal * sample_weights).mean()


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        alpha_soft_f1: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha_soft_f1 = alpha_soft_f1
        self.eps = eps

    def _asymmetric_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1.0 - pos_prob
        if self.clip > 0.0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)
        pos_loss = targets * torch.log(pos_prob.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(neg_prob.clamp(min=self.eps))
        pt = pos_prob * targets + neg_prob * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        weight = torch.pow(1.0 - pt, gamma)
        return -((pos_loss + neg_loss) * weight)

    def _soft_f1_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        soft_f1 = 2.0 * tp / (2.0 * tp + fp + fn + self.eps)
        return 1.0 - soft_f1

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._asymmetric_loss(logits, targets)
        if quality_scores is not None:
            sample_weights = 0.6 + 0.4 * quality_scores.clamp(0.0, 1.0)
            loss = loss * sample_weights
        asl = loss.mean()
        if self.alpha_soft_f1 <= 0.0:
            return asl
        return (1.0 - self.alpha_soft_f1) * asl + self.alpha_soft_f1 * self._soft_f1_loss(logits, targets)


class DistillationLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        alpha_teacher: float = 0.35,
        alpha_distill: float = 0.20,
        alpha_aux: float = 0.15,
        gamma: float = 1.5,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.alpha_teacher = alpha_teacher
        self.alpha_distill = alpha_distill
        self.alpha_aux = alpha_aux
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.aux_loss = nn.SmoothL1Loss(reduction="none")

    def _student_classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        quality_scores = quality_scores.clamp(0.0, 1.0)
        smooth_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smooth_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(logits.device),
        )
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        sample_weights = 0.6 + 0.4 * quality_scores
        return (focal * sample_weights).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        quality_scores: torch.Tensor,
        aux_targets: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        student_cls = self._student_classification_loss(outputs["student_logit"], labels, quality_scores)
        teacher_cls = F.binary_cross_entropy_with_logits(
            outputs["teacher_logit"],
            labels,
            pos_weight=self.pos_weight.to(outputs["teacher_logit"].device),
        )

        student_embedding = F.normalize(outputs["student_embedding"], dim=1)
        teacher_embedding = F.normalize(outputs["teacher_embedding"].detach(), dim=1)
        distill = 1.0 - F.cosine_similarity(student_embedding, teacher_embedding, dim=1).mean()

        aux_residual = self.aux_loss(outputs["aux_prediction"], aux_targets)
        aux_mask = aux_mask.to(aux_residual.dtype)
        aux = (aux_residual * aux_mask).sum() / aux_mask.sum().clamp(min=1.0)

        total = student_cls + self.alpha_teacher * teacher_cls + self.alpha_distill * distill + self.alpha_aux * aux
        loss_parts = {
            "student_cls": float(student_cls.detach().item()),
            "teacher_cls": float(teacher_cls.detach().item()),
            "distill": float(distill.detach().item()),
            "aux": float(aux.detach().item()),
        }
        return total, loss_parts


class RecordLevelHardMiningLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        segment_aux_weight: float = 0.1,
        hard_negative_scale: float = 1.5,
        hard_positive_scale: float = 0.75,
        segment_quality_power: float = 2.0,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.segment_aux_weight = segment_aux_weight
        self.hard_negative_scale = hard_negative_scale
        self.hard_positive_scale = hard_positive_scale
        self.segment_quality_power = segment_quality_power

    def forward(
        self,
        record_logits: torch.Tensor,
        segment_logits: torch.Tensor,
        labels: torch.Tensor,
        quality_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        record_targets = labels.float()
        record_bce = F.binary_cross_entropy_with_logits(
            record_logits,
            record_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(record_logits.device),
        )
        record_prob = torch.sigmoid(record_logits).detach()
        record_hard_weight = (
            1.0
            + self.hard_negative_scale * ((1.0 - record_targets) * record_prob.square())
            + self.hard_positive_scale * (record_targets * (1.0 - record_prob).square())
        )
        record_loss = (record_bce * record_hard_weight).mean()

        segment_targets = record_targets.unsqueeze(1).expand_as(segment_logits)
        segment_bce = F.binary_cross_entropy_with_logits(
            segment_logits,
            segment_targets,
            reduction="none",
            pos_weight=self.pos_weight.to(segment_logits.device),
        )
        segment_prob = torch.sigmoid(segment_logits).detach()
        segment_hard_weight = (
            1.0
            + self.hard_negative_scale * ((1.0 - segment_targets) * segment_prob.square())
            + self.hard_positive_scale * (segment_targets * (1.0 - segment_prob).square())
        )
        segment_quality_confidence = quality_features.mean(dim=-1).clamp(0.0, 1.0)
        segment_quality_weight = 0.25 + 0.75 * segment_quality_confidence.pow(self.segment_quality_power)
        masked_segment_loss = (
            segment_bce * segment_hard_weight * segment_quality_weight * mask.float()
        ).sum() / mask.float().sum().clamp_min(1.0)
        return record_loss + self.segment_aux_weight * masked_segment_loss
