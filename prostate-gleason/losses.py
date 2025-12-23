"""
Focal Loss implementation for handling class imbalance.

Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (tensor of shape [num_classes] or None)
        gamma: Focusing parameter. gamma=0 is equivalent to CE.
               Higher gamma puts more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]

        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to CE loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.

    Helps prevent overconfident predictions and can improve generalization.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
        weight: Class weights (optional)
    """

    def __init__(self, smoothing=0.1, weight=None, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

        if weight is not None:
            if isinstance(weight, (list, tuple)):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]
        """
        num_classes = inputs.size(1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(inputs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute loss
        loss = -smooth_labels * log_probs

        # Apply class weights if provided
        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0).expand_as(loss)
            loss = loss * weight_expanded

        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)
    https://arxiv.org/abs/1901.05555

    Args:
        samples_per_class: List/array of sample counts per class
        num_classes: Number of classes
        loss_type: 'focal', 'softmax', or 'sigmoid'
        beta: Parameter for effective number calculation
        gamma: Focal loss gamma (only used if loss_type='focal')
    """

    def __init__(
        self, samples_per_class, num_classes, loss_type="focal", beta=0.999, gamma=2.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.gamma = gamma

        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(
            torch.tensor(beta), torch.tensor(samples_per_class, dtype=torch.float32)
        )

        # Compute weights
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # Normalize

        self.register_buffer("weights", weights)

        print(f"\nClass-Balanced Loss Weights (beta={beta}):")
        for i, w in enumerate(weights):
            print(f"  Class {i}: {w.item():.4f}")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]
        """
        if self.loss_type == "focal":
            return self._focal_loss(inputs, targets)
        elif self.loss_type == "softmax":
            return F.cross_entropy(inputs, targets, weight=self.weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _focal_loss(self, inputs, targets):
        """Focal loss with class-balanced weights."""
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # Apply class-balanced weights
        alpha_t = self.weights.gather(0, targets)
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


def get_criterion(
    strategy,
    class_weights=None,
    class_counts=None,
    num_classes=4,
    focal_gamma=2.0,
    effective_beta=0.999,
    label_smoothing=0.0,
):
    """
    Factory function to create the appropriate loss criterion.

    Args:
        strategy: "none", "inverse", "inverse_sqrt", "effective", "focal", "cb_focal"
        class_weights: Pre-computed class weights (for inverse/inverse_sqrt)
        class_counts: Sample counts per class (for cb_focal)
        num_classes: Number of classes
        focal_gamma: Gamma for focal loss
        effective_beta: Beta for effective number calculation
        label_smoothing: Label smoothing factor (0 = disabled)

    Returns:
        Loss criterion (nn.Module)
    """
    # Handle label smoothing
    if label_smoothing > 0 and strategy not in ["focal", "cb_focal"]:
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing, weight=weight_tensor
        )

    if strategy == "none":
        return nn.CrossEntropyLoss()

    elif strategy in ["inverse", "inverse_sqrt", "effective"]:
        if class_weights is None:
            raise ValueError(f"class_weights required for strategy '{strategy}'")
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight_tensor)

    elif strategy == "focal":
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return FocalLoss(alpha=weight_tensor, gamma=focal_gamma)

    elif strategy == "cb_focal":
        if class_counts is None:
            raise ValueError("class_counts required for 'cb_focal' strategy")
        return ClassBalancedLoss(
            samples_per_class=class_counts,
            num_classes=num_classes,
            loss_type="focal",
            beta=effective_beta,
            gamma=focal_gamma,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
