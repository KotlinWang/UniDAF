import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableLogitAdjustedCE(nn.Module):
    """
    Logit Adjustment Cross Entropy without class frequency.
    Supports logits of shape [N, C] or [N, C, H, W].
    """
    def __init__(self, num_classes, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits, targets):
        # 判断 logits 维度
        if logits.dim() == 2:
            # [N, C]
            adjusted_logits = logits + self.bias.unsqueeze(0)
        elif logits.dim() == 4:
            # [N, C, H, W]
            adjusted_logits = logits + self.bias.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

        loss = F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)
        return loss

