import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np


class PrototypeContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.3, ignore_label=-1):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, Proto, feat, labels):
        """
        Args:
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Proto.requires_grad and not labels.requires_grad and feat.requires_grad
        if feat.dim() != 2:
            k = feat.size(1)
            feat = feat.permute(0, 2, 3, 1).reshape(-1, k)
        if labels.dim() != 1:
            labels = labels.reshape(-1, )
        assert feat.dim() == 2 and labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != self.ignore_label)
        labels = labels[mask]
        feat = feat[mask]

        feat = tnf.normalize(feat, p=2, dim=1)
        Proto = tnf.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.temperature
        
        loss = self.ce_criterion(logits, labels)
        return loss