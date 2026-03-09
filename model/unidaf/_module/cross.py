import torch
import torch.nn as nn
from typing import Optional


class CrossAttentionFusion(nn.Module):
    """
    Fuse two modality feature maps using cross-attention.
    Supports single-modality input.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mha_sar2opt = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mha_opt2sar = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats_opt: Optional[torch.Tensor], feats_sar: Optional[torch.Tensor], index: int):
        """
        feat_sar, feat_opt: (B,C,H,W)
        returns: fused feature (B,C,H,W)
        """
        # Only one modality present
        if feats_sar is None and feats_opt is None:
            raise ValueError("At least one modality must be provided.")
        if feats_sar is None:
            return feats_opt[index]
        if feats_opt is None:
            return feats_sar[index]

        # Get the feature maps for the current index
        feat_sar = feats_sar[index]
        feat_opt = feats_opt[index]

        B, C, H, W = feat_sar.shape
        N = H * W

        # flatten spatial dimension
        sar_flat = feat_sar.flatten(2).transpose(1, 2)  # (B, N, C)
        opt_flat = feat_opt.flatten(2).transpose(1, 2)  # (B, N, C)

        # Cross-attention: SAR queries, OPT keys/values
        sar_attn, _ = self.mha_sar2opt(query=sar_flat, key=opt_flat, value=opt_flat)
        opt_attn, _ = self.mha_opt2sar(query=opt_flat, key=sar_flat, value=sar_flat)

        # Fuse: add residual + LayerNorm
        fused_sar = self.norm(sar_flat + sar_attn)
        fused_opt = self.norm(opt_flat + opt_attn)

        # combine two modalities (average)
        fused = 0.5 * (fused_sar + fused_opt)  # (B, N, C)

        # reshape back to (B,C,H,W)
        fused = fused.transpose(1,2).view(B, C, H, W)
        fused = self.proj(fused)
        return fused


class GatedFusion(nn.Module):
    """
    Fuse two modalities with dynamic gating.
    If only one modality present, returns that feature (no gating needed).
    If both present: compute scalar gates via pooled features -> MLP -> sigmoid.
    Optionally returns fused feature for each scale.
    """
    def __init__(self, channels, hidden=128):
        super().__init__()
        # per-modality gating MLPs (shared across spatial scales)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_sar = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.mlp_opt = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # small residual after fusion
        self.post = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(inplace=True))

    def forward(self, feats_opt: Optional[torch.Tensor], feats_sar: Optional[torch.Tensor], index: int):
        # Cases: only one modality present
        if feats_sar is None and feats_opt is None:
            raise ValueError("At least one modality must be provided.")
        if feats_sar is None:
            return feats_opt[index]
        if feats_opt is None:
            return feats_sar[index]

        # Get the feature maps for the current index
        feat_sar = feats_sar[index]
        feat_opt = feats_opt[index]

        # both present: compute gate scalars per feature map
        # input shapes: (B, C, H, W)
        b, c, _, _ = feat_sar.shape
        s_pool = self.pool(feat_sar).view(b, c)  # (B, C)
        o_pool = self.pool(feat_opt).view(b, c)
        s_logit = self.mlp_sar(s_pool)  # (B,1)
        o_logit = self.mlp_opt(o_pool)
        # normalize to sum-to-1 weights across modalities (softmax)
        gates = torch.cat([s_logit, o_logit], dim=1)  # (B,2)
        gates = torch.softmax(gates, dim=1)  # (B,2)
        alpha = gates[:, 0].view(b, 1, 1, 1)
        beta = gates[:, 1].view(b, 1, 1, 1)
        fused = alpha * feat_sar + beta * feat_opt
        fused = fused + self.post(fused)
        return fused


class GatedFusion2(nn.Module):
    """
    Fuse two modalities with dynamic gating.
    If only one modality present, returns that feature (no gating needed).
    If both present: compute scalar gates via pooled features -> MLP -> sigmoid.
    Optionally returns fused feature for each scale.
    """
    def __init__(self, h_dim):
        super().__init__()
        # per-modality gating MLPs (shared across spatial scales)
        self.post = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=h_dim * 2, out_channels=h_dim, padding=1),
            nn.BatchNorm2d(h_dim), 
            nn.Mish(inplace=True)
            )

    def forward(self, feats_opt: Optional[torch.Tensor], feats_sar: Optional[torch.Tensor], index: int):
        # Cases: only one modality present
        if feats_sar is None and feats_opt is None:
            raise ValueError("At least one modality must be provided.")
        if feats_sar is None:
            return feats_opt[index]
        if feats_opt is None:
            return feats_sar[index]

        # Get the feature maps for the current index
        feat_sar = feats_sar[index]
        feat_opt = feats_opt[index]

        sar_opt = torch.cat([feat_sar, feat_opt], dim=1)
        fused = self.post(sar_opt)
        return fused
