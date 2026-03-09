import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ._module.base import BasicConv2d, ResBlock
from ._module.msla import GFE
from ._module.cross import GatedFusion2

class ClfHead(nn.Module):
    def __init__(self, h_dim, out_classes):
        super().__init__()
        self.h_dim = h_dim
        self.conv = BasicConv2d(h_dim, h_dim * 3, kernel_size=1)
        
        self.head1 = nn.Conv2d(in_channels=h_dim, out_channels=1, kernel_size=1)
        self.head2 = nn.Conv2d(in_channels=h_dim, out_channels=1, kernel_size=1)
        self.head3 = nn.Conv2d(in_channels=h_dim, out_channels=1, kernel_size=1)

        self.att = BasicConv2d(3, 1, kernel_size=3)
        
        self.head = nn.Sequential(
            BasicConv2d(h_dim, h_dim, kernel_size=3),
            nn.Dropout2d(p=0.3, inplace=True),  # 更强的Dropout
            nn.Conv2d(in_channels=h_dim, out_channels=out_classes, kernel_size=1)
        )

    def forward(self, x, hw):
        x_rep = self.conv(x)
        x1, x2, x3 = torch.split(x_rep, self.h_dim, dim=1)
        
        # 使用共享卷积来进一步提取特征
        head1 = self.head1(x1)
        head2 = self.head2(x2)
        head3 = self.head3(x3)

        head_weight = nn.Sigmoid()(self.att(torch.cat([head1, head2, head3], dim=1)))
        x_head = x * head_weight.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法
        
        head = self.head(x_head)

        # 上采样到目标大小
        out_1 = F.interpolate(head1, size=hw, mode='bilinear')
        out_2 = F.interpolate(head2, size=hw, mode='bilinear')
        out_3 = F.interpolate(head3, size=hw, mode='bilinear')
        out = F.interpolate(head, size=hw, mode='bilinear')

        return out_1, out_2, out_3, out

        

class ClfNet(nn.Module):
    def __init__(self, h_dim, out_classes):
        super().__init__()
        self.post_4 = GatedFusion2(h_dim)
        self.post_3 = GatedFusion2(h_dim)
        self.post_2 = GatedFusion2(h_dim)
        self.post_1 = GatedFusion2(h_dim)

        self.msla_4 = GFE(in_dim=h_dim * 2, out_dim=h_dim, num_heads=8)
        self.msla_3 = GFE(in_dim=h_dim * 2, out_dim=h_dim, num_heads=8)
        self.msla_2 = GFE(in_dim=h_dim * 2, out_dim=h_dim, num_heads=8)
        self.msla_1 = GFE(in_dim=h_dim * 2, out_dim=h_dim, num_heads=8)

        self.smooth_layer_23 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 
        self.smooth_layer_22 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 
        self.smooth_layer_21 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 

        self.clf_head = ClfHead(h_dim, out_classes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_opt_feats: Optional[torch.Tensor], post_opt_feats: Optional[torch.Tensor]=None, post_sar_feats: Optional[torch.Tensor]=None , hw: tuple=None):
        post_output_4 = self.post_4(post_opt_feats, post_sar_feats, -1)
        combined_4 = torch.cat([pre_opt_feats[-1], post_output_4], dim=1)
        p4 = self.msla_4(combined_4)

        post_output_3 = self.post_3(post_opt_feats, post_sar_feats, -2)
        combined_3 = torch.cat([pre_opt_feats[-2], post_output_3], dim=1)
        p3 = self.msla_3(combined_3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_23(p3)

        post_output_2 = self.post_2(post_opt_feats, post_sar_feats, -3)
        combined_2 = torch.cat([pre_opt_feats[-3], post_output_2], dim=1)
        p2 = self.msla_2(combined_2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_22(p2)

        post_output_1 = self.post_1(post_opt_feats, post_sar_feats, -4)
        combined_1 = torch.cat([pre_opt_feats[-4], post_output_1], dim=1)
        p1 = self.msla_1(combined_1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_21(p1)


        out_1, out_2, out_3, clf_out = self.clf_head(p1, hw)

        return out_1, out_2, out_3, clf_out
