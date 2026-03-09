import torch.nn as nn
import torch.nn.functional as F

from ._module.base import ResBlock


class LocNet(nn.Module):
    def __init__(self, h_dim, out_classes):
        super(LocNet, self).__init__()
        self.trans_layer_4 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=h_dim, out_channels=h_dim, padding=1),
            nn.BatchNorm2d(h_dim), 
            nn.Mish(inplace=True)
            )
        self.trans_layer_3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=h_dim, out_channels=h_dim, padding=1),
            nn.BatchNorm2d(h_dim), 
            nn.Mish(inplace=True)
            )
        self.trans_layer_2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=h_dim, out_channels=h_dim, padding=1),
            nn.BatchNorm2d(h_dim), 
            nn.Mish(inplace=True)
            )
        self.trans_layer_1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=h_dim, out_channels=h_dim, padding=1),
            nn.BatchNorm2d(h_dim), 
            nn.Mish(inplace=True)
            )

        self.smooth_layer_13 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 
        self.smooth_layer_12 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 
        self.smooth_layer_11 = ResBlock(in_channels=h_dim, out_channels=h_dim, stride=1) 

        self.loc_out = nn.Conv2d(in_channels=h_dim, out_channels=out_classes, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, feats, hw):
        p4_loc = self.trans_layer_4(feats[-1])

        p3_loc = self.trans_layer_3(feats[-2])
        p3_loc = self._upsample_add(p4_loc, p3_loc)
        p3_loc = self.smooth_layer_13(p3_loc)

        p2_loc = self.trans_layer_2(feats[-3])
        p2_loc = self._upsample_add(p3_loc, p2_loc)
        p2_loc = self.smooth_layer_12(p2_loc)

        p1_loc = self.trans_layer_1(feats[-4])
        p1_loc = self._upsample_add(p2_loc, p1_loc)
        p1_loc = self.smooth_layer_11(p1_loc)

        output_loc = self.loc_out(p1_loc)
        output_loc = F.interpolate(output_loc, size=hw, mode='bilinear')

        return output_loc
