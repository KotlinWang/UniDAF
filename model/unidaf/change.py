import torch.nn.functional as F
import torch.nn as nn

import timm

from .locnet import LocNet
from .clfnet import ClfNet
from ._module.encoder import HybridEncoder

class Change(nn.Module):
    def __init__(self, backbone, loc_classes, clf_classes, h_dim, eval_size=1024, pretrained=True):
        super(Change, self).__init__()
        self.train_dino = 'dino' in backbone

        self.opt_backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        self.sar_backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        self.channels = self.opt_backbone.feature_info.channels()
        self.strides = self.opt_backbone.feature_info.reduction()

        # if self.train_dino:
        #     for param in self.opt_backbone.parameters():
        #         param.requires_grad = False
        #     for param in self.sar_backbone.parameters():
        #         param.requires_grad = False

        self.opt_encoder = HybridEncoder(in_channels=self.channels, h_dim=h_dim, feat_strides=self.strides, use_encoder_idx=[-1], eval_spatial_size=eval_size)
        self.sar_encoder = HybridEncoder(in_channels=self.channels, h_dim=h_dim, feat_strides=self.strides, use_encoder_idx=[-1], eval_spatial_size=eval_size)

        self.locnet = LocNet(h_dim, loc_classes)
        self.clfnet = ClfNet(h_dim, clf_classes)

    # def forward(self, pre_opt, post_opt, post_sar):
    def forward(self, inputs):
        bs = inputs.shape[0] // 3
        pre_opt, post_opt, post_sar = inputs[0:bs, ...], inputs[bs:2*bs, ...], inputs[2*bs:3*bs, ...]
        
        _, _, h, w = pre_opt.shape

        pre_opt_feats = self.opt_backbone(pre_opt)
        post_opt_feats = self.opt_backbone(post_opt)
        post_sar_feats = self.sar_backbone(post_sar)

        pre_opt_feats = self.opt_encoder(pre_opt_feats)
        post_opt_feats = self.opt_encoder(post_opt_feats)
        post_sar_feats = self.sar_encoder(post_sar_feats)

        loc = self.locnet(pre_opt_feats, (h, w))
        out_1, out_2, out_3, clf = self.clfnet(pre_opt_feats, post_opt_feats, post_sar_feats, (h, w))

        return loc, clf, out_1, out_2, out_3
