import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from .Mamba_backbone import Backbone_VSSM
from .classification.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ChangeDecoder import ChangeDecoder
from .SemanticDecoder import SemanticDecoder


class STMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, pretrained, **kwargs):
        super(STMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


       
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )
      
        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_opt_data, post_sar_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_opt_features = self.encoder(post_opt_data)
        post_sar_features = self.encoder(post_sar_data)

        post_features = []
        for i in range(len(post_opt_features)):
            post_features.append(post_opt_features[i] + post_sar_features[i])

        # Decoder processing - passing encoder outputs to the decoder
        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)
        
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_opt_data.size()[-2:], mode='bilinear')
       
        return output_building, output_damage