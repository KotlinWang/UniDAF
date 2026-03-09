import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.functional import interpolate

import timm
from .help_funcs import TransformerDecoder, TwoLayerConv2d


###############################################################################
# main Functions
###############################################################################

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class BASE_Transformer_UNet(nn.Module):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        with_pos,
        resnet_stages_num=5,
        token_len=4,
        token_trans=True,
        enc_depth=1,
        dec_depth=1,
        dim_head=64,
        decoder_dim_head=64,
        tokenizer=True,
        if_upsample_2x=True,
        pool_mode="max",
        pool_size=2,
        backbone="resnet18",
        decoder_softmax=True,
        with_decoder_pos=None,
        with_decoder=True,
    ):
        super(BASE_Transformer_UNet, self).__init__()

        print("using UNet Transformer")

        self.token_len = token_len
        self.tokenizer = tokenizer
        self.token_trans = token_trans
        self.with_decoder = with_decoder
        self.with_pos = with_pos

        self.pre_backbone = timm.create_model(backbone, output_stride=32, out_indices=(1, 2, 3, 4), features_only=True, pretrained=True)
        self.post_backbone = timm.create_model(backbone, output_stride=32, out_indices=(1, 2, 3, 4), in_chans=input_nc, features_only=True, pretrained=True)
        
        self.upsamplex2 = nn.Upsample(scale_factor=2)

        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        # conv squeeze layers before transformer
        dim_5, dim_4, dim_3, dim_2 = 32, 32, 32, 32
        self.conv_squeeze_5 = nn.Sequential(
            nn.Conv2d(512, dim_5, kernel_size=1, padding=0, bias=False), nn.ReLU()
        )
        self.conv_squeeze_4 = nn.Sequential(
            nn.Conv2d(256, dim_4, kernel_size=1, padding=0, bias=False), nn.ReLU()
        )
        self.conv_squeeze_3 = nn.Sequential(
            nn.Conv2d(128, dim_3, kernel_size=1, padding=0, bias=False), nn.ReLU()
        )
        self.conv_squeeze_2 = nn.Sequential(
            nn.Conv2d(64, dim_2, kernel_size=1, padding=0, bias=False), nn.ReLU()
        )
        self.conv_squeeze_layers = [
            self.conv_squeeze_2,
            self.conv_squeeze_3,
            self.conv_squeeze_4,
            self.conv_squeeze_5,
        ]

        self.conv_token_5 = nn.Conv2d(
            dim_5, self.token_len, kernel_size=1, padding=0, bias=False
        )
        self.conv_token_4 = nn.Conv2d(
            dim_4, self.token_len, kernel_size=1, padding=0, bias=False
        )
        self.conv_token_3 = nn.Conv2d(
            dim_3, self.token_len, kernel_size=1, padding=0, bias=False
        )
        self.conv_token_2 = nn.Conv2d(
            dim_2, self.token_len, kernel_size=1, padding=0, bias=False
        )
        self.conv_tokens_layers = [
            self.conv_token_2,
            self.conv_token_3,
            self.conv_token_4,
            self.conv_token_5,
        ]

        self.conv_decode_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.conv_decode_layers = [
            self.conv_decode_2,
            self.conv_decode_3,
            self.conv_decode_4,
            self.conv_decode_5,
        ]

        if with_pos == "learned":
            self.pos_embedding_5 = nn.Parameter(
                torch.randn(1, self.token_len * 2, dim_5)
            )
            self.pos_embedding_4 = nn.Parameter(
                torch.randn(1, self.token_len * 2, dim_4)
            )
            self.pos_embedding_3 = nn.Parameter(
                torch.randn(1, self.token_len * 2, dim_3)
            )
            self.pos_embedding_2 = nn.Parameter(
                torch.randn(1, self.token_len * 2, dim_2)
            )
            self.pos_embedding_layers = [
                self.pos_embedding_2,
                self.pos_embedding_3,
                self.pos_embedding_4,
                self.pos_embedding_5,
            ]

        decoder_pos_size = 512 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == "learned":
            self.pos_embedding_decoder_5 = nn.Parameter(torch.randn(1, dim_5, 16, 16))
            self.pos_embedding_decoder_4 = nn.Parameter(torch.randn(1, dim_4, 32, 32))
            self.pos_embedding_decoder_3 = nn.Parameter(torch.randn(1, dim_3, 64, 64))
            self.pos_embedding_decoder_2 = nn.Parameter(
                torch.randn(1, dim_2, decoder_pos_size, decoder_pos_size)
            )
            self.pos_embedding_decoder_layers = [
                self.pos_embedding_decoder_2,
                self.pos_embedding_decoder_3,
                self.pos_embedding_decoder_4,
                self.pos_embedding_decoder_5,
            ]

        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer_5 = Transformer(
            dim=dim_5,
            depth=self.enc_depth,
            heads=4,
            dim_head=self.dim_head,
            mlp_dim=dim_5,
            dropout=0,
        )
        self.transformer_decoder_5 = TransformerDecoder(
            dim=dim_5,
            depth=4,
            heads=4,
            dim_head=self.decoder_dim_head,
            mlp_dim=dim_5,
            dropout=0,
            softmax=decoder_softmax,
        )
        self.transformer_4 = Transformer(
            dim=dim_4,
            depth=self.enc_depth,
            heads=4,
            dim_head=self.dim_head,
            mlp_dim=dim_4,
            dropout=0,
        )
        self.transformer_decoder_4 = TransformerDecoder(
            dim=dim_4,
            depth=4,
            heads=4,
            dim_head=self.decoder_dim_head,
            mlp_dim=dim_4,
            dropout=0,
            softmax=decoder_softmax,
        )
        self.transformer_3 = Transformer(
            dim=dim_3,
            depth=self.enc_depth,
            heads=8,
            dim_head=self.dim_head,
            mlp_dim=dim_3,
            dropout=0,
        )
        self.transformer_decoder_3 = TransformerDecoder(
            dim=dim_3,
            depth=8,
            heads=8,
            dim_head=self.decoder_dim_head,
            mlp_dim=dim_3,
            dropout=0,
            softmax=decoder_softmax,
        )
        self.transformer_2 = Transformer(
            dim=dim_2,
            depth=self.enc_depth,
            heads=1,
            dim_head=32,
            mlp_dim=dim_2,
            dropout=0,
        )
        self.transformer_decoder_2 = TransformerDecoder(
            dim=dim_2,
            depth=1,
            heads=1,
            dim_head=32,
            mlp_dim=dim_2,
            dropout=0,
            softmax=decoder_softmax,
        )
        self.transformer_layers = [
            self.transformer_2,
            self.transformer_3,
            self.transformer_4,
            self.transformer_5,
        ]
        self.transformer_decoder_layers = [
            self.transformer_decoder_2,
            self.transformer_decoder_3,
            self.transformer_decoder_4,
            self.transformer_decoder_5,
        ]

        self.conv_layer2_0 = TwoLayerConv2d(
            in_channels=128, out_channels=32, kernel_size=3
        )

        # # EXP NEW : Worked Better than BiT and changeformer
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(
            in_channels=32, out_channels=output_nc, kernel_size=3, padding=1
        )

        # FINALIZED CLASSIFIER: Works Best
        # self.classifier = nn.Conv2d(in_channels=32, out_channels=output_nc, kernel_size=3, padding=1)

        # # EXP NEW
        # self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=40, out_channels=16, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        #                                 nn.ReLU())
        # self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        #                                 nn.ReLU())
        # self.conv_layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #                                 nn.ReLU())
        # self.classifier = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(in_channels=5, out_channels=output_nc, kernel_size=3, padding=1))

    def _forward_semantic_tokens(self, x, layer=None):
        b, c, h, w = x.shape
        spatial_attention = self.conv_tokens_layers[layer](x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum("bln,bcn->blc", spatial_attention, x)
        return tokens

    def _forward_transformer(self, x, layer):
        if self.with_pos:
            x += self.pos_embedding_layers[layer]
        x = self.transformer_layers[layer](x)
        return x

    def _forward_transformer_decoder(self, x, m, layer):
        b, c, h, w = x.shape
        if self.with_decoder_pos == "learned":
            """print("Shape of x:", x.shape)
            print(
                "Shape of pos_embedding_decoder_layers:",
                self.pos_embedding_decoder_layers[layer].shape,
            )"""
            if self.with_decoder_pos == "learned":
                if x.shape[-2:] != self.pos_embedding_decoder_layers[layer].shape[-2:]:
                    resized_pos_embedding = interpolate(
                        self.pos_embedding_decoder_layers[layer],
                        size=x.shape[-2:],  # Match spatial dimensions of x
                        mode="bilinear",
                        align_corners=False,
                    )
                    x = x + resized_pos_embedding
                else:
                    x = x + self.pos_embedding_decoder_layers[layer]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer_decoder_layers[layer](x, m)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def _forward_trans_module(self, x1, x2, layer):
        x1 = self.conv_squeeze_layers[layer](x1)
        x2 = self.conv_squeeze_layers[layer](x2)
        token1 = self._forward_semantic_tokens(x1, layer)
        token2 = self._forward_semantic_tokens(x2, layer)
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens = self._forward_transformer(self.tokens_, layer)
        token1, token2 = self.tokens.chunk(2, dim=1)
        x1 = self._forward_transformer_decoder(x1, token1, layer)
        x2 = self._forward_transformer_decoder(x2, token2, layer)
        # return torch.abs(x1 - x2)

        # V1, V2
        # x1 = self._forward_transformer_decoder(x1, token2, layer)
        # x2 = self._forward_transformer_decoder(x2, token1, layer)
        # return torch.add(x1, x2)

        # V3
        diff_token = torch.abs(token2 - token1)
        diff_x = self.conv_decode_layers[layer](torch.cat([x1, x2], axis=1))
        x = self._forward_transformer_decoder(diff_x, diff_token, layer)
        return x

    def forward(self, x1, x2):
        # forward backbone resnet

        a_128, a_64, a_32, a_16 = self.pre_backbone(x1)
        b_128, b_64, b_32, b_16 = self.post_backbone(x2)

        #  level 5 in=256x16x16 out=32x16x16
        x1, x2 = a_16, b_16
        out_5 = self._forward_trans_module(x1, x2, layer=3)
        out_5 = self.upsamplex2(out_5)

        # level 4: in=128x32x32 out=32x32x32
        x1, x2 = a_32, b_32
        out_4 = self._forward_trans_module(x1, x2, layer=2)
        out_4 = out_4 + out_5
        out_4 = self.upsamplex2(out_4)
        out_4 = self.conv_layer4(out_4)

        # level 3: in=64x64x64 out=32x64x64
        x1, x2 = a_64, b_64
        out_3 = self._forward_trans_module(x1, x2, layer=1)
        out_3 = out_3 + out_4
        # out_3 = self.conv_layer3(torch.cat([out_3, out_4], axis=1))
        out_3 = self.upsamplex2(out_3)
        out_3 = self.conv_layer3(out_3)

        # level 2: in=64x128x128
        out_2 = self.conv_layer2_0(torch.cat([a_128, b_128], 1))
        out_2 = out_2 + out_3
        # out_2 = self.conv_layer2(torch.cat([out_2, out_3], axis=1))
        out_2 = self.upsamplex2(out_2)
        out_2 = self.conv_layer2(out_2)

        # print(out_2.shape, out_3.shape, out_4.shape, out_5.shape)
        # forward small cnn
        x = self.classifier(out_2)
        x = self.upsamplex2(x)

        return x


### Three layers UNet bottleneck -> siamese ###
""" # Encoder 1
        x_1 = x1
        enc1_1 = self.conv1(x_1)
        enc2_1 = self.conv2(enc1_1)
        enc3_1 = self.conv3(enc2_1)

        # Encoder 2
        x_2 = x2
        enc1_2 = self.conv1(x_2)
        enc2_2 = self.conv2(enc1_2)
        enc3_2 = self.conv3(enc2_2)

        enc3 = self.ca_skip_3(enc3_1, enc3_2)
        B_, C_, H_, W_ = enc3.shape
        enc3_i = enc3.view([B_, C_, H_*W_])
        enc3_i = self.transformer(enc3_i)
        enc3_i = enc3_i.view([B_, C_, H_, W_])
        enc3 = self.ca_skip_3(enc3_i,enc3)

        enc2 = self.ca_skip_2(enc2_1, enc2_2)
        dec8 = self.conv8(F.interpolate(enc3, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        enc1 = self.ca_skip_2(enc1_1, enc1_2)
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, 
                enc1
                ], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        out = self.res(dec10)
        return out
"""
