from torch import nn
import torch.nn.functional as F
import torch
import sys
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2,eps=0.001,momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self, x):
        return self.act(self.conv(x))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Ups(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True,input_same= False,stride=2):
        super().__init__()
        self.input_same = input_same
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up1 = nn.ConvTranspose2d(in_channels//2, in_channels // 2, kernel_size=2, stride=2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up1 = nn.ConvTranspose2d(in_channels//2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.input_same: x2 = self.up1(x2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class Model(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False,**arg):
        super(Model, self).__init__()
        self.encoder = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        return_nodes = {
            'layer1': 'feat1',
            'layer2': 'feat2',
            'layer3': 'feat3',
            'layer4': 'feat4'
        }
        self.extractor = create_feature_extractor(self.encoder, return_nodes=return_nodes)
        if n_channels != 3:
            self.extractor.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.decoder4 = ConvBlock(1024 * 2, 1024)
        self.decoder3 = ConvBlock(512 * 2, 512)
        self.decoder2 = ConvBlock(256* 2, 256)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
  
        factor = 2 if bilinear else 1
        self.cov3 = Conv(256,128)
        self.up3 = Ups(256, 128 // factor, bilinear,input_same=True)

        self.m1 =M(128,32,first=True,n_classes=n_classes)
        self.m2 =M(128,32,n_classes=n_classes)
        self.m3 =M(128,32,n_classes=n_classes)
        self.m4 =M(128,32,end=True,n_classes=n_classes)
        self.m5 =M0(128,32,n_classes=n_classes)


    def param(self):
        sam = torch.load(self.sam_checkpoint, map_location='cpu')
        self.feature.load_state_dict(sam)
        for param in self.feature.parameters():
            param.requires_grad = False

    def forward(self, x, show=None):
        size0 = x.shape[-2:]
        
        features = self.extractor(x)
        x1,x2,x3,x = features["feat1"],features["feat2"],features["feat3"],features["feat4"]
        
        x = self.upconv4(x)
        x = torch.cat((x, x3), dim=1)
        x = self.decoder4(x)

        x = self.upconv3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.decoder3(x)

        x = self.upconv2(x)
        x = torch.cat((x, x1), dim=1)
        x = self.decoder2(x)

        x_ = self.cov3(x)
        outx = self.up3(x, x_)
        size = outx.shape[-2:]
        x,logits1 = self.m1([outx])
        x,logits2 = self.m2([outx,x])
        x,logits3 = self.m3([outx, x])
        x1,logits4 = self.m5(outx)
        x1 = F.interpolate(x1, size=size)
        x, logits5 = self.m4([outx, x,x1])
        
        logits1 = F.interpolate(logits1, size=size0)
        logits2 = F.interpolate(logits2, size=size0)
        logits3 = F.interpolate(logits3, size=size0)
        logits4 = F.interpolate(logits4, size=size0)
        logits5 = F.interpolate(logits5, size=size0)
        return logits1,logits2,logits3,logits4,logits5
    
    
class M0(nn.Module):
    def __init__(self,in_channels, out_channels,n_classes, first=False,end=False):
        super(M0, self).__init__()
        c = 32
        self.out = Conv(in_channels, c)

        self.cov = Conv(c, out_channels)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        x = self.out(x)
        x1 = self.cov(x)
        logits = self.outc(x1)
        return x, logits


class M(nn.Module):
    def __init__(self,in_channels, out_channels,n_classes, first=False,end=False):
        super(M, self).__init__()

        self.out = Conv(in_channels, out_channels)
        c0 = out_channels if first else out_channels*2
        c = out_channels*3 if end else c0
        self.concat = Concat()

        self.cov = Conv(c, out_channels)
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, xs):
        x = self.out(xs[0])

        xs0 = [x]
        for _ in range(1,len(xs)):
            xs0.append(xs[_])
        x1 = self.cov(self.concat(xs0))
        logits2 = self.outc(x1)

        return x, logits2
