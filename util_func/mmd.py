"""
@Project : rsda
@File    : mmd.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午3:14
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf


def gaussian_kernel(x, y, sigma=1.0):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x_expand = x.view(x_size, 1, dim)
    y_expand = y.view(1, y_size, dim)

    distances = torch.sum((x_expand - y_expand) ** 2, 2)
    kernel = torch.exp(-distances / (2 * sigma))

    return kernel

def gaussian_kernel_imagelevel(x, y, sigma=1.0):
    """
    Args:
        x: (bs, h*w, k)
        y: (bs, h*w, k)
        sigma: 1.0

    Returns:

    """
    bs = x.size(0)
    x_size = x.size(1)
    y_size = y.size(1)
    dim = x.size(2)

    x_expand = x.view(bs, x_size, 1, dim)
    y_expand = y.view(bs, 1, y_size, dim)

    distances = torch.sum((x_expand - y_expand) ** 2, 3)
    kernel = torch.exp(-distances / (2 * sigma))

    return kernel


# MMD损失定义
def mmd_loss(source, target, kernel=gaussian_kernel):
    kernels = kernel(source, source) + kernel(target, target) - 2 * kernel(source, target)
    return torch.mean(kernels)


class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_s, feat_t):
        return mmd_loss(feat_s, feat_t, gaussian_kernel)


class ImageLevelMMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_s, feat_t):
        return mmd_loss(feat_s, feat_t, gaussian_kernel_imagelevel)


class MMDLoss2(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.ext_params = kwargs

    @staticmethod
    def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        total1 = total.unsqueeze(1).expand(int(total.size()[0]), int(total.size()[0]), int(total.size()[1]))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** _i) for _i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    @staticmethod
    def forward_linear(f_of_x, f_of_y):
        delta = f_of_x.float().mean(0) - f_of_y.float().mean(0)
        loss = delta.dot(delta.T) / delta.size(0)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.forward_linear(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            xx = torch.mean(kernels[:batch_size, :batch_size])
            yy = torch.mean(kernels[batch_size:, batch_size:])
            xy = torch.mean(kernels[:batch_size, batch_size:])
            yx = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(xx + yy - xy - yx)
            return loss
