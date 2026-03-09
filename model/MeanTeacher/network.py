# encoding: utf-8
import sys
sys.path.append('/home/songjian/project/BRIGHT/essd')

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from model.MeanTeacher.base_model import resnet50
from model.unidaf3.change import Change


class MTNetwork(nn.Module):
    def __init__(self):
        super(MTNetwork, self).__init__()
        # student network
        self.branch1 = Change('resnet18.fb_swsl_ig1b_ft_in1k', 2, 4, 128)
        # teacher network
        self.branch2 = Change('resnet18.fb_swsl_ig1b_ft_in1k', 2, 4, 128)

        self.ema_decay = 0.99

        # detach the teacher model
        for param in self.branch2.parameters():
            param.detach_()

    def forward(self, data, step=1, cur_iter=None):
        if not self.training:
            _, pred1, _, _, _ = self.branch1(data)
            return pred1

        # copy the parameters from teacher to student
        if cur_iter == 0:
            for t_param, s_param in zip(self.branch2.parameters(), self.branch1.parameters()):
                t_param.data.copy_(s_param.data)

        out_loc1, out_clf1, o11, o21, o31 = self.branch1(data)

        with torch.no_grad():
            out_loc2, out_clf2, o12, o22, o32 = self.branch2(data)

        if step == 1:
            self._update_ema_variables(self.ema_decay, cur_iter)

        return (out_loc1, out_clf1, o11, o21, o31), (out_loc2, out_clf2, o12, o22, o32)

    def _update_ema_variables(self, ema_decay, cur_step):
        for t_param, s_param in zip(self.branch2.parameters(), self.branch1.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1 - ema_decay)